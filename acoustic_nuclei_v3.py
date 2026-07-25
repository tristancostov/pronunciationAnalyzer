#!/usr/bin/env python3
"""Reusable text-independent V3 detector and training entry point.

V3 keeps every precision-oriented V2 event, proposes additional regions with
Sylber, classifies those regions from acoustic evidence, and adds only
non-conflicting high-confidence rescues. The shipped research artifact is
trained on reviewed development speakers; 8fori is deliberately excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from joblib import dump, load
import librosa
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acoustic_nuclei_v2 import detect_nuclei_v2, production_v2_config
from evaluate_sylber_zero_shot import (
    annotation,
    encode,
    load_model as load_sylber,
    sylber_segments,
)
from experiment_sylber_segment_classifier import (
    build_candidates,
    candidate_features,
    choose_threshold,
    scores,
)
from prepare_nucleus_annotations import find_audio


V3_SCHEMA_VERSION = 1
V3_NORM_THRESHOLD = 2.80
V3_MERGE_THRESHOLD = 0.80
V3_SCORE_THRESHOLD = 0.030108361862299777
V3_FUSION_DISTANCE_SECONDS = 0.065
V3_ASR_QUOTA_SLACK = 0
V3_TRAINING_RECORDINGS = (
    "3local", "6fori", "7fori", "2fori_chunk",
)
RUSSIAN_VOWELS = frozenset("аеёиоуыэюя")


@dataclass
class NucleusV3Detection:
    duration_seconds: float
    times: list[float]
    event_scores: list[float]
    v2_times: list[float]
    sylber_candidate_times: list[float]
    sylber_candidate_scores: list[float]
    rescued_times: list[float]
    asr_pruned: bool = False
    pruned_times: list[float] | None = None

    def to_dict(self):
        return {
            "algorithm": "acoustic_nuclei_v3",
            "text_independent": True,
            "duration_seconds": self.duration_seconds,
            "count": len(self.times),
            "times_seconds": self.times,
            "event_scores": self.event_scores,
            "v2_count": len(self.v2_times),
            "rescued_count": len(self.rescued_times),
            "rescued_times_seconds": self.rescued_times,
            "asr_pruned": self.asr_pruned,
            "pruned_times_seconds": self.pruned_times or [],
            "sylber_candidates": [
                {"time_seconds": time, "score": score}
                for time, score in zip(
                    self.sylber_candidate_times,
                    self.sylber_candidate_scores,
                )
            ],
        }


def model_factory():
    return make_pipeline(
        StandardScaler(),
        SVC(C=1.5, gamma="scale", class_weight="balanced"),
    )


def automatic_word_nucleus_count(item: dict) -> int:
    """Estimate a loose quota from the ASR token, never from user text."""
    explicit = item.get("textNucleusCount")
    if explicit is not None:
        return max(int(explicit or 0), 0)
    word = str(item.get("word", "")).lower()
    if word:
        return sum(character in RUSSIAN_VOWELS for character in word)
    return max(int(item.get("syllableCount", 0) or 0), 0)


def train_artifact(root: Path, output: Path, device: str):
    sylber = load_sylber(device)
    recordings = []
    for name in V3_TRAINING_RECORDINGS:
        if name == "2fori_chunk":
            audio_path = (
                root / "audio" / "review_chunks"
                / "2fori_0000_1743.wav"
            )
            annotation_path = (
                root / "analysis" / "v3_annotations" / "chunks"
                / "2fori_0000_1743_verified.TextGrid"
            )
        else:
            audio_path = find_audio(root / "audio", name)
            annotation_path = (
                root / "analysis" / "v2_annotations"
                / f"{name}_v2_reviewed.TextGrid"
            )
        signal, sample_rate = librosa.load(
            audio_path, sr=16000, mono=True
        )
        detection = detect_nuclei_v2(
            signal, sample_rate, production_v2_config()
        )
        states = encode(sylber, signal, device)
        reference, span = annotation(annotation_path)
        recordings.append(build_candidates(
            name=name,
            states=states,
            duration=len(signal) / sample_rate,
            detection=detection,
            reference=reference,
            span=span,
            config=(V3_NORM_THRESHOLD, V3_MERGE_THRESHOLD),
        ))
        print(f"Prepared {name}")
    score_threshold, fusion_distance, threshold_cv = choose_threshold(
        model_factory, recordings, "v2_priority_union"
    )
    model = model_factory()
    model.fit(
        np.vstack([item.features for item in recordings]),
        np.concatenate([item.labels for item in recordings]),
    )
    artifact = {
        "schema_version": V3_SCHEMA_VERSION,
        "algorithm": "acoustic_nuclei_v3",
        "text_independent": True,
        "training_recordings": V3_TRAINING_RECORDINGS,
        "sylber_norm_threshold": V3_NORM_THRESHOLD,
        "sylber_merge_threshold": V3_MERGE_THRESHOLD,
        "score_threshold": score_threshold,
        "fusion_distance_seconds": fusion_distance,
        "threshold_cross_validation": threshold_cv,
        "asr_quota_slack": V3_ASR_QUOTA_SLACK,
        "model": model,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, output, compress=3)
    print(f"Saved V3 artifact: {output}")
    return artifact


def detect_signal(
    signal: np.ndarray,
    sample_rate: int,
    artifact: dict,
    sylber,
    device: str,
) -> NucleusV3Detection:
    signal = np.asarray(signal, dtype=np.float32)
    if sample_rate != 16000:
        signal = librosa.resample(
            signal, orig_sr=sample_rate, target_sr=16000
        )
        sample_rate = 16000
    duration = len(signal) / sample_rate
    detection = detect_nuclei_v2(
        signal, sample_rate, production_v2_config()
    )
    states = encode(sylber, signal, device)
    segments = sylber_segments(
        states,
        float(artifact["sylber_norm_threshold"]),
        float(artifact["sylber_merge_threshold"]),
    )
    seconds_per_hidden = duration / max(len(states), 1)
    candidate_times = np.asarray([
        (start + end) * 0.5 * seconds_per_hidden
        for start, end in segments
    ], dtype=float)
    features = candidate_features(
        states, segments, duration, detection
    )
    values = (
        scores(artifact["model"], features)
        if len(features) else np.array([], dtype=float)
    )
    selected_indices = np.flatnonzero(
        values >= float(artifact["score_threshold"])
    )
    v2_times = [float(value) for value in detection.times]
    event_pairs = []
    for base_index, candidate in enumerate(detection.nuclei):
        near = np.flatnonzero(
            np.abs(candidate_times - candidate.time_seconds) <= 0.060
        )
        acoustic_score = (
            float(np.max(values[near])) if len(near)
            else float(artifact["score_threshold"]) - 1.0
        )
        event_pairs.append([
            float(candidate.time_seconds),
            acoustic_score + 0.12 * float(candidate.confidence),
        ])
    distance = float(artifact["fusion_distance_seconds"])
    for index in selected_indices:
        time = float(candidate_times[index])
        if all(abs(time - existing[0]) >= distance for existing in event_pairs):
            event_pairs.append([time, float(values[index])])
    event_pairs.sort()
    combined = [time for time, _ in event_pairs]
    event_scores = [quality for _, quality in event_pairs]
    rescued = [
        value for value in combined
        if all(
            abs(value - base)
            >= float(artifact["fusion_distance_seconds"])
            for base in v2_times
        )
    ]
    return NucleusV3Detection(
        duration_seconds=duration,
        times=combined,
        event_scores=event_scores,
        v2_times=v2_times,
        sylber_candidate_times=candidate_times.astype(float).tolist(),
        sylber_candidate_scores=values.astype(float).tolist(),
        rescued_times=rescued,
    )


def apply_asr_overcount_pruning(
    detection: NucleusV3Detection,
    word_analysis_payload: dict,
    quota_slack: int = V3_ASR_QUOTA_SLACK,
) -> NucleusV3Detection:
    """Remove only low-quality events that exceed an ASR word's loose quota."""
    words = [
        (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            automatic_word_nucleus_count(item),
        )
        for item in word_analysis_payload.get("wordAnalysis", [])
        if float(item.get("end", 0.0)) > float(item.get("start", 0.0))
    ]
    assignments = {index: [] for index in range(len(words))}
    unassigned = []
    for event_index, (time, quality) in enumerate(zip(
        detection.times, detection.event_scores
    )):
        eligible = [
            (
                abs(time - (start + end) * 0.5)
                / max(end - start, 0.05),
                word_index,
            )
            for word_index, (start, end, _) in enumerate(words)
            if start <= time <= end
        ]
        if eligible:
            assignments[min(eligible)[1]].append(
                (quality, time, event_index)
            )
        else:
            unassigned.append((quality, time, event_index))
    kept = {index for _, _, index in unassigned}
    for word_index, values_in_word in assignments.items():
        written_count = words[word_index][2]
        quota = 0 if written_count == 0 else written_count + quota_slack
        for _, _, event_index in sorted(
            values_in_word, reverse=True
        )[:quota]:
            kept.add(event_index)
    original_times = detection.times
    detection.times = [
        time for index, time in enumerate(original_times) if index in kept
    ]
    detection.event_scores = [
        value for index, value in enumerate(detection.event_scores)
        if index in kept
    ]
    detection.pruned_times = [
        time for index, time in enumerate(original_times)
        if index not in kept
    ]
    detection.rescued_times = [
        time for time in detection.rescued_times
        if time in detection.times
    ]
    detection.asr_pruned = True
    return detection


def detect_file(path: Path, artifact_path: Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    artifact = load(artifact_path)
    if artifact.get("schema_version") != V3_SCHEMA_VERSION:
        raise ValueError("Unsupported V3 model artifact schema")
    signal, sample_rate = librosa.load(path, sr=16000, mono=True)
    sylber = load_sylber(device)
    return detect_signal(
        signal, sample_rate, artifact, sylber, device
    )


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument(
        "--output", type=Path,
        default=root / "models" / "nuclei_v3_svm.joblib",
    )
    detect = subparsers.add_parser("detect")
    detect.add_argument("audio", type=Path)
    detect.add_argument(
        "--model", type=Path,
        default=root / "models" / "nuclei_v3_svm.joblib",
    )
    detect.add_argument("--output", type=Path)
    detect.add_argument(
        "--word-analysis-json", type=Path,
        help="Optional automatic ASR wordAnalysis JSON for conservative pruning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.command == "train":
        train_artifact(root, args.output, device)
        return
    result = detect_file(args.audio, args.model, device)
    if args.word_analysis_json:
        payload = json.loads(
            args.word_analysis_json.read_text(encoding="utf-8")
        )
        result = apply_asr_overcount_pruning(result, payload)
    rendered = json.dumps(
        result.to_dict(), ensure_ascii=False, indent=2
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
