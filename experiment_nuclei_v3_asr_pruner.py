#!/usr/bin/env python3
"""Conservative ASR over-count pruning on top of the best acoustic V3."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acoustic_nuclei_v2 import detect_nuclei_v2, production_v2_config
from acoustic_nuclei_v3 import automatic_word_nucleus_count
from evaluate_nuclei_v2 import match_events
from evaluate_sylber_zero_shot import annotation, encode, load_model
from experiment_sylber_segment_classifier import (
    build_candidates,
    choose_threshold,
    fast_event_metrics,
    pooled,
    scores,
)
from prepare_nucleus_annotations import find_audio


@dataclass
class Data:
    candidate: object
    words: list[tuple[float, float, int]]
    base_confidence: np.ndarray


@dataclass(frozen=True)
class RecordingSpec:
    name: str
    audio: Path
    annotation: Path
    word_analysis: Path


def factory():
    return make_pipeline(
        StandardScaler(),
        SVC(C=1.5, gamma="scale", class_weight="balanced"),
    )


def load_words(path, span):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        (
            max(float(item["start"]), span[0]),
            min(float(item["end"]), span[1]),
            automatic_word_nucleus_count(item),
        )
        for item in payload.get("wordAnalysis", [])
        if float(item["end"]) >= span[0]
        and float(item["start"]) <= span[1]
    ]


def acoustic_events(data, values, threshold, distance):
    candidate = data.candidate
    selected_indices = np.flatnonzero(values >= threshold)
    events = []
    for base_index, time in enumerate(candidate.base_times):
        near = np.flatnonzero(np.abs(candidate.times - time) <= 0.060)
        acoustic_score = (
            float(np.max(values[near])) if len(near)
            else float(threshold) - 1.0
        )
        confidence = (
            float(data.base_confidence[base_index])
            if base_index < len(data.base_confidence) else 0.0
        )
        events.append([float(time), acoustic_score + 0.12 * confidence])
    for index in selected_indices:
        time = float(candidate.times[index])
        if all(abs(time - existing[0]) >= distance for existing in events):
            events.append([time, float(values[index])])
    return sorted(events)


def prune_overcounts(events, words, padding, slack):
    if slack >= 99:
        return [time for time, _ in events]
    assignments = {index: [] for index in range(len(words))}
    unassigned = []
    for event_index, (time, quality) in enumerate(events):
        eligible = [
            (
                abs(time - (start + end) * 0.5)
                / max(end - start, 0.05),
                word_index,
            )
            for word_index, (start, end, _) in enumerate(words)
            if start - padding <= time <= end + padding
        ]
        if eligible:
            assignments[min(eligible)[1]].append(
                (quality, time, event_index)
            )
        else:
            unassigned.append((quality, time, event_index))
    kept_indices = {event_index for _, _, event_index in unassigned}
    for word_index, values in assignments.items():
        written_count = words[word_index][2]
        quota = 0 if written_count == 0 else written_count + slack
        for _, _, event_index in sorted(values, reverse=True)[:quota]:
            kept_indices.add(event_index)
    return sorted(
        time for index, (time, _) in enumerate(events)
        if index in kept_indices
    )


def inner_predictions(data, training_indices, threshold, distance):
    output = {}
    for held_index in training_indices:
        train = [
            data[index].candidate for index in training_indices
            if index != held_index
        ]
        model = factory()
        model.fit(
            np.vstack([item.features for item in train]),
            np.concatenate([item.labels for item in train]),
        )
        held = data[held_index]
        value = scores(model, held.candidate.features)
        output[held_index] = acoustic_events(
            held, value, threshold, distance
        )
    return output


def choose_pruner(data, training_indices, event_map):
    best = None
    for padding in (0.00, 0.02, 0.05):
        for slack in (0, 1, 2, 99):
            metrics = []
            for index in training_indices:
                predicted = prune_overcounts(
                    event_map[index], data[index].words, padding, slack
                )
                metrics.append(fast_event_metrics(
                    predicted, data[index].candidate.reference
                ))
            aggregate = pooled(metrics)
            rank = (
                aggregate["f1"], aggregate["precision"],
                aggregate["recall"], -slack, -padding,
            )
            if best is None or rank > best[0]:
                best = (rank, padding, slack, aggregate)
    return best[1:]


def recording_specs(root: Path, include_2fori_chunk: bool):
    specs = [
        RecordingSpec(
            name=name,
            audio=find_audio(root / "audio", name),
            annotation=(
                root / "analysis" / "v2_annotations"
                / f"{name}_v2_reviewed.TextGrid"
            ),
            word_analysis=(
                root / "analysis" / f"{name}_syllable_analysis.json"
            ),
        )
        for name in ("3local", "6fori", "7fori")
    ]
    if include_2fori_chunk:
        specs.append(RecordingSpec(
            name="2fori_chunk",
            audio=(
                root / "audio" / "review_chunks"
                / "2fori_0000_1743.wav"
            ),
            annotation=(
                root / "analysis" / "v3_annotations" / "chunks"
                / "2fori_0000_1743_verified.TextGrid"
            ),
            word_analysis=(
                root / "analysis" / "2fori_syllable_analysis.json"
            ),
        ))
    return specs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-2fori-chunk",
        action="store_true",
        help="Add the manually reviewed 17.43-second 2fori speaker sample.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compact_metrics(metrics):
    return {
        key: metrics[key] for key in (
            "tp", "fp", "fn", "precision", "recall", "f1",
            "timing_median_ms",
        )
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    specs = recording_specs(root, args.include_2fori_chunk)
    names = tuple(spec.name for spec in specs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sylber = load_model(device)
    data = []
    for spec in specs:
        signal, sample_rate = librosa.load(
            spec.audio, sr=16000, mono=True
        )
        states = encode(sylber, signal, device)
        detection = detect_nuclei_v2(
            signal, sample_rate, production_v2_config()
        )
        reference, span = annotation(
            spec.annotation
        )
        candidate = build_candidates(
            spec.name, states, len(signal) / sample_rate, detection,
            reference, span, (2.8, 0.8),
        )
        confidence = np.asarray([
            item.confidence for item in detection.nuclei
            if span[0] <= item.time_seconds <= span[1]
        ])
        data.append(Data(
            candidate,
            load_words(
                spec.word_analysis,
                span,
            ),
            confidence,
        ))
        print(f"Prepared {spec.name}")

    folds = []
    for held_index, held_name in enumerate(names):
        training_indices = [
            index for index in range(len(names)) if index != held_index
        ]
        training = [data[index].candidate for index in training_indices]
        threshold, distance, acoustic_inner = choose_threshold(
            factory, training, "v2_priority_union"
        )
        inner_map = inner_predictions(
            data, training_indices, threshold, distance
        )
        padding, slack, pruner_inner = choose_pruner(
            data, training_indices, inner_map
        )
        model = factory()
        model.fit(
            np.vstack([item.features for item in training]),
            np.concatenate([item.labels for item in training]),
        )
        held = data[held_index]
        value = scores(model, held.candidate.features)
        raw_events = acoustic_events(held, value, threshold, distance)
        raw_predicted = [time for time, _ in raw_events]
        predicted = prune_overcounts(
            raw_events, held.words, padding, slack
        )
        acoustic_metrics = match_events(
            raw_predicted, held.candidate.reference, 0.050
        )
        metrics = match_events(
            predicted, held.candidate.reference, 0.050
        )
        folds.append({
            "heldout": held_name,
            "threshold": threshold,
            "fusion_distance_seconds": distance,
            "word_padding_seconds": padding,
            "quota_slack": slack,
            "events_before_pruning": len(raw_events),
            "events_after_pruning": len(predicted),
            "acoustic_inner_validation": acoustic_inner,
            "pruner_inner_validation": pruner_inner,
            "acoustic_metrics_50ms": compact_metrics(acoustic_metrics),
            "metrics_50ms": compact_metrics(metrics),
        })
    payload = {
        "experiment": "automatic_asr_conservative_overcount_pruner",
        "user_supplied_transcript": False,
        "text_independent": False,
        "evaluation": "nested leave-one-recording-out; +/-50 ms",
        "recordings": list(names),
        "folds": folds,
        "acoustic_aggregate_50ms": pooled([
            fold["acoustic_metrics_50ms"] for fold in folds
        ]),
        "aggregate_50ms": pooled([
            fold["metrics_50ms"] for fold in folds
        ]),
    }
    output = args.output
    if output is None:
        filename = (
            "nuclei_v3_asr_pruner_four_recordings.json"
            if args.include_2fori_chunk
            else "nuclei_v3_asr_pruner_true_manual.json"
        )
        output = root / "results" / filename
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
