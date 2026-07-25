#!/usr/bin/env python3
"""Automatic-ASR-assisted nucleus detection without user-provided text.

The recognized word stream supplies only time windows and a spelling-derived
syllable quota. A speaker-held-out acoustic SVM ranks Sylber regions inside
each word; precision-oriented V2 events get priority. This is reported
separately from the text-independent detector because ASR errors can influence
the result, even though the user never enters a transcript.
"""

from __future__ import annotations

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
from evaluate_nuclei_v2 import match_events
from evaluate_sylber_zero_shot import annotation, encode, load_model
from experiment_sylber_segment_classifier import (
    build_candidates,
    fast_event_metrics,
    pooled,
    scores,
)
from prepare_nucleus_annotations import find_audio


@dataclass
class AssistedRecording:
    name: str
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    reference: list[float]
    base_times: np.ndarray
    base_confidence: np.ndarray
    words: list[tuple[float, float, int]]


def factory():
    return make_pipeline(
        StandardScaler(),
        SVC(C=1.5, gamma="scale", class_weight="balanced"),
    )


def load_words(path: Path, span):
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for item in payload.get("wordAnalysis", []):
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        count = int(item.get("syllableCount", 0) or 0)
        if end < span[0] or start > span[1] or count <= 0:
            continue
        words.append((max(start, span[0]), min(end, span[1]), count))
    return words


def make_recording(root, name, states, duration, detection, reference, span):
    candidate = build_candidates(
        name, states, duration, detection, reference, span, (2.8, 0.8)
    )
    base = [
        item for item in detection.nuclei
        if span[0] <= item.time_seconds <= span[1]
    ]
    return AssistedRecording(
        name=name,
        times=candidate.times,
        features=candidate.features,
        labels=candidate.labels,
        reference=reference,
        base_times=np.asarray([item.time_seconds for item in base]),
        base_confidence=np.asarray([item.confidence for item in base]),
        words=load_words(
            root / "analysis" / f"{name}_syllable_analysis.json", span
        ),
    )


def word_quota_predictions(
    item, values, threshold, padding, slack, distance,
):
    kept = []
    used_sylber = set()
    used_base = set()
    for start, end, expected in item.words:
        quota = expected + slack
        base_indices = [
            index for index, time in enumerate(item.base_times)
            if index not in used_base
            and start - padding <= time <= end + padding
        ]
        sylber_indices = [
            index for index, time in enumerate(item.times)
            if index not in used_sylber
            and values[index] >= threshold
            and start - padding <= time <= end + padding
        ]
        ranked = [
            (2.0 + float(item.base_confidence[index]),
             float(item.base_times[index]), "base", index)
            for index in base_indices
        ]
        ranked.extend([
            (float(values[index]), float(item.times[index]), "sylber", index)
            for index in sylber_indices
        ])
        selected = 0
        for _, time, source, index in sorted(ranked, reverse=True):
            if selected >= quota:
                break
            if any(abs(time - existing) < distance for existing in kept):
                continue
            kept.append(time)
            selected += 1
            if source == "base":
                used_base.add(index)
            else:
                used_sylber.add(index)
    return sorted(kept)


def choose_rule(training):
    cross = []
    for held_index, held in enumerate(training):
        train = [item for index, item in enumerate(training) if index != held_index]
        model = factory()
        model.fit(
            np.vstack([item.features for item in train]),
            np.concatenate([item.labels for item in train]),
        )
        cross.append((held, scores(model, held.features)))
    all_scores = np.concatenate([value for _, value in cross])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.02, 0.98, 81)))
    best = None
    for padding in (0.00, 0.03, 0.06, 0.09):
        for slack in (0, 1):
            for distance in (0.045, 0.055, 0.065):
                for threshold in thresholds:
                    metrics = [
                        fast_event_metrics(
                            word_quota_predictions(
                                item, value, float(threshold),
                                padding, slack, distance,
                            ),
                            item.reference,
                        )
                        for item, value in cross
                    ]
                    aggregate = pooled(metrics)
                    rank = (
                        aggregate["f1"], aggregate["precision"],
                        aggregate["recall"], -slack, -padding,
                    )
                    if best is None or rank > best[0]:
                        best = (
                            rank, float(threshold), padding,
                            slack, distance, aggregate,
                        )
    return best[1:]


def evaluate(recordings):
    folds = []
    for held_index, held in enumerate(recordings):
        training = [
            item for index, item in enumerate(recordings) if index != held_index
        ]
        threshold, padding, slack, distance, inner = choose_rule(training)
        model = factory()
        model.fit(
            np.vstack([item.features for item in training]),
            np.concatenate([item.labels for item in training]),
        )
        value = scores(model, held.features)
        predicted = word_quota_predictions(
            held, value, threshold, padding, slack, distance
        )
        metrics = match_events(predicted, held.reference, 0.050)
        folds.append({
            "heldout": held.name,
            "threshold": threshold,
            "word_padding_seconds": padding,
            "quota_slack": slack,
            "min_distance_seconds": distance,
            "recognized_word_count": len(held.words),
            "predicted_count": len(predicted),
            "inner_validation": inner,
            "metrics_50ms": {
                key: metrics[key] for key in (
                    "tp", "fp", "fn", "precision", "recall", "f1",
                    "timing_median_ms",
                )
            },
        })
    return {
        "folds": folds,
        "aggregate_50ms": pooled([
            fold["metrics_50ms"] for fold in folds
        ]),
    }


def main():
    root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    recordings = []
    for name in ("3local", "6fori", "7fori"):
        signal, sample_rate = librosa.load(
            find_audio(root / "audio", name), sr=16000, mono=True
        )
        states = encode(model, signal, device)
        detection = detect_nuclei_v2(
            signal, sample_rate, production_v2_config()
        )
        reference, span = annotation(
            root / "analysis" / "v2_annotations"
            / f"{name}_v2_reviewed.TextGrid"
        )
        item = make_recording(
            root, name, states, len(signal) / sample_rate,
            detection, reference, span,
        )
        recordings.append(item)
        print(
            f"{name}: {len(item.words)} recognized words, "
            f"{len(item.reference)} reference nuclei"
        )
    result = evaluate(recordings)
    payload = {
        "experiment": "automatic_asr_assisted_nucleus_detection",
        "user_supplied_transcript": False,
        "text_independent": False,
        "evaluation": "nested leave-one-recording-out; +/-50 ms",
        "result": result,
    }
    output = root / "results" / "nuclei_v3_asr_assisted_true_manual.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
