#!/usr/bin/env python3
"""Classify zero-shot Sylber segments with Russian acoustic evidence.

This is a development experiment, not a production model.  Sylber proposes a
small set of syllable-like regions; a lightweight classifier then rejects
regions whose embedding shape and vowel evidence look non-nuclear.  All
evaluation and threshold selection is grouped by recording/speaker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import librosa
import numpy as np
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acoustic_nuclei_v2 import detect_nuclei_v2, production_v2_config
from evaluate_nuclei_v2 import match_events
from evaluate_sylber_zero_shot import (
    annotation,
    cosine,
    encode,
    load_model,
    sylber_segments,
)
from prepare_nucleus_annotations import find_audio


TRACKS = (
    "likelihood", "envelope", "rms_db", "spectral_sonority",
    "periodicity", "vowel_likeness", "zcr_score", "flatness_score",
    "flux_score", "sonorant_penalty", "spectral_entropy",
    "formant_energy_db", "energy_entropy_ratio",
)
CONFIGS = (
    (2.4, 0.75),
    (2.6, 0.75),
    (2.6, 0.80),
    (2.8, 0.80),
    (2.9, 0.80),
    (3.0, 0.80),
    (3.0, 0.85),
)


@dataclass
class CandidateData:
    name: str
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    reference: list[float]
    raw_metrics: dict
    base_times: list[float]


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    q25, q75 = np.percentile(values, (25, 75))
    scale = max(float((q75 - q25) / 1.349), float(np.std(values)), 1e-6)
    return np.clip((values - med) / scale, -6.0, 6.0)


def summarize(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)), float(np.max(values)),
        float(np.min(values)), float(np.std(values)),
    ]


def candidate_features(states, segments, duration, detection) -> np.ndarray:
    seconds_per_hidden = duration / max(len(states), 1)
    tracks = dict(detection.feature_tracks)
    tracks["likelihood"] = detection.likelihood
    normalized = {
        name: robust_z(tracks[name]) if name in tracks
        else np.zeros(len(detection.frame_times))
        for name in TRACKS
    }
    band_z = tracks.get("band_z", np.zeros((0, len(detection.frame_times))))
    rows = []
    for start_frame, end_frame in segments:
        hidden = states[start_frame:end_frame]
        start = start_frame * seconds_per_hidden
        end = end_frame * seconds_per_hidden
        indices = np.flatnonzero(
            (detection.frame_times >= start) & (detection.frame_times <= end)
        )
        norms = np.sqrt(np.sum(hidden * hidden, axis=1) + 1e-8)
        adjacent = cosine(hidden[:-1], hidden[1:]) if len(hidden) > 1 else np.ones(1)
        mean_state = np.mean(hidden, axis=0) if len(hidden) else np.zeros(states.shape[1])
        row = [
            end - start,
            float(len(hidden)),
            *summarize(norms),
            *summarize(adjacent),
            float(np.sqrt(np.sum(mean_state * mean_state) + 1e-8)),
            float(cosine(hidden[0], hidden[-1])) if len(hidden) else 0.0,
        ]
        for name in TRACKS:
            row.extend(summarize(normalized[name][indices]))
        for band in band_z:
            values = band[indices]
            row.extend([
                float(np.mean(values)) if len(values) else 0.0,
                float(np.max(values)) if len(values) else 0.0,
            ])
        for name in ("likelihood", "vowel_likeness", "formant_energy_db"):
            values = normalized[name][indices]
            peak_ratio = float(np.argmax(values) / max(len(values) - 1, 1)) if len(values) else 0.5
            row.append(peak_ratio)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def build_candidates(name, states, duration, detection, reference, span, config) -> CandidateData:
    segments = sylber_segments(states, *config)
    seconds_per_hidden = duration / max(len(states), 1)
    times = np.asarray([
        (start + end) * 0.5 * seconds_per_hidden
        for start, end in segments
    ])
    inside = (times >= span[0]) & (times <= span[1])
    times = times[inside]
    segments = segments[inside]
    features = candidate_features(states, segments, duration, detection)
    metrics = match_events(times.tolist(), reference, 0.050)
    positive = {
        round(float(value), 9) for value, _ in metrics["matched_pairs_seconds"]
    }
    labels = np.asarray([
        round(float(value), 9) in positive for value in times
    ], dtype=np.int8)
    base_times = [
        float(value) for value in detection.times
        if span[0] <= value <= span[1]
    ]
    return CandidateData(
        name, times, features, labels, reference, metrics, base_times
    )


def factories():
    return {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=3000, random_state=23),
        ),
        "svm": lambda: make_pipeline(
            StandardScaler(), SVC(C=1.5, gamma="scale")
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=23,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=0.65,
            class_weight="balanced_subsample", n_jobs=-1, random_state=23,
        ),
        "hist_gradient": lambda: HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.045, max_leaf_nodes=15,
            min_samples_leaf=8, l2_regularization=1.0,
            class_weight="balanced", random_state=23,
        ),
    }


def scores(model, features):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    return np.asarray(model.predict_proba(features)[:, 1])


def pooled(items):
    tp, fp, fn = (sum(item[key] for item in items) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def fast_event_metrics(predicted, reference, tolerance=0.050):
    """Maximum-cardinality 1-D event counts without the slower DP traceback.

    For equal symmetric tolerance windows, the ordered two-pointer match has
    maximum cardinality. Threshold search only needs TP/FP/FN; final fold
    reporting still uses ``match_events`` to retain exact timing errors.
    """
    predicted = sorted(float(value) for value in predicted)
    reference = sorted(float(value) for value in reference)
    left = right = matched = 0
    while left < len(predicted) and right < len(reference):
        delta = predicted[left] - reference[right]
        if delta < -tolerance:
            left += 1
        elif delta > tolerance:
            right += 1
        else:
            matched += 1
            left += 1
            right += 1
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "tp": matched,
        "fp": len(predicted) - matched,
        "fn": len(reference) - matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def priority_union(primary, secondary, distance=0.055):
    kept = [float(value) for value in primary]
    for value in secondary:
        value = float(value)
        if all(abs(value - existing) >= distance for existing in kept):
            kept.append(value)
    return sorted(kept)


def combined_predictions(item, values, threshold, strategy, distance=0.055):
    selected = item.times[values >= threshold].tolist()
    if strategy == "standalone":
        return selected
    if strategy == "v2_priority_union":
        return priority_union(item.base_times, selected, distance)
    if strategy == "sylber_priority_union":
        return priority_union(selected, item.base_times, distance)
    raise ValueError(f"Unknown strategy: {strategy}")


def choose_threshold(factory, training, strategy):
    cross_scores = []
    for held_index, held in enumerate(training):
        train = [item for index, item in enumerate(training) if index != held_index]
        model = factory()
        model.fit(
            np.vstack([item.features for item in train]),
            np.concatenate([item.labels for item in train]),
        )
        cross_scores.append((held, scores(model, held.features)))
    all_scores = np.concatenate([value for _, value in cross_scores])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.0, 0.98, 100)))
    distances = (0.055,) if strategy == "standalone" else (
        0.035, 0.045, 0.055, 0.065, 0.075,
    )
    best = None
    for distance in distances:
        for threshold in thresholds:
            metrics = [
                fast_event_metrics(
                    combined_predictions(
                        item, value, threshold, strategy, distance
                    ),
                    item.reference, 0.050
                )
                for item, value in cross_scores
            ]
            aggregate = pooled(metrics)
            rank = (
                aggregate["f1"], aggregate["recall"],
                aggregate["precision"], -abs(distance - 0.055),
            )
            if best is None or rank > best[0]:
                best = (rank, float(threshold), float(distance), aggregate)
    return best[1], best[2], best[3]


def evaluate(factory, recordings, strategy):
    folds = []
    for held_index, held in enumerate(recordings):
        training = [item for index, item in enumerate(recordings) if index != held_index]
        threshold, distance, inner = choose_threshold(factory, training, strategy)
        model = factory()
        model.fit(
            np.vstack([item.features for item in training]),
            np.concatenate([item.labels for item in training]),
        )
        value = scores(model, held.features)
        predicted = combined_predictions(
            held, value, threshold, strategy, distance
        )
        metrics = match_events(predicted, held.reference, 0.050)
        folds.append({
            "heldout": held.name,
            "threshold": threshold,
            "fusion_distance_seconds": distance,
            "inner_validation": inner,
            "raw_candidate_count": len(held.times),
            "selected_count": len(predicted),
            "metrics_50ms": {key: metrics[key] for key in (
                "tp", "fp", "fn", "precision", "recall", "f1", "timing_median_ms"
            )},
        })
    return {"folds": folds,
            "aggregate_50ms": pooled([fold["metrics_50ms"] for fold in folds])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--focused", action="store_true",
        help="Run only the previously best ExtraTrees/config refinement.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    names = ("3local", "6fori", "7fori")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    base = {}
    for name in names:
        signal, sr = librosa.load(find_audio(root / "audio", name), sr=16000, mono=True)
        reference, span = annotation(
            root / "analysis" / "v2_annotations" / f"{name}_v2_reviewed.TextGrid"
        )
        base[name] = {
            "states": encode(model, signal, device),
            "duration": len(signal) / sr,
            "detection": detect_nuclei_v2(signal, sr, production_v2_config()),
            "reference": reference,
            "span": span,
        }
        print(f"Encoded {name}")

    configs = ((2.6, 0.8),) if args.focused else CONFIGS
    factory_map = factories()
    if args.focused:
        factory_map = {"extra_trees": factory_map["extra_trees"]}
    results = {}
    for config in configs:
        key = f"norm={config[0]:.2f},merge={config[1]:.2f}"
        recordings = [
            build_candidates(name, config=config, **base[name]) for name in names
        ]
        results[key] = {
            "raw_candidates": [{
                "recording": item.name,
                "count": len(item.times),
                "metrics_50ms": {metric: item.raw_metrics[metric] for metric in (
                    "tp", "fp", "fn", "precision", "recall", "f1"
                )},
            } for item in recordings],
            "models": {
                name: {
                    strategy: evaluate(factory, recordings, strategy)
                    for strategy in (
                        "standalone", "v2_priority_union",
                        "sylber_priority_union",
                    )
                }
                for name, factory in factory_map.items()
            },
        }
    payload = {
        "experiment": "sylber_segment_acoustic_classifier",
        "text_independent": True,
        "evaluation": "nested leave-one-recording-out; ±50 ms",
        "search_scope": "focused refinement" if args.focused else "full comparison",
        "results": results,
    }
    output = root / "results" / "nuclei_v3_sylber_segment_classifier.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
