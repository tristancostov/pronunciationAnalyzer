#!/usr/bin/env python3
"""Nested speaker-held-out ensemble of the strongest complete V3 detectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from acoustic_nuclei_v2 import detect_nuclei_v2, production_v2_config
from evaluate_nuclei_v2 import match_events
from evaluate_sylber_zero_shot import annotation, encode, load_model
from experiment_sylber_segment_classifier import (
    build_candidates,
    choose_threshold,
    combined_predictions,
    fast_event_metrics,
    factories,
    pooled,
    scores,
)
from prepare_nucleus_annotations import find_audio


METHODS = (
    ("svm_280_080", (2.8, 0.8), "svm"),
    ("svm_290_080", (2.9, 0.8), "svm"),
    ("trees_300_080", (3.0, 0.8), "extra_trees"),
)


@dataclass
class MethodData:
    name: str
    recordings: list
    factory: object


def cluster_events(method_events, cluster_distance, minimum_support, min_distance):
    raw = sorted(
        (float(time), method_index)
        for method_index, values in enumerate(method_events)
        for time in values
    )
    clusters = []
    for time, method_index in raw:
        eligible = [
            (abs(time - float(np.median(list(cluster["times"].values())))), index)
            for index, cluster in enumerate(clusters)
            if method_index not in cluster["times"]
            and abs(time - float(np.median(list(cluster["times"].values()))))
            <= cluster_distance
        ]
        if eligible:
            _, index = min(eligible)
            clusters[index]["times"][method_index] = time
        else:
            clusters.append({"times": {method_index: time}})
    candidates = []
    for cluster in clusters:
        support = len(cluster["times"])
        if support < minimum_support:
            continue
        values = list(cluster["times"].values())
        candidates.append((
            support,
            -float(np.std(values)),
            float(np.median(values)),
        ))
    kept = []
    for support, stability, time in sorted(candidates, reverse=True):
        if all(abs(time - existing) >= min_distance for existing in kept):
            kept.append(time)
    return sorted(kept)


def method_predictions(method, outer_held):
    recordings = method.recordings
    training_indices = [
        index for index in range(len(recordings)) if index != outer_held
    ]
    training = [recordings[index] for index in training_indices]
    threshold, distance, _ = choose_threshold(
        method.factory, training, "v2_priority_union"
    )
    inner = {}
    for validation_pos, validation_index in enumerate(training_indices):
        train = [
            recordings[index] for index in training_indices
            if index != validation_index
        ]
        model = method.factory()
        model.fit(
            np.vstack([item.features for item in train]),
            np.concatenate([item.labels for item in train]),
        )
        held = recordings[validation_index]
        value = scores(model, held.features)
        inner[validation_index] = combined_predictions(
            held, value, threshold, "v2_priority_union", distance
        )
    model = method.factory()
    model.fit(
        np.vstack([item.features for item in training]),
        np.concatenate([item.labels for item in training]),
    )
    held = recordings[outer_held]
    value = scores(model, held.features)
    outer = combined_predictions(
        held, value, threshold, "v2_priority_union", distance
    )
    return inner, outer, threshold, distance


def choose_ensemble_rule(inner_by_method, recordings, training_indices):
    best = None
    for cluster_distance in (0.020, 0.030, 0.040, 0.050):
        for support in (1, 2, 3):
            for min_distance in (0.055, 0.065, 0.075):
                metrics = []
                for recording_index in training_indices:
                    predicted = cluster_events(
                        [
                            method_values[recording_index]
                            for method_values in inner_by_method
                        ],
                        cluster_distance, support, min_distance,
                    )
                    metrics.append(fast_event_metrics(
                        predicted, recordings[recording_index].reference
                    ))
                aggregate = pooled(metrics)
                rank = (
                    aggregate["f1"], aggregate["precision"],
                    aggregate["recall"], support,
                    -abs(min_distance - 0.065),
                )
                if best is None or rank > best[0]:
                    best = (
                        rank, cluster_distance, support,
                        min_distance, aggregate,
                    )
    return best[1:]


def main():
    root = Path(__file__).resolve().parent
    names = ("3local", "6fori", "7fori")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sylber = load_model(device)
    base = {}
    for name in names:
        signal, sample_rate = librosa.load(
            find_audio(root / "audio", name), sr=16000, mono=True
        )
        reference, span = annotation(
            root / "analysis" / "v2_annotations"
            / f"{name}_v2_reviewed.TextGrid"
        )
        base[name] = {
            "states": encode(sylber, signal, device),
            "duration": len(signal) / sample_rate,
            "detection": detect_nuclei_v2(
                signal, sample_rate, production_v2_config()
            ),
            "reference": reference,
            "span": span,
        }
        print(f"Encoded {name}")

    factory_map = factories()
    methods = []
    for method_name, config, factory_name in METHODS:
        recordings = [
            build_candidates(name, config=config, **base[name])
            for name in names
        ]
        methods.append(MethodData(
            method_name, recordings, factory_map[factory_name]
        ))

    folds = []
    for outer_held, held_name in enumerate(names):
        training_indices = [
            index for index in range(len(names)) if index != outer_held
        ]
        inner_by_method = []
        outer_by_method = []
        method_rules = []
        for method in methods:
            inner, outer, threshold, distance = method_predictions(
                method, outer_held
            )
            inner_by_method.append(inner)
            outer_by_method.append(outer)
            method_rules.append({
                "method": method.name,
                "threshold": threshold,
                "fusion_distance_seconds": distance,
            })
        cluster_distance, support, min_distance, inner = choose_ensemble_rule(
            inner_by_method, methods[0].recordings, training_indices
        )
        predicted = cluster_events(
            outer_by_method, cluster_distance, support, min_distance
        )
        held = methods[0].recordings[outer_held]
        metrics = match_events(predicted, held.reference, 0.050)
        folds.append({
            "heldout": held_name,
            "cluster_distance_seconds": cluster_distance,
            "minimum_method_support": support,
            "min_event_distance_seconds": min_distance,
            "method_rules": method_rules,
            "inner_validation": inner,
            "predicted_count": len(predicted),
            "metrics_50ms": {
                key: metrics[key] for key in (
                    "tp", "fp", "fn", "precision", "recall", "f1",
                    "timing_median_ms",
                )
            },
        })
        print(f"{held_name}: F1={metrics['f1']:.4f}")
    payload = {
        "experiment": "top_v3_detector_ensemble",
        "text_independent": True,
        "evaluation": "nested leave-one-recording-out; +/-50 ms",
        "methods": [method.name for method in methods],
        "folds": folds,
        "aggregate_50ms": pooled([
            fold["metrics_50ms"] for fold in folds
        ]),
    }
    output = root / "results" / "nuclei_v3_model_ensemble_true_manual.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
