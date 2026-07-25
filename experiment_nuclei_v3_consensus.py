#!/usr/bin/env python3
"""Multi-scale Sylber consensus plus Russian acoustic candidate classification.

Every Sylber threshold pair gives a slightly different syllabic segmentation.
Instead of selecting one pair, this experiment converts a fixed grid into a
frame-level vote contour. Candidate peaks carry vote stability across norm and
merge thresholds, local HuBERT geometry, and the existing Russian acoustic
features. Evaluation is nested leave-one-recording-out.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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
from experiment_nuclei_v3_modes import (
    build_features as acoustic_features,
    candidate_frames as acoustic_candidate_frames,
)
from experiment_sylber_segment_classifier import fast_event_metrics, pooled
from prepare_nucleus_annotations import find_audio


NORMS = tuple(round(value, 2) for value in np.arange(2.2, 3.11, 0.1))
MERGES = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
GRID = tuple((norm, merge) for norm in NORMS for merge in MERGES)
CONTEXT = np.asarray((-10, -7, -4, -2, 0, 2, 4, 7, 10))


@dataclass
class Recording:
    name: str
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    reference: list[float]
    base_times: list[float]
    ceiling: dict


def segment_midpoints(states, duration, norm, merge):
    segments = sylber_segments(states, norm, merge)
    seconds_per_frame = duration / max(len(states), 1)
    return np.asarray([
        (start + end) * 0.5 * seconds_per_frame
        for start, end in segments
    ], dtype=float)


def vote_features(states, duration, detection):
    frames = len(detection.frame_times)
    all_events = []
    vote = np.zeros(frames, dtype=float)
    norm_vote = np.zeros((len(NORMS), frames), dtype=float)
    merge_vote = np.zeros((len(MERGES), frames), dtype=float)
    for norm_index, norm in enumerate(NORMS):
        for merge_index, merge in enumerate(MERGES):
            events = segment_midpoints(states, duration, norm, merge)
            all_events.append(events)
            impulses = np.zeros(frames, dtype=float)
            if len(events):
                indices = np.searchsorted(detection.frame_times, events)
                indices = np.clip(indices, 0, frames - 1)
                impulses[np.unique(indices)] = 1.0
            smoothed = gaussian_filter1d(impulses, sigma=1.4, mode="nearest")
            vote += smoothed
            norm_vote[norm_index] += smoothed
            merge_vote[merge_index] += smoothed
    vote /= len(GRID)
    norm_vote /= len(MERGES)
    merge_vote /= len(NORMS)
    return vote, norm_vote, merge_vote, all_events


def local_prominence(values, frame, radius=10):
    left = np.min(values[max(0, frame - radius):frame + 1])
    right = np.min(values[frame:min(len(values), frame + radius + 1)])
    return max(0.0, float(values[frame] - max(left, right)))


def build_recording(name, states, duration, detection, reference, span):
    vote, norm_vote, merge_vote, all_events = vote_features(
        states, duration, detection
    )
    consensus_peaks, _ = find_peaks(vote, distance=3, prominence=0.0005)
    acoustic = acoustic_candidate_frames(detection)
    base_frames = np.asarray([item.frame for item in detection.nuclei], dtype=int)
    frames = np.unique(np.concatenate([
        consensus_peaks.astype(int), acoustic.astype(int), base_frames,
    ]))
    valid = (
        (frames > 0)
        & (frames < len(detection.frame_times) - 1)
        & detection.speech_mask[frames]
    )
    frames = frames[valid]
    times = detection.frame_times[frames]
    inside = (times >= span[0]) & (times <= span[1])
    frames, times = frames[inside], times[inside]

    base_feature_matrix = acoustic_features(detection, frames)
    hidden_seconds = duration / max(len(states), 1)
    base_times = [
        float(value) for value in detection.times
        if span[0] <= value <= span[1]
    ]
    extra_rows = []
    for frame, time in zip(frames, times):
        indices = np.clip(frame + CONTEXT, 0, len(vote) - 1)
        row = [
            float(vote[frame]),
            local_prominence(vote, int(frame)),
            *vote[indices].astype(float).tolist(),
            *norm_vote[:, frame].astype(float).tolist(),
            *merge_vote[:, frame].astype(float).tolist(),
        ]
        distances = np.asarray([
            np.min(np.abs(events - time)) if len(events) else 9.0
            for events in all_events
        ])
        row.extend([
            float(np.mean(distances <= 0.020)),
            float(np.mean(distances <= 0.035)),
            float(np.mean(distances <= 0.050)),
            float(np.median(np.clip(distances, 0.0, 0.25))),
            float(np.std(np.clip(distances, 0.0, 0.25))),
        ])
        if base_times:
            base_distance = min(abs(time - value) for value in base_times)
        else:
            base_distance = 1.0
        row.extend([
            float(min(base_distance, 0.25)),
            float(base_distance <= 0.025),
            float(base_distance <= 0.050),
        ])
        hidden_frame = int(round(time / hidden_seconds))
        hidden_frame = int(np.clip(hidden_frame, 0, len(states) - 1))
        hidden = states[hidden_frame]
        left = states[max(0, hidden_frame - 2)]
        right = states[min(len(states) - 1, hidden_frame + 2)]
        region = states[
            max(0, hidden_frame - 3):min(len(states), hidden_frame + 4)
        ]
        norms = np.sqrt(np.sum(region * region, axis=1) + 1e-8)
        row.extend([
            float(np.sqrt(np.sum(hidden * hidden) + 1e-8)),
            float(np.mean(norms)),
            float(np.std(norms)),
            float(cosine(left, hidden)),
            float(cosine(hidden, right)),
            float(cosine(left, right)),
        ])
        extra_rows.append(row)
    features = np.hstack([
        base_feature_matrix,
        np.asarray(extra_rows, dtype=np.float32),
    ])

    ceiling = match_events(times.tolist(), reference, 0.050)
    # Multiple nearby candidates can validly represent the same broad target;
    # NMS later keeps only one. A soft spatial label avoids arbitrary negative
    # labels for a candidate only a few milliseconds from the DP-selected one.
    labels = np.asarray([
        min((abs(time - value) for value in reference), default=9.0) <= 0.040
        for time in times
    ], dtype=np.int8)
    return Recording(
        name, times, features, labels, reference, base_times, ceiling
    )


def factories():
    return {
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=650, min_samples_leaf=2, max_features=0.65,
            class_weight="balanced", n_jobs=-1, random_state=71,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=650, min_samples_leaf=2, max_features=0.55,
            class_weight="balanced_subsample", n_jobs=-1, random_state=71,
        ),
        "hist_gradient": lambda: HistGradientBoostingClassifier(
            max_iter=240, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=8, l2_regularization=1.5,
            class_weight="balanced", random_state=71,
        ),
        "svm": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=1.2, gamma="scale", class_weight="balanced"),
        ),
    }


def scores(model, features):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def suppress(times, values, min_distance):
    kept = []
    for index in np.argsort(values)[::-1]:
        time = float(times[index])
        if all(abs(time - existing) >= min_distance for existing in kept):
            kept.append(time)
    return sorted(kept)


def predictions(item, values, threshold, distance, base_priority):
    selected = values >= threshold
    classified = suppress(item.times[selected], values[selected], distance)
    if not base_priority:
        return classified
    kept = list(item.base_times)
    for time in classified:
        if all(abs(time - existing) >= distance for existing in kept):
            kept.append(time)
    return sorted(kept)


def choose_rule(factory, training, base_priority):
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
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.03, 0.99, 121)))
    best = None
    for distance in (0.045, 0.055, 0.065, 0.075):
        for threshold in thresholds:
            metrics = [
                fast_event_metrics(
                    predictions(
                        item, value, float(threshold), distance, base_priority
                    ),
                    item.reference,
                )
                for item, value in cross
            ]
            aggregate = pooled(metrics)
            rank = (
                aggregate["f1"], aggregate["recall"],
                aggregate["precision"], -abs(distance - 0.055),
            )
            if best is None or rank > best[0]:
                best = (rank, float(threshold), distance, aggregate)
    return best[1], best[2], best[3]


def evaluate(factory, recordings, base_priority):
    folds = []
    for held_index, held in enumerate(recordings):
        training = [
            item for index, item in enumerate(recordings) if index != held_index
        ]
        threshold, distance, inner = choose_rule(
            factory, training, base_priority
        )
        model = factory()
        model.fit(
            np.vstack([item.features for item in training]),
            np.concatenate([item.labels for item in training]),
        )
        value = scores(model, held.features)
        predicted = predictions(
            held, value, threshold, distance, base_priority
        )
        metrics = match_events(predicted, held.reference, 0.050)
        folds.append({
            "heldout": held.name,
            "threshold": threshold,
            "min_distance_seconds": distance,
            "inner_validation": inner,
            "candidate_count": len(held.times),
            "selected_count": len(predicted),
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
        item = build_recording(
            name, states, len(signal) / sample_rate,
            detection, reference, span,
        )
        recordings.append(item)
        print(
            f"{name}: {len(item.times)} candidates, "
            f"ceiling recall={item.ceiling['recall']:.3f}"
        )
    result = {
        model_name: {
            "standalone": evaluate(factory, recordings, False),
            "v2_priority": evaluate(factory, recordings, True),
        }
        for model_name, factory in factories().items()
    }
    payload = {
        "experiment": "multiscale_sylber_consensus",
        "text_independent": True,
        "evaluation": "nested leave-one-recording-out; +/-50 ms",
        "grid": {"norms": NORMS, "merges": MERGES},
        "candidate_ceiling": [{
            "recording": item.name,
            "count": len(item.times),
            "recall": item.ceiling["recall"],
        } for item in recordings],
        "models": result,
    }
    output = root / "results" / "nuclei_v3_consensus_true_manual.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
