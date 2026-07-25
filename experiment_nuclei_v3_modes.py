#!/usr/bin/env python3
"""Speaker-held-out V3 experiment: classify acoustic peak shapes.

The experiment follows two ideas from the literature without copying a model:

* Yarra et al. (Speech Communication, 2016): classify the complete shape of a
  peak and its neighbourhood instead of accepting/rejecting it by peak height.
* Landsiedel et al. (ICASSP, 2011): use temporally contextualized perceptual
  features and evaluate with disjoint speakers.

No transcript, word timing, expected syllable count, dictionary, or forced
alignment is used. Human nucleus points are used only as training/evaluation
targets. Hyperparameters are intentionally fixed; the score threshold is
selected inside each outer fold using the two remaining recordings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acoustic_nuclei_v2 import detect_file, production_v2_config
from evaluate_nuclei_v2 import match_events
from nuclei_annotations import (
    read_interval_tier,
    read_point_tier,
    read_textgrid_file,
)
from prepare_nucleus_annotations import find_audio


CONTEXT_OFFSETS = np.asarray((-10, -7, -4, -2, 0, 2, 4, 7, 10))
SCALAR_TRACKS = (
    "likelihood",
    "envelope",
    "rms_db",
    "spectral_sonority",
    "periodicity",
    "vowel_likeness",
    "zcr_score",
    "flatness_score",
    "flux_score",
    "sonorant_penalty",
    "spectral_entropy",
    "formant_energy_db",
    "energy_entropy_ratio",
)
CONTEXT_TRACKS = (
    "likelihood",
    "envelope",
    "periodicity",
    "vowel_likeness",
    "formant_energy_db",
    "energy_entropy_ratio",
)


@dataclass
class RecordingData:
    name: str
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    rescue_mask: np.ndarray
    rescue_labels: np.ndarray
    base_times: list[float]
    base_metrics: dict
    reference: list[float]
    span: tuple[float, float]
    candidate_ceiling: dict
    rescue_ceiling: dict


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values)
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, (25.0, 75.0))
    scale = max(float((q75 - q25) / 1.349), float(np.std(finite)), 1e-6)
    return np.clip((values - median) / scale, -6.0, 6.0)


def candidate_frames(detection) -> np.ndarray:
    """Generate a high-recall, text-independent candidate pool."""
    tracks = detection.feature_tracks
    pools = []
    for name in ("envelope", "likelihood", "vowel_likeness", "periodicity"):
        values = tracks[name] if name in tracks else detection.likelihood
        peaks, _ = find_peaks(values, distance=2)
        pools.append(peaks)
    frames = np.unique(np.concatenate(pools)).astype(int)
    valid = (
        (frames > 0)
        & (frames < len(detection.frame_times) - 1)
        & detection.speech_mask[frames]
    )
    return frames[valid]


def resample(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(length, dtype=float)
    if len(values) == 1:
        return np.full(length, float(values[0]))
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, values)


def shape_features(contour: np.ndarray, frame: int, radius: int = 12) -> list[float]:
    """Normalized left/right mode shape plus valley geometry."""
    left_edge = max(0, frame - radius)
    right_edge = min(len(contour) - 1, frame + radius)
    left_slice = contour[left_edge:frame + 1]
    right_slice = contour[frame:right_edge + 1]
    left_valley_local = int(np.argmin(left_slice))
    right_valley_local = int(np.argmin(right_slice))
    left = left_slice[left_valley_local:]
    right = right_slice[:right_valley_local + 1]
    peak = float(contour[frame])
    left_base = float(left[0])
    right_base = float(right[-1])
    scale = max(peak - min(left_base, right_base), 1e-4)
    normalized_left = (resample(left, 7) - left_base) / scale
    normalized_right = (resample(right, 7) - right_base) / scale
    return [
        *normalized_left.tolist(),
        *normalized_right.tolist(),
        float(len(left) - 1),
        float(len(right) - 1),
        (peak - left_base) / scale,
        (peak - right_base) / scale,
    ]


def build_features(detection, frames: np.ndarray) -> np.ndarray:
    tracks = dict(detection.feature_tracks)
    tracks["likelihood"] = detection.likelihood
    normalized = {}
    for name in SCALAR_TRACKS:
        if name in tracks:
            normalized[name] = robust_z(tracks[name])
        else:
            normalized[name] = np.zeros(len(detection.frame_times))

    band_z = tracks.get("band_z")
    if band_z is None:
        band_z = np.zeros((0, len(detection.frame_times)))
    rows = []
    for candidate_index, frame in enumerate(frames):
        row = []
        # Point measurements, including the paper-inspired formant/entropy cues.
        row.extend(float(normalized[name][frame]) for name in SCALAR_TRACKS)
        row.extend(float(value) for value in band_z[:, frame])

        # Fixed temporal context approximates the delta/modulation information
        # used by the BLSTM paper while keeping this small-data model simple.
        indices = np.clip(frame + CONTEXT_OFFSETS, 0, len(detection.frame_times) - 1)
        for name in CONTEXT_TRACKS:
            row.extend(float(value) for value in normalized[name][indices])

        envelope = normalized["envelope"]
        likelihood = normalized["likelihood"]
        row.extend(shape_features(envelope, frame))
        row.extend(shape_features(likelihood, frame))

        # Peak prominence and distances to neighbouring permissive candidates.
        left_min = float(np.min(envelope[max(0, frame - 12):frame + 1]))
        right_min = float(np.min(envelope[frame:min(len(envelope), frame + 13)]))
        prominence = max(0.0, float(envelope[frame]) - max(left_min, right_min))
        previous_gap = frame - frames[candidate_index - 1] if candidate_index else 50
        next_gap = frames[candidate_index + 1] - frame if candidate_index + 1 < len(frames) else 50
        row.extend((prominence, float(previous_gap), float(next_gap)))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def reference_and_span(annotation: Path) -> tuple[list[float], tuple[float, float]]:
    grid = read_textgrid_file(annotation)
    reference = [
        point.time for point in read_point_tier(grid, "nucleus")
        if point.mark.strip() and "?" not in point.mark
    ]
    intervals = read_interval_tier(grid, "syllable", include_empty=True)
    nonempty = [item for item in intervals if item.text.strip()]
    selected = nonempty or intervals
    span = (
        min(item.start for item in selected),
        max(item.end for item in selected),
    )
    return reference, span


def load_recording(root: Path, name: str) -> RecordingData:
    annotation = root / "analysis" / "v2_annotations" / f"{name}_v2_reviewed.TextGrid"
    reference, span = reference_and_span(annotation)
    detection = detect_file(find_audio(root / "audio", name), production_v2_config())
    frames = candidate_frames(detection)
    times = detection.frame_times[frames]
    inside = (times >= span[0]) & (times <= span[1])
    frames, times = frames[inside], times[inside]
    features = build_features(detection, frames)

    ceiling = match_events(times.tolist(), reference, 0.050)
    positive_times = {round(float(value), 9) for value, _ in ceiling["matched_pairs_seconds"]}
    labels = np.asarray(
        [round(float(value), 9) in positive_times for value in times], dtype=np.int8
    )
    base_times = [
        float(time) for time in detection.times if span[0] <= time <= span[1]
    ]
    base_metrics = match_events(base_times, reference, 0.050)
    rescue_mask = np.asarray([
        all(abs(float(time) - base) >= 0.055 for base in base_times)
        for time in times
    ], dtype=bool)
    rescue_times = times[rescue_mask]
    missed = base_metrics["missed_reference_times_seconds"]
    rescue_ceiling = match_events(rescue_times.tolist(), missed, 0.050)
    rescue_positive_times = {
        round(float(value), 9)
        for value, _ in rescue_ceiling["matched_pairs_seconds"]
    }
    rescue_labels = np.asarray([
        round(float(value), 9) in rescue_positive_times
        for value in rescue_times
    ], dtype=np.int8)
    return RecordingData(
        name=name,
        times=times,
        features=features,
        labels=labels,
        rescue_mask=rescue_mask,
        rescue_labels=rescue_labels,
        base_times=base_times,
        base_metrics=base_metrics,
        reference=reference,
        span=span,
        candidate_ceiling=ceiling,
        rescue_ceiling=rescue_ceiling,
    )


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "logistic_context": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=3000,
                random_state=17,
            ),
        ),
        "svm_mode_shape": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=1.5, gamma="scale", class_weight="balanced"),
        ),
        "extra_trees_context": lambda: ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=17,
        ),
    }


def score_model(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def suppress(times: np.ndarray, scores: np.ndarray, min_distance: float = 0.055) -> list[float]:
    ranked = np.argsort(scores)[::-1]
    kept = []
    for index in ranked:
        time = float(times[index])
        if all(abs(time - existing) >= min_distance for existing in kept):
            kept.append(time)
    return sorted(kept)


def add_rescues(
    base_times: list[float], times: np.ndarray, scores: np.ndarray,
    min_distance: float = 0.055,
) -> list[float]:
    """Keep every precision-tuned V2 event and add only non-conflicting rescues."""
    kept = list(base_times)
    for index in np.argsort(scores)[::-1]:
        time = float(times[index])
        if all(abs(time - existing) >= min_distance for existing in kept):
            kept.append(time)
    return sorted(kept)


def aggregate_metrics(items: list[dict]) -> dict:
    tp = sum(item["tp"] for item in items)
    fp = sum(item["fp"] for item in items)
    fn = sum(item["fn"] for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def choose_threshold(
    factory: Callable[[], object], training: list[RecordingData]
) -> tuple[float, dict]:
    """Two-way inner speaker validation on the outer fold's training set."""
    scored = []
    for validation_index in range(len(training)):
        train_parts = [item for i, item in enumerate(training) if i != validation_index]
        validation = training[validation_index]
        model = factory()
        model.fit(
            np.vstack([item.features for item in train_parts]),
            np.concatenate([item.labels for item in train_parts]),
        )
        scored.append((validation, score_model(model, validation.features)))

    all_scores = np.concatenate([scores for _, scores in scored])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.05, 0.95, 73)))
    best = None
    for threshold in thresholds:
        metrics = []
        for recording, scores in scored:
            selected = scores >= threshold
            predicted = suppress(recording.times[selected], scores[selected])
            metrics.append(match_events(predicted, recording.reference, 0.050))
        aggregate = aggregate_metrics(metrics)
        key = (aggregate["f1"], aggregate["recall"], aggregate["precision"])
        if best is None or key > best[0]:
            best = (key, float(threshold), aggregate)
    assert best is not None
    return best[1], best[2]


def outer_evaluation(
    factory: Callable[[], object], recordings: list[RecordingData]
) -> dict:
    folds = []
    for heldout_index, heldout in enumerate(recordings):
        training = [item for i, item in enumerate(recordings) if i != heldout_index]
        threshold, inner = choose_threshold(factory, training)
        model = factory()
        model.fit(
            np.vstack([item.features for item in training]),
            np.concatenate([item.labels for item in training]),
        )
        scores = score_model(model, heldout.features)
        selected = scores >= threshold
        predicted = suppress(heldout.times[selected], scores[selected])
        metrics = match_events(predicted, heldout.reference, 0.050)
        folds.append({
            "heldout": heldout.name,
            "threshold": threshold,
            "inner_validation": inner,
            "predicted_count": len(predicted),
            "metrics_50ms": {
                key: value for key, value in metrics.items()
                if key in {"tp", "fp", "fn", "precision", "recall", "f1",
                           "timing_median_ms"}
            },
        })
    return {"folds": folds, "aggregate_50ms": aggregate_metrics([
        fold["metrics_50ms"] for fold in folds
    ])}


def choose_rescue_threshold(
    factory: Callable[[], object], training: list[RecordingData]
) -> tuple[float, dict]:
    scored = []
    for validation_index in range(len(training)):
        train_parts = [item for i, item in enumerate(training) if i != validation_index]
        validation = training[validation_index]
        model = factory()
        model.fit(
            np.vstack([item.features[item.rescue_mask] for item in train_parts]),
            np.concatenate([item.rescue_labels for item in train_parts]),
        )
        scores = score_model(model, validation.features[validation.rescue_mask])
        scored.append((validation, scores))

    all_scores = np.concatenate([scores for _, scores in scored])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.10, 0.995, 121)))
    # The no-rescue option is a legitimate outcome when a classifier does not
    # transfer to the other development speaker.
    thresholds = np.append(thresholds, np.nextafter(np.max(all_scores), np.inf))
    best = None
    for threshold in thresholds:
        metrics = []
        for recording, scores in scored:
            times = recording.times[recording.rescue_mask]
            selected = scores >= threshold
            predicted = add_rescues(
                recording.base_times, times[selected], scores[selected]
            )
            metrics.append(match_events(predicted, recording.reference, 0.050))
        aggregate = aggregate_metrics(metrics)
        # When F1 ties, prefer fewer changes (precision before recall).
        key = (aggregate["f1"], aggregate["precision"], aggregate["recall"])
        if best is None or key > best[0]:
            best = (key, float(threshold), aggregate)
    assert best is not None
    return best[1], best[2]


def rescue_outer_evaluation(
    factory: Callable[[], object], recordings: list[RecordingData]
) -> dict:
    folds = []
    for heldout_index, heldout in enumerate(recordings):
        training = [item for i, item in enumerate(recordings) if i != heldout_index]
        threshold, inner = choose_rescue_threshold(factory, training)
        model = factory()
        model.fit(
            np.vstack([item.features[item.rescue_mask] for item in training]),
            np.concatenate([item.rescue_labels for item in training]),
        )
        features = heldout.features[heldout.rescue_mask]
        times = heldout.times[heldout.rescue_mask]
        scores = score_model(model, features)
        selected = scores >= threshold
        predicted = add_rescues(heldout.base_times, times[selected], scores[selected])
        metrics = match_events(predicted, heldout.reference, 0.050)
        folds.append({
            "heldout": heldout.name,
            "threshold": threshold,
            "inner_validation": inner,
            "base_predicted_count": len(heldout.base_times),
            "rescues_added": len(predicted) - len(heldout.base_times),
            "predicted_count": len(predicted),
            "metrics_50ms": {
                key: value for key, value in metrics.items()
                if key in {"tp", "fp", "fn", "precision", "recall", "f1",
                           "timing_median_ms"}
            },
        })
    return {"folds": folds, "aggregate_50ms": aggregate_metrics([
        fold["metrics_50ms"] for fold in folds
    ])}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", default=["3local", "6fori", "7fori"])
    parser.add_argument(
        "--output", type=Path,
        default=root / "results" / "nuclei_v3_mode_shape_experiment.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    recordings = [load_recording(root, name) for name in args.recordings]
    payload = {
        "experiment": "text_independent_v3_mode_shape",
        "recordings": args.recordings,
        "evaluation": "nested leave-one-recording-out; event tolerance ±50 ms",
        "candidate_pool": [
            {
                "recording": item.name,
                "candidates": len(item.times),
                "reference": len(item.reference),
                "ceiling_precision": item.candidate_ceiling["precision"],
                "ceiling_recall": item.candidate_ceiling["recall"],
                "ceiling_f1": item.candidate_ceiling["f1"],
                "base_f1": item.base_metrics["f1"],
                "base_false_negatives": item.base_metrics["fn"],
                "recoverable_false_negatives": item.rescue_ceiling["tp"],
            }
            for item in recordings
        ],
        "models": {
            name: outer_evaluation(factory, recordings)
            for name, factory in model_factories().items()
        },
        "rescue_models": {
            name: rescue_outer_evaluation(factory, recordings)
            for name, factory in model_factories().items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
