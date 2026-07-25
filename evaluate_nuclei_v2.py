#!/usr/bin/env python3
"""Evaluate independent V2 nucleus events against reviewed Praat points."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median

import numpy as np

from acoustic_nuclei_v2 import detect_file
from nuclei_annotations import read_interval_tier, read_point_tier, read_textgrid_file
from prepare_nucleus_annotations import find_audio, has_written_vowel, load_config


def match_events(
    predicted: list[float],
    reference: list[float],
    tolerance_seconds: float,
) -> dict:
    """Maximum-cardinality ordered matching, then minimum timing error."""
    predicted = sorted(predicted)
    reference = sorted(reference)
    rows, columns = len(predicted) + 1, len(reference) + 1
    matches = np.zeros((rows, columns), dtype=np.int32)
    errors = np.zeros((rows, columns), dtype=float)
    choice = np.zeros((rows, columns), dtype=np.uint8)  # 1 skip pred, 2 skip ref, 3 match

    def better(m1: int, e1: float, m2: int, e2: float) -> bool:
        return m1 > m2 or (m1 == m2 and e1 < e2 - 1e-12)

    for i in range(1, rows):
        for j in range(1, columns):
            best_m = int(matches[i - 1, j])
            best_e = float(errors[i - 1, j])
            best_c = 1
            candidate_m = int(matches[i, j - 1])
            candidate_e = float(errors[i, j - 1])
            if better(candidate_m, candidate_e, best_m, best_e):
                best_m, best_e, best_c = candidate_m, candidate_e, 2
            distance = abs(predicted[i - 1] - reference[j - 1])
            if distance <= tolerance_seconds:
                candidate_m = int(matches[i - 1, j - 1]) + 1
                candidate_e = float(errors[i - 1, j - 1]) + distance
                if better(candidate_m, candidate_e, best_m, best_e):
                    best_m, best_e, best_c = candidate_m, candidate_e, 3
            matches[i, j], errors[i, j], choice[i, j] = best_m, best_e, best_c

    pairs = []
    i, j = len(predicted), len(reference)
    while i > 0 and j > 0:
        current = int(choice[i, j])
        if current == 3:
            pairs.append((predicted[i - 1], reference[j - 1]))
            i -= 1
            j -= 1
        elif current == 2:
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    matched_predicted = [left for left, _ in pairs]
    matched_reference = [right for _, right in pairs]
    unmatched_predicted = [
        value for value in predicted if value not in matched_predicted
    ]
    unmatched_reference = [
        value for value in reference if value not in matched_reference
    ]
    timing_errors_ms = [abs(left - right) * 1000.0 for left, right in pairs]
    tp = len(pairs)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_ms": tolerance_seconds * 1000.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "timing_median_ms": median(timing_errors_ms) if timing_errors_ms else None,
        "timing_errors_ms": timing_errors_ms,
        "matched_pairs_seconds": pairs,
        "false_positive_times_seconds": unmatched_predicted,
        "missed_reference_times_seconds": unmatched_reference,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", help="Recording stems, e.g. 3local")
    parser.add_argument("--audio-dir", type=Path, default=root / "audio")
    parser.add_argument(
        "--annotation-dir", type=Path, default=root / "analysis" / "v2_annotations"
    )
    parser.add_argument("--tier", default="nucleus")
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Calibration JSON containing the development candidate to evaluate",
    )
    parser.add_argument("--output", type=Path, default=root / "results" / "nuclei_v2_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_name = load_config(args.config_json)
    recordings = []
    aggregate: dict[int, dict[str, int | list[float]]] = {
        30: {"tp": 0, "fp": 0, "fn": 0, "errors": []},
        50: {"tp": 0, "fp": 0, "fn": 0, "errors": []},
    }
    for recording in args.recordings:
        reviewed = args.annotation_dir / f"{recording}_v2_reviewed.TextGrid"
        if not reviewed.is_file():
            raise FileNotFoundError(
                f"Reviewed annotation not found: {reviewed}. "
                "Open *_v2_to_review.TextGrid in Praat, correct the nucleus tier, "
                "then save it with the *_v2_reviewed.TextGrid name."
            )
        grid_text = read_textgrid_file(reviewed)
        reference = [
            point.time for point in read_point_tier(grid_text, args.tier)
            if point.mark.strip() and "?" not in point.mark
        ]
        detection = detect_file(find_audio(args.audio_dir, recording), config)
        predicted = detection.times
        intervals = []
        try:
            labelled_intervals = read_interval_tier(grid_text, "syllable")
            intervals = [
                item for item in labelled_intervals if has_written_vowel(item.text)
            ]
            if intervals:
                annotated_start = min(item.start for item in labelled_intervals)
                annotated_end = max(item.end for item in labelled_intervals)
                predicted = [
                    time for time in predicted
                    if annotated_start <= time <= annotated_end
                ]
        except ValueError:
            annotated_start, annotated_end = 0.0, detection.duration_seconds

        tolerance_metrics = {}
        for tolerance_ms in (30, 50):
            metrics = match_events(predicted, reference, tolerance_ms / 1000.0)
            tolerance_metrics[str(tolerance_ms)] = metrics
            bucket = aggregate[tolerance_ms]
            bucket["tp"] = int(bucket["tp"]) + metrics["tp"]
            bucket["fp"] = int(bucket["fp"]) + metrics["fp"]
            bucket["fn"] = int(bucket["fn"]) + metrics["fn"]
            bucket["errors"].extend(metrics["timing_errors_ms"])
        metrics_50 = tolerance_metrics["50"]
        def describe(time: float) -> dict:
            containing = [
                item.text for item in intervals
                if item.start <= time <= item.end
            ]
            return {"time_seconds": time, "syllable_labels": containing}
        recordings.append({
            "recording": recording,
            "reference_count": len(reference),
            "predicted_count_in_annotated_span": len(predicted),
            "annotated_span_seconds": [annotated_start, annotated_end],
            "metrics": tolerance_metrics,
            "missed_reference_details_50ms": [
                describe(time) for time in metrics_50["missed_reference_times_seconds"]
            ],
            "false_positive_details_50ms": [
                describe(time) for time in metrics_50["false_positive_times_seconds"]
            ],
        })

    aggregate_metrics = {}
    for tolerance_ms, bucket in aggregate.items():
        tp, fp, fn = int(bucket["tp"]), int(bucket["fp"]), int(bucket["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate_metrics[str(tolerance_ms)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "timing_median_ms": median(bucket["errors"]) if bucket["errors"] else None,
        }

    payload = {
        "algorithm": "acoustic_nuclei_v2",
        "text_independent": True,
        "config_name": config_name,
        "config": asdict(config),
        "recordings": recordings,
        "aggregate": aggregate_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
