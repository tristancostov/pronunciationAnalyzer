#!/usr/bin/env python3
"""Search conservative V2 recall/precision presets on reviewed nucleus points.

This is a development-set search, not a final accuracy report.  Parameters
selected on one recording must be validated on different speakers before they
become defaults or are integrated into the application.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from acoustic_nuclei_v2 import NucleusV2Config, detect_file
from evaluate_nuclei_v2 import match_events
from nuclei_annotations import read_interval_tier, read_point_tier, read_textgrid_file
from prepare_nucleus_annotations import find_audio


PRESETS = (
    {"name": "baseline"},
    {
        "name": "recall_1",
        "strong_prominence": 0.50,
        "weak_prominence": 0.16,
        "strong_min_confidence": 0.40,
        "weak_min_confidence": 0.56,
    },
    {
        "name": "recall_2",
        "strong_prominence": 0.45,
        "weak_prominence": 0.12,
        "strong_min_confidence": 0.38,
        "weak_min_confidence": 0.52,
    },
    {
        "name": "recall_3",
        "strong_prominence": 0.40,
        "weak_prominence": 0.10,
        "strong_min_confidence": 0.36,
        "weak_min_confidence": 0.50,
    },
    {
        "name": "recall_4",
        "strong_prominence": 0.35,
        "weak_prominence": 0.08,
        "strong_min_confidence": 0.34,
        "weak_min_confidence": 0.48,
    },
    {
        "name": "weak_vowels_1",
        "strong_prominence": 0.50,
        "weak_prominence": 0.10,
        "strong_min_confidence": 0.40,
        "weak_min_confidence": 0.48,
    },
    {
        "name": "close_nuclei_1",
        "strong_prominence": 0.45,
        "weak_prominence": 0.12,
        "strong_min_confidence": 0.38,
        "weak_min_confidence": 0.52,
        "min_distance_seconds": 0.045,
    },
    {
        "name": "close_nuclei_2",
        "strong_prominence": 0.35,
        "weak_prominence": 0.08,
        "strong_min_confidence": 0.34,
        "weak_min_confidence": 0.48,
        "min_distance_seconds": 0.045,
    },
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", help="Reviewed development recordings")
    parser.add_argument("--audio-dir", type=Path, default=root / "audio")
    parser.add_argument(
        "--annotation-dir", type=Path, default=root / "analysis" / "v2_annotations"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "results" / "nuclei_v2_calibration.json"
    )
    return parser.parse_args()


def load_reference(annotation_dir: Path, recording: str) -> tuple[list[float], float, float]:
    path = annotation_dir / f"{recording}_v2_reviewed.TextGrid"
    if not path.is_file():
        raise FileNotFoundError(f"Reviewed annotation not found: {path}")
    text = read_textgrid_file(path)
    points = [
        item.time for item in read_point_tier(text, "nucleus")
        if item.mark.strip() and "?" not in item.mark
    ]
    intervals = read_interval_tier(text, "syllable")
    if intervals:
        start = min(item.start for item in intervals)
        end = max(item.end for item in intervals)
    else:
        start, end = min(points), max(points)
    return points, start, end


def main() -> None:
    args = parse_args()
    references = {
        recording: load_reference(args.annotation_dir, recording)
        for recording in args.recordings
    }
    rows = []
    for preset in PRESETS:
        name = preset["name"]
        overrides = {key: value for key, value in preset.items() if key != "name"}
        config = replace(NucleusV2Config(), **overrides)
        pooled_predicted = []
        pooled_reference = []
        offset = 0.0
        per_recording = []
        for recording in args.recordings:
            reference, start, end = references[recording]
            detection = detect_file(find_audio(args.audio_dir, recording), config)
            predicted = [time for time in detection.times if start <= time <= end]
            metrics_30 = match_events(predicted, reference, 0.030)
            metrics_50 = match_events(predicted, reference, 0.050)
            per_recording.append({
                "recording": recording,
                "predicted_count": len(predicted),
                "reference_count": len(reference),
                "f1_30": metrics_30["f1"],
                "f1_50": metrics_50["f1"],
                "precision_50": metrics_50["precision"],
                "recall_50": metrics_50["recall"],
            })
            pooled_predicted.extend(time + offset for time in predicted)
            pooled_reference.extend(time + offset for time in reference)
            offset += max(end, detection.duration_seconds) + 1.0
        pooled_30 = match_events(pooled_predicted, pooled_reference, 0.030)
        pooled_50 = match_events(pooled_predicted, pooled_reference, 0.050)
        rows.append({
            "name": name,
            "overrides": overrides,
            "config": asdict(config),
            "development_only": True,
            "f1_30": pooled_30["f1"],
            "f1_50": pooled_50["f1"],
            "precision_50": pooled_50["precision"],
            "recall_50": pooled_50["recall"],
            "tp_50": pooled_50["tp"],
            "fp_50": pooled_50["fp"],
            "fn_50": pooled_50["fn"],
            "per_recording": per_recording,
        })
        print(
            f"{name:16s} F1@30={pooled_30['f1']:.3f} "
            f"F1@50={pooled_50['f1']:.3f} "
            f"P/R@50={pooled_50['precision']:.3f}/{pooled_50['recall']:.3f}"
        )

    eligible = [row for row in rows if row["precision_50"] >= 0.80]
    ranked = sorted(
        eligible or rows,
        key=lambda row: (row["f1_50"], row["f1_30"], row["precision_50"]),
        reverse=True,
    )
    payload = {
        "recordings": args.recordings,
        "selection_rule": "max F1@50, then F1@30, with development precision@50 >= 0.80",
        "warning": "Development-set result; validate on held-out speakers before adoption.",
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best development preset: {ranked[0]['name']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
