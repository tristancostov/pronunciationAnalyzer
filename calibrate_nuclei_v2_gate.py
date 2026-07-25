#!/usr/bin/env python3
"""Calibrate a noise gate after permissive V2 candidate generation.

After this script uses a recording, that recording becomes development data.
Keep at least one different speaker untouched for final evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
from statistics import mean

from acoustic_nuclei_v2 import NucleusCandidate, NucleusV2Config, detect_file
from evaluate_nuclei_v2 import match_events
from calibrate_nuclei_v2 import load_reference
from prepare_nucleus_annotations import find_audio, load_config


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", help="Reviewed development recordings")
    parser.add_argument("--audio-dir", type=Path, default=root / "audio")
    parser.add_argument(
        "--annotation-dir", type=Path, default=root / "analysis" / "v2_annotations"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=root / "results" / "nuclei_v2_calibration_3local.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "nuclei_v2_gate_calibration.json",
    )
    return parser.parse_args()


def apply_gate(
    candidates: list[NucleusCandidate],
    periodicity: float,
    strength: float,
    vowel: float,
) -> list[float]:
    return [
        item.time_seconds for item in candidates
        if item.weak_recovery or (
            item.periodicity >= periodicity
            and item.strength >= strength
            and item.vowel_likeness >= vowel
        )
    ]


def main() -> None:
    args = parse_args()
    base_config, base_name = load_config(args.base_config)
    data = {}
    for recording in args.recordings:
        reference, start, end = load_reference(args.annotation_dir, recording)
        detection = detect_file(find_audio(args.audio_dir, recording), base_config)
        candidates = [
            item for item in detection.nuclei if start <= item.time_seconds <= end
        ]
        data[recording] = {
            "reference": reference,
            "candidates": candidates,
        }

    rows = []
    for periodicity, strength, vowel in itertools.product(
        (0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        (0.0, 0.25, 0.30, 0.35, 0.40, 0.45),
        (0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
    ):
        per_recording = []
        pooled_predicted = []
        pooled_reference = []
        offset = 0.0
        for recording in args.recordings:
            candidates = data[recording]["candidates"]
            reference = data[recording]["reference"]
            predicted = apply_gate(candidates, periodicity, strength, vowel)
            metrics = match_events(predicted, reference, 0.050)
            per_recording.append({
                "recording": recording,
                "predicted_count": len(predicted),
                "precision_50": metrics["precision"],
                "recall_50": metrics["recall"],
                "f1_50": metrics["f1"],
            })
            pooled_predicted.extend(value + offset for value in predicted)
            pooled_reference.extend(value + offset for value in reference)
            offset += max(reference + predicted, default=0.0) + 1.0
        pooled = match_events(pooled_predicted, pooled_reference, 0.050)
        f1_values = [item["f1_50"] for item in per_recording]
        rows.append({
            "gate": {
                "strong_min_periodicity": periodicity,
                "strong_min_strength": strength,
                "strong_min_vowel_likeness": vowel,
            },
            "macro_f1_50": mean(f1_values),
            "minimum_recording_f1_50": min(f1_values),
            "pooled_f1_50": pooled["f1"],
            "pooled_precision_50": pooled["precision"],
            "pooled_recall_50": pooled["recall"],
            "per_recording": per_recording,
        })

    rows.sort(
        key=lambda row: (
            row["macro_f1_50"],
            row["minimum_recording_f1_50"],
            row["pooled_precision_50"],
        ),
        reverse=True,
    )
    base_payload = json.loads(args.base_config.read_text(encoding="utf-8"))
    base_overrides = dict((base_payload.get("best") or {}).get("overrides") or {})
    best_gate = rows[0]["gate"]
    combined_overrides = {**base_overrides, **best_gate}
    best_config = replace(NucleusV2Config(), **combined_overrides)
    best = {
        "name": f"{base_name}_noise_gate",
        "overrides": combined_overrides,
        "config": asdict(best_config),
        **rows[0],
    }
    payload = {
        "recordings": args.recordings,
        "development_only": True,
        "warning": "Reserve a different speaker for final untouched evaluation.",
        "base_config": str(args.base_config.resolve()),
        "best": best,
        "top_20": rows[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
