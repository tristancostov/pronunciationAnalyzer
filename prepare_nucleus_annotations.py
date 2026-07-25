#!/usr/bin/env python3
"""Generate non-destructive Praat preannotations for V2 nucleus review.

When a manual ``syllable`` tier exists, it is used only to place one editable
seed inside each already annotated syllable.  The independent V2 predictions
are always saved separately in JSON and never become ground truth until a
human reviews and saves the TextGrid as ``*_v2_reviewed.TextGrid``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from acoustic_nuclei_v2 import NucleusDetection, NucleusV2Config, detect_file
from nuclei_annotations import (
    TextGridPoint,
    append_point_tier,
    create_textgrid,
    read_interval_tier,
    read_textgrid_file,
)


AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm")
RUSSIAN_VOWELS = frozenset("аеёиоуыэюя")


def find_audio(audio_dir: Path, recording: str) -> Path:
    for suffix in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{recording}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No audio file found for {recording!r} in {audio_dir}")


def has_written_vowel(label: str) -> bool:
    """Return whether a manual syllable label can contain an acoustic nucleus."""
    return any(character.lower() in RUSSIAN_VOWELS for character in label)


def seed_from_syllables(textgrid: str, detection: NucleusDetection) -> list[TextGridPoint]:
    intervals = [
        interval for interval in read_interval_tier(textgrid, "syllable")
        if has_written_vowel(interval.text)
    ]
    return [
        TextGridPoint(
            time=detection.best_time_in_interval(interval.start, interval.end),
            mark="N",
        )
        for interval in intervals
    ]


def prediction_points(detection: NucleusDetection) -> list[TextGridPoint]:
    return [TextGridPoint(item.time_seconds, "N") for item in detection.nuclei]


def interval_diagnostics(textgrid: str, detection: NucleusDetection) -> dict:
    """Provisional count diagnostics from existing syllable intervals.

    This is useful before nucleus centres are reviewed, but it is deliberately
    not called event F1: an interval only tells us whether a prediction landed
    somewhere inside a manually segmented syllable.
    """
    labelled_intervals = read_interval_tier(textgrid, "syllable")
    intervals = [item for item in labelled_intervals if has_written_vowel(item.text)]
    if not intervals or not labelled_intervals:
        return {}
    counts = []
    for interval in intervals:
        counts.append(sum(
            interval.start <= candidate.time_seconds <= interval.end
            for candidate in detection.nuclei
        ))
    span_start = min(item.start for item in labelled_intervals)
    span_end = max(item.end for item in labelled_intervals)
    predictions_in_span = [
        candidate.time_seconds for candidate in detection.nuclei
        if span_start <= candidate.time_seconds <= span_end
    ]
    predictions_in_gaps = sum(
        not any(item.start <= time <= item.end for item in intervals)
        for time in predictions_in_span
    )
    return {
        "provisional_only": True,
        "annotated_span_seconds": [span_start, span_end],
        "manual_syllable_intervals": len(intervals),
        "vowelless_labelled_intervals_excluded": len(labelled_intervals) - len(intervals),
        "predictions_in_annotated_span": len(predictions_in_span),
        "intervals_with_exactly_one_prediction": sum(value == 1 for value in counts),
        "intervals_with_no_prediction": sum(value == 0 for value in counts),
        "intervals_with_multiple_predictions": sum(value > 1 for value in counts),
        "predictions_in_unlabelled_gaps": predictions_in_gaps,
        "exactly_one_interval_accuracy_pct": (
            sum(value == 1 for value in counts) / len(counts) * 100.0
        ),
        "interval_coverage_recall_pct": (
            sum(value >= 1 for value in counts) / len(counts) * 100.0
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", default=["3local", "6fori", "7fori"])
    parser.add_argument("--audio-dir", type=Path, default=root / "audio")
    parser.add_argument("--textgrid-dir", type=Path, default=root / "analysis")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "analysis" / "v2_annotations"
    )
    parser.add_argument("--tier", default="nucleus")
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Do not seed one point per existing manual syllable interval",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Calibration JSON whose best overrides should seed validation files",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> tuple[NucleusV2Config, str]:
    if path is None:
        return NucleusV2Config(), "default"
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best") or {}
    overrides = best.get("overrides") or {}
    return replace(NucleusV2Config(), **overrides), str(best.get("name", path.stem))


def main() -> None:
    args = parse_args()
    config, config_name = load_config(args.config_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for recording in args.recordings:
        audio = find_audio(args.audio_dir, recording)
        detection = detect_file(audio, config)
        source_grid = args.textgrid_dir / f"{recording}.TextGrid"
        source_text = read_textgrid_file(source_grid) if source_grid.is_file() else None

        seeded_from_manual_intervals = False
        provisional_diagnostics = {}
        if source_text is not None and not args.predictions_only:
            try:
                points = seed_from_syllables(source_text, detection)
                seeded_from_manual_intervals = True
                provisional_diagnostics = interval_diagnostics(source_text, detection)
            except ValueError:
                points = prediction_points(detection)
        else:
            points = prediction_points(detection)

        duration = detection.duration_seconds
        if source_text is not None:
            output_text = append_point_tier(source_text, args.tier, points)
        else:
            output_text = create_textgrid(0.0, duration, args.tier, points)

        review_path = args.output_dir / f"{recording}_v2_to_review.TextGrid"
        review_path.write_text(output_text, encoding="utf-8")
        candidates_path = args.output_dir / f"{recording}_v2_candidates.json"
        candidate_payload = detection.to_dict(include_tracks=False)
        candidate_payload.update({
            "recording": recording,
            "audio": str(audio.resolve()),
            "review_textgrid": str(review_path.resolve()),
            "seeded_from_manual_syllable_intervals": seeded_from_manual_intervals,
            "seed_count": len(points),
            "review_required": True,
            "interval_diagnostics": provisional_diagnostics,
            "config_name": config_name,
            "config": asdict(config),
        })
        candidates_path.write_text(
            json.dumps(candidate_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append({
            "recording": recording,
            "independent_prediction_count": len(detection.nuclei),
            "review_point_count": len(points),
            "seeded_from_manual_syllable_intervals": seeded_from_manual_intervals,
            "interval_diagnostics": provisional_diagnostics,
            "config_name": config_name,
            "textgrid": str(review_path.resolve()),
            "candidates": str(candidates_path.resolve()),
        })
        print(
            f"{recording}: {len(detection.nuclei)} independent predictions; "
            f"{len(points)} points to review -> {review_path}"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"recordings": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
