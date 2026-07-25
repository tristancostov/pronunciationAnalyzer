#!/usr/bin/env python3
"""Create a short, text-assisted Praat chunk for manual V3 review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from add_alignment_hints import complete_partition
from nuclei_annotations import (
    TextGridInterval,
    TextGridPoint,
    append_interval_tier,
    create_textgrid,
    read_point_tier,
    read_textgrid_file,
)
from prepare_nucleus_annotations import has_written_vowel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_textgrid", type=Path)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    return parser.parse_args()


def clipped_interval(
    start: float, end: float, label: str,
    chunk_start: float, chunk_end: float,
) -> TextGridInterval | None:
    start = max(start, chunk_start)
    end = min(end, chunk_end)
    if end <= start:
        return None
    return TextGridInterval(
        start - chunk_start, end - chunk_start, label.strip()
    )


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise ValueError("--end must be after --start")
    duration = args.end - args.start
    source = read_textgrid_file(args.source_textgrid)
    source_points = read_point_tier(source, "nucleus")
    points = [
        TextGridPoint(point.time - args.start, "?")
        for point in source_points
        if args.start <= point.time <= args.end
    ]
    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    words: list[TextGridInterval] = []
    syllables: list[TextGridInterval] = []
    for word in payload.get("wordAnalysis", []):
        word_start = float(word.get("start", 0.0))
        word_end = float(word.get("end", word_start))
        interval = clipped_interval(
            word_start, word_end, str(word.get("word", "")),
            args.start, args.end,
        )
        if interval is not None:
            words.append(interval)
        for syllable in word.get("syllableAnalysis", []):
            label = str(syllable.get("syllable", "")).strip()
            if not has_written_vowel(label):
                continue
            start = word_start + float(syllable.get("startSec", 0.0))
            end = min(
                word_end,
                word_start + float(syllable.get("endSec", 0.0)),
            )
            interval = clipped_interval(
                start, end, label, args.start, args.end
            )
            if interval is not None:
                syllables.append(interval)

    text = create_textgrid(0.0, duration, "nucleus", points)
    text = append_interval_tier(
        text, "word_hint", complete_partition(words, 0.0, duration)
    )
    text = append_interval_tier(
        text, "syllable", complete_partition(syllables, 0.0, duration)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        f"Saved {duration:.2f}s chunk with {len(points)} nucleus candidates, "
        f"{len(words)} word hints and {len(syllables)} syllable hints: {args.output}"
    )


if __name__ == "__main__":
    main()
