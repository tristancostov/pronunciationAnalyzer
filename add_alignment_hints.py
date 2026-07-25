#!/usr/bin/env python3
"""Add ASR word/syllable hint tiers to a nucleus-review TextGrid.

The hint tiers make difficult manual nucleus review possible without changing
the ground-truth rule: only the ``nucleus`` point tier is human-edited and used
for evaluation.  Recognized text and automatically divided syllables are
context hints, never reference labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nuclei_annotations import (
    TextGridInterval,
    TextGridPoint,
    append_interval_tier,
    append_point_tier,
    read_point_tier,
    read_textgrid_file,
    textgrid_bounds,
)


def complete_partition(
    labelled: list[TextGridInterval], xmin: float, xmax: float,
) -> list[TextGridInterval]:
    """Clip possibly overlapping ASR spans into a valid Praat partition."""
    output: list[TextGridInterval] = []
    cursor = xmin
    for item in sorted(labelled, key=lambda value: (value.start, value.end)):
        start = min(xmax, max(xmin, float(item.start)))
        end = min(xmax, max(start, float(item.end)))
        if end <= cursor + 1e-7:
            continue
        start = max(start, cursor)
        if start > cursor + 1e-7:
            output.append(TextGridInterval(cursor, start, ""))
        output.append(TextGridInterval(start, end, item.text))
        cursor = end
    if cursor < xmax - 1e-7:
        output.append(TextGridInterval(cursor, xmax, ""))
    if not output:
        output = [TextGridInterval(xmin, xmax, "")]
    # Remove floating-point hairline gaps at tier boundaries.
    normalized = []
    cursor = xmin
    for item in output:
        normalized.append(TextGridInterval(cursor, item.end, item.text))
        cursor = item.end
    normalized[-1] = TextGridInterval(
        normalized[-1].start, xmax, normalized[-1].text
    )
    return normalized


def hint_intervals(payload: dict, xmin: float, xmax: float):
    words = []
    syllables = []
    for word in payload.get("wordAnalysis", []):
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        label = str(word.get("word", "")).strip()
        words.append(TextGridInterval(start, end, label))
        for syllable in word.get("syllableAnalysis", []):
            relative_start = float(syllable.get("startSec", 0.0))
            relative_end = float(syllable.get("endSec", relative_start))
            # Syllable times in the analysis JSON are relative to the word.
            syllables.append(TextGridInterval(
                start + relative_start,
                min(end, start + relative_end),
                str(syllable.get("syllable", "")).strip(),
            ))
    return (
        complete_partition(words, xmin, xmax),
        complete_partition(syllables, xmin, xmax),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("textgrid", type=Path)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--base-textgrid", type=Path,
        help="Use this TextGrid as the tier base and copy nucleus points from the input.",
    )
    parser.add_argument(
        "--unreviewed-after", type=float,
        help="Relabel copied nucleus points after this time as '?' pending review.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_text = read_textgrid_file(args.textgrid)
    text = (
        read_textgrid_file(args.base_textgrid)
        if args.base_textgrid else source_text
    )
    if args.base_textgrid:
        points = read_point_tier(source_text, "nucleus")
        if args.unreviewed_after is not None:
            points = [
                TextGridPoint(
                    point.time,
                    "?" if point.time > args.unreviewed_after else point.mark,
                )
                for point in points
            ]
        text = append_point_tier(text, "nucleus", points)
    elif args.unreviewed_after is not None:
        raise ValueError("--unreviewed-after requires --base-textgrid")
    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    xmin, xmax = textgrid_bounds(text)
    word_tier, syllable_tier = hint_intervals(payload, xmin, xmax)
    text = append_interval_tier(text, "word_hint", word_tier, xmin, xmax)
    text = append_interval_tier(
        text, "syllable_hint", syllable_tier, xmin, xmax
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        f"Saved {len(word_tier)} word intervals and "
        f"{len(syllable_tier)} syllable intervals to {args.output}"
    )


if __name__ == "__main__":
    main()
