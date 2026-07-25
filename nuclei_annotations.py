"""Small Praat TextGrid helpers for V2 nucleus point annotations.

Only the documented long-text TextGrid representation is written. Existing
TextGrids are never modified in place by these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class TextGridPoint:
    time: float
    mark: str = "N"


@dataclass(frozen=True)
class TextGridInterval:
    start: float
    end: float
    text: str


def _unescape_praat(value: str) -> str:
    return value.replace('""', '"')


def _escape_praat(value: str) -> str:
    return value.replace('"', '""').replace("\r", " ").replace("\n", " ")


def read_textgrid_file(path: str | Path) -> str:
    """Read UTF-8 or UTF-16 TextGrids written by different Praat versions."""
    payload = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeError(f"Unsupported TextGrid encoding: {path}")


def _tier_blocks(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^\s*item \[\d+\]:\s*$", text))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end])
    return blocks


def _tier_name(block: str) -> str | None:
    match = re.search(r'(?m)^\s*name\s*=\s*"((?:[^"]|"")*)"\s*$', block)
    return _unescape_praat(match.group(1)) if match else None


def textgrid_bounds(text: str) -> tuple[float, float]:
    xmin = re.search(rf"(?m)^xmin\s*=\s*({NUMBER})\s*$", text)
    xmax = re.search(rf"(?m)^xmax\s*=\s*({NUMBER})\s*$", text)
    if not xmin or not xmax:
        raise ValueError("TextGrid does not contain top-level xmin/xmax")
    return float(xmin.group(1)), float(xmax.group(1))


def read_point_tier(text: str, tier_name: str = "nucleus") -> list[TextGridPoint]:
    for block in _tier_blocks(text):
        if _tier_name(block) != tier_name:
            continue
        if not re.search(r'(?m)^\s*class\s*=\s*"TextTier"\s*$', block):
            raise ValueError(f"Tier {tier_name!r} exists but is not a TextTier")
        pattern = re.compile(
            rf"points \[\d+\]:\s*"
            rf"number\s*=\s*({NUMBER})\s*"
            rf'mark\s*=\s*"((?:[^"]|"")*)"',
            re.MULTILINE,
        )
        return [
            TextGridPoint(float(number), _unescape_praat(mark))
            for number, mark in pattern.findall(block)
        ]
    raise ValueError(f"Point tier {tier_name!r} was not found")


def read_interval_tier(
    text: str,
    tier_name: str = "syllable",
    include_empty: bool = False,
) -> list[TextGridInterval]:
    for block in _tier_blocks(text):
        if _tier_name(block) != tier_name:
            continue
        if not re.search(r'(?m)^\s*class\s*=\s*"IntervalTier"\s*$', block):
            raise ValueError(f"Tier {tier_name!r} exists but is not an IntervalTier")
        pattern = re.compile(
            rf"intervals \[\d+\]:\s*"
            rf"xmin\s*=\s*({NUMBER})\s*"
            rf"xmax\s*=\s*({NUMBER})\s*"
            rf'text\s*=\s*"((?:[^"]|"")*)"',
            re.MULTILINE,
        )
        intervals = [
            TextGridInterval(float(start), float(end), _unescape_praat(mark))
            for start, end, mark in pattern.findall(block)
        ]
        return intervals if include_empty else [item for item in intervals if item.text.strip()]
    raise ValueError(f"Interval tier {tier_name!r} was not found")


def create_textgrid(
    xmin: float,
    xmax: float,
    tier_name: str,
    points: Iterable[TextGridPoint],
) -> str:
    base = (
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n\n'
        f"xmin = {xmin:.9f}\n"
        f"xmax = {xmax:.9f}\n"
        "tiers? <exists>\n"
        "size = 0\n"
        "item []:\n"
    )
    return append_point_tier(base, tier_name, points, xmin=xmin, xmax=xmax)


def append_point_tier(
    text: str,
    tier_name: str,
    points: Iterable[TextGridPoint],
    xmin: float | None = None,
    xmax: float | None = None,
) -> str:
    if any(_tier_name(block) == tier_name for block in _tier_blocks(text)):
        raise ValueError(f"Tier {tier_name!r} already exists")
    if xmin is None or xmax is None:
        xmin, xmax = textgrid_bounds(text)
    assert xmin is not None and xmax is not None

    size_match = re.search(r"(?m)^size\s*=\s*(\d+)\s*$", text)
    if not size_match:
        raise ValueError("TextGrid does not contain a top-level tier count")
    old_size = int(size_match.group(1))
    new_size = old_size + 1
    updated = text[:size_match.start()] + f"size = {new_size}" + text[size_match.end():]

    ordered = sorted(points, key=lambda item: item.time)
    lines = [
        f"    item [{new_size}]:",
        '        class = "TextTier"',
        f'        name = "{_escape_praat(tier_name)}"',
        f"        xmin = {xmin:.9f}",
        f"        xmax = {xmax:.9f}",
        f"        points: size = {len(ordered)}",
    ]
    for index, point in enumerate(ordered, start=1):
        if point.time < xmin - 1e-9 or point.time > xmax + 1e-9:
            raise ValueError(f"Point {point.time} lies outside TextGrid bounds")
        lines.extend([
            f"        points [{index}]:",
            f"            number = {point.time:.9f}",
            f'            mark = "{_escape_praat(point.mark)}"',
        ])
    return updated.rstrip() + "\n" + "\n".join(lines) + "\n"


def append_interval_tier(
    text: str,
    tier_name: str,
    intervals: Iterable[TextGridInterval],
    xmin: float | None = None,
    xmax: float | None = None,
) -> str:
    """Append a complete Praat IntervalTier without changing existing tiers.

    The caller must provide a contiguous partition of the TextGrid domain.
    Enforcing this here prevents subtly malformed hint tiers that Praat may
    display but later refuse to save.
    """
    if any(_tier_name(block) == tier_name for block in _tier_blocks(text)):
        raise ValueError(f"Tier {tier_name!r} already exists")
    if xmin is None or xmax is None:
        xmin, xmax = textgrid_bounds(text)
    assert xmin is not None and xmax is not None

    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    if not ordered:
        ordered = [TextGridInterval(xmin, xmax, "")]
    tolerance = 1e-7
    if abs(ordered[0].start - xmin) > tolerance:
        raise ValueError("Interval tier does not start at the TextGrid xmin")
    if abs(ordered[-1].end - xmax) > tolerance:
        raise ValueError("Interval tier does not end at the TextGrid xmax")
    for index, interval in enumerate(ordered):
        if interval.end <= interval.start:
            raise ValueError("Interval tier contains an empty/reversed interval")
        if index and abs(interval.start - ordered[index - 1].end) > tolerance:
            raise ValueError("Interval tier contains an overlap or gap")

    size_match = re.search(r"(?m)^size\s*=\s*(\d+)\s*$", text)
    if not size_match:
        raise ValueError("TextGrid does not contain a top-level tier count")
    old_size = int(size_match.group(1))
    new_size = old_size + 1
    updated = text[:size_match.start()] + f"size = {new_size}" + text[size_match.end():]

    lines = [
        f"    item [{new_size}]:",
        '        class = "IntervalTier"',
        f'        name = "{_escape_praat(tier_name)}"',
        f"        xmin = {xmin:.9f}",
        f"        xmax = {xmax:.9f}",
        f"        intervals: size = {len(ordered)}",
    ]
    for index, interval in enumerate(ordered, start=1):
        lines.extend([
            f"        intervals [{index}]:",
            f"            xmin = {interval.start:.9f}",
            f"            xmax = {interval.end:.9f}",
            f'            text = "{_escape_praat(interval.text)}"',
        ])
    return updated.rstrip() + "\n" + "\n".join(lines) + "\n"
