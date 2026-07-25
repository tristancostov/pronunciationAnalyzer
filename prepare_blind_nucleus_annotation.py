#!/usr/bin/env python3
"""Create an empty Praat nucleus tier for a genuinely blind test recording.

Unlike ``prepare_nucleus_annotations.py``, this helper never runs a detector
and never inserts automatic candidates.  It is intended for held-out test
recordings whose human annotation must not be biased by model predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import wave

from audio_input import prepare_audio
from nuclei_annotations import create_textgrid
from prepare_nucleus_annotations import find_audio


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Recording stem, for example 8fori")
    parser.add_argument("--audio-dir", type=Path, default=root / "audio")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "analysis" / "v2_annotations",
    )
    parser.add_argument("--tier", default="nucleus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio = find_audio(args.audio_dir, args.recording)
    prepared = prepare_audio(audio)
    with wave.open(prepared.analysis_path, "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.recording}_v2_blind.TextGrid"
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite blind annotation: {output}"
        )
    output.write_text(
        create_textgrid(0.0, duration, args.tier, []), encoding="utf-8"
    )
    print(f"Blind empty annotation ({duration:.3f} s): {output.resolve()}")


if __name__ == "__main__":
    main()
