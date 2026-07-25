"""Measure reference-free ASR word error rate on recordings with transcripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from asr_backends import recognize_whisper
from audio_input import prepare_audio


ROOT = Path(__file__).resolve().parent


def words(text):
    # Treat е/ё as the same grapheme for ASR benchmarking. Stress and vowel
    # quality are evaluated separately by the pronunciation analyzer.
    return re.findall(r"[0-9a-zа-я-]+", text.lower().replace("ё", "е"), re.I)


def edit_counts(reference, hypothesis):
    """Return Levenshtein substitutions, deletions and insertions."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    dp = [[None] * cols for _ in range(rows)]
    dp[0][0] = (0, 0, 0)
    for i in range(1, rows):
        dp[i][0] = (0, i, 0)
    for j in range(1, cols):
        dp[0][j] = (0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            candidates = [
                (sub[0] + 1, sub[1], sub[2]),
                (delete[0], delete[1] + 1, delete[2]),
                (insert[0], insert[1], insert[2] + 1),
            ]
            dp[i][j] = min(candidates, key=lambda item: (sum(item), item))
    return dp[-1][-1]


def metrics(reference_text, recognized_text):
    ref, hyp = words(reference_text), words(recognized_text)
    substitutions, deletions, insertions = edit_counts(ref, hyp)
    errors = substitutions + deletions + insertions
    return {
        "referenceWords": len(ref),
        "recognizedWords": len(hyp),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "werPct": round(errors / len(ref) * 100, 2) if ref else None,
    }


def load_vosk_text(name):
    path = ROOT / "analysis" / f"{name}_syllable_analysis.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("recognizedText", "")


def benchmark(recordings, include_whisper=True):
    rows = []
    for name in recordings:
        audio = ROOT / "audio" / f"{name}.wav"
        transcript = ROOT / "text" / f"{name}.txt"
        if not audio.is_file() or not transcript.is_file():
            print(f"Пропуск {name}: нет audio/text пары")
            continue
        reference = transcript.read_text(encoding="utf-8").strip()

        vosk_text = load_vosk_text(name)
        if vosk_text is not None:
            row = {"recording": name, "engine": "vosk",
                   **metrics(reference, vosk_text), "text": vosk_text}
            rows.append(row)
            print(f"{name:8} VOSK    WER {row['werPct']:6.2f}%")

        if include_whisper:
            prepared = prepare_audio(audio)
            result = recognize_whisper(prepared.analysis_path, reference_text="")
            row = {"recording": name, "engine": result.engine,
                   **metrics(reference, result.text), "text": result.text}
            rows.append(row)
            print(f"{name:8} Whisper WER {row['werPct']:6.2f}%")
    return rows


def aggregate(rows, engine):
    selected = [row for row in rows if row["engine"] == engine]
    refs = sum(row["referenceWords"] for row in selected)
    errors = sum(row["errors"] for row in selected)
    return {
        "engine": engine,
        "recordings": len(selected),
        "referenceWords": refs,
        "errors": errors,
        "werPct": round(errors / refs * 100, 2) if refs else None,
    }


def save_report(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    engines = list(dict.fromkeys(row["engine"] for row in rows))
    totals = [aggregate(rows, engine) for engine in engines]
    payload = {"rows": rows, "aggregate": totals,
               "normalization": "lowercase, punctuation removed, ё=е"}
    json_path = out_dir / "asr_benchmark.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    lines = [
        "# Сравнение распознавания речи",
        "",
        "WER = (замены + удаления + вставки) / число слов эталона. Ниже — лучше.",
        "Эталон используется только для измерения после распознавания; Whisper",
        "получает пустой prompt и работает как в режиме свободной речи.",
        "",
        "| Запись | Движок | Слов | S | D | I | WER |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['recording']} | {row['engine']} | {row['referenceWords']} | "
            f"{row['substitutions']} | {row['deletions']} | {row['insertions']} | "
            f"{row['werPct']:.2f}% |")
    lines.extend(["", "## Micro-average", "",
                  "| Движок | Записей | Слов | Ошибок | WER |",
                  "|---|---:|---:|---:|---:|"])
    for total in totals:
        lines.append(
            f"| {total['engine']} | {total['recordings']} | "
            f"{total['referenceWords']} | {total['errors']} | "
            f"{total['werPct']:.2f}% |")
    md_path = out_dir / "asr_benchmark.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*",
                        default=["3local", "6fori", "7fori"])
    parser.add_argument("--no-whisper", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    rows = benchmark(args.recordings, include_whisper=not args.no_whisper)
    paths = save_report(rows, Path(args.out_dir))
    print(f"Сохранено: {paths[1]}")


if __name__ == "__main__":
    main()
