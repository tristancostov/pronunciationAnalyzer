#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетный запуск pronunciationAnalyzer.py для всех записей.

Использование:
    python batch_analyze.py                          # все записи, hybrid ASR
    python batch_analyze.py 1local 2local            # указанные записи
    python batch_analyze.py --asr vosk               # только VOSK
    python batch_analyze.py 3local --free-speech     # без reference/prompt
    python batch_analyze.py --outdir analysis_test   # в отдельную папку
"""

import os
import glob
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR  = os.path.join(SCRIPT_DIR, "audio")
TEXT_DIR   = os.path.join(SCRIPT_DIR, "text")

import pronunciationAnalyzer as pa


def find_pairs(names=None):
    wavs = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    pairs = []
    for wav in sorted(wavs):
        name = os.path.basename(wav).replace(".wav", "")
        if names and name not in names:
            continue
        txt = os.path.join(TEXT_DIR, f"{name}.txt")
        pairs.append((wav, txt, name))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Имена записей без расширения")
    parser.add_argument("--outdir", default=os.path.join(SCRIPT_DIR, "analysis"))
    parser.add_argument("--asr", choices=("vosk", "whisper"), default="whisper")
    parser.add_argument(
        "--free-speech", action="store_true",
        help="Не передавать эталон распознавателю и не считать wordAccuracy")
    args = parser.parse_args()
    out_dir = args.outdir
    names = args.names or None

    pairs = find_pairs(names)
    if not pairs:
        print("❌ Нет пар WAV+TXT для обработки.")
        return

    print(f"📁 Выходная папка: {out_dir}")
    print(f"Загрузка моделей (ASR={args.asr} + pymorphy2 + ruaccent)…")
    models = pa.loadModels()
    print(f"🔍 Найдено {len(pairs)} записей для анализа.\n")

    for idx, (wav, txt, name) in enumerate(pairs, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(pairs)}] {name}")
        print(f"{'='*60}")

        if args.free_speech:
            reference = ""
        elif os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as handle:
                reference = handle.read().strip()
        else:
            reference = ""
        pa.main(models=models, outDir=out_dir, inputAudio=wav,
                referenceText=reference, asrEngine=args.asr)

    print(f"\n✅ Готово. Обработано {len(pairs)} записей → {out_dir}")


if __name__ == "__main__":
    main()
