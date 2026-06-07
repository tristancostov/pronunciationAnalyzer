#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетный запуск pronunciationAnalyzer.py для всех записей.

Использование:
    python batch_analyze.py                          # все записи → analysis/
    python batch_analyze.py 1local 2local            # указанные записи
    python batch_analyze.py --outdir analysis_f0stab  # в отдельную папку
    python batch_analyze.py 3local --outdir test_out
"""

import os
import sys
import glob

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
    args = sys.argv[1:]
    out_dir = None
    if "--outdir" in args:
        idx = args.index("--outdir")
        out_dir = args[idx + 1] if idx + 1 < len(args) else None
        args = args[:idx] + args[idx + 2:]
    out_dir = out_dir or os.path.join(SCRIPT_DIR, "analysis")
    names = args if args else None

    pairs = find_pairs(names)
    if not pairs:
        print("❌ Нет пар WAV+TXT для обработки.")
        return

    print(f"📁 Выходная папка: {out_dir}")
    print("Загрузка моделей (VOSK + pymorphy2 + ruaccent)…")
    models = pa.loadModels()
    print(f"🔍 Найдено {len(pairs)} записей для анализа.\n")

    for idx, (wav, txt, name) in enumerate(pairs, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(pairs)}] {name}")
        print(f"{'='*60}")

        pa.audioFile = wav
        pa.textFile  = txt
        pa.main(models=models, outDir=out_dir)

    print(f"\n✅ Готово. Обработано {len(pairs)} записей → {out_dir}")


if __name__ == "__main__":
    main()
