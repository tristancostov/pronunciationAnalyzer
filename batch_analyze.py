#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетный запуск pronunciationAnalyzer.py для всех записей.
Перегенерирует все *_syllable_analysis.json с обновлёнными алгоритмами
(улучшенное вокалическое ядро, core-based stress detection).

Использование:
    python batch_analyze.py           # все записи
    python batch_analyze.py 1local    # одна запись
"""

import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR  = os.path.join(SCRIPT_DIR, "audio")
TEXT_DIR   = os.path.join(SCRIPT_DIR, "text")

import pronunciationAnalyzer as pa


def find_pairs(names=None):
    """Возвращает список (wav_path, txt_path, name)."""
    wavs = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    pairs = []
    for wav in sorted(wavs):
        name = os.path.basename(wav).replace(".wav", "")
        if names and name not in names:
            continue
        txt = os.path.join(TEXT_DIR, f"{name}.txt")
        if not os.path.exists(txt):
            print(f"⚠ Пропущена {name}: нет {txt}")
            continue
        pairs.append((wav, txt, name))
    return pairs


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else None
    pairs = find_pairs(names)
    if not pairs:
        print("❌ Нет пар WAV+TXT для обработки.")
        return

    print(f"🔍 Найдено {len(pairs)} записей для анализа.\n")

    for idx, (wav, txt, name) in enumerate(pairs, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(pairs)}] {name}")
        print(f"{'='*60}")

        pa.audioFile = wav
        pa.textFile  = txt

        # main() печатает отчёт и сохраняет JSON рядом с WAV
        pa.main()

    print(f"\n✅ Готово. Обработано {len(pairs)} записей.")
    print(f"   JSON-файлы лежат в: {AUDIO_DIR}")


if __name__ == "__main__":
    main()
