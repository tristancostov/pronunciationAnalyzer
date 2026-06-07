#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нарезка слогов в отдельные .wav из готового *_syllable_analysis.json.

Использование:
    python extraSegAudio.py                          # все записи → syllables_export/
    python extraSegAudio.py 3local                   # одна запись
    python extraSegAudio.py 3local --max 50          # максимум 50 слогов
    python extraSegAudio.py 3local --out my_syllables
"""

import json, wave, os, sys, glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR   = os.path.join(SCRIPT_DIR, "audio")
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")
EXPORT_BASE  = os.path.join(SCRIPT_DIR, "syllables_export")


def export_recording(name, max_count=None, out_dir=None):
    json_path = os.path.join(ANALYSIS_DIR, f"{name}_syllable_analysis.json")
    wav_path  = os.path.join(AUDIO_DIR,  f"{name}.wav")

    if not os.path.exists(json_path):
        print(f"❌ Нет JSON: {json_path}"); return
    if not os.path.exists(wav_path):
        print(f"❌ Нет WAV: {wav_path}"); return

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    wf = wave.open(wav_path, "rb")
    sr, nch, sampW = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    allBytes = wf.readframes(wf.getnframes())
    wf.close()

    out_dir = out_dir or os.path.join(EXPORT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    words = report.get("wordAnalysis", [])
    for wordItem in words:
        if max_count and count >= max_count:
            break
        wordStart = wordItem.get("start", 0)
        wordText  = wordItem.get("word", "?")
        for sylItem in wordItem.get("syllableAnalysis", []):
            if max_count and count >= max_count:
                break
            sylText = sylItem.get("syllable", "?")
            absStart = wordStart + sylItem.get("startSec", 0)
            absEnd   = wordStart + sylItem.get("endSec", 0)

            startByte = int(absStart * sr) * sampW * nch
            endByte   = int(absEnd   * sr) * sampW * nch
            chunk = allBytes[startByte:endByte]

            if len(chunk) < 100:
                continue

            fname = os.path.join(out_dir, f"{count+1:03d}_{wordText}_{sylText}.wav")
            out = wave.open(fname, "wb")
            out.setnchannels(nch)
            out.setsampwidth(sampW)
            out.setframerate(sr)
            out.writeframes(chunk)
            out.close()

            print(f"  [{count+1:03d}] {wordText} → «{sylText}»  "
                  f"{absStart:.3f}–{absEnd:.3f}s")
            count += 1

    print(f"\n✅ {name}: {count} слогов → {out_dir}")


def main():
    args = sys.argv[1:]
    out_dir = None; max_count = None
    if "--out" in args:
        idx = args.index("--out")
        out_dir = args[idx + 1] if idx + 1 < len(args) else None
        args = args[:idx] + args[idx + 2:]
    if "--max" in args:
        idx = args.index("--max")
        try: max_count = int(args[idx + 1])
        except: pass
        args = args[:idx] + args[idx + 2:]

    if args:
        names = args
    else:
        files = glob.glob(os.path.join(ANALYSIS_DIR, "*_syllable_analysis.json"))
        names = sorted(os.path.basename(p).replace("_syllable_analysis.json", "")
                      for p in files)

    if not names:
        print("❌ Нет записей для экспорта.")
        return

    print(f"🔪 Нарезка слогов для {len(names)} записей → {out_dir or EXPORT_BASE}\n")
    for name in names:
        export_recording(name, max_count=max_count, out_dir=out_dir)


if __name__ == "__main__":
    main()
