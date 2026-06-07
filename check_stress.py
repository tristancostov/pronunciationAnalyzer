#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_stress.py — Проверка детектора ударения без ручной разметки границ.
=====================================================================
Для каждой многосложной записи выводит таблицу: слово | слоги |
словарное ударение (ruaccent) | акустическое ударение.

Вывод: консоль + stress_check_report.md (таблица со всеми расхождениями).

ИСПОЛЬЗОВАНИЕ:
    python check_stress.py                    # все записи
    python check_stress.py 3local             # одна запись
    python check_stress.py 3local --detail    # показать все слова (не только расхождения)
"""

import json, os, sys, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR   = os.path.join(SCRIPT_DIR, "analysis")
OUTPUT_MD  = os.path.join(SCRIPT_DIR, "stress_check_report.md")


def check_recording(name, show_detail=False):
    path = os.path.join(JSON_DIR, f"{name}_syllable_analysis.json")
    if not os.path.exists(path):
        return None, f"❌ Нет файла: {path}"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    words = data["wordAnalysis"]
    multi = [w for w in words if w["syllableCount"] > 1]
    if not multi:
        return None, f"  (нет многосложных слов)"

    match = sum(1 for w in multi
                if w.get("expectedStressedIdx", -1) == w.get("actualStressedIdx", -2))
    total = len(multi)
    rate  = match / total * 100

    kind = "носитель" if "local" in name else "неноситель"

    lines = []
    lines.append(f"\n## {name} ({kind})")
    lines.append(f"**Совпадений: {match}/{total} = {rate:.1f}%**\n")

    # Статистика по длине слова
    by_len = {}
    for w in multi:
        n = w["syllableCount"]
        ok = w.get("expectedStressedIdx", -1) == w.get("actualStressedIdx", -2)
        by_len.setdefault(n, {"ok": 0, "all": 0})
        by_len[n]["all"] += 1
        if ok: by_len[n]["ok"] += 1
    lines.append("| Сложность | Совпало | Всего | % |")
    lines.append("|---|---|---|---|")
    for n in sorted(by_len):
        d = by_len[n]
        pct = f"{d['ok']/d['all']*100:.0f}" if d['all'] > 0 else "—"
        lines.append(f"| {n}-сложные | {d['ok']} | {d['all']} | {pct} |")
    lines.append("")

    mismatches = [w for w in multi
                  if w.get("expectedStressedIdx", -1) != w.get("actualStressedIdx", -2)]

    if show_detail:
        lines.append("| Слово | Слоги | Словарь | Акустика | ✓/✗ |")
        lines.append("|---|---|---|---|---|")
        for w in multi:
            exp = w.get("expectedStressedIdx", -1)
            act = w.get("actualStressedIdx", -1)
            ok = "✓" if exp == act else "✗"
            lines.append(f"| {w['word']} | {w['syllableStr']} | слог {exp+1} | слог {act+1} | {ok} |")
    else:
        lines.append(f"**Расхождения ({len(mismatches)} из {total}):**\n")
        lines.append("| Слово | Слоги | Словарь | Акустика |")
        lines.append("|---|---|---|---|")
        for w in mismatches:
            exp = w.get("expectedStressedIdx", -1)
            act = w.get("actualStressedIdx", -1)
            lines.append(f"| {w['word']} | {w['syllableStr']} | слог {exp+1} | слог {act+1} |")
        if not mismatches:
            lines.append("| ✓ | Все совпали | — | — |")

    console = f"\n{'='*70}\n  {name} ({kind}) | {match}/{total} = {rate:.1f}%\n{'='*70}"

    # Статистика по длине слова
    by_len = {}
    for w in multi:
        n = w["syllableCount"]
        ok = w.get("expectedStressedIdx", -1) == w.get("actualStressedIdx", -2)
        by_len.setdefault(n, {"ok": 0, "all": 0})
        by_len[n]["all"] += 1
        if ok: by_len[n]["ok"] += 1
    console += "\n  По числу слогов:"
    for n in sorted(by_len):
        d = by_len[n]
        console += f"  {n}-сложные: {d['ok']}/{d['all']} ({d['ok']/d['all']*100:.0f}%)"

    if mismatches and not show_detail:
        # Группируем расхождения по длине
        console += f"\n  Расхождения (первые 10):"
        for w in mismatches[:10]:
            exp = w.get("expectedStressedIdx", -1)
            act = w.get("actualStressedIdx", -1)
            console += f"\n    {w['word']:<15} {w['syllableStr']:<28} словарь→{exp+1} акуст→{act+1}"
        if len(mismatches) > 10:
            console += f"\n    ... и ещё {len(mismatches)-10} слов (см. отчёт)"

    return (match, total, kind, lines), console


def main():
    global JSON_DIR
    # Поддержка --jsondir
    if "--jsondir" in sys.argv:
        idx = sys.argv.index("--jsondir")
        JSON_DIR = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else JSON_DIR
        sys.argv = [a for a in sys.argv if a != "--jsondir" and a != JSON_DIR]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        names = [sys.argv[1]]
    else:
        files = glob.glob(os.path.join(JSON_DIR, "*_syllable_analysis.json"))
        names = sorted({os.path.basename(p)[:-len("_syllable_analysis.json")]
                       for p in files})
    show_detail = "--detail" in sys.argv

    total_match, total_all = 0, 0
    all_lines = [
        "# Проверка детектора ударения",
        f"Словарное ударение (ruaccent) vs акустический детектор.",
        f"У носителей (local) расхождение = ошибка детектора.",
        f"У неносителей (fori) расхождение = возможная ошибка произношения.\n",
    ]

    for name in names:
        res, console = check_recording(name, show_detail)
        print(console)
        if res:
            total_match += res[0]
            total_all += res[1]
            all_lines.extend(res[3])

    if total_all > 0:
        summary = f"\n---\n## ИТОГО: {total_match}/{total_all} = {total_match/total_all*100:.1f}%"
        all_lines.append(summary)
        print(f"\n{'='*70}\n  ИТОГО: {total_match}/{total_all} = {total_match/total_all*100:.1f}%\n{'='*70}")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\n📄 Отчёт сохранён: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
