#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_annotations.py
=======================
Сравнивает ручную разметку (Praat .TextGrid) с автоматическим выходом
системы (*_syllable_analysis.json) и считает воспроизводимые метрики:

  1. Точность текстового слогоделения (число слогов из текста);
  2. Точность числа акустически найденных ядер;
  3. Ошибка границ слогов (Boundary Error, мс) — среднее, медиана, P90;
  4. Точность ударения — только при явной ручной отметке ударного слога.

Ударение в TextGrid отмечается знаком +, ´, combining acute или буквой ё
в подписи нужного слога, например «мо», «ло», «к+о». Без такой отметки
скрипт не выдаёт псевдо-точность по правилу «самый длинный = ударный».

ИСПОЛЬЗОВАНИЕ
-------------

Вариант 1 — указать конкретные записи (имена без расширения):
    python EvaluateAnnotations.py 6fori 7fori

Вариант 2 — автоматически найти все записи, у которых есть и .TextGrid,
            и _syllable_analysis.json в текущей папке:
    python EvaluateAnnotations.py

Вариант 3 — задать свои каталоги:
    python EvaluateAnnotations.py --textgrid-dir ./annotations \\
                                   --json-dir ./analysis_results \\
                                   --ground ./my_ground_truth.json \\
                                   6fori 7fori

КОНФИГ
------
Эталонный список слов для каждой записи лежит в `ground_truth_words.json`
(можно переопределить через --ground). Чтобы добавить новую размеченную
запись, отредактируйте этот JSON — Python-код менять не надо.

ВЫВОД
-----
Консольный отчёт + `evaluation_results.md` + `evaluation_results.csv`.

"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from statistics import median, mean
from typing import List, Tuple, Dict


# ---------------------------------------------------------------- ground
def load_ground_truth(path: str) -> Dict[str, List[Tuple[str, int]]]:
    """Грузим словарь {имя_записи: [(слово, число_слогов), ...]} из JSON."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, items in raw.items():
        if key.startswith("_"):    # комментарии типа "_comment"
            continue
        out[key] = [(it[0], int(it[1])) for it in items]
    return out


# ------------------------- парсинг TextGrid -----------------------------
def read_textgrid(path: str) -> List[Tuple[float, float, str]]:
    """Возвращает список (xmin, xmax, text) только для непустых интервалов."""
    # Praat пишет в UTF-16 по умолчанию
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    pattern = re.compile(
        r"xmin\s*=\s*([\d.]+)\s*"
        r"xmax\s*=\s*([\d.]+)\s*"
        r"text\s*=\s*\"([^\"]*)\""
    )
    intervals = [(float(a), float(b), t) for a, b, t in pattern.findall(content)]
    # отбрасываем пустые и слой-обёртку (длинный пустой первый интервал)
    return [(a, b, t) for a, b, t in intervals if t.strip() and (b - a) < 5.0]


# ------------------------- группировка слогов в слова -------------------
def allocate_syllables_to_words(syllables, word_counts):
    """По известному списку слов [(name, n_sylls), ...] раскидать слоги
    последовательно: первые n0 слогов — слово 0, следующие n1 — слово 1, и т.д."""
    words, idx = [], 0
    for name, n in word_counts:
        if idx + n > len(syllables):
            break
        words.append({
            "name": name,
            "sylls": syllables[idx:idx + n],
            "start": syllables[idx][0],
            "end":   syllables[idx + n - 1][1],
        })
        idx += n
    return words, idx


def align_system_to_ground(system_words, ground_words):
    """Сопоставляет системные слова (из VOSK) с эталонными по тексту слова.
    VOSK может пропускать или добавлять слова; используем DP-выравнивание
    по равенству нормализованного текста."""
    n, m = len(ground_words), len(system_words)
    # DP: dp[i][j] = (стоимость, шаг)
    INF = float("inf")
    dp = [[INF]*(m+1) for _ in range(n+1)]
    bp = [[None]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(n+1):
        for j in range(m+1):
            if dp[i][j] == INF: continue
            # match
            if i < n and j < m:
                g_name = ground_words[i]["name"].lower()
                s_name = system_words[j]["word"].lower()
                cost = 0 if g_name == s_name else 2
                if dp[i][j] + cost < dp[i+1][j+1]:
                    dp[i+1][j+1] = dp[i][j] + cost
                    bp[i+1][j+1] = ("match" if cost==0 else "sub", i, j)
            # skip ground (VOSK пропустил)
            if i < n and dp[i][j] + 1 < dp[i+1][j]:
                dp[i+1][j] = dp[i][j] + 1
                bp[i+1][j] = ("skip_g", i, j)
            # skip system (VOSK вставил лишнее)
            if j < m and dp[i][j] + 1 < dp[i][j+1]:
                dp[i][j+1] = dp[i][j] + 1
                bp[i][j+1] = ("skip_s", i, j)

    # восстанавливаем
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        op, pi, pj = bp[i][j]
        if op in ("match", "sub"):
            pairs.append((ground_words[pi], system_words[pj], op))
            i, j = pi, pj
        elif op == "skip_g":
            pairs.append((ground_words[pi], None, "skip_g"))
            i = pi
        else:
            pairs.append((None, system_words[pj], "skip_s"))
            j = pj
    pairs.reverse()
    return pairs


# ------------------------- метрики ---------------------------------------
def boundary_errors_for_word(manual_sylls, system_sylls):
    """Для слова с РАВНЫМ числом слогов в разметке и в системе —
    возвращает абсолютные ошибки границ (старт каждого слога) в мс."""
    if len(manual_sylls) != len(system_sylls):
        return []
    errs = []
    # сравниваем стартовые точки каждого слога (для k=0 — старт слова)
    for k in range(len(manual_sylls)):
        m_start = manual_sylls[k][0]
        s_start = system_sylls[k]["startSec"] + system_sylls[k].get("_wordStart", 0)
        errs.append(abs(m_start - s_start) * 1000.0)
    # и конец последнего слога
    m_end = manual_sylls[-1][1]
    s_end = system_sylls[-1]["endSec"] + system_sylls[-1].get("_wordStart", 0)
    errs.append(abs(m_end - s_end) * 1000.0)
    return errs


def manually_marked_stress_idx(manual_sylls):
    """Индекс явно отмеченного ударного слога или None.

    Длительность намеренно не используется: в быстрой русской речи самый
    длинный слог далеко не всегда ударный, поэтому такая эвристика не может
    служить ground truth для оценки акустического детектора.
    """
    marked = []
    for idx, (_, _, label) in enumerate(manual_sylls):
        lower = label.lower()
        if ("+" in label or "\u0301" in label or "´" in label or
                "ё" in lower):
            marked.append(idx)
    return marked[0] if len(marked) == 1 else None


# ------------------------- основной анализ одной записи -------------------
def evaluate_recording(name: str, tg_path: str, js_path: str,
                       ground_truth: Dict[str, List[Tuple[str, int]]]) -> dict:
    """Оценка одной записи. tg_path и js_path — пути к разметке и JSON."""
    if name not in ground_truth:
        raise ValueError(f"Для записи '{name}' нет эталонного списка слов в ground_truth_words.json")

    # 1) Ручная разметка: слоги → слова по эталонному списку
    manual_sylls = read_textgrid(tg_path)
    manual_words, used = allocate_syllables_to_words(manual_sylls, ground_truth[name])
    expected_sylls = sum(n for _, n in ground_truth[name])
    if used != len(manual_sylls):
        print(f"⚠ {name}: размечено {len(manual_sylls)} слогов, "
              f"эталон ожидает {expected_sylls}, "
              f"использовано {used}.")

    # 2) Системные слова
    with open(js_path, encoding="utf-8") as f:
        system_data = json.load(f)
    system_words = []
    for w in system_data["wordAnalysis"]:
        # каждое слогу в системе уже хранит startSec относительно слова — переведём в абсолют
        abs_sylls = []
        for s in w["syllableAnalysis"]:
            abs_sylls.append({
                "syllable": s["syllable"],
                "abs_start": w["start"] + s["startSec"],
                "abs_end":   w["start"] + s["endSec"],
            })
        system_words.append({
            "word": w["word"], "start": w["start"], "end": w["end"],
            "syllableCount": w["syllableCount"],
            "detectedNuclei": w.get("detectedNuclei"),
            "expectedStressedIdx": w.get("expectedStressedIdx",
                                          w.get("stressedIdx", -1)),
            "actualStressedIdx": w.get("actualStressedIdx"),
            "sylls": abs_sylls,
        })

    # 3) Выравнивание по тексту слова
    pairs = align_system_to_ground(system_words, manual_words)

    # 4) Метрики
    text_count_match = 0
    acoustic_count_match = 0
    acoustic_count_total = 0
    paired_for_count = 0
    boundary_errs = []
    stress_total = 0
    stress_dict_hits = 0
    stress_acoustic_hits = 0
    dict_acoustic_total = 0
    dict_acoustic_hits = 0
    per_word_rows = []

    for g, s, op in pairs:
        if g is None or s is None:
            # пропуск
            per_word_rows.append({"word": (g["name"] if g else f"[VOSK:{s['word']}]"),
                                  "op": op, "manual_count": (len(g["sylls"]) if g else "—"),
                                  "text_count": (s["syllableCount"] if s else "—"),
                                  "acoustic_count": (s.get("detectedNuclei") if s else "—"),
                                  "boundary_err_ms": None,
                                  "stress_dict_ok": None,
                                  "stress_acoustic_ok": None})
            continue

        paired_for_count += 1
        m_count = len(g["sylls"])
        text_count = s["syllableCount"]
        text_count_ok = (m_count == text_count)
        detected_count = s.get("detectedNuclei")
        if detected_count is not None:
            acoustic_count_total += 1
            if m_count == detected_count:
                acoustic_count_match += 1

        if text_count_ok:
            text_count_match += 1
            # boundary errors: старт каждого слога + конец последнего
            errs = []
            for k in range(m_count):
                m_start = g["sylls"][k][0]
                ss_start = s["sylls"][k]["abs_start"]
                errs.append(abs(m_start - ss_start) * 1000)
            m_end = g["sylls"][-1][1]
            ss_end = s["sylls"][-1]["abs_end"]
            errs.append(abs(m_end - ss_end) * 1000)
            boundary_errs.extend(errs)
            mean_err = sum(errs) / len(errs)
        else:
            mean_err = None

        stress_dict_ok = None
        stress_acoustic_ok = None
        expected_stress = s.get("expectedStressedIdx")
        actual_stress = s.get("actualStressedIdx")
        valid_stress_indices = (
            text_count_ok and m_count > 1 and
            isinstance(expected_stress, int) and 0 <= expected_stress < m_count and
            isinstance(actual_stress, int) and 0 <= actual_stress < m_count
        )
        if valid_stress_indices:
            dict_acoustic_total += 1
            if expected_stress == actual_stress:
                dict_acoustic_hits += 1

            manual_stress = manually_marked_stress_idx(g["sylls"])
            if manual_stress is not None:
                stress_total += 1
                stress_dict_ok = (manual_stress == expected_stress)
                stress_acoustic_ok = (manual_stress == actual_stress)
                if stress_dict_ok:
                    stress_dict_hits += 1
                if stress_acoustic_ok:
                    stress_acoustic_hits += 1

        per_word_rows.append({
            "word": g["name"], "op": op,
            "manual_count": m_count,
            "text_count": text_count,
            "acoustic_count": detected_count,
            "text_count_ok": text_count_ok,
            "boundary_err_ms": round(mean_err, 1) if mean_err is not None else None,
            "stress_dict_ok": stress_dict_ok,
            "stress_acoustic_ok": stress_acoustic_ok,
        })

    return {
        "name": name,
        "manual_words": len(manual_words),
        "system_words": len(system_words),
        "paired": paired_for_count,
        "text_count_match": text_count_match,
        "text_count_accuracy": (text_count_match / paired_for_count * 100
                                if paired_for_count else 0),
        "acoustic_count_match": acoustic_count_match,
        "acoustic_count_total": acoustic_count_total,
        "acoustic_count_accuracy": (acoustic_count_match / acoustic_count_total * 100
                                    if acoustic_count_total else 0),
        "boundary_n": len(boundary_errs),
        "boundary_mean_ms":   mean(boundary_errs)   if boundary_errs else 0,
        "boundary_median_ms": median(boundary_errs) if boundary_errs else 0,
        "boundary_p90_ms":    sorted(boundary_errs)[int(0.9*len(boundary_errs))] if boundary_errs else 0,
        "_boundary_errors": boundary_errs,
        "stress_total": stress_total,
        "stress_dict_hits": stress_dict_hits,
        "stress_dict_accuracy": (stress_dict_hits / stress_total * 100
                                 if stress_total else None),
        "stress_acoustic_hits": stress_acoustic_hits,
        "stress_acoustic_accuracy": (stress_acoustic_hits / stress_total * 100
                                     if stress_total else None),
        "dict_acoustic_total": dict_acoustic_total,
        "dict_acoustic_hits": dict_acoustic_hits,
        "dict_acoustic_agreement": (dict_acoustic_hits / dict_acoustic_total * 100
                                    if dict_acoustic_total else None),
        "per_word": per_word_rows,
    }


# ------------------------- вывод -----------------------------------------
def print_report(results, output_md="evaluation_results.md",
                 output_csv="evaluation_results.csv"):
    """Печатает и сохраняет отчёт без смешивания разных задач оценки."""
    width = 100
    total_paired = sum(r["paired"] for r in results)
    total_text_hits = sum(r["text_count_match"] for r in results)
    total_acoustic = sum(r["acoustic_count_total"] for r in results)
    total_acoustic_hits = sum(r["acoustic_count_match"] for r in results)
    all_boundary_errors = [err for r in results for err in r["_boundary_errors"]]
    total_stress = sum(r["stress_total"] for r in results)
    total_stress_dict = sum(r["stress_dict_hits"] for r in results)
    total_stress_acoustic = sum(r["stress_acoustic_hits"] for r in results)
    total_agreement = sum(r["dict_acoustic_total"] for r in results)
    total_agreement_hits = sum(r["dict_acoustic_hits"] for r in results)

    boundary_mean = mean(all_boundary_errors) if all_boundary_errors else 0.0
    boundary_median = median(all_boundary_errors) if all_boundary_errors else 0.0
    boundary_p90 = (sorted(all_boundary_errors)[int(0.9 * len(all_boundary_errors))]
                    if all_boundary_errors else 0.0)

    print("=" * width)
    print("  РЕЗУЛЬТАТЫ ОЦЕНКИ — РУЧНАЯ РАЗМЕТКА vs АВТОМАТИЧЕСКИЙ ВЫХОД")
    print("=" * width)
    for result in results:
        print(f"\n[{result['name']}] сопоставлено слов: {result['paired']}")
        print(f"  Текстовое слогоделение:  {result['text_count_match']}/{result['paired']} "
              f"= {result['text_count_accuracy']:.1f}%")
        print(f"  Число акустических ядер: {result['acoustic_count_match']}/"
              f"{result['acoustic_count_total']} = {result['acoustic_count_accuracy']:.1f}%")
        print(f"  Ошибка границ: mean {result['boundary_mean_ms']:.1f} мс, "
              f"median {result['boundary_median_ms']:.1f} мс, "
              f"P90 {result['boundary_p90_ms']:.1f} мс (n={result['boundary_n']})")
        agreement = result["dict_acoustic_agreement"]
        if agreement is not None:
            print(f"  Акустика vs словарь:     {result['dict_acoustic_hits']}/"
                  f"{result['dict_acoustic_total']} = {agreement:.1f}% (не ground truth)")
        if result["stress_total"]:
            print(f"  Ударение vs ручная метка: акустика "
                  f"{result['stress_acoustic_hits']}/{result['stress_total']} = "
                  f"{result['stress_acoustic_accuracy']:.1f}%; словарь "
                  f"{result['stress_dict_hits']}/{result['stress_total']} = "
                  f"{result['stress_dict_accuracy']:.1f}%")
        else:
            print("  Ударение vs ручная метка: — (в TextGrid нет явных меток ударения)")

    print("\n" + "=" * width)
    print(f"  СВОДНО ПО {len(results)} ЗАПИСЯМ")
    print("=" * width)
    if total_paired:
        print(f"  Текстовое слогоделение:  {total_text_hits}/{total_paired} "
              f"= {total_text_hits / total_paired * 100:.1f}%")
    if total_acoustic:
        print(f"  Число акустических ядер: {total_acoustic_hits}/{total_acoustic} "
              f"= {total_acoustic_hits / total_acoustic * 100:.1f}%")
    print(f"  Ошибка границ: mean {boundary_mean:.1f} мс, median {boundary_median:.1f} мс, "
          f"P90 {boundary_p90:.1f} мс (n={len(all_boundary_errors)})")
    if total_agreement:
        print(f"  Акустика vs словарь:     {total_agreement_hits}/{total_agreement} "
              f"= {total_agreement_hits / total_agreement * 100:.1f}% (диагностика)")
    if not total_stress:
        print("  StressAcc не рассчитан: сначала разметьте ударный слог явно.")

    with open(output_md, "w", encoding="utf-8") as report:
        report.write("# Результаты оценки системы\n\n")
        report.write("Текстовое слогоделение, акустический поиск ядер и ручная разметка "
                     "ударения оцениваются отдельно. Правило «самый длинный слог = ударный» "
                     "не используется как ground truth.\n\n")
        report.write(f"## Сводно по {len(results)} записям\n\n")
        report.write("| Метрика | Значение |\n|---|---|\n")
        report.write(f"| Сопоставлено слов | {total_paired} |\n")
        if total_paired:
            report.write(f"| **Точность текстового слогоделения** | "
                         f"**{total_text_hits / total_paired * 100:.1f}%** "
                         f"({total_text_hits}/{total_paired}) |\n")
        if total_acoustic:
            report.write(f"| **Точность числа акустических ядер** | "
                         f"**{total_acoustic_hits / total_acoustic * 100:.1f}%** "
                         f"({total_acoustic_hits}/{total_acoustic}) |\n")
        report.write(f"| **Ошибка границ** | mean **{boundary_mean:.1f} мс**, "
                     f"median **{boundary_median:.1f} мс**, P90 **{boundary_p90:.1f} мс** |\n")
        if total_agreement:
            report.write(f"| Акустика vs словарь (диагностика, не ground truth) | "
                         f"{total_agreement_hits / total_agreement * 100:.1f}% "
                         f"({total_agreement_hits}/{total_agreement}) |\n")
        if total_stress:
            report.write(f"| **StressAcc акустики vs ручная метка** | "
                         f"**{total_stress_acoustic / total_stress * 100:.1f}%** "
                         f"({total_stress_acoustic}/{total_stress}) |\n")
            report.write(f"| Словарь vs ручная метка | "
                         f"{total_stress_dict / total_stress * 100:.1f}% "
                         f"({total_stress_dict}/{total_stress}) |\n")
        else:
            report.write("| StressAcc | —; в TextGrid нет явных ручных меток ударения |\n")

        report.write("\n## По отдельным записям\n\n")
        report.write("| Запись | Слов | TextSylAcc | AcousticNucleiAcc | "
                     "Boundary mean | median | P90 | Dict↔Acoustic | Manual StressAcc |\n")
        report.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for result in results:
            agreement = (f"{result['dict_acoustic_agreement']:.1f}%"
                         if result["dict_acoustic_agreement"] is not None else "—")
            manual_stress = (f"{result['stress_acoustic_accuracy']:.1f}%"
                             if result["stress_acoustic_accuracy"] is not None else "—")
            report.write(f"| {result['name']} | {result['paired']} | "
                         f"{result['text_count_accuracy']:.1f}% | "
                         f"{result['acoustic_count_accuracy']:.1f}% | "
                         f"{result['boundary_mean_ms']:.1f} мс | "
                         f"{result['boundary_median_ms']:.1f} мс | "
                         f"{result['boundary_p90_ms']:.1f} мс | {agreement} | "
                         f"{manual_stress} |\n")

        report.write("\n## Как получить честный StressAcc\n\n")
        report.write("Отметьте ударный слог в TextGrid знаком `+`, `´`, combining acute "
                     "или буквой `ё`. Например: `мо`, `ло`, `к+о`. Если явной отметки нет, "
                     "метрика не вычисляется.\n")

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "recording", "paired_words", "text_syllable_accuracy_pct",
            "acoustic_nuclei_accuracy_pct", "boundary_mean_ms",
            "boundary_median_ms", "boundary_p90_ms",
            "dict_acoustic_agreement_pct", "manual_stress_accuracy_pct",
            "manual_stress_total",
        ])
        for result in results:
            writer.writerow([
                result["name"], result["paired"], round(result["text_count_accuracy"], 1),
                round(result["acoustic_count_accuracy"], 1),
                round(result["boundary_mean_ms"], 1),
                round(result["boundary_median_ms"], 1),
                round(result["boundary_p90_ms"], 1),
                (round(result["dict_acoustic_agreement"], 1)
                 if result["dict_acoustic_agreement"] is not None else ""),
                (round(result["stress_acoustic_accuracy"], 1)
                 if result["stress_acoustic_accuracy"] is not None else ""),
                result["stress_total"],
            ])
    print(f"\n💾 Сохранено: {output_md}, {output_csv}")


# ══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════

# Якорим пути к местоположению самого скрипта (как в pronunciation_analyzer.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Папки, где лежат файлы. По умолчанию — рядом со скриптом.
TEXTGRID_DIR = os.path.join(SCRIPT_DIR, "analysis")
JSON_DIR     = os.path.join(SCRIPT_DIR, "analysis")
GROUND_FILE  = os.path.join(SCRIPT_DIR, "ground_truth_words.json")
OUTPUT_MD    = os.path.join(SCRIPT_DIR, "results", "evaluation_results.md")
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "results", "evaluation_results.csv")

# Какие записи сравнивать. Поставьте имена (без расширения) — или
# оставьте пустой список, и скрипт сам найдёт все доступные.
RECORDINGS = []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Сверка TextGrid с JSON-выходом анализатора произношения."
    )
    parser.add_argument("recordings", nargs="*", help="Имена записей без расширения")
    parser.add_argument("--textgrid-dir", default=TEXTGRID_DIR)
    parser.add_argument("--json-dir", default=JSON_DIR)
    parser.add_argument("--ground", default=GROUND_FILE)
    parser.add_argument("--output-md", default=OUTPUT_MD)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    return parser.parse_args(argv)


# ------------------------- main ------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    # Проверка наличия эталона
    if not os.path.exists(args.ground):
        print(f"❌ Не найден файл с эталонными словами: {args.ground}")
        sys.exit(1)
    ground_truth = load_ground_truth(args.ground)

    # Какие записи обрабатывать
    requested = args.recordings or RECORDINGS
    if requested:
        names = requested
    else:
        tg_names = {os.path.basename(p)[:-len(".TextGrid")]
                    for p in glob.glob(os.path.join(args.textgrid_dir, "*.TextGrid"))}
        js_suffix = "_syllable_analysis.json"
        js_names = {os.path.basename(p)[:-len(js_suffix)]
                    for p in glob.glob(os.path.join(args.json_dir, f"*{js_suffix}"))}
        names = sorted(tg_names & js_names & set(ground_truth.keys()))
        if not names:
            print("❌ Не найдено ни одной записи с парой .TextGrid + _syllable_analysis.json")
            sys.exit(1)
        print(f"Автообнаружено записей: {names}\n")

    # Проверяем, что все нужные файлы есть
    missing = []
    for n in names:
        tg = os.path.join(args.textgrid_dir, f"{n}.TextGrid")
        js = os.path.join(args.json_dir, f"{n}_syllable_analysis.json")
        if not os.path.exists(tg): missing.append(("TextGrid", tg))
        if not os.path.exists(js): missing.append(("JSON", js))
    if missing:
        print("❌ Не найдены следующие файлы:")
        for label, p in missing:
            print(f"   {label}: {p}")
        print(f"\n   Папка скрипта: {SCRIPT_DIR}")
        print(f"   Текущая CWD:   {os.getcwd()}")
        sys.exit(1)

    # Прогоняем
    results = []
    for n in names:
        tg = os.path.join(args.textgrid_dir, f"{n}.TextGrid")
        js = os.path.join(args.json_dir, f"{n}_syllable_analysis.json")
        try:
            results.append(evaluate_recording(n, tg, js, ground_truth))
        except Exception as e:
            print(f"⚠ Ошибка при обработке {n}: {e}")

    if results:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        print_report(results, args.output_md, args.output_csv)


if __name__ == "__main__":
    main()
