#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_annotations.py
=======================
Сравнивает ручную разметку (Praat .TextGrid) с автоматическим выходом
системы (*_syllable_analysis.json) и считает «реальные» метрики:

  1. Точность числа слогов (Syllable Count Accuracy);
  2. Ошибка границ слогов (Boundary Error, мс) — среднее, медиана, P90;
  3. Точность определения ударения (Stress Hit Rate).

ИСПОЛЬЗОВАНИЕ
-------------

Вариант 1 — указать конкретные записи (имена без расширения):
    python evaluate_annotations.py 6fori 7fori

Вариант 2 — автоматически найти все записи, у которых есть и .TextGrid,
            и _syllable_analysis.json в текущей папке:
    python evaluate_annotations.py

Вариант 3 — задать свои каталоги:
    python evaluate_annotations.py --textgrid-dir ./annotations \\
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


def stressed_idx_by_duration(manual_sylls):
    """Самый длинный слог = ударный (грубое приближение)."""
    if len(manual_sylls) <= 1:
        return 0
    durations = [(b - a) for a, b, _ in manual_sylls]
    return int(max(range(len(durations)), key=lambda i: durations[i]))


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
            "stressedIdx": w.get("stressedIdx", -1),
            "sylls": abs_sylls,
        })

    # 3) Выравнивание по тексту слова
    pairs = align_system_to_ground(system_words, manual_words)

    # 4) Метрики
    count_match = 0
    paired_for_count = 0
    boundary_errs = []
    stress_total = 0
    stress_hits = 0
    stress_actual_hits = 0    # новый: для акустического детектора
    per_word_rows = []

    for g, s, op in pairs:
        if g is None or s is None:
            # пропуск
            per_word_rows.append({"word": (g["name"] if g else f"[VOSK:{s['word']}]"),
                                  "op": op, "manual_count": (len(g["sylls"]) if g else "—"),
                                  "system_count": (s["syllableCount"] if s else "—"),
                                  "boundary_err_ms": None, "stress_ok": None})
            continue

        paired_for_count += 1
        m_count = len(g["sylls"])
        s_count = s["syllableCount"]
        count_ok = (m_count == s_count)
        if count_ok:
            count_match += 1
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

        stress_ok = None
        stress_actual_ok = None
        if count_ok and m_count > 1:
            stress_total += 1
            durs = [(b - a) for a, b, _ in g["sylls"]]
            m_stress = int(max(range(len(durs)), key=lambda i: durs[i]))
            # Метрика 1 (как раньше): СЛОВАРНОЕ ударение vs ручная разметка
            #   — измеряет, правильно ли говорящий произнёс по нормам языка.
            s_stress_expected = s.get("expectedStressedIdx", s["stressedIdx"])
            stress_ok = (m_stress == s_stress_expected)
            if stress_ok: stress_hits += 1
            # Метрика 2 (новая): АКУСТИЧЕСКОЕ ударение vs ручная разметка
            #   — измеряет, правильно ли алгоритм находит реальное ударение.
            s_stress_actual = s.get("actualStressedIdx", s["stressedIdx"])
            stress_actual_ok = (m_stress == s_stress_actual)
            if stress_actual_ok: stress_actual_hits += 1

        per_word_rows.append({
            "word": g["name"], "op": op,
            "manual_count": m_count, "system_count": s_count,
            "count_ok": count_ok,
            "boundary_err_ms": round(mean_err, 1) if mean_err is not None else None,
            "stress_ok": stress_ok,
        })

    return {
        "name": name,
        "manual_words": len(manual_words),
        "system_words": len(system_words),
        "paired": paired_for_count,
        "count_match": count_match,
        "count_accuracy": count_match / paired_for_count * 100 if paired_for_count else 0,
        "boundary_n": len(boundary_errs),
        "boundary_mean_ms":   mean(boundary_errs)   if boundary_errs else 0,
        "boundary_median_ms": median(boundary_errs) if boundary_errs else 0,
        "boundary_p90_ms":    sorted(boundary_errs)[int(0.9*len(boundary_errs))] if boundary_errs else 0,
        "stress_total": stress_total,
        "stress_hits": stress_hits,
        "stress_accuracy": stress_hits / stress_total * 100 if stress_total else 0,
        # Новая метрика: акустический детектор (то, насколько алгоритм
        # правильно ловит фактически выделенный слог)
        "stress_actual_hits": stress_actual_hits,
        "stress_actual_accuracy": stress_actual_hits / stress_total * 100 if stress_total else 0,
        "per_word": per_word_rows,
    }


# ------------------------- вывод -----------------------------------------
def print_report(results, output_md="evaluation_results.md", output_csv="evaluation_results.csv"):
    W = 92
    print("=" * W)
    print("  РЕЗУЛЬТАТЫ ОЦЕНКИ — РУЧНАЯ РАЗМЕТКА vs АВТОМАТИЧЕСКИЙ ВЫХОД СИСТЕМЫ")
    print("=" * W)

    for r in results:
        print(f"\n[{r['name']}]  слов в разметке: {r['manual_words']}  |  в системе: {r['system_words']}  |  сопоставлено: {r['paired']}")
        print(f"  Точность числа слогов:    {r['count_match']}/{r['paired']} = {r['count_accuracy']:.1f}%")
        print(f"  Ошибка границ слогов:     среднее {r['boundary_mean_ms']:.1f} мс, "
              f"медиана {r['boundary_median_ms']:.1f} мс, P90 {r['boundary_p90_ms']:.1f} мс  "
              f"(n={r['boundary_n']})")
        print(f"  Ударение (словарь vs разметка): {r['stress_hits']}/{r['stress_total']} = {r['stress_accuracy']:.1f}%  "
              f"← правильность произношения говорящим")
        print(f"  Ударение (акустика vs разметка): {r['stress_actual_hits']}/{r['stress_total']} = {r['stress_actual_accuracy']:.1f}%  "
              f"← точность акустического детектора")

    # Сводно
    total_p = sum(r["paired"] for r in results)
    total_cm = sum(r["count_match"] for r in results)
    total_st = sum(r["stress_total"] for r in results)
    total_sh = sum(r["stress_hits"] for r in results)
    total_sah = sum(r["stress_actual_hits"] for r in results)
    total_bn = sum(r["boundary_n"] for r in results)
    bmean = sum(r["boundary_mean_ms"] * r["boundary_n"] for r in results) / max(1, total_bn)

    print("\n" + "=" * W)
    print(f"  ОБЩИЕ ПОКАЗАТЕЛИ (по {len(results)} записям)")
    print("=" * W)
    print(f"  Сопоставлено слов:        {total_p}")
    print(f"  Точность числа слогов:    {total_cm}/{total_p} = {total_cm/total_p*100:.1f}%" if total_p else "  —")
    print(f"  Средняя ошибка границ:    {bmean:.1f} мс")
    if total_st:
        print(f"  Ударение, словарь:        {total_sh}/{total_st} = {total_sh/total_st*100:.1f}%  ← правильность произношения говорящим")
        print(f"  Ударение, акустика:       {total_sah}/{total_st} = {total_sah/total_st*100:.1f}%  ← точность акустического детектора")

    # Markdown
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Результаты оценки системы (ручная разметка vs автоматический выход)\n\n")
        f.write("Метрики, полученные при сверке ручной разметки границ слогов (TextGrid в Praat) "
                "с автоматическим выходом системы.\n\n")
        f.write(f"## Сводно по {len(results)} записям\n\n")
        f.write("| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Сопоставлено слов | {total_p} |\n")
        f.write(f"| **Точность числа слогов** | **{total_cm/total_p*100:.1f}%** ({total_cm}/{total_p}) |\n")
        f.write(f"| **Средняя ошибка границ слогов** | **{bmean:.1f} мс** |\n")
        if total_st:
            f.write(f"| **Ударение: акустический детектор vs ручная разметка** | **{total_sah/total_st*100:.1f}%** ({total_sah}/{total_st}) — точность алгоритма |\n")
            f.write(f"| **Ударение: словарь ruaccent vs ручная разметка** | **{total_sh/total_st*100:.1f}%** ({total_sh}/{total_st}) — правильность произношения говорящим |\n")
        f.write("\n## По отдельным записям\n\n")
        f.write("| Запись | Слов | SylCountAcc | Bound.err mean (мс) | Bound.err median | Bound.err P90 | StressAcc (акустика) | StressAcc (словарь) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            sa_act = f"{r['stress_actual_accuracy']:.1f}%" if r["stress_total"] else "—"
            sa_exp = f"{r['stress_accuracy']:.1f}%" if r["stress_total"] else "—"
            f.write(f"| {r['name']} | {r['paired']} | {r['count_accuracy']:.1f}% | "
                    f"{r['boundary_mean_ms']:.1f} | {r['boundary_median_ms']:.1f} | "
                    f"{r['boundary_p90_ms']:.1f} | {sa_act} | {sa_exp} |\n")
        f.write("\n## Метрики — определения\n\n")
        f.write("- **SylCountAcc** — доля слов с правильным числом слогов.\n")
        f.write("- **Bound.err** — абсолютная разница (мс) границ слогов в разметке и в системе. "
                "Считается только по совпавшим по числу слогов словам.\n")
        f.write("- **StressAcc (акустика)** — доля многосложных слов, где наш АКУСТИЧЕСКИЙ детектор "
                "ударного слога (по длительности, энергии, F1 и MFCC) совпал с самым длинным "
                "слогом в ручной разметке. **Это точность нашего алгоритма.**\n")
        f.write("- **StressAcc (словарь)** — доля многосложных слов, где СЛОВАРНОЕ (ruaccent) "
                "ожидаемое ударение совпало с фактическим (по разметке). **Это правильность "
                "произношения говорящим — у неносителей закономерно ниже из-за акцента.**\n")
        f.write("- Разница между двумя метриками = сигнал об ошибках произношения, которые "
                "система может указать ученику.\n")
        f.write("- **StressAcc** — доля многосложных слов, в которых ударный слог системы совпал "
                "с самым длинным слогом в ручной разметке (приближение).\n")

    # CSV
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "paired_words", "count_accuracy_pct",
                    "boundary_mean_ms", "boundary_median_ms", "boundary_p90_ms",
                    "stress_actual_accuracy_pct", "stress_dict_accuracy_pct",
                    "stress_total"])
        for r in results:
            w.writerow([r["name"], r["paired"], round(r["count_accuracy"],1),
                        round(r["boundary_mean_ms"],1), round(r["boundary_median_ms"],1),
                        round(r["boundary_p90_ms"],1),
                        round(r["stress_actual_accuracy"],1),
                        round(r["stress_accuracy"],1),
                        r["stress_total"]])

    print(f"\n💾 Сохранено: {output_md}, {output_csv}")


# ══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ — ПРАВИТЕ ЗДЕСЬ, КАК В pronunciation_analyzer.py
# ══════════════════════════════════════════════════════════════════

# Якорим пути к местоположению самого скрипта (как в pronunciation_analyzer.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Папки, где лежат файлы. По умолчанию — рядом со скриптом.
TEXTGRID_DIR = os.path.join(SCRIPT_DIR, "analysis")
JSON_DIR     = os.path.join(SCRIPT_DIR, "analysis")
GROUND_FILE  = os.path.join(SCRIPT_DIR, "ground_truth_words.json")
OUTPUT_MD    = os.path.join(SCRIPT_DIR, "fori_evaluation_results.md")
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "fori_evaluation_results.csv")

# Какие записи сравнивать. Поставьте имена (без расширения) — или
# оставьте пустой список, и скрипт сам найдёт все доступные.
RECORDINGS = ["6fori","7fori"]


# ------------------------- main ------------------------------------------
def main():
    # Проверка наличия эталона
    if not os.path.exists(GROUND_FILE):
        print(f"❌ Не найден файл с эталонными словами: {GROUND_FILE}")
        sys.exit(1)
    ground_truth = load_ground_truth(GROUND_FILE)

    # Какие записи обрабатывать
    if RECORDINGS:
        names = RECORDINGS
    else:
        tg_names = {os.path.basename(p)[:-len(".TextGrid")]
                    for p in glob.glob(os.path.join(TEXTGRID_DIR, "*.TextGrid"))}
        js_suffix = "_syllable_analysis.json"
        js_names = {os.path.basename(p)[:-len(js_suffix)]
                    for p in glob.glob(os.path.join(JSON_DIR, f"*{js_suffix}"))}
        names = sorted(tg_names & js_names & set(ground_truth.keys()))
        if not names:
            print("❌ Не найдено ни одной записи с парой .TextGrid + _syllable_analysis.json")
            sys.exit(1)
        print(f"Автообнаружено записей: {names}\n")

    # Проверяем, что все нужные файлы есть
    missing = []
    for n in names:
        tg = os.path.join(TEXTGRID_DIR, f"{n}.TextGrid")
        js = os.path.join(JSON_DIR, f"{n}_syllable_analysis.json")
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
        tg = os.path.join(TEXTGRID_DIR, f"{n}.TextGrid")
        js = os.path.join(JSON_DIR, f"{n}_syllable_analysis.json")
        try:
            results.append(evaluate_recording(n, tg, js, ground_truth))
        except Exception as e:
            print(f"⚠ Ошибка при обработке {n}: {e}")

    if results:
        print_report(results, OUTPUT_MD, OUTPUT_CSV)


if __name__ == "__main__":
    main()