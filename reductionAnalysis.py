#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reductionAnalysis.py — Кросс-записный анализ редукции гласных
                       (с объединением неносителей и графиками)
==============================================================
Новое:
  - POOL_NONNATIVE: объединить всех *fori в одну группу «nonnative_pooled»
  - PLOT_FILE: путь к сохраняемому графику (PNG)
  - В таблицу добавлен Cohen's d для длительности и централизации
  - Автоматическая генерация:
       * Boxplot (длительность гласного ядра): носители vs неносители, ударный vs безударный
       * Forest plot (длительность, отношение безударный/ударный с 95% ДИ)
"""

import json, os, sys, glob, math
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

# ========== НАСТРОЙКИ ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR   = os.path.join(SCRIPT_DIR, "analysis")
OUTPUT_MD  = os.path.join(SCRIPT_DIR, "reduction_comparison2.md")
PLOT_FILE  = os.path.join(SCRIPT_DIR, "reduction_plots.png")   # новое

RECORDINGS = []                     # пусто = автообнаружение

MIN_SAMPLES_FOR_REPORT = 3
MIN_SAMPLES_FOR_TEST   = 10
COLLAPSE_UNSTRESSED    = False
POOL_NONNATIVE         = True        # ← объединить всех *fori в одну группу

TARGET_VOWELS = set("аоеияёэюуы")
# ================================

def main_vowel(syl):
    syl = syl.lower()
    for ch in syl:
        if ch in TARGET_VOWELS: return ch
    return None

def position_label(i, st_idx):
    if i == st_idx: return "stressed"
    if i == st_idx - 1: return "pretonic"
    if i == st_idx + 1: return "posttonic"
    return "other_unstressed"

def collect_recording(json_path):
    USE_CORE = True
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pool = defaultdict(lambda: defaultdict(list))
    core_used = 0; full_used = 0
    for w in data["wordAnalysis"]:
        if w["syllableCount"] < 2: continue
        st_idx = w.get("expectedStressedIdx", w.get("stressedIdx", -1))
        if not (0 <= st_idx < w["syllableCount"]): continue
        for i, syl in enumerate(w["syllableAnalysis"]):
            ac = syl.get("acoustics", {})
            src = ac.get("core") if (USE_CORE and ac.get("core")) else ac
            if src is ac.get("core"): core_used += 1
            else: full_used += 1
            dur = src.get("duration", 0)
            ener = src.get("energyMean", 0)
            central = src.get("central", -1)
            # При использовании core длительность ядра короче целого слога.
            # 25 мс — минимум для вокалического ядра (редуцированный гласный).
            min_dur = 0.025 if (USE_CORE and ac.get("core")) else 0.040
            if dur < min_dur or central < 0: continue
            vowel = main_vowel(syl.get("syllable", ""))
            if vowel is None: continue
            pool[vowel][position_label(i, st_idx)].append((dur, ener, central))
    if core_used + full_used > 0:
        print(f"   ({os.path.basename(json_path)}: ядро={core_used}, полный слог={full_used})")
    return {v: dict(p) for v, p in pool.items()}

# ─── Статистика без scipy ───
def welch_t(a, b):
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2: return None
    m_a, m_b = mean(a), mean(b)
    v_a = stdev(a)**2 if n_a > 1 else 0
    v_b = stdev(b)**2 if n_b > 1 else 0
    if v_a == 0 and v_b == 0: return None
    se = math.sqrt(v_a/n_a + v_b/n_b)
    if se == 0: return None
    t = (m_a - m_b) / se
    if v_a == 0 or v_b == 0:
        df = max(n_a, n_b) - 1
    else:
        num = (v_a/n_a + v_b/n_b)**2
        den = (v_a/n_a)**2/(n_a-1) + (v_b/n_b)**2/(n_b-1)
        df = num/den if den > 0 else 1
    p = 2.0 * (1.0 - _stu_cdf(abs(t), df))
    return t, p

def _stu_cdf(t, df):
    if df <= 0: return 0.5
    x = df / (df + t*t)
    return 1.0 - 0.5 * _ibeta(df/2.0, 0.5, x)

def _ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = (math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b)
             + a*math.log(x) + b*math.log(1-x))
    if x < (a+1)/(a+b+2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1-x) / b

def _betacf(a, b, x):
    qab = a+b; qap = a+1; qam = a-1
    c, d = 1.0, 1.0 - qab*x/qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0/d; h = d
    for m in range(1, 200):
        m2 = 2*m
        aa = m*(b-m)*x / ((qam+m2)*(a+m2))
        d = 1.0 + aa*d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa/c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        d = 1.0 + aa*d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa/c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0/d; delta = d*c; h *= delta
        if abs(delta - 1) < 3e-7: return h
    return h

def sig_marker(p, can_test=True):
    if not can_test or p is None:
        return f"N<{MIN_SAMPLES_FOR_TEST}"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def ratio_with_ci(num_samples, den_samples):
    m_n, m_d = mean(num_samples), mean(den_samples)
    if m_d == 0: return float("nan"), float("nan"), float("nan")
    n_n, n_d = len(num_samples), len(den_samples)
    v_n = stdev(num_samples)**2 if n_n > 1 else 0
    v_d = stdev(den_samples)**2 if n_d > 1 else 0
    r = m_n / m_d
    var_r = (math.sqrt(v_n/n_n)/m_d)**2 + (m_n*math.sqrt(v_d/n_d)/(m_d**2))**2
    se = math.sqrt(var_r)
    return r, r - 1.96*se, r + 1.96*se

def cohens_d(x, y):
    """Cohen's d для двух независимых выборок (пуллированное стандартное отклонение)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    mean_x, mean_y = mean(x), mean(y)
    var_x = stdev(x)**2
    var_y = stdev(y)**2
    pooled_sd = math.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0 if mean_x == mean_y else float('inf')
    return (mean_x - mean_y) / pooled_sd

def compare_feature(stressed, unstressed, idx, min_for_report, min_for_test):
    n_st = len(stressed)
    n_un = len(unstressed)
    if n_st < min_for_report or n_un < min_for_report:
        return None

    st = [x[idx] for x in stressed]
    un = [x[idx] for x in unstressed]

    r, lo, hi = ratio_with_ci(un, st)
    d_val = cohens_d(un, st)   # сравнение: безударный - ударный

    result = {
        "n_st": n_st,
        "n_un": n_un,
        "mean_st": round(mean(st), 4),
        "mean_un": round(mean(un), 4),
        "ratio": round(r, 2),
        "ci_lo": round(lo, 2),
        "ci_hi": round(hi, 2),
        "cohens_d": round(d_val, 2) if d_val is not None else None,
    }

    can_test = (n_st >= min_for_test and n_un >= min_for_test)
    if can_test:
        t_res = welch_t(un, st)
        if t_res is not None:
            t, p = t_res
            result["p"] = p
            result["sig"] = sig_marker(p, can_test=True)
        else:
            result["sig"] = sig_marker(None, can_test=False)
    else:
        result["sig"] = sig_marker(None, can_test=False)
        result["p"] = None
    return result

def interpret(by_feat, can_test):
    d = by_feat.get("duration"); c = by_feat.get("central")
    if d is None and c is None:
        return "недостаточно данных"
    if not can_test:
        return "описательная оценка (без теста)"

    strong = 0
    if d and d.get("p") is not None and d["p"] < 0.05 and d["ratio"] < 0.80: strong += 1
    if c and c.get("p") is not None and c["p"] < 0.05 and c["ratio"] < 0.85: strong += 1
    if strong >= 2: return "сильная редукция (значимо)"
    if (d and d.get("p") is not None and d["p"] < 0.05 and d["ratio"] < 0.90) or \
       (c and c.get("p") is not None and c["p"] < 0.05 and c["ratio"] < 0.92):
        return "умеренная редукция (значимо)"
    return "редукция не доказана"

# ─── Построение графиков ───
def plot_results(all_results, all_pools, plot_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib не установлен — графики пропущены.")
        return

    # Определяем, какие записи относятся к носителям/неносителям
    native_names = [name for name in all_pools if "local" in name or "native" in name]
    nonnative_names = [name for name in all_pools if "fori" in name or "nonnative" in name]

    # 1. Boxplots: длительность ядра для основных гласных
    vowels_to_plot = ["о", "е", "а"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Длительность вокалического ядра: ударный vs безударный", fontsize=16)

    for ax, vowel in zip(axes, vowels_to_plot):
        data = {"native_stressed": [], "native_unstressed": [],
                "nonnative_stressed": [], "nonnative_unstressed": []}
        for rec_type, name_list, key_prefix in [("native", native_names, "native"),
                                                ("nonnative", nonnative_names, "nonnative")]:
            stressed_durs = []
            unstressed_durs = []
            for name in name_list:
                pool = all_pools.get(name, {})
                pos = pool.get(vowel, {})
                stressed_durs.extend([x[0] for x in pos.get("stressed", [])])
                for p in ("pretonic", "posttonic", "other_unstressed"):
                    unstressed_durs.extend([x[0] for x in pos.get(p, [])])
            data[f"{key_prefix}_stressed"] = stressed_durs
            data[f"{key_prefix}_unstressed"] = unstressed_durs

        positions = [1, 2, 4, 5]
        box_data = [data["native_stressed"], data["native_unstressed"],
                    data["nonnative_stressed"], data["nonnative_unstressed"]]
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"), showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
        colors = ["#7fc97f", "#7fc97f", "#fdc086", "#fdc086"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(f"Гласный «{vowel}»")
        ax.set_ylabel("Длительность (с)")
        ax.set_xticks([1.5, 4.5])
        ax.set_xticklabels(["Носители", "Неносители"])
        ax.legend([bp['boxes'][0], bp['boxes'][2]], ["Ударный", "Безударный"],
                  loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 2. Forest plot: отношение длительностей (безударный / ударный) с 95% ДИ
    # Показываем только строки с N≥10 (те, где t-тест применим), чтобы
    # график был читаемым. N<10 — описательные оценки, их слишком много.
    pos_names = {"pretonic": "предуд.", "posttonic": "зауд.",
                 "other_unstressed": "проч.безуд."}
    for side_name, side_names in [("Носители", native_names),
                                    ("Неносители", nonnative_names)]:
        # Собираем только тестируемые строки (N≥10 с обеих сторон)
        rows = []
        for name in side_names:
            res = all_results.get(name, {})
            for v in sorted(res.keys()):
                for pos, info in res[v].items():
                    d = info.get("by_feat", {}).get("duration", {})
                    if not d:
                        continue
                    r = d.get("ratio")
                    if r is None or math.isnan(r):
                        continue
                    sig = d.get("sig", "")
                    # Показываем только если t-тест был выполнен
                    if sig.startswith("N<"):
                        continue
                    rows.append((name, v, pos, r, d.get("ci_lo", r),
                                 d.get("ci_hi", r), sig))

        n_rows = len(rows)
        if n_rows == 0:
            continue

        fig_h = max(4, n_rows * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        fig.suptitle(f"Отношение длительности безударный/ударный — {side_name}", fontsize=13)

        # Сортируем по ratio
        rows.sort(key=lambda x: x[3])

        for i, (name, v, pos, r, lo, hi, sig) in enumerate(rows):
            y = i + 1
            col = "green" if sig in ("*", "**", "***") else ("orange" if r < 0.90 else "gray")
            ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt='o', capsize=3,
                       color=col, markersize=6)
            lbl = f"{v} {pos_names.get(pos, pos)} [{name}]"
            ax.text(0.02, y, lbl, va='center', fontsize=8,
                   fontfamily='monospace')

        ax.axvline(1.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        # Серая полоса: зона неопределённости (0.85–1.0)
        ax.axvspan(0.85, 1.0, alpha=0.08, color='gray')
        ax.set_yticks([])
        ax.set_ylim(0, n_rows + 1)
        ax.set_xlabel("Отношение длительности (безударный / ударный)", fontsize=11)
        ax.set_xlim(0, max(2.0, max(r[5] for r in rows) + 0.3))
        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='Значимо (p<0.05)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=8, label='Тренд (ratio<0.90, не знач.)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=8, label='Не значимо'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        plt.tight_layout(rect=[0.22, 0, 1, 0.95])
        fname = plot_path.replace(".png", f"_{side_name}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"📈 Forest plot ({side_name}): {fname}  ({n_rows} строк)")

def main():
    # Поддержка --jsondir через командную строку
    global JSON_DIR
    args = sys.argv[1:]
    if "--jsondir" in args:
        idx = args.index("--jsondir")
        JSON_DIR = args[idx + 1] if idx + 1 < len(args) else JSON_DIR
    # Автообнаружение файлов
    if RECORDINGS:
        names = RECORDINGS
    else:
        files = glob.glob(os.path.join(JSON_DIR, "*_syllable_analysis.json"))
        names = sorted({os.path.basename(p)[:-len("_syllable_analysis.json")] for p in files})
    if not names:
        print("❌ Нет файлов *_syllable_analysis.json"); sys.exit(1)

    # Загружаем все пулы для объединения и графиков
    all_pools = {}
    for name in names:
        path = os.path.join(JSON_DIR, f"{name}_syllable_analysis.json")
        if os.path.exists(path):
            all_pools[name] = collect_recording(path)

    # Объединение неносителей (если включено)
    if POOL_NONNATIVE:
        nonnative_names_in_data = [n for n in names if "fori" in n]
        if nonnative_names_in_data:
            # Сливаем пулы
            merged = defaultdict(lambda: defaultdict(list))
            for name in nonnative_names_in_data:
                pool = all_pools.pop(name, {})
                for v, pos in pool.items():
                    for p, lst in pos.items():
                        merged[v][p].extend(lst)
            # Добавляем как отдельную запись
            merged_name = "nonnative_pooled"
            all_pools[merged_name] = {v: dict(p) for v, p in merged.items()}
            names = [n for n in names if n not in nonnative_names_in_data] + [merged_name]
            print(f"🗂 Объединены неносители ({', '.join(nonnative_names_in_data)}) → {merged_name}")

    # Анализ каждой записи
    all_results = {}
    for name in names:
        pool = all_pools.get(name, {})
        if not pool:
            continue
        # Применяем сжатие безударных позиций, если нужно
        if COLLAPSE_UNSTRESSED:
            for vowel in pool:
                pos = pool[vowel]
                if "unstressed" not in pos:
                    unstressed = []
                    for k in ("pretonic", "posttonic", "other_unstressed"):
                        unstressed.extend(pos.pop(k, []))
                    pos["unstressed"] = unstressed

        rec = {}
        for vowel in sorted(pool.keys()):
            positions = pool[vowel]
            stressed = positions.get("stressed", [])
            if COLLAPSE_UNSTRESSED:
                unstressed_positions = ["unstressed"]
            else:
                unstressed_positions = ["pretonic", "posttonic", "other_unstressed"]
            v_res = {}
            for pos in unstressed_positions:
                un = positions.get(pos, [])
                if len(un) < MIN_SAMPLES_FOR_REPORT or len(stressed) < MIN_SAMPLES_FOR_REPORT:
                    continue
                by_feat = {}
                for feat, idx in (("duration", 0), ("energy", 1), ("central", 2)):
                    r = compare_feature(stressed, un, idx,
                                        MIN_SAMPLES_FOR_REPORT, MIN_SAMPLES_FOR_TEST)
                    if r: by_feat[feat] = r
                if by_feat:
                    can_test = all(
                        (feat not in by_feat) or (by_feat[feat].get("p") is not None)
                        for feat in ("duration", "central")
                    )
                    v_res[pos] = {"by_feat": by_feat, "verdict": interpret(by_feat, can_test)}
            if v_res: rec[vowel] = v_res
        all_results[name] = rec

    # ── Консольный и Markdown отчёт ──
    print("="*100)
    print(f"  СТРОГИЙ АНАЛИЗ РЕДУКЦИИ (t-тест Уэлча, 95% ДИ, Cohen's d)")
    print(f"  Мин. образцов для отчёта: {MIN_SAMPLES_FOR_REPORT} | для t-теста: {MIN_SAMPLES_FOR_TEST}")
    if COLLAPSE_UNSTRESSED:
        print("  Режим: все безударные позиции объединены")
    if POOL_NONNATIVE:
        print("  Неносители объединены в группу 'nonnative_pooled'")
    print("="*100)

    for name, res in all_results.items():
        if name == "nonnative_pooled":
            kind = "неносители (объединённые)"
        else:
            kind = "НОСИТЕЛЬ" if "local" in name else "неноситель"
        print(f"\n[{name}]   ({kind})")
        if not res:
            print("  (недостаточно данных)"); continue
        for v, positions in res.items():
            print(f"  «{v}»:")
            for pos, info in positions.items():
                d = info["by_feat"].get("duration", {})
                c = info["by_feat"].get("central", {})
                dur_str = f"длит.={d.get('ratio','?')} [{d.get('ci_lo','?')}..{d.get('ci_hi','?')}] {d.get('sig','')} d={d.get('cohens_d','-')}"
                cent_str = f"центр.={c.get('ratio','?')} {c.get('sig','')} d={c.get('cohens_d','-')}" if c else ""
                print(f"    {pos:<17} N(уд)={d.get('n_st','?'):>3} N(б/у)={d.get('n_un','?'):>3}  "
                      f"{dur_str}  {cent_str} → {info['verdict']}")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Кросс-записной анализ редукции гласных (с Cohen's d и объединением)\n\n")
        f.write("**Метод.** ...\n")
        f.write(f"**Пороги:** отчёт от {MIN_SAMPLES_FOR_REPORT} слогов, t‑тест от {MIN_SAMPLES_FOR_TEST}.\n")
        if POOL_NONNATIVE:
            f.write("**Неносители объединены** в одну группу.\n\n")
        f.write("**Значимость:** `***` p<0.001, `**` p<0.01, `*` p<0.05, "
                f"`N<{MIN_SAMPLES_FOR_TEST}` — образцов мало для теста.\n\n")
        f.write("| Гласн | Позиция | N(уд) | N(б/у) | длит.отн. [95% ДИ] | Cohen's d | "
                "центр.отн. | Cohen's d | Вывод |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for name, res in all_results.items():
            if name == "nonnative_pooled":
                kind = "неносители (объединённые)"
            else:
                kind = "носитель" if "local" in name else "неноситель"
            f.write(f"## {name} ({kind})\n\n")
            if not res:
                f.write("Недостаточно данных.\n\n"); continue
            for v, positions in res.items():
                for pos, info in positions.items():
                    d = info["by_feat"].get("duration", {})
                    c = info["by_feat"].get("central", {})
                    f.write(f"| {v} | {pos} | {d.get('n_st','?')} | {d.get('n_un','?')} | "
                            f"{d.get('ratio','—')} [{d.get('ci_lo','?')}..{d.get('ci_hi','?')}] {d.get('sig','')} | "
                            f"{d.get('cohens_d','—')} | "
                            f"{c.get('ratio','—')} {c.get('sig','')} | "
                            f"{c.get('cohens_d','—')} | "
                            f"{info['verdict']} |\n")
            f.write("\n")

    # ── Графики ──
    plot_results(all_results, all_pools, PLOT_FILE)

    print(f"\n💾 Отчёт: {OUTPUT_MD}")
    print(f"📈 График: {PLOT_FILE}")

if __name__ == "__main__":
    main()