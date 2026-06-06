#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reductionAnalysis.py — Кросс-записный анализ редукции гласных
                       (академически строгая версия)
==============================================================

Что делает:
  Для каждой записи отдельно собирает ВСЕ слоги по всей записи,
  группирует их по «главному гласному» и по позиции относительно
  ударного слога (ударный / предударный / заударный / прочий),
  затем сравнивает средние характеристики групп с проверкой
  статистической значимости (t-тест Уэлча) и доверительными
  интервалами 95%.

Подход предложен мной 23.05.2026 как улучшение исходной идеи
научного руководителя «сопоставление слогов для редуцированных
звуков» (15.04.2026): не сравнивать соседние слоги внутри слова
(где сегментация даёт большую погрешность), а собирать статистику
по всей записи и сравнивать одинаковые гласные в разных позициях.

Метрики (для каждого гласного V в записи R):
  Сравниваем безударные группы (предударные, заударные, прочие)
  с ударной группой по:
    duration   — длительность слога целиком
    energyMean — средняя энергия
    central    — расстояние формантной точки (F1,F2) до центра
                 гласного пространства (схва). Меньше = ближе
                 к [ə] = сильнее редуцирован.

  Для каждой пары групп считаем:
    ratio        = mean_unstressed / mean_stressed
    ci_lo, ci_hi = 95%-доверительный интервал отношения
    p_value      = двусторонний t-тест Уэлча (H0: средние равны)

  Значимость:
    p < 0.001  → ***   очень сильное доказательство
    p < 0.01   → **
    p < 0.05   → *
    иначе     → n.s.  (не значимо)
"""

import json, os, sys, glob, math
from collections import defaultdict
from statistics import mean, stdev

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR   = os.path.join(SCRIPT_DIR, "analysis")
OUTPUT_MD  = os.path.join(SCRIPT_DIR, "reduction_comparison.md")

# Какие записи. Пусто → авто-обнаружение всех записей.
RECORDINGS = []

# Минимум 10 — академический минимум для t-теста.
MIN_SAMPLES_PER_GROUP = 10

# Использовать признаки ВОКАЛИЧЕСКОГО ЯДРА (если оно выделено в JSON
# в поле acoustics.core), а не всего слога. Это позволяет отделить
# гласный от соседних согласных и должно улучшить детектирование
# редукции «а», «е», «и» (для «о» уже работает по слогу целиком).
# Если в JSON ядра нет (старые файлы) — автоматически откатываемся
# на полный слог.
USE_CORE = True

TARGET_VOWELS = set("аоеияёэюуы")


def main_vowel(syl):
    syl = syl.lower()
    for ch in syl:
        if ch in TARGET_VOWELS: return ch
    return None


def position_label(i, st_idx):
    """ударный / предударный / заударный_1 / другой безударный"""
    if i == st_idx: return "stressed"
    if i == st_idx - 1: return "pretonic"
    if i == st_idx + 1: return "posttonic"
    return "other_unstressed"


def collect_recording(json_path):
    """{vowel: {position: [(dur, ener, central), ...]}}

    Когда USE_CORE=True и в JSON есть acoustics.core — берём признаки
    из ядра гласного (без согласных хвостов). Иначе — из полного слога.
    """
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
            # Выбираем источник признаков: ядро или весь слог
            src = ac.get("core") if (USE_CORE and ac.get("core")) else ac
            if src is ac.get("core"): core_used += 1
            else: full_used += 1
            dur = src.get("duration", 0)
            ener = src.get("energyMean", 0)
            central = src.get("central", -1)
            if dur < 0.040 or central < 0: continue
            vowel = main_vowel(syl.get("syllable", ""))
            if vowel is None: continue
            pool[vowel][position_label(i, st_idx)].append((dur, ener, central))
    # для диагностики — выведем, сколько слогов взяли из ядра
    if core_used + full_used > 0:
        print(f"   ({os.path.basename(json_path)}: ядро={core_used}, "
              f"полный слог={full_used})")
    return {v: dict(p) for v, p in pool.items()}


# ── статистика без scipy ──
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


def sig_marker(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def ratio_with_ci(num_samples, den_samples):
    """отношение средних и 95%-CI (метод дельт)"""
    m_n, m_d = mean(num_samples), mean(den_samples)
    if m_d == 0: return float("nan"), float("nan"), float("nan")
    n_n, n_d = len(num_samples), len(den_samples)
    v_n = stdev(num_samples)**2 if n_n > 1 else 0
    v_d = stdev(den_samples)**2 if n_d > 1 else 0
    r = m_n / m_d
    var_r = (math.sqrt(v_n/n_n)/m_d)**2 + (m_n*math.sqrt(v_d/n_d)/(m_d**2))**2
    se = math.sqrt(var_r)
    return r, r - 1.96*se, r + 1.96*se


def compare_feature(stressed, unstressed, idx):
    """idx: 0=dur, 1=ener, 2=central"""
    if len(stressed) < MIN_SAMPLES_PER_GROUP or len(unstressed) < MIN_SAMPLES_PER_GROUP:
        return None
    st = [x[idx] for x in stressed]
    un = [x[idx] for x in unstressed]
    r, lo, hi = ratio_with_ci(un, st)
    t_res = welch_t(un, st)
    if t_res is None: return None
    t, p = t_res
    return {
        "n_st": len(stressed), "n_un": len(unstressed),
        "mean_st": round(mean(st), 4),
        "mean_un": round(mean(un), 4),
        "ratio": round(r, 2),
        "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
        "p": p, "sig": sig_marker(p),
    }


def interpret(by_feat):
    """Вывод по комбинации признаков."""
    d = by_feat.get("duration"); c = by_feat.get("central")
    strong = 0
    if d and d["p"] < 0.05 and d["ratio"] < 0.80: strong += 1
    if c and c["p"] < 0.05 and c["ratio"] < 0.85: strong += 1
    if strong >= 2: return "сильная редукция (значимо)"
    if (d and d["p"] < 0.05 and d["ratio"] < 0.90) or \
       (c and c["p"] < 0.05 and c["ratio"] < 0.92):
        return "умеренная редукция (значимо)"
    return "редукция не доказана"


def main():
    if RECORDINGS:
        names = RECORDINGS
    else:
        files = glob.glob(os.path.join(JSON_DIR, "*_syllable_analysis.json"))
        names = sorted({os.path.basename(p)[:-len("_syllable_analysis.json")] for p in files})
    if not names:
        print("❌ Нет файлов *_syllable_analysis.json"); sys.exit(1)

    all_results = {}
    for name in names:
        path = os.path.join(JSON_DIR, f"{name}_syllable_analysis.json")
        if not os.path.exists(path):
            print(f"⚠ Пропускаю {name}: нет файла"); continue
        pool = collect_recording(path)
        rec = {}
        for vowel in sorted(pool.keys()):
            positions = pool[vowel]
            stressed = positions.get("stressed", [])
            v_res = {}
            for pos in ("pretonic", "posttonic", "other_unstressed"):
                un = positions.get(pos, [])
                if len(un) < MIN_SAMPLES_PER_GROUP or len(stressed) < MIN_SAMPLES_PER_GROUP:
                    continue
                by_feat = {}
                for feat, idx in (("duration", 0), ("energy", 1), ("central", 2)):
                    r = compare_feature(stressed, un, idx)
                    if r: by_feat[feat] = r
                if by_feat:
                    v_res[pos] = {"by_feat": by_feat, "verdict": interpret(by_feat)}
            if v_res: rec[vowel] = v_res
        all_results[name] = rec

    # ── консольный отчёт ──
    print("="*100)
    print(f"  СТРОГИЙ АНАЛИЗ РЕДУКЦИИ (t-тест Уэлча, 95% ДИ, мин. {MIN_SAMPLES_PER_GROUP} образцов в группе)")
    print(f"  Значимость: ***=p<0.001, **=p<0.01, *=p<0.05, n.s.=не значимо")
    print("="*100)

    for name, res in all_results.items():
        kind = "НОСИТЕЛЬ" if "local" in name else "неноситель"
        print(f"\n[{name}]   ({kind})")
        if not res:
            print("  (недостаточно данных)"); continue
        for v, positions in res.items():
            print(f"  «{v}»:")
            for pos, info in positions.items():
                d = info["by_feat"].get("duration", {})
                c = info["by_feat"].get("central", {})
                print(f"    {pos:<17} N(уд)={d.get('n_st','?'):>3} N(б/у)={d.get('n_un','?'):>3}  "
                      f"длит.={d.get('ratio','?')} [{d.get('ci_lo','?')}..{d.get('ci_hi','?')}] {d.get('sig',''):4} "
                      f"центр.={c.get('ratio','?')} {c.get('sig',''):4} → {info['verdict']}")

    # ── Markdown ──
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Кросс-записной анализ редукции гласных (строгая версия)\n\n")
        f.write("**Метод.** Для каждой записи собираем все слоги, группируем по "
                "главному гласному и позиции относительно ударного (предударный, "
                "заударный, прочие). Сравниваем средние характеристики безударных "
                "групп с ударной двусторонним t-тестом Уэлча; для отношений "
                "средних приводится 95% доверительный интервал (метод дельт).\n\n")
        f.write(f"**Минимальный размер группы:** {MIN_SAMPLES_PER_GROUP} образцов.\n\n")
        f.write("**Значимость:** `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` не значимо.\n\n")
        f.write("**Признаки:**\n")
        f.write("- `длит.отн.` — отношение средней длительности безударного слога к ударному. <1 = безударный короче.\n")
        f.write("- `центр.отн.` — отношение расстояния (F1,F2) до нейтрального центра. <1 = ближе к [ə].\n\n")

        for name, res in all_results.items():
            kind = "носитель русского" if "local" in name else "неноситель"
            f.write(f"## {name} ({kind})\n\n")
            if not res:
                f.write("Недостаточно данных.\n\n"); continue
            f.write("| Гласн | Позиция | N(уд) | N(б/у) | длит.отн. [95% ДИ] | центр.отн. | Вывод |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for v, positions in res.items():
                for pos, info in positions.items():
                    d = info["by_feat"].get("duration", {})
                    c = info["by_feat"].get("central", {})
                    f.write(f"| {v} | {pos} | {d.get('n_st','?')} | {d.get('n_un','?')} | "
                            f"{d.get('ratio','—')} [{d.get('ci_lo','?')}..{d.get('ci_hi','?')}] {d.get('sig','')} | "
                            f"{c.get('ratio','—')} {c.get('sig','')} | "
                            f"{info['verdict']} |\n")
            f.write("\n")

    print(f"\n💾 Сохранено: {OUTPUT_MD}")


if __name__ == "__main__":
    main()