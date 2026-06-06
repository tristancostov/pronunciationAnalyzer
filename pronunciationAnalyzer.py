from __future__ import annotations

"""
Анализатор произношения — сигнальный поиск ядер слогов (без forced alignment)
=============================================================================
Что изменилось по сравнению с прошлой версией:

  1. Слоги строятся по РЕАЛЬНОЙ (произнесённой) форме слова, а не по лемме.
     Раньше: splitIntoSyllables(lemma) — число и структура слогов не совпадали
     с тем, что реально сказано (русский язык флективный, ударение подвижное).

  2. Новый поиск ядер слогов (метод de Jong & Wempe, чистый сигнал, без MFA):
       • голосовой фильтр (voicing gate) — пики ищем только на озвонченных
         кадрах, что убирает ложные пики от глухих согласных (с, ш, ф, к, п, т);
       • низкочастотная (≤1000 Гц) энергия вместо широкополосной RMS —
         подчёркивает гласные, гасит высокочастотный шум фрикативов;
       • порог провала (prominence/dip): два пика считаются разными ядрами
         только если между ними есть достаточный провал энергии — иначе один
         долгий гласный не дробится на два слога;
       • число ядер НЕ навязывается числом слогов из текста — текстовое число
         используется только как ориентир; расхождение само по себе сигнал;
       • границы слогов ставятся в МИНИМУМАХ энергии (в провалах), а не в
         середине между пиками.

  3. Убрано сравнение с одним внешним эталонным WAV (оно было сломано: весь
     эталон нарезался на «слоги одного слова»). Вместо этого — ВНУТРЕННЕЕ
     сопоставление слогов (ударный vs безударный) для анализа редукции,
     как и предлагал научный руководитель.

  4. Редукция оценивается по ДЛИТЕЛЬНОСТИ и КАЧЕСТВУ гласного (форманты F1/F2,
     централизация к шва), а не только по энергии, плюс глобальная таблица
     контраста одного и того же слога в ударной и безударной позиции.

Установка:
  pip install vosk pymorphy2 pymorphy2-dicts-ru librosa scipy ruaccent
"""

import wave
import json
import sys
import re
import os
import difflib

import numpy as np
import librosa
from scipy.signal import find_peaks, butter, sosfiltfilt


# ══════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════

# Якорим все пути к местоположению самого скрипта — независимо от того,
# из какой папки его запускают (терминал, кнопка «Run», debug).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

audioFile  = os.path.join(SCRIPT_DIR, "audio", "5fori.wav")
textFile   = os.path.join(SCRIPT_DIR, "text",  "5fori.txt")
modelPath  = os.path.join(SCRIPT_DIR, "vosk-model-ru-0.42")

SR                  = 16000
N_MFCC              = 13
pauseThreshold      = 2.0
normalSpeakingRate  = (2.0, 3.5)

# --- параметры поиска ядер слогов ---
LOWPASS_HZ          = 1000.0   # энергию берём из полосы гласных (≤ ~F1+)
NUCLEUS_FRAME_S     = 0.030    # окно энергии 30 мс
NUCLEUS_HOP_S       = 0.010    # шаг 10 мс
DIP_DB              = 3.0      # минимальный провал между ядрами, дБ
PEAK_FLOOR_DB       = 30.0     # пики ниже (max - 30 дБ) игнорируем (тишина)
MIN_NUCLEUS_DIST_S  = 0.080    # минимум 80 мс между ядрами
VOWEL_PEAK_GAIN_DB  = 4.0      # на сколько дБ «гласность» (MFCC/MEL) двигает контур при поиске пиков

# --- голосовой фильтр (voicing gate) ---
VOICE_LOWBAND_RATIO = 0.45     # доля низкочастотной энергии для «озвончённого»
VOICE_ZCR_MAX       = 0.18     # выше — скорее глухой шум

# --- редукция ---
REDUCIBLE_VOWELS    = set("аоеяи")
DUR_RATIO_NO_RED    = 0.85     # безударный ≥85% длины ударного → нет редукции
ENER_RATIO_NO_RED   = 0.80     # и ≥80% энергии ударного → подозрение
CENTRAL_RATIO_NO_RED = 0.90    # и централизация ≥90% от ударного → гласный не «съехал» к шва
SCHWA_F1            = 500.0    # центр гласного пространства (приближённо)
SCHWA_F2            = 1500.0


# ══════════════════════════════════════════════════════════════════
# 2. РАЗБИЕНИЕ НА СЛОГИ (адаптация PHP-алгоритма с Хабра)
# ══════════════════════════════════════════════════════════════════

VOWELS = set("аеёиоуыэюя")

SONORITY_MAP = {
    **{ch: 4 for ch in "аеёиоуыэюя"},  # Гласные
    **{ch: 3 for ch in "лмнрй"},        # Сонорные
    **{ch: 2 for ch in "бвгджз"},       # Звонкие
    **{ch: 1 for ch in "кпстфхцчшщ"},   # Глухие
    "ь": 0, "ъ": 0,
}


def hasVowel(fragment: str) -> bool:
    return any(ch in VOWELS for ch in fragment.lower())


def cleanWord(word: str) -> str:
    """Оставляет только кириллицу, в нижнем регистре."""
    return re.sub(r"[^а-яёА-ЯЁ]", "", word).lower()


def splitIntoSyllables(word: str) -> list[str]:
    """
    Разбивает слово на слоги по принципу восходящей звучности.
    ВАЖНО: подавать сюда РЕАЛЬНУЮ форму слова, а не лемму.
    Каждый слог гарантированно содержит гласную.
    """
    wordLower   = word.lower()
    sonoritySeq = [SONORITY_MAP.get(ch, -1) for ch in wordLower]
    splitPoints = []

    for i in range(len(wordLower) - 1):
        cur = sonoritySeq[i]
        nxt = sonoritySeq[i + 1]
        if cur <= 0 or nxt <= 0:
            continue
        if cur == 4 and nxt < 4:
            if wordLower[i + 1] not in ("ь", "ъ"):
                splitPoints.append(i + 1)

    rawFragments, prevIdx = [], 0
    for point in splitPoints:
        rawFragments.append(word[prevIdx:point])
        prevIdx = point
    rawFragments.append(word[prevIdx:])
    rawFragments = [s for s in rawFragments if s]

    mergedSyllables: list[str] = []
    for fragment in rawFragments:
        if mergedSyllables and not hasVowel(fragment):
            mergedSyllables[-1] += fragment
        else:
            mergedSyllables.append(fragment)

    return mergedSyllables if mergedSyllables else [word]


def syllableCharSpans(syllables: list[str]) -> list[tuple[int, int]]:
    """Для каждого слога — диапазон индексов символов в склеенном слове."""
    spans, pos = [], 0
    for syl in syllables:
        spans.append((pos, pos + len(syl)))
        pos += len(syl)
    return spans


def compareSyllables(refWord: str, recWord: str) -> dict:
    """Сравнивает слоги двух слов, находит позиции ошибок."""
    refSyl = splitIntoSyllables(cleanWord(refWord))
    recSyl = splitIntoSyllables(cleanWord(recWord))
    matcher = difflib.SequenceMatcher(None, refSyl, recSyl)
    errors  = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        elif op == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                exp = refSyl[i1 + k] if (i1 + k) < len(refSyl) else "—"
                got = recSyl[j1 + k] if (j1 + k) < len(recSyl) else "—"
                errors.append({"position": i1 + k + 1, "expected": exp,
                               "got": got, "type": "replace",
                               "hint": f"Ожидался слог «{exp}», произнесено «{got}»"})
        elif op == "delete":
            for k in range(i2 - i1):
                errors.append({"position": i1 + k + 1, "expected": refSyl[i1 + k],
                               "got": "—", "type": "missing",
                               "hint": f"Слог «{refSyl[i1 + k]}» пропущен"})
        elif op == "insert":
            for k in range(j2 - j1):
                errors.append({"position": i1 + 1, "expected": "—",
                               "got": recSyl[j1 + k], "type": "extra",
                               "hint": f"Лишний слог «{recSyl[j1 + k]}»"})

    total      = max(len(refSyl), len(recSyl))
    matchRatio = round(1.0 - len(errors) / total, 3) if total > 0 else 1.0
    return {"refSyllables": refSyl, "recSyllables": recSyl,
            "errors": errors, "matchRatio": matchRatio}


# ══════════════════════════════════════════════════════════════════
# 3. СИГНАЛЬНЫЙ ПОИСК ЯДЕР СЛОГОВ (de Jong & Wempe, без alignment)
# ══════════════════════════════════════════════════════════════════

def _lowpass(signal: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Фильтр низких частот — оставляем полосу гласных."""
    if len(signal) < 27:
        return signal
    nyq = sr / 2.0
    wc  = min(cutoff / nyq, 0.99)
    sos = butter(4, wc, btype="low", output="sos")
    return sosfiltfilt(sos, signal)


def computeContours(wordSegment: np.ndarray, sr: int):
    """
    Возвращает dict с покадровыми признаками слова.
    Ключи: energyDb, voiced, vowelScore, hopLen, frameLen, melDb, mfcc, zcr.
    """
    frameLen = max(int(sr * NUCLEUS_FRAME_S), 64)
    hopLen   = max(int(sr * NUCLEUS_HOP_S), 16)

    low = _lowpass(wordSegment, sr, LOWPASS_HZ)

    rmsLow  = librosa.feature.rms(y=low,         frame_length=frameLen, hop_length=hopLen)[0]
    rmsFull = librosa.feature.rms(y=wordSegment, frame_length=frameLen, hop_length=hopLen)[0]
    zcr     = librosa.feature.zero_crossing_rate(wordSegment,
                                                 frame_length=frameLen, hop_length=hopLen)[0]

    nFft = max(frameLen, 256)
    mfcc = librosa.feature.mfcc(y=wordSegment, sr=sr,
                                n_mfcc=13, n_fft=nFft,
                                hop_length=hopLen,
                                n_mels=40, fmax=sr/2)
    melS = librosa.feature.melspectrogram(y=wordSegment, sr=sr,
                                          n_fft=nFft, hop_length=hopLen,
                                          n_mels=40, fmax=sr/2)
    melDb = librosa.power_to_db(melS + 1e-10, ref=np.max)

    n = min(len(rmsLow), len(rmsFull), len(zcr), mfcc.shape[1], melDb.shape[1])
    rmsLow, rmsFull, zcr = rmsLow[:n], rmsFull[:n], zcr[:n]
    mfcc, melDb = mfcc[:, :n], melDb[:, :n]

    energyDb = librosa.amplitude_to_db(rmsLow + 1e-8, ref=np.max)

    lowbandRatio = rmsLow / (rmsFull + 1e-9)
    voiced = (lowbandRatio >= VOICE_LOWBAND_RATIO) & (zcr <= VOICE_ZCR_MAX)

    melFreqs = librosa.mel_frequencies(n_mels=40, fmax=sr/2)
    maskF1 = (melFreqs >= 250) & (melFreqs <= 900)
    maskMid = (melFreqs > 900) & (melFreqs <= 2500)
    maskHi = (melFreqs > 2500)
    eF1 = melDb[maskF1, :].mean(axis=0)
    eMid = melDb[maskMid, :].mean(axis=0)
    eHi = melDb[maskHi, :].mean(axis=0)

    lowHiContrast = eF1 - eHi
    midContribution = eMid - eHi
    mfcc_norm = np.abs(mfcc[1, :]) + np.abs(mfcc[2, :])

    def _normFrame(v):
        v = v - v.min()
        m = v.max()
        return v / m if m > 1e-9 else np.zeros_like(v)

    vowelScore = (_normFrame(lowHiContrast)
                  + _normFrame(midContribution)
                  + _normFrame(mfcc_norm)) / 3.0

    return {
        "energyDb": energyDb, "voiced": voiced, "vowelScore": vowelScore,
        "hopLen": hopLen, "frameLen": frameLen,
        "melDb": melDb, "mfcc": mfcc, "zcr": zcr, "rmsLow": rmsLow,
    }


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def selectNuclei(energyDb: np.ndarray, voiced: np.ndarray,
                 vowelScore: np.ndarray, expectedCount: int,
                 hopLen: int, sr: int) -> tuple[np.ndarray, int]:
    """
    Ищет ядра слогов и приводит их к ИЗВЕСТНОМУ числу слогов из текста.

    Алгоритм:
      1. на озвончённых кадрах ищем пики энергии (provал/prominence);
      2. «сырое» число пиков (с порогом DIP_DB) запоминаем для прозрачности;
      3. из кандидатов выбираем ровно expectedCount самых «гласных»
         по комбинированному score: энергия + выраженность пика +
         MFCC/MEL-«гласность» (vowelScore). MFCC-компонента помогает
         отбрасывать ложные пики на сонорных (р/м/н/л/й), которые
         по энергии могут конкурировать с настоящими гласными,
         но по тембру явно отличаются.
      4. если кандидатов меньше, чем нужно, добор делаем уже на границах.

    Возвращает (выбранные_пики, сырое_число_пиков).
    """
    if len(energyDb) == 0:
        return np.array([], dtype=int), 0

    # ── C: контур для поиска пиков теперь «взвешен по гласности» ──
    # К энергии прибавляем бонус за высокий vowelScore (MFCC/MEL говорят
    # «это гласный») и штраф за низкий. Это в самом детектировании пиков
    # подавляет ложные пики на звонких согласных (р/м/н/л) и помогает
    # проявиться редуцированным гласным, у которых энергия низкая, но
    # спектр гласный. Усиление умеренное (±VOWEL_PEAK_GAIN_DB дБ).
    contour = energyDb.astype(float).copy()
    if len(vowelScore) == len(contour):
        contour = contour + (vowelScore - 0.5) * 2.0 * VOWEL_PEAK_GAIN_DB
    contour[~voiced] = contour.min() - 10.0          # глушим неозвончённые кадры
    floor   = float(np.max(energyDb)) - PEAK_FLOOR_DB
    minDist = max(1, int(MIN_NUCLEUS_DIST_S * sr / hopLen))

    # «сырое» обнаружение — сколько ядер видно акустике без подсказки текста
    rawPeaks, _ = find_peaks(contour, prominence=DIP_DB, distance=minDist, height=floor)
    rawCount = len(rawPeaks)

    # кандидаты с пониженным порогом — чтобы было из чего выбирать
    cand, props = find_peaks(contour, prominence=0.5, distance=minDist, height=floor)
    if len(cand) == 0:
        return np.array([int(np.argmax(contour))]), rawCount

    # Комбинированный score: энергия + выраженность + «гласность» по MFCC/MEL
    vs = vowelScore[cand] if len(vowelScore) >= len(cand) else np.ones(len(cand))
    score = _norm(energyDb[cand]) + _norm(props["prominences"]) + 0.8 * _norm(vs)

    if len(cand) >= expectedCount:
        keep = cand[np.argsort(score)[-expectedCount:]]   # самые «гласные» N
        return np.sort(keep), rawCount
    return np.sort(cand), rawCount                          # меньше — добьём ниже


def nucleiToBoundaries(peaks: np.ndarray, energyDb: np.ndarray, hopLen: int,
                       totalSamples: int, sr: int, expectedCount: int,
                       vowelScore: np.ndarray | None = None) -> list[tuple[int, int]]:
    """
    Границы слогов — в минимуме ЭНЕРГИИ между соседними ядрами.

    (Пробовали добавлять «гласность» (vowelScore) в поиск провала, но это
    ухудшило медианную ошибку границ на всех записях: между двумя гласными
    и так стоит согласный с низкой гласностью, поэтому добавление vowelScore
    смещало границу к середине согласного, а не к реальному стыку слогов.
    Поэтому здесь оставлена чистая энергия. vowelScore используется только
    при ОТБОРЕ ядер, где он полезен.)

    Гарантирует РОВНО expectedCount непустых сегментов.
    """
    if len(peaks) == 0:
        peaks = np.array([len(energyDb) // 2 if len(energyDb) else 0])

    cuts = [0]
    for k in range(len(peaks) - 1):
        a, b   = int(peaks[k]), int(peaks[k + 1])
        valley = a + int(np.argmin(energyDb[a:b + 1]))
        cuts.append(int(valley * hopLen))
    cuts.append(totalSamples)
    segs = list(zip(cuts[:-1], cuts[1:]))

    # добор до нужного числа слогов: дробим самый длинный сегмент
    while len(segs) < expectedCount:
        i      = max(range(len(segs)), key=lambda j: segs[j][1] - segs[j][0])
        s, e   = segs[i]
        mid    = (s + e) // 2
        segs[i:i + 1] = [(s, mid), (mid, e)]

    minSeg, prev, out = max(int(sr * 0.020), 1), 0, []
    for s, e in segs:
        s = max(prev, min(s, totalSamples))
        e = max(s + minSeg, min(e, totalSamples))
        out.append((s, e))
        prev = e
    return out


# ══════════════════════════════════════════════════════════════════
# 4. АКУСТИЧЕСКИЕ ПРИЗНАКИ + ФОРМАНТЫ
# ══════════════════════════════════════════════════════════════════

def estimateFormants(signal: np.ndarray, sr: int, maxFormants: int = 2) -> list[float]:
    """Оценка формант (F1, F2) через LPC — для анализа качества гласного."""
    if len(signal) < int(sr * 0.020):
        return []
    y = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])  # преэмфазис
    y = y * np.hamming(len(y))
    order = int(2 + sr / 1000)
    try:
        a = librosa.lpc(y, order=order)
    except Exception:
        return []
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.01]
    if len(roots) == 0:
        return []
    angz  = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angz * (sr / (2 * np.pi))
    bw    = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + 1e-9)
    formants = sorted(f for f, b in zip(freqs, bw)
                      if 90 < f < (sr / 2 - 100) and b < 400)
    return [round(float(f), 1) for f in formants[:maxFormants]]


def centralization(formants: list[float]) -> float:
    """Расстояние (F1,F2) до центра гласного пространства (шва).
       Меньше → гласный более редуцирован/централизован."""
    if len(formants) < 2:
        return -1.0
    f1, f2 = formants[0], formants[1]
    return round(float(np.hypot(f1 - SCHWA_F1, f2 - SCHWA_F2)), 1)


def findVowelNucleus(signal: np.ndarray, sr: int) -> tuple[int, int]:
    """
    Внутри уже выделенного слога ищет ВОКАЛИЧЕСКОЕ ЯДРО (вокалическое ядро) —
    самую «гласноподобную» подпоследовательность кадров.

    Использует три признака (без pyin — тот нестабилен на коротких
    сегментах <0.5 с, из которых состоит отдельный слог):
      1. Низкочастотная энергия (≤1000 Гц) — у гласных энергия
         сконцентрирована внизу, у согласных рассеивается вверх.
      2. MEL-контраст F1 vs высокие — у гласного полоса 250–900 Гц
         (первые две форманты) доминирует над шумом >2500 Гц.
      3. ZCR (частота пересечения нуля) — у гласных низкая (<0.2),
         у фрикативов/взрывных высокая.

    Возвращает (start_sample, end_sample) — границы ядра внутри
    переданного signal. Если ядро не выделяется — возвращает весь
    сегмент (запасной вариант).
    """
    n = len(signal)
    if n < int(sr * 0.040):
        return 0, n

    frameLen = max(int(sr * 0.025), 128)
    hopLen   = max(frameLen // 2, 32)
    nFft     = max(frameLen, 256)

    # ── 1. Энергия и ZCR ────────────────────────────────────────
    low = _lowpass(signal, sr, LOWPASS_HZ)
    rmsLow  = librosa.feature.rms(y=low,    frame_length=frameLen, hop_length=hopLen)[0]
    rmsFull = librosa.feature.rms(y=signal, frame_length=frameLen, hop_length=hopLen)[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=frameLen,
                                              hop_length=hopLen)[0]

    # ── 2. MEL-спектрограмма: контраст F1 vs высокие ────────────
    melS = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=nFft,
                                           hop_length=hopLen,
                                           n_mels=40, fmax=sr/2)
    melDb = librosa.power_to_db(melS + 1e-10, ref=np.max)

    # Выравниваем длины
    nFrames = min(len(rmsLow), len(zcr), melDb.shape[1])
    rmsLow = rmsLow[:nFrames]
    zcr    = zcr[:nFrames]
    melDb  = melDb[:, :nFrames]

    if nFrames < 3:
        return 0, n

    # ── 3. Три компоненты score ─────────────────────────────────
    # (a) Энергия низкочастотной полосы (дБ) — гласные громкие внизу
    e_db = librosa.amplitude_to_db(rmsLow + 1e-8, ref=np.max)
    e_score = np.clip((e_db + 35.0) / 35.0, 0.0, 1.0)

    # (b) ZCR — у гласных низкая
    z_score = 1.0 - np.clip(zcr / 0.22, 0.0, 1.0)

    # (c) MEL-контраст: F1-область (250–900 Гц) минус шум (>2500 Гц).
    # У гласного этот контраст большой (>10 дБ), у согласных малый/отрицательный.
    # Это ключевой признак для отличия гласного от сонорного (р/м/н/л):
    # у сонорного энергия в F1-полосе слабее, а в высокой полосе — сильнее.
    melFreqs = librosa.mel_frequencies(n_mels=40, fmax=sr/2)
    maskF1 = (melFreqs >= 250) & (melFreqs <= 900)
    maskHi = (melFreqs > 2500)
    eF1 = melDb[maskF1, :].mean(axis=0) if maskF1.any() else np.zeros(nFrames)
    eHi = melDb[maskHi, :].mean(axis=0) if maskHi.any() else np.zeros(nFrames)
    lowHiContrast = eF1 - eHi
    m_score = np.clip((lowHiContrast + 10.0) / 30.0, 0.0, 1.0)

    # Итоговый score: MEL-контраст и энергия — основные (по 0.35),
    # ZCR — вспомогательный (0.30).
    # MEL-контраст получил больший вес т.к. он лучше всего отделяет
    # гласные от сонорных согласных.
    score = 0.35 * m_score + 0.35 * e_score + 0.30 * z_score

    # Порог: 55% от максимума, но не ниже 0.30
    threshold = max(0.30, 0.55 * score.max())
    above = score >= threshold
    if not above.any():
        return 0, n

    # Самая длинная связная область выше порога — это ядро
    best_s, best_e = 0, 0
    cur_s = None
    for i, a in enumerate(above):
        if a and cur_s is None:
            cur_s = i
        elif not a and cur_s is not None:
            if i - cur_s > best_e - best_s:
                best_s, best_e = cur_s, i
            cur_s = None
    if cur_s is not None:
        if len(above) - cur_s > best_e - best_s:
            best_s, best_e = cur_s, len(above)

    s_sample = best_s * hopLen
    e_sample = min(n, best_e * hopLen + frameLen)

    # минимум 30 мс ядра — снижено для коротких редуцированных гласных
    if (e_sample - s_sample) < int(sr * 0.030):
        return 0, n
    return s_sample, e_sample


def extractFeatures(signal: np.ndarray, sr: int) -> dict:
    """Признаки короткого сегмента (слога).

    Считает признаки в ДВУХ вариантах:
      • полный слог (как раньше) — основной набор полей;
      • только вокалическое ядро ('core') — для анализа редукции,
        чтобы не разбавлять сигнал согласными.
    """
    if len(signal) < 16:
        return {}

    nFft      = max(min(2048, len(signal)), 64)
    hopLength = nFft // 4
    nMels     = min(128, nFft // 2)

    mfccMatrix = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                      n_fft=nFft, hop_length=hopLength, n_mels=nMels)
    mfccMean = mfccMatrix.mean(axis=1).tolist()

    rmsVal   = float(librosa.feature.rms(y=signal, frame_length=nFft,
                                         hop_length=hopLength).mean())
    zcrVal   = float(librosa.feature.zero_crossing_rate(signal).mean())
    formants = estimateFormants(signal, sr)

    out = {
        "mfcc":       [round(v, 4) for v in mfccMean],
        "energyMean": round(rmsVal, 6),
        "zcrMean":    round(zcrVal, 6),
        "formants":   formants,
        "central":    centralization(formants),
        "duration":   round(len(signal) / sr, 3),
    }

    # ── НОВОЕ: вокалическое ядро внутри этого слога ──
    # Считаем признаки только по самой «гласноподобной» подпоследовательности.
    # Это устраняет влияние соседних согласных и должно улучшить
    # детектирование редукции для гласных «а», «е», «и» (для «о» оно
    # и так уже работает, поэтому ожидаем сохранение значимости).
    n_signal = len(signal)
    ns, ne = findVowelNucleus(signal, sr)
    if (ne - ns) >= int(sr * 0.040) and (ne - ns) < n_signal:
        core_sig  = signal[ns:ne]
        core_rms  = float(librosa.feature.rms(y=core_sig).mean())
        core_form = estimateFormants(core_sig, sr)
        out["core"] = {
            "duration":   round((ne - ns) / sr, 3),
            "energyMean": round(core_rms, 6),
            "formants":   core_form,
            "central":    centralization(core_form),
            "startInSyl": round(ns / sr, 3),     # для отладки
            "endInSyl":   round(ne / sr, 3),
        }
    else:
        # ядро не выделилось — пометим, чтобы анализ редукции мог откатиться на полный слог
        out["core"] = None

    return out


# ─── Быстрые версии: используют предвычисленные контуры слова ───

def findVowelNucleusFast(signal: np.ndarray, sr: int,
                         wc: dict, f0: int, f1: int) -> tuple:
    """
    То же что findVowelNucleus, но не вычисляет mel-спектрограмму заново —
    берёт её из предвычисленных контуров слова wc.
    f0, f1 — кадровые индексы слога внутри контуров.
    Возвращает (start_sample, end_sample) относительно signal.
    """
    n = len(signal)
    if n < int(sr * 0.040) or f1 <= f0:
        return 0, n

    hopLen = wc["hopLen"]
    nFrames = f1 - f0
    if nFrames < 3:
        return 0, n

    mel_slice = wc["melDb"][:, f0:f1]
    zcr_slice = wc["zcr"][f0:f1]
    rms_slice = wc["rmsLow"][f0:f1]

    e_db = librosa.amplitude_to_db(rms_slice + 1e-8, ref=np.max)
    e_score = np.clip((e_db + 35.0) / 35.0, 0.0, 1.0)
    z_score = 1.0 - np.clip(zcr_slice / 0.22, 0.0, 1.0)

    melFreqs = librosa.mel_frequencies(n_mels=40, fmax=sr/2)
    maskF1 = (melFreqs >= 250) & (melFreqs <= 900)
    maskHi = (melFreqs > 2500)
    eF1 = mel_slice[maskF1, :].mean(axis=0) if maskF1.any() else np.zeros(nFrames)
    eHi = mel_slice[maskHi, :].mean(axis=0) if maskHi.any() else np.zeros(nFrames)
    m_score = np.clip((eF1 - eHi + 10.0) / 30.0, 0.0, 1.0)

    score = 0.35 * m_score + 0.35 * e_score + 0.30 * z_score
    threshold = max(0.30, 0.55 * score.max())
    above = score >= threshold
    if not above.any():
        return 0, n

    best_s, best_e, cur_s = 0, 0, None
    for i, a in enumerate(above):
        if a and cur_s is None: cur_s = i
        elif not a and cur_s is not None:
            if i - cur_s > best_e - best_s: best_s, best_e = cur_s, i
            cur_s = None
    if cur_s is not None and len(above) - cur_s > best_e - best_s:
        best_s, best_e = cur_s, len(above)

    s_sample = best_s * hopLen
    e_sample = min(n, best_e * hopLen + wc["frameLen"])
    if (e_sample - s_sample) < int(sr * 0.030):
        return 0, n
    return s_sample, e_sample


def extractFeaturesFast(signal: np.ndarray, sr: int,
                        wc: dict, f0: int, f1: int) -> dict:
    """
    То же что extractFeatures, но MFCC и RMS/ZCR берутся из контуров слова.
    Форманты (LPC) по-прежнему считаются по raw signal (быстро).
    """
    if len(signal) < 16 or f1 <= f0:
        return {}
    nFrames = f1 - f0
    if nFrames < 1:
        return {}
    mfccMean = wc["mfcc"][:, f0:f1].mean(axis=1).tolist()
    rmsVal = float(wc["rmsLow"][f0:f1].mean())
    zcrVal = float(wc["zcr"][f0:f1].mean())
    formants = estimateFormants(signal, sr)
    return {
        "mfcc":       [round(v, 4) for v in mfccMean],
        "energyMean": round(rmsVal, 6),
        "zcrMean":    round(zcrVal, 6),
        "formants":   formants,
        "central":    centralization(formants),
        "duration":   round(len(signal) / sr, 3),
    }


def trimSilenceBounds(signal: np.ndarray, sr: int,
                      floorDb: float = 40.0) -> tuple[int, int]:
    """
    Возвращает индексы начала и конца «звучащего ядра» слова: отсекает
    ведущую и хвостовую тишину, чтобы она не отнимала «слот» под слог.
    Предохранитель: не отрезаем больше половины слова в сумме.
    """
    if len(signal) < int(sr * 0.04):
        return 0, len(signal)
    frameLen = max(int(sr * 0.020), 64)
    hopLen   = max(frameLen // 2, 16)
    rms = librosa.feature.rms(y=signal, frame_length=frameLen, hop_length=hopLen)[0]
    if len(rms) == 0 or rms.max() <= 0:
        return 0, len(signal)
    db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    above = db > -floorDb
    if not above.any():
        return 0, len(signal)
    first = int(np.argmax(above))
    last  = len(above) - 1 - int(np.argmax(above[::-1]))
    s = max(0, first * hopLen)
    e = min(len(signal), (last + 1) * hopLen + frameLen)
    # Не отрезаем слишком агрессивно
    if (e - s) < int(len(signal) * 0.5):
        return 0, len(signal)
    return s, e


def analyzeSyllablesInWord(wordSegment: np.ndarray, sr: int,
                           syllables: list[str]) -> tuple[list[dict], int]:
    """
    Находит ядра слогов, приводит их к известному числу слогов из текста,
    режет по провалам энергии, считает признаки. Возвращает
    (результаты_по_слогам, сырое_акустическое_число_ядер).

    Перед анализом отсекает ведущую и хвостовую тишину внутри слова —
    это предотвращает ситуацию, когда последний слог получает в свой
    сегмент молчание (а сам гласный остался слева от среза).
    """
    expected = len(syllables)
    trimStart, trimEnd = trimSilenceBounds(wordSegment, sr, floorDb=40.0)
    core = wordSegment[trimStart:trimEnd]

    # Контуры слова — melDb, mfcc и пр. считаются ОДИН РАЗ
    wc = computeContours(core, sr)
    energyDb, voiced, vowelScore, hopLen = (
        wc["energyDb"], wc["voiced"], wc["vowelScore"], wc["hopLen"])

    peaks, rawDetected = selectNuclei(energyDb, voiced, vowelScore, expected, hopLen, sr)
    bounds = nucleiToBoundaries(peaks, energyDb, hopLen, len(core), sr, expected, vowelScore)
    bounds = [(s + trimStart, e + trimStart) for s, e in bounds]

    # Переводим границы в кадры (относительно core = wordSegment[trimStart:trimEnd])
    frame_from_sample = lambda s: int((s - trimStart) / hopLen)

    results = []
    for i, sylText in enumerate(syllables):
        segStart, segEnd = bounds[i] if i < len(bounds) else bounds[-1]
        seg = wordSegment[segStart:segEnd]

        # Кадровые индексы этого слога внутри контуров слова
        f0 = max(0, frame_from_sample(segStart))
        f1 = min(wc["melDb"].shape[1], max(f0 + 1, frame_from_sample(segEnd)))

        feats = extractFeaturesFast(seg, sr, wc, f0, f1)
        # Вокалическое ядро — используем предвычисленный melDb
        ns, ne = findVowelNucleusFast(seg, sr, wc, f0, f1)
        if ns is not None and (ne - ns) >= int(sr * 0.030) and (ne - ns) < len(seg):
            core_sig = seg[ns:ne]
            core_rms  = float(librosa.feature.rms(y=core_sig).mean())
            core_form = estimateFormants(core_sig, sr)
            feats["core"] = {
                "duration":   round((ne - ns) / sr, 3),
                "energyMean": round(core_rms, 6),
                "formants":   core_form,
                "central":    centralization(core_form),
                "startInSyl": round(ns / sr, 3),
                "endInSyl":   round(ne / sr, 3),
            }
        else:
            feats["core"] = None

        remarks = []
        if feats.get("energyMean", 0) < 0.004:
            remarks.append(f"Слог «{sylText}» произнесён очень тихо")

        results.append({
            "syllable": sylText,
            "startSec": round(segStart / sr, 3),
            "endSec":   round(segEnd / sr, 3),
            "acoustics": feats,
            "isProblematic": len(remarks) > 0,
            "remarks": remarks,
        })

    return results, rawDetected


# ══════════════════════════════════════════════════════════════════
# 5. УДАРЕНИЕ И РЕДУКЦИЯ (внутреннее сопоставление слогов)
# ══════════════════════════════════════════════════════════════════

def detectAcousticStress(syllableResults: list[dict],
                         f0_by_syllable: list[float] | None = None) -> tuple[int, list[float]]:
    """
    Определяет, какой слог говорящий ФАКТИЧЕСКИ выделил голосом
    (фактическое ударение / actual stress).

    УЛУЧШЕННАЯ ВЕРСИЯ: если для всех слогов доступно вокалическое
    ядро (core), использует признаки ЯДРА вместо целого слога.
    Длительность и энергия целого слога включают согласные, которые
    не несут информации об ударении и «размывают» сигнал. Ядро
    содержит только гласный — его длительность и F1 напрямую
    отражают ударность.

    Комбинирует пять признаков (нормируются внутри слова):
      • F0 (ЧОТ / частота основного тона) — сильнейший коррелят
        ударения в русском; у ударного гласного F0 выше или
        заметно меняется.
      • длительность — ударный гласный длиннее. При использовании
        core: длительность ТОЛЬКО вокалического ядра, без согласных.
      • энергия (RMS) — ударный громче
      • F1 (первая форманта) — ударный гласный «открытее», F1 выше
      • MFCC «открытость» — через MFCC[1] (чем меньше |MFCC[1]|,
        тем «открытее» артикуляция)

    Веса: F0 и длительность — основные, энергия и F1 — вспомогательные,
    MFCC — корректирующий.
    """
    n = len(syllableResults)
    if n == 0:
        return 0, []
    if n == 1:
        return 0, [1.0]

    def _norm01(vals):
        vals = np.asarray(vals, float)
        rng = vals.max() - vals.min()
        return (vals - vals.min()) / rng if rng > 1e-9 else np.zeros_like(vals)

    # Проверяем, у всех ли слогов есть вокалическое ядро
    useCore = all(
        (r.get("acoustics") or {}).get("core") is not None
        for r in syllableResults
    )

    durs, eners, f1s, mfcc1_abs = [], [], [], []
    for r in syllableResults:
        a = r.get("acoustics") or {}
        if useCore:
            c = a["core"]
            durs.append(float(c.get("duration", 0.0)))
            eners.append(float(c.get("energyMean", 0.0)))
            core_formants = c.get("formants") or [0.0, 0.0]
            f1s.append(float(core_formants[0]) if core_formants and core_formants[0] else 0.0)
        else:
            durs.append(float(a.get("duration", 0.0)))
            eners.append(float(a.get("energyMean", 0.0)))
            formants = a.get("formants") or [0.0, 0.0]
            f1s.append(float(formants[0]) if formants and formants[0] else 0.0)
        # MFCC берём всегда от целого слога — кепстральные коэффициенты
        # на коротком ядре могут быть нестабильны
        mfcc = a.get("mfcc") or []
        mfcc1_abs.append(abs(float(mfcc[1])) if len(mfcc) > 1 else 0.0)

    durN  = _norm01(durs)
    enerN = _norm01(eners)
    f1N   = _norm01(f1s)
    mfccN_inv = 1.0 - _norm01(mfcc1_abs)   # меньше |MFCC[1]| → «открытее»

    if f0_by_syllable is not None and len(f0_by_syllable) == n:
        f0N = _norm01([(v if v and v > 0 else 0.0) for v in f0_by_syllable])
        # F0 доступна — она главный сигнал ударения.
        # При использовании core длительность ядра надёжнее → повышаем вес.
        if useCore:
            scores = (0.30 * f0N + 0.30 * durN + 0.18 * enerN
                      + 0.15 * f1N + 0.07 * mfccN_inv).tolist()
        else:
            scores = (0.30 * f0N + 0.25 * durN + 0.20 * enerN
                      + 0.15 * f1N + 0.10 * mfccN_inv).tolist()
    else:
        # F0 нет — длительность и энергия основные
        if useCore:
            scores = (0.40 * durN + 0.25 * enerN + 0.20 * f1N + 0.15 * mfccN_inv).tolist()
        else:
            scores = (0.35 * durN + 0.30 * enerN + 0.20 * f1N + 0.15 * mfccN_inv).tolist()

    return int(np.argmax(scores)), scores


def computeSyllableF0(wordSegment: np.ndarray, sr: int,
                      syllableResults: list[dict], wordStart: float) -> list[float]:
    """
    Считает медианную F0 для каждого слога слова через librosa.pyin.
    ВАЖНО: этот вариант больше НЕ вызывается в основном цикле.
    Используйте sliceSyllableF0() с предварительно вычисленным fullF0.
    """
    n = len(syllableResults)
    if n == 0 or len(wordSegment) < int(sr * 0.05):
        return [0.0] * n
    try:
        f0, voiced_flag, _ = librosa.pyin(
            wordSegment, sr=sr,
            fmin=70, fmax=400,
            frame_length=max(int(sr * 0.04), 256))
    except Exception:
        return [0.0] * n
    hop = int(sr * 0.04) // 4 if int(sr*0.04) >= 256 else 512
    times = librosa.times_like(f0, sr=sr, hop_length=hop)

    out = []
    for r in syllableResults:
        s = r.get("startSec", 0.0)
        e = r.get("endSec", 0.0)
        mask = (times >= s) & (times < e)
        vals = f0[mask]
        vals = vals[~np.isnan(vals)] if vals.size else vals
        out.append(float(np.median(vals)) if vals.size else 0.0)
    return out


def computeFullF0(fullSignal: np.ndarray, sr: int) -> tuple:
    """
    ОДИН РАЗ вычисляет F0 для всего аудиосигнала.
    Возвращает (f0_array, times_array, hop_length).
    Это главная оптимизация: pyin дёшев на длинных сигналах,
    но крайне дорог при вызове на 100+ коротких сегментах.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            fullSignal, sr=sr,
            fmin=70, fmax=400,
            frame_length=max(int(sr * 0.04), 256))
        hop = int(sr * 0.04) // 4 if int(sr*0.04) >= 256 else 512
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        return f0, times, hop
    except Exception:
        return None, None, None


def sliceSyllableF0(fullF0: np.ndarray, f0Times: np.ndarray,
                    wordStartSec: float,
                    syllableResults: list[dict]) -> list[float]:
    """
    Из предвычисленного F0 всего аудио вырезает медианные F0 для
    каждого слога слова. wordStartSec — абсолютное время начала слова
    в аудио. syllableResults[i]["startSec"/"endSec"] — относительно
    начала слова.
    """
    n = len(syllableResults)
    if n == 0 or fullF0 is None:
        return [0.0] * n
    out = []
    for r in syllableResults:
        abs_s = wordStartSec + r.get("startSec", 0.0)
        abs_e = wordStartSec + r.get("endSec", 0.0)
        mask = (f0Times >= abs_s) & (f0Times < abs_e)
        vals = fullF0[mask]
        vals = vals[~np.isnan(vals)] if vals.size else vals
        out.append(float(np.median(vals)) if vals.size else 0.0)
    return out


def findStressedSyllableByDict(word: str, syllables: list[str], accentizer) -> int:
    """Индекс ударного слога по словарю ruaccent. ВАЖНО: word — реальная форма."""
    if len(syllables) <= 1 or accentizer is None:
        return 0
    try:
        accented = accentizer.process_all(word.lower())
    except Exception:
        return -1

    mark = "+" if "+" in accented else ("\u0301" if "\u0301" in accented else None)
    if mark is None:
        return -1
    markPos = accented.index(mark)
    cleanBefore = accented[:markPos].replace("+", "").replace("\u0301", "")
    stressedCharIdx = len(cleanBefore) - 1

    count = 0
    for sylIdx, syl in enumerate(syllables):
        count += len(syl)
        if stressedCharIdx < count:
            return sylIdx
    return len(syllables) - 1


def analyzeVowelReduction(syllableResults: list[dict], syllables: list[str],
                          stressedIdx: int) -> dict:
    """
    Сопоставляет безударные слоги с УДАРНЫМ в пределах одного слова (по
    заданию научного руководителя — «сопоставление слогов для редуцированных
    звуков»).

    УЛУЧШЕННАЯ ВЕРСИЯ: если для всех слогов доступно вокалическое ядро
    (core), использует признаки ЯДРА (длительность гласного, энергия
    гласного, централизацию по формантам гласного) вместо целого слога.
    Это принципиально: длительность слога включает согласные, которые
    редукции не подвергаются и «размывают» сигнал. Например, слог «страх»
    длинный за счёт консонантного кластера, хотя гласный [а] короткий.

    Признаки редукции безударного гласного по сравнению с ударным:
      • длительность ↓ (безударный короче)
      • энергия ↓      (безударный тише)
      • централизация ↑ — ГЛАВНЫЙ признак: гласный смещается к нейтральному
        [ə] (шва), т.е. расстояние (F1,F2) до центра гласного пространства
        уменьшается. У ударного гласного оно большое (чёткий [а]/[о]/[е]),
        у редуцированного — маленькое.

    «Редукции нет», если безударный слог по ВСЕМ трём признакам почти не
    отличается от ударного: он и не короче, и не тише, и его гласный
    не централизован. Раньше использовались только длительность и энергия;
    добавление централизации делает оценку устойчивее и ближе к фонетике.
    """
    if len(syllableResults) <= 1 or not (0 <= stressedIdx < len(syllableResults)):
        return {"stressedIdx": max(stressedIdx, 0),
                "stressedSyl": syllables[stressedIdx] if 0 <= stressedIdx < len(syllables) else "",
                "reductionErrors": [], "reductionScore": 100.0,
                "hasReductionIssue": False}

    # Проверяем, у всех ли слогов есть вокалическое ядро
    useCore = all(
        (r.get("acoustics") or {}).get("core") is not None
        for r in syllableResults
    )

    st = syllableResults[stressedIdx]["acoustics"]
    if useCore and st.get("core"):
        stCore = st["core"]
        stDur    = stCore.get("duration", 0) or 1e-6
        stEner   = stCore.get("energyMean", 0) or 1e-6
        stCentral = stCore.get("central", -1)
    else:
        stDur    = st.get("duration", 0) or 1e-6
        stEner   = st.get("energyMean", 0) or 1e-6
        stCentral = st.get("central", -1)

    errors, checked, correct = [], 0, 0
    for i, (sd, sylText) in enumerate(zip(syllableResults, syllables)):
        if i == stressedIdx:
            continue
        if not any(ch in REDUCIBLE_VOWELS for ch in sylText.lower()):
            continue
        ac = sd["acoustics"]
        if useCore and ac.get("core"):
            c = ac["core"]
            dur = c.get("duration", 0)
            ener = c.get("energyMean", 0)
            unCentral = c.get("central", -1)
        else:
            dur = ac.get("duration", 0)
            ener = ac.get("energyMean", 0)
            unCentral = ac.get("central", -1)
        if dur < 0.025:    # снижено с 40 мс: ядро может быть короче слога
            continue
        checked += 1
        durRatio  = dur / stDur
        enerRatio = ener / stEner

        # централизация: отношение «насколько безударный ближе к шва, чем ударный».
        # Меньше central → ближе к центру → сильнее редукция.
        # centralRatio < 1 означает, что безударный централизован сильнее ударного (хорошо).
        if stCentral > 0 and unCentral >= 0:
            centralRatio = unCentral / stCentral      # <1 = безударный ближе к шва (есть редукция)
        else:
            centralRatio = None                       # форманты не определились

        # «нет редукции»: безударный не короче, не тише И не централизован
        dur_no_red = durRatio  >= DUR_RATIO_NO_RED
        ener_no_red = enerRatio >= ENER_RATIO_NO_RED
        central_no_red = (centralRatio is not None and centralRatio >= CENTRAL_RATIO_NO_RED)

        # Если централизация доступна — требуем ВСЕ три признака для «нет редукции».
        # Если форманты не определились — откатываемся на старую логику (длит.+энергия).
        if centralRatio is not None:
            no_reduction = dur_no_red and ener_no_red and central_no_red
        else:
            no_reduction = dur_no_red and ener_no_red

        if no_reduction:
            hint = (f"Безударный слог «{sylText}»: длительность {durRatio*100:.0f}%, "
                    f"энергия {enerRatio*100:.0f}% от ударного «{syllables[stressedIdx]}»")
            if centralRatio is not None:
                hint += f", централизация {centralRatio*100:.0f}% (гласный не сместился к нейтральному)"
            hint += ". Похоже, гласный не редуцирован."
            errors.append({
                "syllable": sylText, "position": i + 1,
                "durRatio": round(durRatio, 2), "enerRatio": round(enerRatio, 2),
                "centralRatio": round(centralRatio, 2) if centralRatio is not None else None,
                "hint": hint,
            })
        else:
            correct += 1

    score = (correct / checked * 100) if checked > 0 else 100.0
    return {"stressedIdx": stressedIdx, "stressedSyl": syllables[stressedIdx],
            "reductionErrors": errors, "reductionScore": round(score, 1),
            "hasReductionIssue": len(errors) > 0}


# ══════════════════════════════════════════════════════════════════
# 6. МОРФОЛОГИЯ (только для части речи и леммы в отчёте)
# ══════════════════════════════════════════════════════════════════

POS_NAMES = {
    "NOUN": "существительное", "ADJF": "прилагательное", "VERB": "глагол",
    "INFN": "инфинитив", "NUMR": "числительное", "ADVB": "наречие",
    "NPRO": "местоимение", "PREP": "предлог", "CONJ": "союз",
    "PRCL": "частица", "INTJ": "междометие", "COMP": "сравн. степень",
    "UNKN": "неизвестно",
}


def analyzeWordMorph(word: str, morphAnalyzer) -> dict:
    cw = cleanWord(word)
    if not cw or morphAnalyzer is None:
        return {"lemma": word, "pos": "UNKN", "posName": POS_NAMES["UNKN"]}
    best   = morphAnalyzer.parse(cw)[0]
    posTag = best.tag.POS or "UNKN"
    return {"lemma": best.normal_form, "pos": posTag,
            "posName": POS_NAMES.get(posTag, posTag)}


# ══════════════════════════════════════════════════════════════════
# 7. ОСНОВНОЙ КОНВЕЙЕР
# ══════════════════════════════════════════════════════════════════

def cleanTextToWords(text: str) -> list[str]:
    text = re.sub(r"[^\w\sа-яА-ЯёЁ]", " ", text).lower()
    return text.split()


# Кеш моделей (загружаются один раз при первом вызове)
_cached_models = None


def loadModels(verbose=True):
    """Загружает VOSK, pymorphy2, ruaccent ОДИН РАЗ и кеширует."""
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    if not os.path.exists(modelPath):
        print(f"❌ Модель VOSK не найдена: {modelPath}")
        sys.exit(1)
    from vosk import Model, KaldiRecognizer
    import pymorphy2
    morphAnalyzer = pymorphy2.MorphAnalyzer()
    accentizer = None
    try:
        from ruaccent import RUAccent
        accentizer = RUAccent()
        accentizer.load(omograph_model_size="turbo", use_dictionary=True)
        if verbose: print("ruaccent готов.")
    except Exception as e:
        if verbose: print(f"ruaccent недоступен ({e}); ударение — по энергии (резерв).")
    voskModel = Model(modelPath)
    if verbose: print("VOSK модель загружена.")
    _cached_models = (voskModel, KaldiRecognizer, morphAnalyzer, accentizer)
    return _cached_models


def main(models=None):
    """Основной конвейер. models=(voskModel, KaldiRecognizer, morphAnalyzer, accentizer)."""
    if not os.path.exists(audioFile):
        print(f"❌ Аудио не найдено: {audioFile}"); sys.exit(1)

    if models is not None:
        voskModel, KaldiRecognizer, morphAnalyzer, accentizer = models
    else:
        voskModel, KaldiRecognizer, morphAnalyzer, accentizer = loadModels()

    wf            = wave.open(audioFile, "rb")
    fileSr        = wf.getframerate()
    audioDuration = wf.getnframes() / fileSr
    rec = KaldiRecognizer(voskModel, fileSr)
    rec.SetWords(True)

    wordDetails, textParts = [], []
    while True:
        chunk = wf.readframes(4000)
        if not chunk:
            break
        if rec.AcceptWaveform(chunk):
            res = json.loads(rec.Result())
            wordDetails.extend(res.get("result", []))
            if res.get("text"):
                textParts.append(res["text"])
    finalRes = json.loads(rec.FinalResult())
    wordDetails.extend(finalRes.get("result", []))
    if finalRes.get("text"):
        textParts.append(finalRes["text"])
    recognizedText = " ".join(textParts)
    wf.close()
    print(f"VOSK: получено меток для {len(wordDetails)} слов.")

    # --- аудио в librosa (16 кГц моно) ---
    fullSignal, librosaSR = librosa.load(audioFile, sr=SR, mono=True, res_type="kaiser_fast")

    # --- ОПТИМИЗАЦИЯ: F0 один раз для всего аудио ---
    print("Вычисление F0 (pyin) для всего аудио…")
    fullF0, f0Times, _f0Hop = computeFullF0(fullSignal, librosaSR)
    print("F0 готово.")

    # --- главный цикл ---
    fullWordAnalysis = []
    for idx, wd in enumerate(wordDetails):
        wordText  = wd.get("word", "")
        startTime = wd.get("start", 0)
        endTime   = wd.get("end", 0)
        confScore = wd.get("conf", 0)

        surface   = cleanWord(wordText)
        morphInfo = analyzeWordMorph(wordText, morphAnalyzer)
        syllables = splitIntoSyllables(surface) if surface else [wordText]

        # VOSK часто обрезает хвост слова (особенно с редуцированным
        # последним гласным). Добавим до 80 мс в конец, но не больше
        # половины интервала до следующего слова, чтобы не залезть в него.
        padEnd = 0.080
        if idx + 1 < len(wordDetails):
            nextStart = wordDetails[idx + 1].get("start", endTime)
            padEnd    = min(padEnd, max(0.0, (nextStart - endTime) * 0.5))

        s = int(startTime * librosaSR)
        e = min(len(fullSignal), int((endTime + padEnd) * librosaSR))
        wordSegment = fullSignal[s:e]
        if len(wordSegment) < int(librosaSR * 0.04):
            continue

        syllableResults, detectedNuclei = analyzeSyllablesInWord(
            wordSegment, librosaSR, syllables)

        # ── 1. ОЖИДАЕМОЕ ударение (по словарю ruaccent) ──
        expectedStressedIdx = -1
        if accentizer is not None and surface:
            expectedStressedIdx = findStressedSyllableByDict(surface, syllables, accentizer)
        if not (0 <= expectedStressedIdx < len(syllableResults)):
            expectedStressedIdx = 0  # резерв

        # ── 2. ФАКТИЧЕСКОЕ ударение (то, что сказал говорящий) ──
        # Детектор на основе F0 + длительности + энергии + F1 + MFCC.
        # F0 (частота основного тона) — сильнейший коррелят ударения
        # в русском; считаем её по слогам отдельно через pyin.
        f0_syl = sliceSyllableF0(fullF0, f0Times, startTime, syllableResults)
        actualStressedIdx, _stressScores = detectAcousticStress(syllableResults, f0_syl)

        # Для редукции используем ожидаемое ударение (словарное)
        # — мы оцениваем, реализована ли редукция В ОТНОШЕНИИ К нормативному ударному слогу
        stressedIdx = expectedStressedIdx
        reductionInfo = analyzeVowelReduction(syllableResults, syllables, stressedIdx)

        # Несовпадение ожидаемого и фактического ударения = подозрение
        # на ошибку произношения у неносителя
        stressMismatch = (len(syllables) > 1 and
                         expectedStressedIdx != actualStressedIdx)

        countMismatch = (detectedNuclei != len(syllables))
        hasIssue = (any(r["isProblematic"] for r in syllableResults)
                    or reductionInfo["hasReductionIssue"] or countMismatch
                    or stressMismatch)

        fullWordAnalysis.append({
            "word": wordText, "start": startTime, "end": endTime,
            "confidence": confScore, "lemma": morphInfo["lemma"], "pos": morphInfo["pos"],
            "syllableStr": "-".join(syllables), "syllableCount": len(syllables),
            "detectedNuclei": detectedNuclei, "countMismatch": countMismatch,
            # Обратная совместимость: stressedIdx = ожидаемое (как было раньше)
            "stressedIdx": reductionInfo["stressedIdx"],
            # Новые поля:
            "expectedStressedIdx": expectedStressedIdx,   # словарь (ruaccent)
            "actualStressedIdx":   actualStressedIdx,     # акустика (MFCC+время+F1+E)
            "stressMismatch":      stressMismatch,        # подозрение на ошибку произношения
            "syllableAnalysis": syllableResults, "reductionInfo": reductionInfo,
            "hasIssue": hasIssue,
        })
    print(f"Проанализировано слов: {len(fullWordAnalysis)}.")

    # --- глобальная таблица контраста слога: ударный vs безударный ---
    contrast: dict[str, dict] = {}
    for w in fullWordAnalysis:
        sIdx = w["stressedIdx"]
        for i, r in enumerate(w["syllableAnalysis"]):
            key = r["syllable"]
            slot = contrast.setdefault(key, {"stressDur": [], "unstrDur": [],
                                             "stressEn": [], "unstrEn": [],
                                             "stressCentral": [], "unstrCentral": []})
            ac = r["acoustics"]
            dur, en = ac.get("duration", 0), ac.get("energyMean", 0)
            cen = ac.get("central", -1)
            if i == sIdx:
                slot["stressDur"].append(dur); slot["stressEn"].append(en)
                if cen >= 0: slot["stressCentral"].append(cen)
            else:
                slot["unstrDur"].append(dur); slot["unstrEn"].append(en)
                if cen >= 0: slot["unstrCentral"].append(cen)

    contrastTable = []
    for syl, d in contrast.items():
        if d["stressDur"] and d["unstrDur"]:   # слог встречался в обеих позициях
            contrastTable.append({
                "syllable": syl,
                "stressDur": round(float(np.mean(d["stressDur"])), 3),
                "unstrDur":  round(float(np.mean(d["unstrDur"])), 3),
                "stressEn":  round(float(np.mean(d["stressEn"])), 5),
                "unstrEn":   round(float(np.mean(d["unstrEn"])), 5),
            })

    # --- сравнение с эталонным текстом ---
    referenceText = ""
    if os.path.exists(textFile):
        with open(textFile, "r", encoding="utf-8") as f:
            referenceText = f.read().strip()
    refWordList = cleanTextToWords(referenceText) if referenceText else []
    recWordList = cleanTextToWords(recognizedText)

    wm = difflib.SequenceMatcher(None, refWordList, recWordList)
    cmp = {"correct": [], "substituted": [], "missed": [], "inserted": []}
    for op, i1, i2, j1, j2 in wm.get_opcodes():
        if op == "equal":
            cmp["correct"].extend(refWordList[i1:i2])
        elif op == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                r = refWordList[i1 + k] if (i1 + k) < i2 else "—"
                g = recWordList[j1 + k] if (j1 + k) < j2 else "—"
                cmp["substituted"].append((r, g))
        elif op == "delete":
            cmp["missed"].extend(refWordList[i1:i2])
        elif op == "insert":
            cmp["inserted"].extend(recWordList[j1:j2])
    textAccuracy = (len(cmp["correct"]) / len(refWordList) * 100) if refWordList else 0

    syllableErrors = []
    for refW, recW in cmp["substituted"]:
        if refW == "—" or recW == "—":
            continue
        c = compareSyllables(refW, recW)
        syllableErrors.append({"referenceWord": refW, "recognizedWord": recW,
                               "refStr": "-".join(c["refSyllables"]),
                               "recStr": "-".join(c["recSyllables"]),
                               "errors": c["errors"]})

    # --- темп и паузы ---
    if wordDetails:
        speechDur = wordDetails[-1].get("end", 0) - wordDetails[0].get("start", 0)
        wordCount = len(wordDetails)
        speakingRate = wordCount / speechDur if speechDur > 0 else 0
    else:
        speechDur = speakingRate = wordCount = 0
    pauses = []
    for i in range(1, len(wordDetails)):
        gap = wordDetails[i].get("start", 0) - wordDetails[i - 1].get("end", 0)
        if gap >= pauseThreshold:
            pauses.append({"afterWord": wordDetails[i - 1].get("word", ""),
                           "beforeWord": wordDetails[i].get("word", ""),
                           "duration": round(gap, 2)})

    # --- частотный словарь слогов ---
    counter: dict[str, int] = {}
    for w in fullWordAnalysis:
        for syl in w["syllableStr"].split("-"):
            if syl:
                counter[syl] = counter.get(syl, 0) + 1
    sortedSyl = sorted(counter.items(), key=lambda x: -x[1])

    printReport(audioFile, audioDuration, librosaSR, fullWordAnalysis,
                contrastTable, speechDur, speakingRate, wordCount, pauses,
                normalSpeakingRate, referenceText, recognizedText, refWordList,
                cmp, textAccuracy, syllableErrors, sortedSyl)

    saveJson(audioFile, audioDuration, librosaSR, referenceText, recognizedText,
             fullWordAnalysis, contrastTable, syllableErrors, pauses, cmp)


# ══════════════════════════════════════════════════════════════════
# 8. ОТЧЁТ И JSON
# ══════════════════════════════════════════════════════════════════

def printReport(audioFile, audioDuration, sr, words, contrastTable, speechDur,
                rate, wordCount, pauses, normRate, refText, recText, refWords,
                cmp, textAcc, sylErrors, sortedSyl):
    W = 70
    print("\n" + "=" * W)
    print("  ОТЧЁТ — СЛОГОВЫЙ АНАЛИЗ (сигнальный поиск ядер, без alignment)")
    print("=" * W)
    print(f"  Файл: {audioFile} | {audioDuration:.1f}с | {sr} Гц\n")

    good = [w for w in words if not w["hasIssue"]]
    print(f"  Слов: {len(words)} | без замечаний: {len(good)} | "
          f"с замечаниями: {len(words) - len(good)}\n")
    print("─" * W)
    print("  РАЗДЕЛ 1: Слоги в аудиосигнале (ядра по провалам энергии)")
    print("─" * W)
    for w in words:
        st = "⚠" if w["hasIssue"] else "✓"
        mm = f"  [ядер:{w['detectedNuclei']}≠слогов:{w['syllableCount']}]" if w["countMismatch"] else ""
        print(f"  {st} «{w['word']}» ({w['start']:.1f}–{w['end']:.1f}с) "
              f"{w['syllableStr']} conf:{w['confidence']:.2f}{mm}")
        for j, s in enumerate(w["syllableAnalysis"]):
            ac = s["acoustics"]
            stress = " ´" if j == w["stressedIdx"] else ""
            f = ac.get("formants", [])
            ftxt = f"F1{f[0]:.0f}/F2{f[1]:.0f}" if len(f) >= 2 else "—"
            print(f"      [{s['syllable']}{stress}] {s['startSec']:.3f}–{s['endSec']:.3f}с "
                  f"эн:{ac.get('energyMean',0):.4f} {ftxt}")
            for rm in s["remarks"]:
                print(f"         → {rm}")
        for err in w["reductionInfo"]["reductionErrors"]:
            print(f"      ↻ {err['hint']}")

    print(f"\n{'─' * W}")
    print("  РАЗДЕЛ 1б: Контраст слога ударный/безударный (среднее)")
    print("─" * W)
    if contrastTable:
        for c in contrastTable[:15]:
            print(f"  «{c['syllable']}»  длит. удар {c['stressDur']:.3f}с / безуд "
                  f"{c['unstrDur']:.3f}с | энергия удар {c['stressEn']:.4f} / безуд {c['unstrEn']:.4f}")
    else:
        print("  (нет слогов, встретившихся и в ударной, и в безударной позиции)")

    print(f"\n{'─' * W}")
    print(f"  РАЗДЕЛ 2: Темп {rate:.2f} сл/с | речь {speechDur:.1f}с")
    print("─" * W)
    if rate < normRate[0]:
        print("  ⚠ Темп ниже нормы")
    elif rate > normRate[1]:
        print("  ⚠ Темп выше нормы")
    else:
        print("  ✓ Темп в норме")
    for p in pauses:
        print(f"  ⚠ Пауза {p['duration']:.1f}с после «{p['afterWord']}»")

    if refText:
        print(f"\n{'─' * W}")
        print(f"  РАЗДЕЛ 3: Точность слов {textAcc:.1f}% "
              f"({len(cmp['correct'])}/{len(refWords)})")
        print("─" * W)
        for e in sylErrors:
            print(f"    «{e['referenceWord']}»: {e['refStr']} → {e['recStr']}")
            for err in e["errors"]:
                print(f"      ✗ {err['hint']}")

    print(f"\n{'─' * W}")
    print("  РАЗДЕЛ 4: Частотный словарь слогов (топ-10)")
    print("─" * W)
    for syl, cnt in sortedSyl[:10]:
        print(f"  {syl:<8} {'█' * min(cnt, 25)} {cnt}")

    # --- итоговый балл ---
    allSyl = [s for w in words for s in w["syllableAnalysis"]]
    sylScore = (sum(1 for s in allSyl if not s["isProblematic"]) / len(allSyl) * 100) if allSyl else 100
    multi = [w for w in words if w["syllableCount"] > 1]
    redScore = (sum(1 for w in multi if not w["reductionInfo"]["hasReductionIssue"]) / len(multi) * 100) if multi else 100
    avgConf = (sum(w["confidence"] for w in words) / len(words)) if words else 0
    rateScore = 100 if normRate[0] <= rate <= normRate[1] else max(
        0, 100 - min(abs(rate - normRate[0]), abs(rate - normRate[1])) * 30)

    components = [
        ("Слоговая акустика", sylScore, 0.30),
        ("Редукция гласных",  redScore, 0.30),
        ("Уверенность VOSK",  avgConf * 100, 0.20),
        ("Темп речи",         rateScore, 0.10),
        ("Точность слов",     textAcc, 0.10),
    ]
    final = sum(s * w for _, s, w in components) / sum(w for _, _, w in components)

    print(f"\n{'═' * W}")
    print("  ИТОГОВАЯ ОЦЕНКА")
    print("═" * W)
    for name, score, weight in components:
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  {name:<20} [{bar}] {score:5.1f} (вес {weight:.0%})")
    print(f"\n  Итоговый балл: {final:5.1f} / 100")


def saveJson(audioFile, audioDuration, sr, refText, recText, words,
             contrastTable, sylErrors, pauses, cmp):
    data = {
        "audioFile": audioFile, "durationSec": round(audioDuration, 2),
        "sampleRate": sr, "referenceText": refText, "recognizedText": recText,
        "wordAnalysis": words, "syllableContrast": contrastTable,
        "syllableErrors": sylErrors, "pauses": pauses,
        "wordComparison": {
            "correct": cmp["correct"],
            "substituted": [{"expected": r, "got": g} for r, g in cmp["substituted"]],
            "missed": cmp["missed"], "inserted": cmp["inserted"]},
    }
    # Сохраняем в analysis/, а не рядом с WAV
    basename = os.path.basename(audioFile).replace(".wav", "_syllable_analysis.json")
    name = os.path.join(SCRIPT_DIR, "analysis", basename)
    with open(name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт сохранён: {name}")


if __name__ == "__main__":
    main()