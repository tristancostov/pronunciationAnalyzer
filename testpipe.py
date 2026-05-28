
import json
import re
import os
from pathlib import Path
import pymorphy2

GOLOS_MANIFEST_PATH = "./trainD/manifest.jsonl"

MAX_TEXTS = 50

OUTPUT_PATH = "pipeline_result.json"


def loadTextsFromGolos(manifestPath: str, maxTexts: int) -> list[str]:

    texts = []
    manifestFile = Path(manifestPath)


    print(f"Загрузка текстов из: {manifestPath}")
    print(f"从以下文件加载文本: {manifestPath}\n")

    with open(manifestFile, "r", encoding="utf-8") as f:
        for lineNum, line in enumerate(f):
            if lineNum >= maxTexts:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
              
                textValue = record.get("text", "")
                if textValue:
                    texts.append(textValue)
            except json.JSONDecodeError as e:
                print(f"  Строка {lineNum + 1}: ошибка JSON — {e}")

    print(f"Загружено текстов: {len(texts)}")
    print(f"已加载文本数量: {len(texts)}\n")
    return texts


# ============================================================
# 3. Алгоритм разбиения на слоги / 音节分割算法
# ============================================================
# Адаптация PHP-алгоритма с Хабра (статья 2008 года)
# 改写自Habr网站2008年的PHP算法
# Принцип: восходящая звучность (принцип Sievers)
# 原理：升调音响原则

# Звучность каждой буквы / 每个字母的响度值
SONORITY_MAP = {
    # Гласные / 元音 — звучность 4
    **{ch: 4 for ch in "аеёиоуыэюя"},
    # Сонорные согласные / 响亮辅音 — звучность 3
    **{ch: 3 for ch in "лмнрй"},
    # Звонкие шумные / 浊音辅音 — звучность 2
    **{ch: 2 for ch in "бвгджз"},
    # Глухие согласные / 清音辅音 — звучность 1
    **{ch: 1 for ch in "кпстфхцчшщ"},
    # Ь и Ъ — не влияют на слогораздел / 软硬音符不影响音节分割
    "ь": 0,
    "ъ": 0,
}


VOWELS = set("аеёиоуыэюя")


def _hasVowel(fragment: str) -> bool:
    """Проверяет, есть ли в фрагменте хотя бы одна гласная. / 检查片段中是否有元音。"""
    return any(ch in VOWELS for ch in fragment.lower())


def splitIntoSyllables(word: str) -> list[str]:
    """
    Разбивает слово на слоги по принципу восходящей звучности.
    按升调音响原则将单词分割为音节。

    Правила / 规则:
    1. Граница слога — там, где звучность падает после гласной
       音节边界在元音后响度下降处
    2. Ь и Ъ не отделяются от предыдущей буквы
       软硬音符不与前面字母分离
    3. КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: каждый слог обязан содержать гласную.
       关键修复：每个音节必须含有元音。
       Фрагменты без гласных (например, конечные -р, -н, -в, -рс)
       присоединяются к предыдущему слогу.
       没有元音的片段（如词尾 -р, -н, -в, -рс）合并到前一个音节。

    Примеры до и после исправления / 修复前后示例:
      эфир:     э-фи-р  →  э-фир     ✓
      сезон:    се-зо-н →  се-зон    ✓
      орлов:    о-рло-в →  ор-лов    ✓
      студент:  сту-дент            ✓ (не изменился)
      молоко:   мо-ло-ко            ✓ (не изменился)
    """
    wordLower = word.lower()

    # Строим список звучностей / 构建响度列表
    sonoritySeq = [SONORITY_MAP.get(ch, -1) for ch in wordLower]

    # Находим точки разбиения / 找分割点
    splitPoints = []

    for i in range(len(wordLower) - 1):
        currentSon = sonoritySeq[i]
        nextSon    = sonoritySeq[i + 1]

        if currentSon <= 0 or nextSon <= 0:
            continue

        if currentSon == 4 and nextSon < 4:
            nextChar = wordLower[i + 1] if i + 1 < len(wordLower) else ""
            if nextChar not in ("ь", "ъ"):
                splitPoints.append(i + 1)

    # Нарезаем слово / 按点切割
    rawSyllables = []
    prevIdx = 0
    for point in splitPoints:
        rawSyllables.append(word[prevIdx:point])
        prevIdx = point
    rawSyllables.append(word[prevIdx:])
    rawSyllables = [s for s in rawSyllables if s]

    # ─────────────────────────────────────────────────────────
    # ИСПРАВЛЕНИЕ: убираем безгласные фрагменты
    # 修复：消除无元音片段
    # Каждый фрагмент без гласной сливается с предыдущим слогом.
    # 每个无元音的片段合并到前一个音节。
    # ─────────────────────────────────────────────────────────
    mergedSyllables: list[str] = []
    for fragment in rawSyllables:
        if mergedSyllables and not _hasVowel(fragment):
            # Нет гласной → прилепляем к предыдущему / 无元音 → 合并到前一个
            mergedSyllables[-1] += fragment
        else:
            mergedSyllables.append(fragment)

    return mergedSyllables if mergedSyllables else [word]


# ============================================================
# 4. Морфологический анализ / 词形分析
# ============================================================

# Инициализируем анализатор один раз (загрузка словаря)
# 只初始化一次分析器（加载词典耗时，只做一次）
print("Инициализация pymorphy2... / 初始化pymorphy2...")
morphAnalyzer = pymorphy2.MorphAnalyzer()
print("pymorphy2 готов. / pymorphy2就绪。\n")

# Словарь: как перевести теги pymorphy2 на человеческий язык
# 词性标签翻译词典（pymorphy2使用OpenCorpora标准）
POS_TRANSLATION = {
    "NOUN": "существительное 名词",
    "ADJF": "прилагательное (полн.) 形容词",
    "ADJS": "прилагательное (кратк.) 短尾形容词",
    "COMP": "сравнительная степень 比较级",
    "VERB": "глагол 动词",
    "INFN": "инфинитив 不定式",
    "PRTF": "причастие (полн.) 分词",
    "PRTS": "причастие (кратк.) 短尾分词",
    "GRND": "деепричастие 副动词",
    "NUMR": "числительное 数词",
    "ADVB": "наречие 副词",
    "NPRO": "местоимение 代词",
    "PRED": "предикатив 谓词",
    "PREP": "предлог 前置词",
    "CONJ": "союз 连词",
    "PRCL": "частица 助词",
    "INTJ": "междометие 感叹词",
    "UNKN": "неизвестно 未知",
}


def analyzeWord(word: str) -> dict:
    """
    Анализирует одно слово с помощью pymorphy2.
    使用pymorphy2分析一个词。

    Возвращает / 返回:
      {
        "original":  исходное слово / 原词,
        "lemma":     нормальная форма / 词根（原形）,
        "pos":       часть речи / 词性,
        "pos_ru":    название части речи / 词性名称,
        "score":     уверенность анализатора / 分析器置信度,
        "syllables": список слогов / 音节列表,
        "syllable_count": количество слогов / 音节数量,
        "syllable_str": слоги через дефис / 带连字符的音节
      }
    """
    # Убираем знаки препинания для анализа
    # 去掉标点符号再分析
    cleanWord = re.sub(r"[^а-яёА-ЯЁa-zA-Z]", "", word)

    if not cleanWord:
        return {
            "original":      word,
            "lemma":         word,
            "pos":           "UNKN",
            "pos_ru":        POS_TRANSLATION["UNKN"],
            "score":         0.0,
            "syllables":     [word],
            "syllable_count": 0,
            "syllable_str":  word,
        }

    # pymorphy2 возвращает список возможных разборов, отсортированных по вероятности
    # pymorphy2返回按概率排序的可能分析列表
    # Берём первый (самый вероятный)
    # 取第一个（最可能的）
    parsedForms = morphAnalyzer.parse(cleanWord)
    bestParse   = parsedForms[0]

    lemma    = bestParse.normal_form          # Нормальная форма / 词根原形
    posTag   = bestParse.tag.POS or "UNKN"   # Часть речи / 词性
    score    = round(float(bestParse.score), 4)  # Уверенность / 置信度

    # Разбиваем лемму (нормальную форму) на слоги
    # 对词根（原形）进行音节分割
    syllables     = splitIntoSyllables(lemma)
    syllableStr   = "-".join(syllables)
    syllableCount = len(syllables)

    return {
        "original":       word,
        "lemma":          lemma,
        "pos":            posTag,
        "pos_ru":         POS_TRANSLATION.get(posTag, posTag),
        "score":          score,
        "syllables":      syllables,
        "syllable_count": syllableCount,
        "syllable_str":   syllableStr,
    }


# ============================================================
# 5. Основной конвейер / 主处理管道
# ============================================================

def processText(text: str) -> dict:
    """
    Обрабатывает одну строку текста через весь конвейер.
    将一条文本通过整个处理管道。

    Возвращает / 返回:
      {
        "original_text": исходный текст / 原始文本,
        "word_count":    количество слов / 词数,
        "words": [
          { ...analyzeWord результат... },
          ...
        ]
      }
    """
    # Разбиваем текст на слова (по пробелам и знакам препинания)
    # 按空格和标点分词
    rawWords  = re.findall(r"[а-яёА-ЯЁ]+", text)
    wordInfos = []

    for word in rawWords:
        wordInfo = analyzeWord(word)
        wordInfos.append(wordInfo)

    return {
        "original_text": text,
        "word_count":    len(rawWords),
        "words":         wordInfos,
    }


def runPipeline(manifestPath: str, maxTexts: int, outputPath: str):
    """
    Запускает весь конвейер от загрузки до сохранения.
    运行从加载到保存的完整管道。
    """
    print("=" * 60)
    print("  ЗАПУСК КОНВЕЙЕРА / 启动处理管道")
    print("=" * 60)

    # Шаг 1: Загрузка текстов / 步骤1：加载文本
    texts = loadTextsFromGolos(manifestPath, maxTexts)

    # Шаг 2: Обработка каждого текста / 步骤2：处理每条文本
    print(f"Обработка {len(texts)} текстов...")
    print(f"处理 {len(texts)} 条文本...\n")

    allResults = []

    for idx, text in enumerate(texts):
        result = processText(text)
        allResults.append(result)

        # Прогресс и красивый вывод / 进度和美观输出
        print(f"[{idx + 1:>3}/{len(texts)}] {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"         Слов: {result['word_count']}")

        # Выводим первые 3 слова для проверки / 显示前3个词作为检查
        for wordInfo in result["words"][:3]:
            print(
                f"         • {wordInfo['original']:<12} → "
                f"лемма: {wordInfo['lemma']:<12} | "
                f"ч.р.: {wordInfo['pos']:<6} | "
                f"слоги: {wordInfo['syllable_str']}"
            )
        if result["word_count"] > 3:
            print(f"         ... и ещё {result['word_count'] - 3} слов")
        print()

    # Шаг 3: Сбор статистики / 步骤3：收集统计
    totalWords     = sum(r["word_count"] for r in allResults)
    allWordInfos   = [w for r in allResults for w in r["words"]]

    # Частотность частей речи / 词性频率统计
    posCounter: dict[str, int] = {}
    for w in allWordInfos:
        pos = w["pos"]
        posCounter[pos] = posCounter.get(pos, 0) + 1

    # Средняя длина слога / 平均音节数
    avgSyllables = (
        sum(w["syllable_count"] for w in allWordInfos) / len(allWordInfos)
        if allWordInfos else 0
    )

    statistics = {
        "total_texts":          len(allResults),
        "total_words":          totalWords,
        "avg_syllables_per_word": round(avgSyllables, 2),
        "pos_distribution":     posCounter,
    }
    # 在 runPipeline 函数的统计部分加入以下代码
    # 放在 "Шаг 3: Сбор статистики" 之后

    # Подсчёт частоты слогов / 统计音节频率
    syllableCounter: dict[str, int] = {}
    for wordInfo in allWordInfos:
        for syllable in wordInfo["syllables"]:
            syl = syllable.lower()
            syllableCounter[syl] = syllableCounter.get(syl, 0) + 1

    # Сортируем по убыванию частоты / 按频率从高到低排序
    sortedSyllables = sorted(syllableCounter.items(), key=lambda x: -x[1])

    # Выводим топ-20 / 打印前20个
    print(f"\n  Топ-20 самых частых слогов / 最常见的20个音节:")
    for syl, count in sortedSyllables[:20]:
        bar = "█" * min(count, 30)
        print(f"    {syl:<6} {bar} {count}")

    # Сохраняем в отдельный JSON / 保存到单独的JSON文件
    syllableReport = {
        "total_unique_syllables": len(syllableCounter),
        "total_syllable_occurrences": sum(syllableCounter.values()),
        "syllables_by_frequency": [
            {"syllable": syl, "count": count}
            for syl, count in sortedSyllables
        ]
    }

    with open("syllable_frequency.json", "w", encoding="utf-8") as f:
        json.dump(syllableReport, f, ensure_ascii=False, indent=2)

    print(f"\n  Частотный словарь слогов сохранён: syllable_frequency.json")
    print(f"  音节频率词典已保存: syllable_frequency.json")
    # Шаг 4: Сохранение в JSON / 步骤4：保存为JSON
    finalOutput = {
        "pipeline_version": "1.0",
        "source_file":      manifestPath,
        "statistics":       statistics,
        "results":          allResults,
    }

    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(finalOutput, f, ensure_ascii=False, indent=2)

    # Итоговая сводка / 最终摘要
    print("=" * 60)
    print("  ИТОГИ / 统计摘要")
    print("=" * 60)
    print(f"  Текстов обработано:      {len(allResults)}")
    print(f"  已处理文本数:              {len(allResults)}")
    print(f"  Слов всего:              {totalWords}")
    print(f"  总词数:                   {totalWords}")
    print(f"  Средн. слогов на слово:  {avgSyllables:.2f}")
    print(f"  平均每词音节数:             {avgSyllables:.2f}")
    print(f"\n  Распределение частей речи / 词性分布:")
    for pos, count in sorted(posCounter.items(), key=lambda x: -x[1]):
        posName = POS_TRANSLATION.get(pos, pos)
        bar     = "█" * min(count, 30)
        print(f"    {pos:<6} {bar:<30} {count:>5}  ({posName.split()[0]})")

    print(f"\n  Результат сохранён: {outputPath}")
    print(f"  结果已保存至: {outputPath}")
    print("=" * 60)


# ============================================================
# 6. Точка входа / 程序入口
# ============================================================
if __name__ == "__main__":
    runPipeline(
        manifestPath=GOLOS_MANIFEST_PATH,
        maxTexts=MAX_TEXTS,
        outputPath=OUTPUT_PATH,
    )

