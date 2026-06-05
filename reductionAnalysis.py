import json
import glob
import os

VOWELS = set("аоуыэяёюие")

def get_main_vowel(syl):
    """Найти главную гласную в слоге"""
    for char in syl.lower():
        if char in VOWELS:
            return char
    return None

def analyze_folder(json_folder="audio"):
    # Ищем все файлы JSON в текущей папке
    files = glob.glob(os.path.join(json_folder, "*_syllable_analysis.json"))
    if not files:
        print("❌ Не найдено ни одного JSON файла.")
        return

    print("📊 Глобальный анализ редукции гласных (Без Praat)\n" + "="*60)

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vowel_stats = {} 

        for word in data.get('wordAnalysis', []):
            syls = word.get('syllableAnalysis', [])
            if len(syls) < 2: 
                continue # Пропускаем односложные слова

            # Используем словарное (нормативное) ударение для проверки
            expected_stress = word.get('expectedStressedIdx', -1)
            if expected_stress < 0:
                continue

            for i, syl in enumerate(syls):
                v = get_main_vowel(syl.get('syllable', ''))
                if not v: continue

                if v not in vowel_stats:
                    vowel_stats[v] = {'str_durs':[], 'unstr_durs':[]}

                dur = syl.get('acoustics', {}).get('duration', 0)
                if dur <= 0: continue

                if i == expected_stress:
                    vowel_stats[v]['str_durs'].append(dur)
                else:
                    vowel_stats[v]['unstr_durs'].append(dur)

        filename = os.path.basename(fpath)
        print(f"📁 Файл: {filename}")
        found_data = False
        for v, stats in sorted(vowel_stats.items()):
            s_durs = stats['str_durs']
            u_durs = stats['unstr_durs']
            
            # Условие: минимум 3 примера для статистики
            if len(s_durs) >= 3 and len(u_durs) >= 3:
                s_mean = sum(s_durs) / len(s_durs) * 1000
                u_mean = sum(u_durs) / len(u_durs) * 1000
                ratio = u_mean / s_mean
                
                # Если отношение >= 0.85, редукция слабая или отсутствует
                status = "⚠ Нет редукции" if ratio >= 0.85 else "✓ Норма"
                
                print(f"  Гласная '{v}':")
                print(f"     Ударные (n={len(s_durs)}): {s_mean:.0f} мс")
                print(f"     Безударные (n={len(u_durs)}): {u_mean:.0f} мс")
                print(f"     Отношение (Безуд/Удар): {ratio:.2f} -> {status}")
                found_data = True
                
        if not found_data:
            print("  (Недостаточно данных для статистики)")
        print("-" * 60)

if __name__ == "__main__":
    analyze_folder()