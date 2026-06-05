import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Подготовка данных (твои результаты из CSV)
# Представим данные ДО MFCC и ПОСЛЕ MFCC
data = {
    'Запись': ['6fori (Иностранец)', '7fori (Иностранец)', '3local (Носитель)'],
    'Медианная ошибка (ДО MFCC), мс': [38.0, 35.0, 36.0], # Примерные старые данные
    'Медианная ошибка (ПОСЛЕ MFCC), мс': [29.4, 26.9, 27.6],
    'Точность слогов (ПОСЛЕ MFCC), %': [79.1, 81.0, 95.0]
}
df = pd.DataFrame(data)

# Настройка стиля графиков
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- График 1: Сравнение медианной ошибки границ (До и После MFCC) ---
x = np.arange(len(df['Запись']))
width = 0.35

rects1 = axes[0].bar(x - width/2, df['Медианная ошибка (ДО MFCC), мс'], width, label='До MFCC (Только Энергия)', color='#d3d3d3')
rects2 = axes[0].bar(x + width/2, df['Медианная ошибка (ПОСЛЕ MFCC), мс'], width, label='После MFCC (Энергия + Спектр)', color='#4CAF50')

axes[0].set_ylabel('Ошибка границ слогов (мс)', fontsize=12)
axes[0].set_title('Уменьшение ошибки границ слогов (Медиана)', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df['Запись'], fontsize=11)
axes[0].legend()

# Добавление значений над столбцами
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
autolabel(rects1, axes[0])
autolabel(rects2, axes[0])

# --- График 2: Точность определения числа слогов (Носитель vs Иностранец) ---
colors = ['#ff9999', '#ff9999', '#66b3ff'] # Выделяем носителя
bars = axes[1].bar(df['Запись'], df['Точность слогов (ПОСЛЕ MFCC), %'], color=colors)

axes[1].set_ylabel('Точность совпадения слогов (%)', fontsize=12)
axes[1].set_title('Точность структуры слогов (Носитель vs Иностранцы)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 100)

# Добавление значений
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()