import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Завантаження даних з файлу
file_path_csv = 'top_10000_1950-now (1).xlsx - top_10000_1950-now.csv'
data = pd.read_csv(file_path_csv, encoding='utf-8')

# Перегляд інформації про дані
data.info()
print(data.head())

# Для прикладу, вибираємо два числових стовпця
selected_columns = ['Loudness', 'Instrumentalness']
data = data[selected_columns].dropna()

# 1. Нормалізація даних
# Min-Max Normalization
dataLoudness = data.copy()
dataLoudness = (dataLoudness - dataLoudness.min()) / (dataLoudness.max() - dataLoudness.min())

# Instrumentalness Normalization
dataInstrumentalness = data.copy()
dataInstrumentalness = (dataInstrumentalness - dataInstrumentalness.mean()) / dataInstrumentalness.std()

# 2. Аналіз та візуалізація для нормалізованих даних
# Створення графіків для Loudness даних
plt.figure(figsize=(15, 10))

# Коробчаста діаграма
plt.subplot(2, 3, 1)
sns.boxplot(data=dataLoudness)
plt.title('Коробчаста діаграма (Loudness)')

# Скрипковий графік
plt.subplot(2, 3, 2)
sns.violinplot(data=dataLoudness)
plt.title('Скрипковий графік (Loudness)')

# Гістограма
plt.subplot(2, 3, 3)
plt.hist(dataLoudness.iloc[:, 0], bins=20, alpha=0.7, label=dataLoudness.columns[0])
plt.hist(dataLoudness.iloc[:, 1], bins=20, alpha=0.7, label=dataLoudness.columns[1])
plt.legend()
plt.title('Гістограма (Loudness)')

# Графік густини
plt.subplot(2, 3, 4)
sns.kdeplot(dataLoudness.iloc[:, 0], fill=True, label=dataLoudness.columns[0])
sns.kdeplot(dataLoudness.iloc[:, 1], fill=True, label=dataLoudness.columns[1])
plt.legend()
plt.title('Графік густини (Loudness)')

# Контурний графік
plt.subplot(2, 3, 5)
sns.kdeplot(x=dataLoudness.iloc[:, 0], y=dataLoudness.iloc[:, 1], cmap='Blues', fill=True)
plt.title('Контурний графік (Loudness)')

plt.tight_layout()
plt.show()

# Створення графіків для Instrumentalness Normalized даних
plt.figure(figsize=(15, 10))

# Коробчаста діаграма
plt.subplot(2, 3, 1)
sns.boxplot(data=dataInstrumentalness)
plt.title('Коробчаста діаграма (Instrumentalness Normalized)')

# Скрипковий графік
plt.subplot(2, 3, 2)
sns.violinplot(data=dataInstrumentalness)
plt.title('Скрипковий графік (Instrumentalness Normalized)')

# Гістограма
plt.subplot(2, 3, 3)
plt.hist(dataInstrumentalness.iloc[:, 0], bins=20, alpha=0.7, label=dataInstrumentalness.columns[0])
plt.hist(dataInstrumentalness.iloc[:, 1], bins=20, alpha=0.7, label=dataInstrumentalness.columns[1])
plt.legend()
plt.title('Гістограма (Instrumentalness Normalized)')

# Графік густини
plt.subplot(2, 3, 4)
sns.kdeplot(dataInstrumentalness.iloc[:, 0], fill=True, label=dataInstrumentalness.columns[0])
sns.kdeplot(dataInstrumentalness.iloc[:, 1], fill=True, label=dataInstrumentalness.columns[1])
plt.legend()
plt.title('Графік густини (Instrumentalness Normalized)')

# Контурний графік
plt.subplot(2, 3, 5)
sns.kdeplot(x=dataInstrumentalness.iloc[:, 0], y=dataInstrumentalness.iloc[:, 1], cmap='Blues', fill=True)
plt.title('Контурний графік (Instrumentalness Normalized)')

plt.tight_layout()
plt.show()

# 3. Кореляційний аналіз
print("Loudness та Instrumentalness дані:")
correlation_matrixLoudness = dataLoudness.corr()
print(correlation_matrixLoudness)
sns.heatmap(correlation_matrixLoudness, annot=True, cmap='coolwarm')
plt.title('Кореляційна матриця (Loudness та Instrumentalness)')
plt.show()


# 4. Мода та стандартна похибка
modeLoudness_1 = dataLoudness.iloc[:, 0].mode()[0]
modeLoudness_2 = dataLoudness.iloc[:, 1].mode()[0]
std_errorLoudness_1 = stats.sem(dataLoudness.iloc[:, 0])
std_errorLoudness_2 = stats.sem(dataLoudness.iloc[:, 1])
print(
    f"Loudness: Мода: {modeLoudness_1}, {modeLoudness_2}, Середня похибка: {std_errorLoudness_1}, {std_errorLoudness_2}")

modeInstrumentalness_1 = dataInstrumentalness.iloc[:, 0].mode()[0]
modeInstrumentalness_2 = dataInstrumentalness.iloc[:, 1].mode()[0]
std_errorInstrumentalness_1 = stats.sem(dataInstrumentalness.iloc[:, 0])
std_errorInstrumentalness_2 = stats.sem(dataInstrumentalness.iloc[:, 1])
print(
    f"Instrumentalness Normalized: Мода: {modeInstrumentalness_1}, {modeInstrumentalness_2}, Середня похибка: {std_errorInstrumentalness_1}, {std_errorInstrumentalness_2}")


# 5. Оцінка розподілу даних на основі вигляду основних розділів з теорії ймовірності
def estimate_distribution(data):
    # Перевірка на нормальний розподіл (Гаусовський розподіл)
    k2, p_value = stats.normaltest(data)
    if p_value > 0.05:
        return "Гаусовський розподіл"

    # Перевірка на рівномірний розподіл
    uniform_statistic, uniform_p_value = stats.kstest(data, 'uniform')
    if uniform_p_value > 0.05:
        return "Рівномірний розподіл"

    # Перевірка на показниковий розподіл
    expon_statistic, expon_p_value = stats.kstest(data, 'expon')
    if expon_p_value > 0.05:
        return "Показниковий розподіл"

    return "Інший розподіл"


distributionLoudness = estimate_distribution(dataLoudness.iloc[:, 0])
distributionInstrumentalness = estimate_distribution(dataInstrumentalness.iloc[:, 0])
print(f"Розподіл Loudness: {distributionLoudness}")
print(f"Розподіл Instrumentalness: {distributionInstrumentalness}")

# 6. Бутстрап-вибірка для нормалізованих даних (наприклад, для Loudness)
bootstrap_samples = 1000
bootstrap_means = [
    np.mean(np.random.choice(dataLoudness.iloc[:, 0], size=len(dataLoudness), replace=True))
    for _ in range(bootstrap_samples)
]
ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"Довірчий інтервал для середнього значення ({dataLoudness.columns[0]}): [{ci_low}, {ci_high}]")

# Візуалізація бутстрап-вибірки
sns.histplot(bootstrap_means, kde=True)
plt.title(f'Бутстрап-розподіл середнього значення ({dataLoudness.columns[0]} )')
plt.axvline(ci_low, color='red', linestyle='--', label=f'2.5%: {ci_low:.2f}')
plt.axvline(ci_high, color='green', linestyle='--', label=f'97.5%: {ci_high:.2f}')
plt.legend()
plt.show()