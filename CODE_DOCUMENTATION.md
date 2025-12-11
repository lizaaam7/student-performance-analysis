#  Полное Руководство по Коду | Code Documentation

##  Содержание
1. [Используемые библиотеки](#1-используемые-библиотеки)
2. [Загрузка и очистка данных](#2-загрузка-и-очистка-данных)
3. [Feature Engineering](#3-feature-engineering)
4. [Описательная статистика](#4-описательная-статистика)
5. [Визуализации](#5-визуализации)
6. [Статистические тесты](#6-статистические-тесты)

---

## 1. Используемые Библиотеки

```python
import pandas as pd          # Библиотека для работы с табличными данными
import numpy as np           # Библиотека для численных вычислений
import seaborn as sns        # Библиотека для статистической визуализации
import matplotlib.pyplot as plt  # Библиотека для построения графиков
from scipy import stats      # Модуль для статистических тестов
import warnings              # Модуль для управления предупреждениями
```

### Описание библиотек:

| Библиотека | Назначение | Основные функции |
|------------|-----------|------------------|
| **pandas** | Обработка данных | `read_csv()`, `groupby()`, `describe()`, `cut()` |
| **numpy** | Численные операции | `mean()`, `linspace()`, массивы |
| **seaborn** | Визуализация | `histplot()`, `boxplot()`, `heatmap()`, `violinplot()` |
| **matplotlib** | Графики | `figure()`, `bar()`, `pie()`, `show()` |
| **scipy.stats** | Статистика | `ttest_ind()`, `f_oneway()` |

---

## 2. Загрузка и Очистка Данных

### 2.1 Загрузка CSV файла
```python
df = pd.read_csv('StudentsPerformance.csv')
```
**Что делает:** Читает CSV файл и создаёт DataFrame — таблицу данных pandas.

**Параметры:**
- `'StudentsPerformance.csv'` — путь к файлу (относительный)

---

### 2.2 Переименование столбцов
```python
df.columns = [
    'gender', 'race_ethnicity', 'parental_level_of_education', 
    'lunch', 'test_preparation_course', 'math_score', 
    'reading_score', 'writing_score'
]
```
**Зачем:** Приводим названия к стандартному формату (snake_case) для удобства работы.

---

### 2.3 Проверка пропущенных значений
```python
df.isnull().sum()
```
**Что делает:** 
- `isnull()` — возвращает True/False для каждой ячейки (True = пропуск)
- `sum()` — суммирует True значения по каждому столбцу

**Результат:** Если все нули — пропусков нет.

---

### 2.4 Удаление дубликатов
```python
df.drop_duplicates(inplace=True)
```
**Параметры:**
- `inplace=True` — изменяет оригинальный DataFrame (без создания копии)

---

## 3. Feature Engineering

### 3.1 Средний балл
```python
df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
```
**Формула:** `average = (math + reading + writing) / 3`

**Зачем:** Единая метрика для сравнения общей успеваемости студентов.

---

### 3.2 Бинарный признак сдачи
```python
PASS_THRESHOLD = 40  # Порог сдачи экзамена

df['pass_all'] = (
    (df['math_score'] >= PASS_THRESHOLD) & 
    (df['reading_score'] >= PASS_THRESHOLD) & 
    (df['writing_score'] >= PASS_THRESHOLD)
).astype(int)
```
**Логика:**
- `&` — логическое И (все условия должны быть True)
- `.astype(int)` — преобразует True/False в 1/0

**Результат:** 1 = сдал все экзамены, 0 = не сдал хотя бы один.

---

### 3.3 Категории успеваемости
```python
bins = [0, 40, 60, 80, 100]
labels = ['Fail', 'Satisfactory', 'Good', 'Excellent']
df['performance_level'] = pd.cut(df['average_score'], bins=bins, labels=labels, right=False)
```
**Функция `pd.cut()`:**
- `bins` — границы интервалов
- `labels` — названия категорий
- `right=False` — интервалы [левая, правая) (не включая правую границу)

**Интервалы:**
| Категория | Диапазон баллов |
|-----------|-----------------|
| Fail | 0-39 |
| Satisfactory | 40-59 |
| Good | 60-79 |
| Excellent | 80-100 |

---

## 4. Описательная Статистика

### 4.1 Метод describe()
```python
df[['math_score', 'reading_score', 'writing_score']].describe()
```
**Возвращает:**
| Метрика | Описание |
|---------|----------|
| count | Количество значений |
| mean | Среднее арифметическое |
| std | Стандартное отклонение |
| min | Минимум |
| 25% | 1-й квартиль (25% данных меньше) |
| 50% | Медиана |
| 75% | 3-й квартиль |
| max | Максимум |

---

### 4.2 Метод value_counts()
```python
df['gender'].value_counts()
```
**Что делает:** Подсчитывает количество каждого уникального значения в столбце.

---

### 4.3 Метод groupby()
```python
df.groupby('gender')['average_score'].mean()
```
**Логика:**
1. `groupby('gender')` — группирует данные по полу
2. `['average_score']` — выбирает столбец
3. `.mean()` — вычисляет среднее для каждой группы

---

## 5. Визуализации

### 5.1 Гистограмма (histplot)
```python
sns.histplot(df['average_score'], bins=25, kde=True, color='steelblue')
```
**Параметры:**
- `bins=25` — количество столбцов
- `kde=True` — добавляет кривую плотности
- `color` — цвет

**Назначение:** Показывает распределение данных.

---

### 5.2 Box Plot (boxplot)
```python
sns.boxplot(x='test_preparation_course', y='average_score', data=df, hue='test_preparation_course')
```
**Что показывает:**
- Медиана (линия внутри)
- Квартили (границы "коробки")
- Выбросы (точки за "усами")

---

### 5.3 Bar Chart
```python
df.groupby('gender')[['math_score']].mean().plot(kind='bar')
```
**Назначение:** Сравнение средних значений между категориями.

---

### 5.4 Pie Chart
```python
plt.pie(values, labels=labels, autopct='%1.1f%%')
```
**Параметры:**
- `autopct='%1.1f%%'` — формат процентов (1 знак после запятой)

---

### 5.5 Heatmap (корреляция)
```python
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='RdYlBu_r')
```
**Параметры:**
- `.corr()` — вычисляет корреляцию между столбцами
- `annot=True` — показывает числа на ячейках
- `cmap` — цветовая палитра

**Интерпретация корреляции:**
| Значение | Сила связи |
|----------|------------|
| 0.9-1.0 | Очень сильная |
| 0.7-0.9 | Сильная |
| 0.5-0.7 | Умеренная |
| 0.3-0.5 | Слабая |
| 0-0.3 | Очень слабая |

---

### 5.6 Violin Plot
```python
sns.violinplot(x='Subject', y='Score', hue='gender', data=df_melted, split=True)
```
**Что показывает:** Комбинация box plot и распределения плотности.

**Параметр `split=True`:** Разделяет "скрипку" пополам для сравнения двух групп.

---

## 6. Статистические Тесты

### 6.1 T-Test (сравнение двух групп)
```python
from scipy import stats

group1 = df[df['test_preparation_course'] == 'completed']['average_score']
group2 = df[df['test_preparation_course'] == 'none']['average_score']

t_stat, p_value = stats.ttest_ind(group1, group2)
```
**Функция `ttest_ind()`:** Independent samples t-test

**Интерпретация:**
- `t_stat` — t-статистика (чем больше по модулю — тем сильнее различие)
- `p_value` — вероятность получить такие данные случайно

**Правило:**
- p < 0.05 → **Статистически значимо** (различие реальное)
- p ≥ 0.05 → Не значимо (различие может быть случайным)

---

### 6.2 ANOVA (сравнение 3+ групп)
```python
groups = [group['average_score'].values for name, group in df.groupby('parental_level_of_education')]
f_stat, p_value = stats.f_oneway(*groups)
```
**Функция `f_oneway()`:** One-way ANOVA

**Когда использовать:** Когда нужно сравнить средние более чем 2 групп.

**Параметр `*groups`:** Распаковывает список в отдельные аргументы.

---

##  Важность Переменных

### Какие переменные влияют на успеваемость?

| Ранг | Переменная | Влияние | Объяснение |
|------|-----------|---------|------------|
| 1 | **lunch** (тип питания) | +10.1 балла | Прокси социально-экономического статуса |
| 2 | **parental_education** | +8.2 балла | Образованные родители = больше поддержки |
| 3 | **test_preparation** | +7.5 балла | Прямая подготовка к экзаменам |
| 4 | **gender** | ±5-7 баллов | Влияет на конкретные предметы |
| 5 | **race_ethnicity** | ±3-5 баллов | Связано с другими факторами |

### Почему `lunch` так важен?
- **standard lunch** = семья может платить за питание = более высокий доход
- Более высокий доход → лучший доступ к образовательным ресурсам
- Это НЕ само питание влияет, а социально-экономический статус, который оно отражает

---

##  Полезные Python-конструкции

### List Comprehension
```python
[group['score'].values for name, group in df.groupby('category')]
```
Создаёт список из значений каждой группы.

### Lambda функции
```python
df['column'].apply(lambda x: x * 2)
```
Применяет функцию к каждому элементу.

### Метод melt() для преобразования данных
```python
df_melted = df.melt(id_vars='gender', value_vars=['math_score', 'reading_score'])
```
Преобразует "широкий" формат в "длинный" для визуализации.

---

##  Шпаргалка по функциям

| Функция | Библиотека | Назначение |
|---------|------------|------------|
| `pd.read_csv()` | pandas | Загрузка CSV |
| `df.describe()` | pandas | Статистика |
| `df.groupby()` | pandas | Группировка |
| `pd.cut()` | pandas | Категоризация |
| `sns.histplot()` | seaborn | Гистограмма |
| `sns.boxplot()` | seaborn | Box-plot |
| `sns.heatmap()` | seaborn | Тепловая карта |
| `plt.pie()` | matplotlib | Круговая диаграмма |
| `stats.ttest_ind()` | scipy | T-тест |
| `stats.f_oneway()` | scipy | ANOVA |


