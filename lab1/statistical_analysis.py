"""Лабораторная №1, пункты 1-8: статистика, сопряжённость и корреляция.

Запуск: python statistical_analysis.py
Результаты сохраняются рядом со скриптом в results/statistics.
"""

import argparse
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COLUMNS = (
    "symboling normalized-losses make fuel-type aspiration num-of-doors "
    "body-style drive-wheels engine-location wheel-base length width height "
    "curb-weight engine-type num-of-cylinders engine-size fuel-system bore "
    "stroke compression-ratio horsepower peak-rpm city-mpg highway-mpg price"
).split()
SELECTED = ["symboling", "drive-wheels", "curb-weight"]
NUMERIC = ["curb-weight"]
LEVELS = {
    "symboling": list(range(-3, 4)),
    "drive-wheels": ["4wd", "fwd", "rwd"],
}
ALPHA = 0.05


def markdown_table(frame):
    """Таблица Markdown без дополнительной зависимости tabulate."""

    def fmt(value):
        return (
            f"{value:.6g}"
            if isinstance(value, (float, np.floating))
            else str(value)
        )

    rows = [[str(frame.index.name or "Показатель"), *map(str, frame.columns)]]
    rows += [[str(idx), *map(fmt, row)] for idx, row in frame.iterrows()]
    return "\n".join(
        [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * len(rows[0])) + " |",
            *["| " + " | ".join(row) + " |" for row in rows[1:]],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=BASE / "data/imports-85.data"
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/statistics"
    )
    args = parser.parse_args()
    raw = pd.read_csv(args.data, header=None, na_values="?")
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"Ожидалось 26 столбцов, получено {raw.shape[1]}")
    raw.columns = COLUMNS
    data = raw[SELECTED].copy()
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="raise")
    for column, levels in LEVELS.items():
        if not data[column].dropna().isin(levels).all():
            raise ValueError(f"Недопустимые значения {column}")
    if not data["curb-weight"].dropna().between(1488, 4066).all():
        raise ValueError("curb-weight вне указанного диапазона")
    if len(data) < 3 or data[SELECTED].isna().any().any():
        raise ValueError(
            "Недостаточно наблюдений или есть пропуски; нужна отдельная обработка"
        )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    k_sturges = int(np.ceil(1 + np.log2(len(data))))

    def save_table(frame, name):
        frame.to_csv(out / f"{name}.csv", encoding="utf-8-sig")

    quality = pd.DataFrame(
        {
            "Тип": ["порядковый", "номинальный", "количественный"],
            "Наблюдений": data.count(),
            "Пропусков": data.isna().sum(),
            "Уникальных значений": data.nunique(),
        },
        index=SELECTED,
    )
    save_table(quality, "data_quality")
    summary = pd.DataFrame(index=NUMERIC)
    for column in NUMERIC:
        x = data[column]
        values = {
            "n": len(x),
            "Среднее": x.mean(),
            "Медиана": x.median(),
            "Моды": ", ".join(map(str, x.mode().tolist())),
            "Минимум": x.min(),
            "Максимум": x.max(),
            "Размах": x.max() - x.min(),
            "Дисперсия (ddof=1)": x.var(ddof=1),
            "Станд. отклонение (ddof=1)": x.std(ddof=1),
        }
        for key, value in values.items():
            summary.loc[column, key] = value
    save_table(summary, "descriptive_statistics")

    frequencies = {}
    for column, levels in LEVELS.items():
        counts = data[column].value_counts().reindex(levels, fill_value=0)
        frequencies[column] = pd.DataFrame(
            {"Частота": counts, "Доля, %": counts / len(data) * 100}
        )
        save_table(frequencies[column], f"frequencies_{column}")
    symbol_counts = frequencies["symboling"]["Частота"]
    most_common_count = int(symbol_counts.max())
    most_common_levels = ", ".join(
        map(str, symbol_counts[symbol_counts == most_common_count].index)
    )
    most_common_percent = most_common_count / len(data) * 100

    # Нормальность имеет содержательный смысл для количественной массы.
    # Для порядкового symboling непрерывная нормальная модель не подходит.
    weight = data["curb-weight"]
    # Нормальность оцениваем описательно, без специальных критериев.
    contingency = pd.crosstab(data["symboling"], data["drive-wheels"]).reindex(
        index=LEVELS["symboling"], columns=LEVELS["drive-wheels"], fill_value=0
    )
    column_percent = contingency.div(contingency.sum(axis=0), axis=1) * 100
    save_table(contingency, "contingency_counts")
    save_table(column_percent, "contingency_column_percent")
    # Пустой уровень -3 показываем в таблице, но исключаем из расчёта V.
    observed = contingency.loc[contingency.sum(axis=1) > 0].to_numpy()
    expected = (
        np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    )
    chi_square = np.sum((observed - expected) ** 2 / expected)
    cramers_v = np.sqrt(
        chi_square / (observed.sum() * min(np.array(observed.shape) - 1))
    )
    save_table(
        pd.DataFrame(
            expected,
            index=contingency.index[contingency.sum(axis=1) > 0],
            columns=contingency.columns,
        ),
        "contingency_expected",
    )
    save_table(
        pd.DataFrame(
            {"V Крамера": [cramers_v]}, index=["symboling / drive-wheels"]
        ),
        "categorical_association",
    )
    rho, p_corr = stats.spearmanr(data["symboling"], weight)
    strength = (
        "слабая"
        if abs(rho) < 0.3
        else "умеренная" if abs(rho) < 0.7 else "сильная"
    )
    significance = (
        "статистически значима"
        if p_corr < ALPHA
        else "статистически незначима"
    )
    correlation = pd.DataFrame(
        {
            "rho Спирмена": [rho],
            "p-value (двустороннее)": [p_corr],
            "n": [len(data)],
            "alpha": [ALPHA],
            "Сила связи": [strength],
            "Значимость": [significance],
        },
        index=["symboling / curb-weight"],
    )
    save_table(correlation, "rank_correlation")
    weight_by_drive = data.groupby("drive-wheels")["curb-weight"].agg(
        ["count", "mean", "median"]
    )
    save_table(weight_by_drive, "weight_by_drive")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        1, len(NUMERIC), figsize=(7, 4.6), layout="constrained", squeeze=False
    )
    bin_rows = []
    for ax, column in zip(axes.flat, NUMERIC):
        x = data[column]
        k = k_sturges
        edges = np.linspace(x.min(), x.max(), k + 1)
        counts, _ = np.histogram(x, bins=edges)
        for i, count in enumerate(counts):
            bin_rows.append(
                {
                    "Признак": column,
                    "Интервал": i + 1,
                    "Левая граница": edges[i],
                    "Правая граница": edges[i + 1],
                    "Частота": int(count),
                    "Правая граница включена": i == k - 1,
                }
            )
        ax.hist(x, bins=edges, edgecolor="white", color="#3576a8")
        ax.set(
            title=f"{column}: k = {k}, h = {(x.max()-x.min())/k:.3f}",
            xlabel=column,
            ylabel="Частота",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "histograms_sturges.png", dpi=170)
    plt.close(fig)
    save_table(pd.DataFrame(bin_rows), "histogram_bins")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, column in zip(axes, LEVELS):
        counts = frequencies[column]["Частота"]
        percentages = frequencies[column]["Доля, %"]
        bars = ax.bar(counts.index.astype(str), percentages, color="#3576a8")
        ax.bar_label(
            bars, labels=[f"{v:.2f}%" for v in percentages], padding=3
        )
        ax.set(
            title=f"Распределение {column}",
            xlabel=column,
            ylabel="Доля автомобилей, %",
        )
        ax.set_ylim(0, percentages.max() * 1.15)
    fig.savefig(out / "category_frequencies.png", dpi=170)
    plt.close(fig)

    report = f"""# Лабораторная работа №1: описательная статистика и корреляционный анализ

Дисциплина: «Методы анализа данных». Выполнены пункты 1-8 задания.

## 1. Данные и переменные

Источник: локальный архив automobile.zip, файлы imports-85.data и imports-85.names.
Набор 1985 Auto Imports содержит {len(raw)} строк и {len(raw.columns)} столбцов, без заголовка; разделитель - запятая, пропуски обозначаются `?`.

- `symboling`: зависимая порядковая переменная, оценка страхового риска. Чем выше значение, тем выше риск относительно ожидаемого по цене автомобиля. Допустимы уровни от -3 до 3; фактически наблюдаются от {data['symboling'].min()} до {data['symboling'].max()}.
- `drive-wheels`: независимая номинальная переменная, тип привода: 4wd - полный, fwd - передний, rwd - задний.
- `curb-weight`: независимая количественная переменная, снаряжённая масса автомобиля. Значения приводятся в исходной шкале: файл .names не указывает единицу измерения.

Назначение переменных зависимыми и независимыми задаёт дальнейшую постановку анализа, само по себе не означает причинную связь.

## Проверка данных

{markdown_table(quality)}

В выбранных столбцах пропусков нет; все {len(data)} строк сохранены. Пропуски других столбцов не являются основанием удалять строки для этого анализа.
Полных повторов строк исходного набора: {int(raw.duplicated().sum())}. Повторов комбинаций трёх выбранных признаков: {int(data.duplicated().sum())}; совпадение признаков не доказывает повтор объекта, поэтому такие строки сохранены.

## 2. Дескриптивный анализ

{markdown_table(summary.T)}

Выборочные дисперсия и стандартное отклонение рассчитаны с делителем n-1.
Числовые описательные статистики рассчитаны только для `curb-weight`.
Категориальные `symboling` (порядковая) и `drive-wheels` (номинальная) описываются частотами и процентными долями. Доля категории равна её частоте, делённой на число автомобилей и умноженной на 100%.

### Частоты symboling

{markdown_table(frequencies['symboling'])}

Самый частый уровень symboling: {most_common_levels}; число автомобилей - {most_common_count}, доля - {most_common_percent:.2f}%.

### Частоты drive-wheels

{markdown_table(frequencies['drive-wheels'])}

![Частоты категорий](category_frequencies.png)

## 3. Гистограммы и формула Стерджесса

Число интервалов k = ceil(1 + log2(n)) = {int(np.ceil(1 + np.log2(len(data))))}; длина интервала h = (max - min) / k.
При n = {len(data)} для массы h = {(weight.max()-weight.min())/k_sturges:.6f}.
Использован наблюдаемый диапазон, а не теоретический. Интервалы замкнуты слева и открыты справа; последний включает максимум.
Границы и частоты сохранены в `histogram_bins.csv`.

![Гистограммы по Стерджессу](histograms_sturges.png)

Для категориальных `symboling` и `drive-wheels` используются столбчатые диаграммы процентных долей без объединения категорий в интервалы. У symboling показаны все допустимые уровни, включая отсутствующий -3.

### Оценка согласованности с нормальным распределением (пункт 2)

Нормальность оценивается визуально по гистограмме. При нормальном распределении ожидается форма симметричного колокола: большинство значений находится около центра, а к краям частоты постепенно уменьшаются.
Гистограмма `curb-weight` не вполне соответствует этой форме: она несимметрична и имеет вытянутый правый хвост.
По визуальной оценке есть отклонения от нормальной формы. Для дальнейшего анализа выбран ранговый коэффициент корреляции Спирмена, который также подходит для порядковой переменной symboling.
Визуальная оценка не является строгой проверкой гипотезы нормальности.
`symboling` имеет конечную порядковую шкалу, а `drive-wheels` - номинальную: проверка их на непрерывное нормальное распределение содержательно не требуется.

## 4. Таблица сопряжённости

В таблице строки - уровни symboling, столбцы - тип привода. В ячейке указано число автомобилей с соответствующим сочетанием признаков.
Принцип сопоставления фактических и ожидаемых частот рассмотрен в [примере из задания](https://excel2.ru/articles/kriteriy-nezavisimosti-hi-kvadrat-v-ms-excel).

{markdown_table(contingency)}

Группы приводов имеют разный размер, поэтому сравниваем доли уровней symboling внутри каждого типа привода (каждый столбец ниже даёт 100%).

{markdown_table(column_percent.round(2))}

Доля symboling = 0: для 4wd - {column_percent.loc[0, '4wd']:.2f}%, fwd - {column_percent.loc[0, 'fwd']:.2f}%, rwd - {column_percent.loc[0, 'rwd']:.2f}%.
Распределения риска по типам привода различаются в данной выборке. В группе 4wd всего {contingency['4wd'].sum()} автомобилей, поэтому её проценты особенно чувствительны к отдельным наблюдениям.

Описательная мера связи категорий: V Крамера = {cramers_v:.3f}. V = sqrt(Σ((O-E)²/E) / (n · min(r-1, c-1))), где E = итог строки · итог столбца / n.
Отсутствующий уровень -3 исключён только из расчёта V. V принимает значения от 0 до 1 и не имеет знака; это мера связи категорий, а не ранговая корреляция.
В {int((expected < 5).sum())} из {expected.size} ячеек ожидаемая частота меньше 5. Здесь V используется описательно; статистическая значимость связи категорий не утверждается.

## 5. Ранговый корреляционный анализ

Для пары `symboling` и `curb-weight` рассчитан коэффициент Спирмена. У symboling категории упорядочены по риску, поэтому их порядок можно использовать в ранговом анализе.
Коэффициент вычисляется как корреляция рангов, совпадающим значениям присваиваются средние ранги. Среднее или медиана самих кодов symboling для этого не нужны.

{markdown_table(correlation)}

Номинальные категории `drive-wheels` не имеют порядка: присваивать им произвольные числовые ранги и вычислять Спирмена некорректно. Их связь с symboling описана таблицей сопряжённости.
Для сопоставления массы по типам привода приведены групповые описательные статистики:

{markdown_table(weight_by_drive.rename(columns={'count': 'Количество', 'mean': 'Средняя масса', 'median': 'Медиана массы'}).round(2))}

## 6. Значимость коэффициента корреляции

H0: ранговая корреляция symboling и curb-weight в генеральной совокупности равна 0. H1: она отличается от 0.
Двустороннее p-значение равно {p_corr:.6g}, уровень значимости alpha = {ALPHA}. Связь {significance}.
p-значение вычислено стандартным асимптотическим приближением `scipy.stats.spearmanr`, поэтому является приближённым.
Вывод предполагает независимые наблюдения; сходство моделей автомобилей ограничивает перенос результатов на более широкую совокупность.

## 7. Сила связи

Для интерпретации Спирмена принята условная шкала: |rho| < 0.3 - слабая связь; 0.3 ≤ |rho| < 0.7 - умеренная; |rho| ≥ 0.7 - сильная.
Для symboling и curb-weight rho = {rho:.3f}: связь {strength}, {'обратная' if rho < 0 else 'прямая'}.
Сильно коррелированных пар среди признаков, для которых здесь применим ранговый коэффициент, не обнаружено.
Для V Крамера эта шкала не применяется; его значение нельзя напрямую сравнивать со Спирменом.

## 8. Интерпретация результатов

1. Средняя масса равна {weight.mean():.2f}, медиана - {weight.median():.2f}, стандартное отклонение - {weight.std(ddof=1):.2f}. Значения лежат от {weight.min()} до {weight.max()}.
2. Гистограмма массы несимметрична и имеет вытянутый правый хвост: визуально есть отклонения от нормальной формы. Использован ранговый анализ.
3. Самый частый уровень symboling - {most_common_levels}: {most_common_count} автомобилей ({most_common_percent:.2f}%). Уровень -3 отсутствует в выборке (0%).
4. Передний привод встречается у {int(frequencies['drive-wheels'].loc['fwd', 'Частота'])} автомобилей ({frequencies['drive-wheels'].loc['fwd', 'Доля, %']:.2f}%). Полный привод представлен всего 9 наблюдениями; группы имеют разный размер.
5. Более тяжёлые автомобили имеют тенденцию к меньшему symboling: rho = {rho:.3f}, p = {p_corr:.6g}. Связь {strength} и {significance}; она не позволяет точно предсказывать риск отдельного автомобиля.
6. Таблица сопряжённости показывает различия распределений symboling по приводам, V Крамера = {cramers_v:.3f}. Причинное влияние массы или привода этим анализом не установлено: возможны другие связанные характеристики автомобилей.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string())
    print(frequencies["symboling"].to_string())
    print(
        f"Самый частый symboling: {most_common_levels} ({most_common_percent:.2f}%)"
    )
    print(contingency.to_string())
    print(correlation.to_string())
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()
