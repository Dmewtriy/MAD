"""Сформировать REPORT.md с разделами 2-8 и примером ложной корреляции."""

import re

from statistical_analysis import BASE, main as analyze, markdown_table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SOURCE = "https://www.tylervigen.com/spurious/correlation/5920_per-capita-consumption-of-margarine_correlates-with_the-divorce-rate-in-maine"


def main():
    analyze()
    original = (BASE / "results/statistics/report.md").read_text(encoding="utf-8")

    def section(start, end=None):
        text = original.split(start + "\n", 1)[1]
        if end:
            text = text.split(end + "\n", 1)[0]
        return text.strip()

    description = section("## 1. Данные и переменные", "## 2. Дескриптивный анализ")
    description = description.replace("## Проверка данных", "### 3.1. Проверка данных")
    descriptive = section("## 2. Дескриптивный анализ", "## 4. Таблица сопряжённости")
    descriptive = descriptive.replace("## 3. Гистограммы и формула Стерджесса",
                                      "### Гистограмма массы и формула Стерджесса")
    descriptive = descriptive.replace(" (пункт 2)", "")
    contingency = section("## 4. Таблица сопряжённости", "## 5. Ранговый корреляционный анализ")
    correlation = section("## 5. Ранговый корреляционный анализ", "## 7. Сила связи")
    correlation = correlation.replace("## 6. Значимость коэффициента корреляции",
                                      "### Значимость коэффициента корреляции")
    interpretation = section("## 7. Сила связи")
    interpretation = interpretation.replace("## 8. Интерпретация результатов", "### Выводы по данным Automobile")

    out = BASE / "results/spurious"
    out.mkdir(parents=True, exist_ok=True)
    example = pd.read_csv(BASE / "data/spurious_margarine_divorce.csv")
    t = example["year"].to_numpy() - 2000
    x = example["margarine_lb_per_person"].to_numpy()
    y = example["maine_divorce_rate"].to_numpy()
    r, p = stats.pearsonr(x, y)
    trends, residuals, standardized = [], [], []
    details = example.copy()
    for label, values in [("margarine", x), ("divorce", y)]:
        coef = np.polyfit(t, values, 1)
        predicted = np.polyval(coef, t)
        residual = values - predicted
        z = (residual - residual.mean()) / residual.std(ddof=1)
        trends.append(coef)
        residuals.append(residual)
        standardized.append(z)
        details[f"{label}_trend"] = predicted
        details[f"{label}_residual"] = residual
        details[f"{label}_standardized_residual"] = z
    r_residual = stats.pearsonr(*residuals).statistic
    # Частная корреляция с контролем времени: n - 1 контролируемая переменная - 2.
    df = len(t) - 3
    p_residual = 2 * stats.t.sf(abs(r_residual) * np.sqrt(df / (1 - r_residual ** 2)), df)
    details.to_csv(out / "trend_and_residuals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"coefficient": [r, r_residual], "p_value": [p, p_residual],
                  "df": [len(t) - 2, df]}, index=["Pearson_raw", "Pearson_controlling_year"]).to_csv(
                      out / "correlations.csv", encoding="utf-8-sig")

    labels = ["Маргарин, фунтов на человека", "Разводы в Мэне, исходный показатель"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, values, coef, label in zip(axes, [x, y], trends, labels):
        ax.plot(example["year"], values, "o-", label="Наблюдения")
        ax.plot(example["year"], np.polyval(coef, t), "--",
                label=f"Тренд: y = {coef[1]:.3f} {coef[0]:+.3f}t")
        ax.set(xlabel="Год (t = год − 2000)", ylabel=label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(f"Динамика за 2000–2009 годы: r = {r:.3f}")
    fig.savefig(out / "time_series.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, z, label in zip(axes, standardized, ["Маргарин", "Разводы"]):
        ax.plot(example["year"], z, "o-")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(title=label, xlabel="Год", ylabel="Стандартизированный остаток")
        ax.grid(alpha=0.2)
    fig.savefig(out / "residuals.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, z, label in zip(axes, standardized, ["Маргарин", "Разводы"]):
        ax.hist(z, bins=int(np.ceil(1 + np.log2(len(z)))), edgecolor="white")
        ax.set(title=label, xlabel="Стандартизированный остаток", ylabel="Частота")
    fig.savefig(out / "residual_histograms.png", dpi=160)
    plt.close(fig)

    example_table = example.rename(columns={"year": "Год", "margarine_lb_per_person": "Маргарин",
                                           "maine_divorce_rate": "Разводы в Мэне"}).set_index("Год")
    report = f"""# Лабораторная работа №1

**Дисциплина:** Методы анализа данных  
**Тема:** Описательная статистика. Корреляционный анализ  
**Среда выполнения:** Python, pandas, NumPy, Matplotlib, SciPy

## 2. Цель работы

Изучить методы дескриптивного и корреляционного анализа: описать выбранные признаки,
оценить форму распределения массы по гистограмме, составить таблицу сопряжённости,
определить направление, силу и значимость связи признаков. На отдельном примере
рассмотреть ложную корреляцию и ограничения причинной интерпретации.

## 3. Описание исходных данных

{description}

## 4. Результаты дескриптивного анализа

{descriptive}

## 5. Анализ результатов таблиц сопряженности

{contingency}

## 6. Результаты корреляционного анализа

{correlation}

## 7. Оценка и интерпретация результатов

{interpretation}

## 8. Пример и интерпретация ложной корреляции

### 8.1. Данные примера

Рассмотрены потребление маргарина на человека в США (фунтов в год) и показатель
разводов в штате Мэн за 2000–2009 годы. Источник чисел: таблица **Data details**
на [странице Tyler Vigen]({SOURCE}); там указаны USDA и CDC как источники рядов.
Использованы именно опубликованные числовые данные. Шуточные объяснения сайта не
используются в качестве научного обоснования.

{markdown_table(example_table)}

Локальная копия: [spurious_margarine_divorce.csv](data/spurious_margarine_divorce.csv).

### 8.2. Корреляция исходных рядов

По 10 парам значений рассчитан коэффициент Пирсона: **r = {r:.6f}**.
Это сильная положительная линейная связь. Формальное двустороннее p-значение
равно **{p:.6g}**, что меньше 0.05. Здесь Пирсон используется для воспроизведения
линейной корреляции из примера; вывод о нормальности временных рядов не делается.

![Исходные ряды и линейные тренды](results/spurious/time_series.png)

### 8.3. Проверка предположения об общем тренде

Проверено предположение: высокая корреляция объясняется тем, что оба показателя
снижаются со временем. Для каждого ряда методом наименьших квадратов построен
простой линейный тренд, где t = год − 2000:

- Маргарин: x̂ = {trends[0][1]:.6f} − {abs(trends[0][0]):.6f} · t.
- Разводы: ŷ = {trends[1][1]:.6f} − {abs(trends[1][0]):.6f} · t.

Линейная форма выбрана как простая модель общего снижения, без подбора сложной
кривой по десяти точкам. Остаток равен наблюдаемому значению минус значение тренда.
Для сопоставления масштабов остатки центрированы и разделены на их выборочное
стандартное отклонение. Это только приведение к общей шкале, не проверка нормальности.

![Остатки после удаления трендов](results/spurious/residuals.png)

![Гистограммы стандартизированных остатков](results/spurious/residual_histograms.png)

Корреляция остатков: **r = {r_residual:.6f}**. После учёта времени формальное
двустороннее p-значение частной корреляции равно **{p_residual:.6g}**
(число степеней свободы: 10 − 1 − 2 = 7; учитывается одна переменная - время).
При расчёте принято обычное приближение с независимыми нормальными ошибками;
для коротких временных рядов эти предпосылки не подтверждены.

Корреляция немного уменьшилась, но осталась высокой. Поэтому предположение,
что связь полностью объясняется общим **линейным** трендом, расчётом не подтверждается.
Результаты прогнозов, остатков и их стандартизации сохранены в
[таблице расчётов](results/spurious/trend_and_residuals.csv).

### 8.4. Интерпретация

Высокий коэффициент не доказывает, что потребление маргарина влияет на разводы.
В данном анализе отсутствуют сведения о питании конкретных семей и об их разводах:
сопоставляются агрегированные показатели по годам, причём для разных территорий.

На [странице источника]({SOURCE}) отмечены массовый поиск совпадающих рядов и
возможная зависимость соседних лет. Поэтому маленькое p-значение отдельной пары
нельзя воспринимать как подтверждение причинной связи.

По нашему расчёту удаление линейных трендов не устранило связь. Это не доказывает
причинность: возможны нелинейная динамика, неучтённые факторы и отбор необычно
похожих рядов. По этим десяти наблюдениям установить конкретную причину совпадения
нельзя. Пример показывает, что даже очень высокая и формально значимая корреляция
требует содержательного объяснения и дополнительной проверки.
"""
    # Отчёт расположен в lab1/, исходные графики и таблицы - глубже.
    for name in ["category_frequencies.png", "histograms_sturges.png", "histogram_bins.csv"]:
        report = report.replace(f"]({name})", f"](results/statistics/{name})")
    report = report.replace("`histogram_bins.csv`", "[таблице интервалов](results/statistics/histogram_bins.csv)")
    # Проверяем локальные ссылки перед сохранением.
    for target in re.findall(r"\]\(([^)]+)\)", report):
        if not target.startswith(("http://", "https://")) and not (BASE / target).exists():
            raise FileNotFoundError(target)
    (BASE / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Отчёт: {BASE / 'REPORT.md'}")


if __name__ == "__main__":
    main()
