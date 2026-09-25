"""Формирование Markdown-отчёта для лабораторной работы 1.2."""

from pathlib import Path

import numpy as np
import pandas as pd

from regression_analysis import BASE, run_analysis


def markdown_table(frame):
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)

    rows = [[str(frame.index.name or "Показатель"), *map(str, frame.columns)]]
    rows.extend([str(index), *map(format_value, row)] for index, row in frame.iterrows())
    return "\n".join(
        [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * len(rows[0])) + " |",
            *["| " + " | ".join(row) + " |" for row in rows[1:]],
        ]
    )


def read_table(results, name, index_col=0):
    return pd.read_csv(results / f"{name}.csv", index_col=index_col)


def coefficient_equation(coefficients, poisson=False):
    values = coefficients["coefficient"]
    parts = [f"{values['intercept']:.6f}"]
    labels = {
        "curb_weight_100": "(curb-weight / 100)",
        "curb_weight_100_squared": "(curb-weight / 100)^2",
        "drive_fwd": "I(drive-wheels = fwd)",
        "drive_rwd": "I(drive-wheels = rwd)",
    }
    for name in coefficients.index:
        if name != "intercept":
            parts.append(f"{values[name]:+.6f} * {labels[name]}")
    expression = " ".join(parts)
    if poisson:
        return f"E(symboling + 2 | X) = exp({expression})"
    return f"symboling_hat = {expression}"


def model_section(title, name, results, explanation):
    coefficients = read_table(results, f"{name}_coefficients")
    metrics = read_table(results, f"{name}_metrics")
    metric_table = metrics.T.rename(columns={name: "Значение"})
    residual_groups = read_table(results, f"{name}_residual_groups")
    maximum_mean = residual_groups["mean"].abs().max()
    positive_variances = residual_groups.loc[residual_groups["var"] > 0, "var"]
    variance_ratio = positive_variances.max() / positive_variances.min()
    stability = (
        "Групповые средние заметно отклоняются от нуля, а дисперсии различаются более чем вдвое; предположения о постоянстве среднего и дисперсии остатков выполняются плохо."
        if maximum_mean > 0.25 or variance_ratio > 2
        else "Групповые средние близки к нулю, а разброс дисперсий невелик; явных нарушений постоянства среднего и дисперсии не обнаружено."
    )
    equation = coefficient_equation(coefficients, poisson=name == "poisson")
    return f"""### {title}

{explanation}

Уравнение модели:

`{equation}`

За базовую категорию типа привода принят `4wd`. Переменные `I(...)` равны 1 при выполнении условия и 0 в остальных случаях.

Коэффициенты и их значимость:

{markdown_table(coefficients)}

Показатели качества и значимости модели:

{markdown_table(metric_table)}

![Диагностика остатков модели](results/{name}_residuals.png)

Среднее и дисперсия остатков по четырём группам предсказанных значений:

{markdown_table(residual_groups)}

Максимальное по модулю групповое среднее остатка равно {maximum_mean:.3f}, отношение наибольшей групповой дисперсии к наименьшей - {variance_ratio:.3f}. Чем ближе средние к нулю и отношение дисперсий к единице, тем лучше выполняются предположения о постоянстве среднего и дисперсии остатков.

{stability}
"""


def build_report(results):
    quality = read_table(results, "data_quality")
    descriptive = read_table(results, "descriptive_statistics")
    symboling = read_table(results, "frequencies_symboling")
    drive = read_table(results, "frequencies_drive-wheels")
    comparison = read_table(results, "model_comparison")
    linear_metrics = read_table(results, "linear_metrics").iloc[0]
    nonlinear_metrics = read_table(results, "nonlinear_metrics").iloc[0]
    poisson_metrics = read_table(results, "poisson_metrics").iloc[0]
    common_symboling = symboling["count"].idxmax()

    linear = model_section(
        "Линейная регрессия с фиктивными переменными",
        "linear",
        results,
        "Зависимая переменная - порядковая категория `symboling`. Модель МНК используется как учебное числовое приближение. Независимые переменные: масса и тип привода.",
    )
    poisson = model_section(
        "Пуассоновская регрессия",
        "poisson",
        results,
        "Распределение Пуассона не допускает отрицательных значений, поэтому использован сдвиг `symboling + 2`, дающий значения от 0 до 5. Поскольку `symboling` является порядковой, а не счётной переменной, модель рассматривается как учебная и интерпретируется осторожно.",
    )
    nonlinear = model_section(
        "Нелинейная регрессия с фиктивными переменными",
        "nonlinear",
        results,
        "К линейной модели добавлен квадрат нормированной массы `(curb-weight / 100)^2`. Модель остаётся линейной по коэффициентам и оценивается методом наименьших квадратов.",
    )

    linear_better = nonlinear_metrics["adjusted_r_squared"] <= linear_metrics["adjusted_r_squared"]
    preferred = "линейная" if linear_better else "нелинейная"
    report = f"""# Лабораторная работа 1.2. Регрессионный анализ

## 2. Цель работы

Изучить построение линейной, пуассоновской и нелинейной регрессионных моделей с фиктивными переменными, провести анализ остатков, оценить качество и статистическую значимость моделей.

## 3. Описание исходных данных

Использован набор **1985 Auto Imports** из первой части лабораторной работы. Рассматриваются 205 автомобилей и три признака:

- `symboling` - зависимая порядковая переменная страхового риска;
- `curb-weight` - независимая количественная переменная массы;
- `drive-wheels` - независимая категориальная переменная типа привода (`4wd`, `fwd`, `rwd`).

{markdown_table(quality)}

Пропусков в выбранных переменных нет.

## 4. Результаты дескриптивного анализа

Для количественной переменной `curb-weight`:

{markdown_table(descriptive.T)}

Для категориальной зависимой переменной `symboling`:

{markdown_table(symboling)}

Самая частая категория `symboling` - **{common_symboling}** ({symboling.loc[common_symboling, 'percent']:.2f}%). Распределение типа привода:

{markdown_table(drive)}

Число интервалов гистограммы массы рассчитано по формуле Стерджесса. Нормальность оценивается визуально: распределение массы несимметрично и имеет правый хвост.

![Гистограммы выбранных переменных](results/selected_variables_histograms.png)

## 5. Регрессионные модели

{linear}

{poisson}

{nonlinear}

## 6. Сравнительный анализ и интерпретация

Сводная таблица рассчитанных показателей:

{markdown_table(comparison)}

Линейная модель: R² = {linear_metrics['r_squared']:.4f}, скорректированный R² = {linear_metrics['adjusted_r_squared']:.4f}, p модели = {linear_metrics['model_p_value']:.6g}.

Нелинейная модель: R² = {nonlinear_metrics['r_squared']:.4f}, скорректированный R² = {nonlinear_metrics['adjusted_r_squared']:.4f}, p модели = {nonlinear_metrics['model_p_value']:.6g}.

Пуассоновская модель сошлась за {int(float(poisson_metrics['iterations']))} итераций; псевдо-R² Мак-Фаддена = {float(poisson_metrics['mcfadden_pseudo_r_squared']):.4f}, p модели = {float(poisson_metrics['model_p_value']):.6g}. Этот показатель нельзя напрямую сравнивать с обычным R² моделей МНК.

По скорректированному R² среди моделей МНК предпочтительна **{preferred} модель**. Качество всех моделей оценивается совместно по метрикам и графикам остатков. Наличие структуры или изменения разброса остатков указывает на неполное описание зависимости.

Обе модели МНК объясняют менее 8% разброса `symboling`, поэтому их практическая предсказательная способность низкая. Добавление квадрата массы повышает скорректированный R² лишь примерно на {nonlinear_metrics['adjusted_r_squared'] - linear_metrics['adjusted_r_squared']:.4f}, а коэффициент квадратного члена статистически незначим. Пуассоновская модель в целом незначима при α = 0.05. На графиках остатки образуют полосы из-за дискретных уровней зависимой переменной; групповые дисперсии заметно различаются.

Полученные связи не доказывают причинного влияния массы или типа привода на страховой риск. Кроме того, `symboling` является порядковой категорией, поэтому линейная и пуассоновская регрессии здесь выполняют учебную функцию; для строгого моделирования естественнее порядковая логистическая регрессия.
"""
    return report


def main():
    data_path = BASE.parent / "lab1/data/imports-85.data"
    results = BASE / "results"
    run_analysis(data_path, results)
    report_path = BASE / "REPORT.md"
    report_path.write_text(build_report(results), encoding="utf-8")
    print(f"Отчёт: {report_path}")


if __name__ == "__main__":
    main()
