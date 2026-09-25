"""Расчёты для лабораторной работы №1.

Скрипт читает данные Automobile, выполняет статистический анализ и сохраняет
таблицы CSV и графики PNG.
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
LEVELS = {"symboling": list(range(-3, 4)), "drive-wheels": ["4wd", "fwd", "rwd"]}
ALPHA = 0.05


def save_table(frame, output, name):
    frame.to_csv(output / f"{name}.csv", encoding="utf-8-sig")


def load_data(data_path):
    raw = pd.read_csv(data_path, header=None, na_values="?")
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"Ожидалось 26 столбцов, получено {raw.shape[1]}")
    raw.columns = COLUMNS
    data = raw[SELECTED].copy()
    data["curb-weight"] = pd.to_numeric(data["curb-weight"], errors="raise")
    for column, levels in LEVELS.items():
        if not data[column].dropna().isin(levels).all():
            raise ValueError(f"Недопустимые значения {column}")
    if not data["curb-weight"].between(1488, 4066).all():
        raise ValueError("curb-weight вне указанного диапазона")
    if len(data) < 3 or data.isna().any().any():
        raise ValueError("Недостаточно наблюдений или есть пропуски")
    return raw, data


def calculate_tables(raw, data, output):
    quality = pd.DataFrame(
        {
            "Тип": ["порядковый", "номинальный", "количественный"],
            "Наблюдений": data.count(),
            "Пропусков": data.isna().sum(),
            "Уникальных значений": data.nunique(),
        },
        index=SELECTED,
    )
    save_table(quality, output, "data_quality")

    weight = data["curb-weight"]
    summary = pd.DataFrame(
        {
            "n": [len(weight)], "Среднее": [weight.mean()], "Медиана": [weight.median()],
            "Моды": [", ".join(map(str, weight.mode().tolist()))],
            "Минимум": [weight.min()], "Максимум": [weight.max()],
            "Размах": [weight.max() - weight.min()],
            "Дисперсия (ddof=1)": [weight.var(ddof=1)],
            "Станд. отклонение (ddof=1)": [weight.std(ddof=1)],
        },
        index=["curb-weight"],
    )
    save_table(summary, output, "descriptive_statistics")

    frequencies = {}
    for column, levels in LEVELS.items():
        counts = data[column].value_counts().reindex(levels, fill_value=0)
        frequencies[column] = pd.DataFrame({"Частота": counts, "Доля, %": counts / len(data) * 100})
        save_table(frequencies[column], output, f"frequencies_{column}")

    contingency = pd.crosstab(data["symboling"], data["drive-wheels"]).reindex(
        index=LEVELS["symboling"], columns=LEVELS["drive-wheels"], fill_value=0
    )
    column_percent = contingency.div(contingency.sum(axis=0), axis=1) * 100
    observed = contingency.loc[contingency.sum(axis=1) > 0].to_numpy()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    chi_square = np.sum((observed - expected) ** 2 / expected)
    cramers_v = np.sqrt(chi_square / (observed.sum() * min(np.array(observed.shape) - 1)))
    save_table(contingency, output, "contingency_counts")
    save_table(column_percent, output, "contingency_column_percent")
    save_table(pd.DataFrame(expected, index=contingency.index[contingency.sum(axis=1) > 0], columns=contingency.columns), output, "contingency_expected")
    save_table(pd.DataFrame({"V Крамера": [cramers_v]}, index=["symboling / drive-wheels"]), output, "categorical_association")

    rho, p_value = stats.spearmanr(data["symboling"], weight)
    strength = "слабая" if abs(rho) < 0.3 else "средняя" if abs(rho) < 0.7 else "сильная"
    significance = "статистически значима" if p_value < ALPHA else "статистически незначима"
    correlation = pd.DataFrame(
        {"rho Спирмена": [rho], "p-value (двустороннее)": [p_value], "n": [len(data)],
         "alpha": [ALPHA], "Сила связи": [strength], "Значимость": [significance]},
        index=["symboling / curb-weight"],
    )
    save_table(correlation, output, "rank_correlation")
    save_table(data.groupby("drive-wheels")["curb-weight"].agg(["count", "mean", "median"]), output, "weight_by_drive")
    return frequencies


def create_charts(data, frequencies, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    weight = data["curb-weight"]
    intervals = int(np.ceil(1 + np.log2(len(data))))
    edges = np.linspace(weight.min(), weight.max(), intervals + 1)
    counts, _ = np.histogram(weight, bins=edges)
    save_table(
        pd.DataFrame({"Признак": "curb-weight", "Интервал": range(1, intervals + 1),
                      "Левая граница": edges[:-1], "Правая граница": edges[1:],
                      "Частота": counts, "Правая граница включена": [False] * (intervals - 1) + [True]}),
        output, "histogram_bins",
    )
    fig, axis = plt.subplots(figsize=(7, 4.6), layout="constrained")
    axis.hist(weight, bins=edges, edgecolor="white", color="#3576a8")
    axis.set(title=f"curb-weight: k = {intervals}, h = {(weight.max() - weight.min()) / intervals:.3f}", xlabel="curb-weight", ylabel="Частота")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "histograms_sturges.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for axis, column in zip(axes, LEVELS):
        percentages = frequencies[column]["Доля, %"]
        bars = axis.bar(percentages.index.astype(str), percentages, color="#3576a8")
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in percentages], padding=3)
        axis.set(title=f"Распределение {column}", xlabel=column, ylabel="Доля автомобилей, %")
        axis.set_ylim(0, percentages.max() * 1.15)
    fig.savefig(output / "category_frequencies.png", dpi=170)
    plt.close(fig)


def run_analysis(data_path=BASE / "data/imports-85.data", output=BASE / "results/statistics"):
    """Выполнить расчёты лабораторной и вернуть данные и путь результатов."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw, data = load_data(data_path)
    frequencies = calculate_tables(raw, data, output)
    create_charts(data, frequencies, output)
    return {"raw": raw, "data": data, "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "data/imports-85.data")
    parser.add_argument("--output", type=Path, default=BASE / "results/statistics")
    args = parser.parse_args()
    print(f"Таблицы и графики: {run_analysis(args.data, args.output)['output']}")


if __name__ == "__main__":
    main()
