"""Расчёт ложной корреляции потребления маргарина и разводов в штате Мэн."""

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


def calculate(data_path, output):
    data = pd.read_csv(data_path)
    required = {"year", "margarine_lb_per_person", "maine_divorce_rate"}
    if set(data.columns) != required:
        raise ValueError(f"Ожидались столбцы: {sorted(required)}")
    if data.isna().any().any() or len(data) < 4:
        raise ValueError("В данных есть пропуски или недостаточно наблюдений")

    output.mkdir(parents=True, exist_ok=True)
    years = data["year"].to_numpy()
    time = years - years.min()
    margarine = data["margarine_lb_per_person"].to_numpy()
    divorce = data["maine_divorce_rate"].to_numpy()
    raw_correlation, raw_p_value = stats.pearsonr(margarine, divorce)

    details = data.copy()
    coefficients = []
    residuals = []
    standardized_residuals = []
    for name, values in (("margarine", margarine), ("divorce", divorce)):
        coefficient = np.polyfit(time, values, 1)
        predicted = np.polyval(coefficient, time)
        residual = values - predicted
        standardized = (residual - residual.mean()) / residual.std(ddof=1)
        coefficients.append(coefficient)
        residuals.append(residual)
        standardized_residuals.append(standardized)
        details[f"{name}_trend"] = predicted
        details[f"{name}_residual"] = residual
        details[f"{name}_standardized_residual"] = standardized

    residual_correlation = stats.pearsonr(*residuals).statistic
    degrees_of_freedom = len(data) - 3
    residual_t = residual_correlation * np.sqrt(
        degrees_of_freedom / (1 - residual_correlation**2)
    )
    residual_p_value = 2 * stats.t.sf(abs(residual_t), degrees_of_freedom)

    details.to_csv(output / "trend_and_residuals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "coefficient": [raw_correlation, residual_correlation],
            "p_value": [raw_p_value, residual_p_value],
            "degrees_of_freedom": [len(data) - 2, degrees_of_freedom],
        },
        index=["Pearson_raw", "Pearson_controlling_year"],
    ).to_csv(output / "correlations.csv", encoding="utf-8-sig")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    labels = ["Маргарин, фунтов на человека", "Разводы в Мэне"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for axis, values, coefficient, label in zip(
        axes, (margarine, divorce), coefficients, labels
    ):
        axis.plot(years, values, "o-", label="Наблюдения")
        axis.plot(
            years,
            np.polyval(coefficient, time),
            "--",
            label=f"Тренд: y = {coefficient[1]:.3f} {coefficient[0]:+.3f}t",
        )
        axis.set(xlabel="Год", ylabel=label)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    fig.suptitle(f"Динамика за {years.min()}-{years.max()} годы: r = {raw_correlation:.3f}")
    fig.savefig(output / "time_series.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for axis, standardized, label in zip(
        axes, standardized_residuals, ("Маргарин", "Разводы")
    ):
        axis.plot(years, standardized, "o-")
        axis.axhline(0, color="gray", linewidth=1)
        axis.set(title=label, xlabel="Год", ylabel="Стандартизированный остаток")
        axis.grid(alpha=0.2)
    fig.savefig(output / "residuals.png", dpi=160)
    plt.close(fig)

    bins = int(np.ceil(1 + np.log2(len(data))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for axis, standardized, label in zip(
        axes, standardized_residuals, ("Маргарин", "Разводы")
    ):
        axis.hist(standardized, bins=bins, edgecolor="white")
        axis.set(title=label, xlabel="Стандартизированный остаток", ylabel="Частота")
    fig.savefig(output / "residual_histograms.png", dpi=160)
    plt.close(fig)

    return raw_correlation, residual_correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE / "data/spurious_margarine_divorce.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/spurious"
    )
    args = parser.parse_args()
    raw, residual = calculate(args.data, args.output)
    print(f"Корреляция исходных рядов: {raw:.6f}")
    print(f"Корреляция остатков: {residual:.6f}")
    print(f"Таблицы и графики: {args.output}")


if __name__ == "__main__":
    main()
