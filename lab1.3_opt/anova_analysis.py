"""Дополнение к ЛР 1.3: ANOVA Уэлча и Брауна–Форсайта.

Оба метода сравнивают средние цены по приводу при неодинаковых дисперсиях.
Формулы независимо сверяются с statsmodels; для Уэлча — также с SciPy.
"""

from __future__ import annotations

import hashlib
import json
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
from statsmodels.stats.oneway import anova_oneway

RESULTS = BASE / "results"
DRIVES = ["4wd", "fwd", "rwd"]
SELECTED = ["price", "drive-wheels"]
COLUMNS = (
    "symboling normalized-losses make fuel-type aspiration num-of-doors "
    "body-style drive-wheels engine-location wheel-base length width height "
    "curb-weight engine-type num-of-cylinders engine-size fuel-system bore "
    "stroke compression-ratio horsepower peak-rpm city-mpg highway-mpg price"
).split()
ALPHA = .05


def csv(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False, encoding="utf-8-sig")


def load_data():
    path = BASE / "data/imports-85.data"
    raw = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    assert raw.shape == (205, 26)
    selected = raw[SELECTED].copy()
    selected.insert(0, "source_row", np.arange(1, len(raw) + 1))
    complete = selected[SELECTED].notna().all(axis=1)
    data, excluded = selected.loc[complete].copy(), selected.loc[~complete].copy()
    assert len(data) == 201 and excluded.source_row.tolist() == [10, 45, 46, 130]
    assert data["drive-wheels"].isin(DRIVES).all()
    assert np.isfinite(data.price).all() and (data.price > 0).all()
    csv(data, "analysis_data")
    csv(excluded, "excluded_rows")
    csv(pd.DataFrame([
        {"variable": name, "type": "количественный" if name == "price" else "номинальный",
         "total": len(raw), "valid": int(raw[name].count()), "missing": int(raw[name].isna().sum()),
         "in_analysis": int(data[name].count())} for name in SELECTED
    ]), "data_quality")
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def descriptive(data):
    rows, normality = [], []
    for group in ["all", *DRIVES]:
        values = data.price if group == "all" else data.loc[data["drive-wheels"] == group, "price"]
        rows.append({
            "group": group, "count": len(values), "mean": values.mean(), "median": values.median(),
            "minimum": values.min(), "maximum": values.max(), "range": values.max() - values.min(),
            "variance": values.var(ddof=1), "standard_deviation": values.std(ddof=1),
            "q1": values.quantile(.25), "q3": values.quantile(.75),
            "iqr": values.quantile(.75) - values.quantile(.25), "skewness": stats.skew(values, bias=False),
        })
        w, p = stats.shapiro(values)
        normality.append({"group": group, "n": len(values), "W": w, "p_value": p})
    csv(pd.DataFrame(rows), "descriptive_statistics")
    csv(pd.DataFrame(normality), "normality_shapiro")


def compare_means(groups):
    n = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    k, total = len(groups), n.sum()
    assert np.all(variances > 0) and np.all(n > 1)

    weights = n / variances
    proportions = weights / weights.sum()
    weighted_mean = proportions @ means
    correction_sum = np.sum((1 - proportions) ** 2 / (n - 1))
    welch_f = (weights @ (means - weighted_mean) ** 2 / (k - 1)) / (
        1 + 2 * (k - 2) * correction_sum / (k ** 2 - 1))
    welch_df = (float(k - 1), float((k ** 2 - 1) / (3 * correction_sum)))
    welch_p = stats.f.sf(welch_f, *welch_df)
    library_welch = anova_oneway(groups, use_var="unequal")
    scipy_welch = stats.f_oneway(*groups, equal_var=False)
    np.testing.assert_allclose([welch_f, welch_p], [library_welch.statistic, library_welch.pvalue], rtol=1e-11)
    np.testing.assert_allclose([welch_f, welch_p], scipy_welch, rtol=1e-11)
    np.testing.assert_allclose(welch_df, library_welch.df, rtol=1e-11)

    # Браун–Форсайт: проверка СРЕДНИХ с поправкой Мехротры.
    # pvalue в statsmodels относится именно к этой поправке; pvalue2 здесь не используется.
    overall_mean = (n @ means) / total
    fractions = n / total
    denominator = np.sum((1 - fractions) * variances)
    bf_f = np.sum(n * (means - overall_mean) ** 2) / denominator
    bf_df2 = denominator ** 2 / np.sum((1 - fractions) ** 2 * variances ** 2 / (n - 1))
    bf_df1 = denominator ** 2 / (
        np.sum(variances ** 2) + np.sum(fractions * variances) ** 2
        - 2 * np.sum(fractions * variances ** 2))
    bf_p = stats.f.sf(bf_f, bf_df1, bf_df2)
    library_bf = anova_oneway(groups, use_var="bf")
    np.testing.assert_allclose([bf_f, bf_p], [library_bf.statistic, library_bf.pvalue], rtol=1e-11)
    np.testing.assert_allclose([bf_df1, bf_df2], library_bf.df, rtol=1e-11)

    results = {
        "welch": {"F": float(welch_f), "df1": welch_df[0], "df2": welch_df[1], "p_value": float(welch_p)},
        "brown_forsythe": {"F": float(bf_f), "df1": float(bf_df1), "df2": float(bf_df2), "p_value": float(bf_p)},
    }
    for name, row in results.items():
        csv(pd.DataFrame([row]), name)
    csv(pd.DataFrame({"drive": DRIVES, "n": n.astype(int), "mean": means, "variance": variances,
                      "welch_weight": weights, "welch_weight_share": proportions}), "group_calculations")
    return results


def figures(data, groups):
    # Геометрия, палитра и подписи графиков соответствуют актуальной ЛР 1.3.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), layout="constrained")
    bins = []
    for ax, group in zip(axes.flat, ["all", *DRIVES]):
        values = data.price if group == "all" else data.loc[data["drive-wheels"] == group, "price"]
        k = int(np.ceil(1 + np.log2(len(values))))
        edges = np.linspace(values.min(), values.max(), k + 1)
        counts, _ = np.histogram(values, bins=edges)
        step = (values.max() - values.min()) / k
        ax.hist(values, bins=edges, edgecolor="white", color="#3576a8")
        label = "Вся выборка" if group == "all" else f"Привод {group}"
        ax.set(title=f"{label}: n = {len(values)}, k = {k}, h = {step:.2f}",
               xlabel="Цена, долл. США", ylabel="Частота")
        ax.ticklabel_format(axis="x", style="plain")
        ax.grid(axis="y", alpha=.18)
        bins.extend({"group": group, "n": len(values), "k": k, "h": step,
                     "bin": j + 1, "left": edges[j], "right": edges[j + 1], "count": int(counts[j])}
                    for j in range(k))
    csv(pd.DataFrame(bins), "histogram_bins")
    fig.savefig(RESULTS / "price_histograms.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    boxes = axis.boxplot(groups, tick_labels=DRIVES, patch_artist=True, showmeans=True)
    for box in boxes["boxes"]:
        box.set_facecolor("#d6e6f1")
    axis.set(title="Цена по типу привода", xlabel="Тип привода", ylabel="Цена, долл. США")
    axis.grid(axis="y", alpha=.18)
    fig.savefig(RESULTS / "price_boxplot.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), layout="constrained")
    for ax, group, values in zip(axes, DRIVES, groups):
        stats.probplot(values, dist="norm", plot=ax)
        ax.set(title=f"Q–Q: {group}", xlabel="Теоретические квантили", ylabel="Цена, долл. США")
    fig.savefig(RESULTS / "price_qq_by_drive.png", dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    data, sha256 = load_data()
    descriptive(data)
    groups = [data.loc[data["drive-wheels"] == group, "price"].to_numpy() for group in DRIVES]
    models = compare_means(groups)
    levene = stats.levene(*groups, center="median")
    summary = {"n": len(data), "excluded_rows": [10, 45, 46, 130], "data_sha256": sha256,
               "models": models, "levene": {"F": float(levene.statistic), "p_value": float(levene.pvalue)},
               "validation": {"welch_checked_with_statsmodels_and_scipy": True,
                              "brown_forsythe_mehrotra_checked_with_statsmodels": True}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(data, groups)
    for name, model in models.items():
        print(f"{name}: F = {model['F']:.6f}, df = ({model['df1']:.6f}; {model['df2']:.6f}), p = {model['p_value']:.6g}")


if __name__ == "__main__":
    main()
