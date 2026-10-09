"""ЛР 1.3: анализ цены Automobile по приводу и типу наддува.

Исходный файл берётся из первой лабораторной; скачивание данных не требуется.
Таблицы, рисунки и машинно-читаемая сводка сохраняются в results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

COLUMNS = (
    "symboling normalized-losses make fuel-type aspiration num-of-doors "
    "body-style drive-wheels engine-location wheel-base length width height "
    "curb-weight engine-type num-of-cylinders engine-size fuel-system bore "
    "stroke compression-ratio horsepower peak-rpm city-mpg highway-mpg price"
).split()
DRIVE_LEVELS = ["4wd", "fwd", "rwd"]
ASPIRATION_LEVELS = ["std", "turbo"]
SELECTED = ["price", "drive-wheels", "aspiration"]
ALPHA = 0.05


def save_csv(frame, output, name, index=False):
    frame.to_csv(output / f"{name}.csv", index=index, encoding="utf-8-sig")


def load_data(path):
    raw = pd.read_csv(path, header=None, na_values="?")
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"Ожидалось 26 столбцов, получено {raw.shape[1]}")
    raw.columns = COLUMNS
    raw["price"] = pd.to_numeric(raw["price"], errors="raise")
    for column, levels in [("drive-wheels", DRIVE_LEVELS), ("aspiration", ASPIRATION_LEVELS)]:
        if not raw[column].dropna().isin(levels).all():
            raise ValueError(f"Неизвестные категории {column}")
    selected = raw[SELECTED].copy()
    selected.insert(0, "source_row", np.arange(1, len(raw) + 1))
    selected.insert(1, "make", raw["make"])
    complete = selected[SELECTED].notna().all(axis=1)
    clean = selected.loc[complete].copy()
    if not np.isfinite(clean["price"]).all() or (clean["price"] <= 0).any():
        raise ValueError("Цена должна быть положительным конечным числом")
    if any((clean["drive-wheels"] == level).sum() < 3 for level in DRIVE_LEVELS):
        raise ValueError("Недостаточно наблюдений для анализа групп")
    return raw, clean, selected.loc[~complete].copy()


def price_summary(values):
    values = pd.Series(values, dtype=float)
    n = len(values)
    critical = stats.t.ppf(1 - ALPHA / 2, n - 1)
    margin = critical * values.std(ddof=1) / np.sqrt(n)
    return {
        "count": n, "mean": values.mean(), "median": values.median(),
        "modes": ", ".join(f"{v:g}" for v in values.mode()),
        "minimum": values.min(), "maximum": values.max(),
        "range": values.max() - values.min(),
        "variance": values.var(ddof=1), "standard_deviation": values.std(ddof=1),
        "q1": values.quantile(0.25), "q3": values.quantile(0.75),
        "iqr": values.quantile(0.75) - values.quantile(0.25),
        "skewness": stats.skew(values, bias=False),
        "excess_kurtosis": stats.kurtosis(values, bias=False),
        "mean_ci95_low": values.mean() - margin,
        "mean_ci95_high": values.mean() + margin,
    }


def descriptive_analysis(raw, clean, excluded, output):
    quality = pd.DataFrame([
        {"variable": column, "type": "quantitative" if column == "price" else "nominal",
         "total_rows": len(raw), "valid_before": int(raw[column].count()),
         "missing_before": int(raw[column].isna().sum()),
         "unique_before": int(raw[column].nunique()), "valid_after": int(clean[column].count())}
        for column in SELECTED
    ])
    save_csv(quality, output, "data_quality")
    save_csv(clean, output, "analysis_data")
    save_csv(excluded, output, "excluded_rows")
    descriptive = pd.DataFrame([
        {"group": "all", **price_summary(clean["price"])},
        *[{"group": level, **price_summary(clean.loc[clean["drive-wheels"] == level, "price"])}
          for level in DRIVE_LEVELS],
    ])
    save_csv(descriptive, output, "descriptive_statistics")
    for column, levels in [("drive-wheels", DRIVE_LEVELS), ("aspiration", ASPIRATION_LEVELS)]:
        counts = clean[column].value_counts().reindex(levels, fill_value=0)
        save_csv(pd.DataFrame({"level": levels, "count": counts.to_numpy(),
                              "percent": counts.to_numpy() / len(clean) * 100}),
                 output, f"frequencies_{column}")
    normality = []
    for _, row in descriptive.iterrows():
        values = clean["price"] if row["group"] == "all" else clean.loc[
            clean["drive-wheels"] == row["group"], "price"]
        w, p = stats.shapiro(values)
        normality.append({"group": row["group"], "count": len(values),
                          "W": w, "p_value": p, "reject_normality": p < ALPHA})
    save_csv(pd.DataFrame(normality), output, "normality_shapiro")
    wide = pd.DataFrame({level: pd.Series(clean.loc[clean["drive-wheels"] == level, "price"].to_numpy())
                         for level in DRIVE_LEVELS})
    save_csv(wide, output, "price_groups")
    return descriptive


def one_way_analysis(clean, output):
    groups = [clean.loc[clean["drive-wheels"] == level, "price"].to_numpy()
              for level in DRIVE_LEVELS]
    n, k = len(clean), len(groups)
    mean = clean["price"].mean()
    ss_between = sum(len(g) * (g.mean() - mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    ss_total = float(np.sum((clean["price"] - mean) ** 2))
    df_between, df_within = k - 1, n - k
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    f = ms_between / ms_within
    p = stats.f.sf(f, df_between, df_within)
    f_critical = stats.f.ppf(1 - ALPHA, df_between, df_within)
    np.testing.assert_allclose(ss_between + ss_within, ss_total, rtol=1e-12)
    scipy_f, scipy_p = stats.f_oneway(*groups)
    np.testing.assert_allclose([f, p], [scipy_f, scipy_p], rtol=1e-11)
    save_csv(pd.DataFrame([
        {"source": "between", "SS": ss_between, "df": df_between,
         "MS": ms_between, "F": f, "p_value": p, "F_critical": f_critical},
        {"source": "within", "SS": ss_within, "df": df_within, "MS": ms_within},
        {"source": "total", "SS": ss_total, "df": n - 1},
    ]), output, "one_way_anova")
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    # Уэлч: отдельная оценка дисперсии для каждой группы, без их объединения.
    sizes = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    weights = sizes / variances
    weighted_mean = np.sum(weights * means) / weights.sum()
    correction = np.sum((1 - weights / weights.sum()) ** 2 / (sizes - 1))
    welch_f = np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)
    welch_f /= 1 + 2 * (k - 2) / (k**2 - 1) * correction
    welch_df2 = (k**2 - 1) / (3 * correction)
    welch_p = stats.f.sf(welch_f, k - 1, welch_df2)
    scipy_welch = stats.f_oneway(*groups, equal_var=False)
    np.testing.assert_allclose([welch_f, welch_p], scipy_welch, rtol=1e-11)
    levene_f, levene_p = stats.levene(*groups, center="median")
    save_csv(pd.DataFrame([{"test": "Levene_median", "statistic": levene_f,
                           "df1": k - 1, "df2": n - k, "p_value": levene_p}]),
             output, "variance_homogeneity")
    save_csv(pd.DataFrame([{"F": welch_f, "df1": k - 1, "df2": welch_df2,
                           "p_value": welch_p,
                           "F_critical": stats.f.ppf(1 - ALPHA, k - 1, welch_df2)}]),
             output, "welch_anova")
    # Уточнение различающихся пар: двусторонние t-тесты Уэлча + поправка Холма.
    pairwise = []
    for i, j in combinations(range(k), 2):
        t, pair_p = stats.ttest_ind(groups[i], groups[j], equal_var=False)
        a, b = variances[i] / sizes[i], variances[j] / sizes[j]
        df = (a + b) ** 2 / (a**2 / (sizes[i] - 1) + b**2 / (sizes[j] - 1))
        pairwise.append({"group1": DRIVE_LEVELS[i], "group2": DRIVE_LEVELS[j],
                         "mean_difference": means[i] - means[j], "t": t,
                         "df": df, "p_value": pair_p})
    pairs = pd.DataFrame(pairwise)
    pairs["reject_holm"], pairs["p_holm"], _, _ = multipletests(pairs["p_value"], alpha=ALPHA, method="holm")
    save_csv(pairs, output, "pairwise_welch_holm")
    return {"F": float(f), "df1": df_between, "df2": df_within, "p_value": float(p),
            "F_critical": float(f_critical), "eta_squared": float(eta_squared),
            "omega_squared": float(omega_squared), "welch_F": float(welch_f),
            "welch_df2": float(welch_df2), "welch_p_value": float(welch_p),
            "levene_F": float(levene_f), "levene_p_value": float(levene_p),
            "variance_ratio": float(variances.max() / variances.min())}


def kruskal_analysis(clean, output):
    n, k = len(clean), len(DRIVE_LEVELS)
    values = clean["price"].to_numpy()
    ranks = stats.rankdata(values, method="average")
    _, tie_counts = np.unique(values, return_counts=True)
    tie_sum = float(np.sum(tie_counts**3 - tie_counts))
    tie_correction = 1 - tie_sum / (n**3 - n)
    rank_rows, groups = [], []
    for level in DRIVE_LEVELS:
        mask = (clean["drive-wheels"] == level).to_numpy()
        rank_rows.append({"group": level, "count": int(mask.sum()),
                          "rank_sum": float(ranks[mask].sum()),
                          "mean_rank": float(ranks[mask].mean())})
        groups.append(values[mask])
    rank_table = pd.DataFrame(rank_rows)
    raw_h = 12 / (n * (n + 1)) * np.sum(rank_table["rank_sum"] ** 2 / rank_table["count"]) - 3 * (n + 1)
    h = raw_h / tie_correction
    p = stats.chi2.sf(h, k - 1)
    np.testing.assert_allclose([h, p], stats.kruskal(*groups), rtol=1e-11)
    save_csv(rank_table, output, "kruskal_ranks")
    save_csv(clean.assign(price_rank=ranks), output, "ranked_data")
    result = {"H": float(h), "H_before_tie_correction": float(raw_h),
              "tie_correction": float(tie_correction), "df": k - 1, "p_value": float(p),
              "chi2_critical": float(stats.chi2.ppf(1 - ALPHA, k - 1)),
              "epsilon_squared": float((h - k + 1) / (n - k))}
    save_csv(pd.DataFrame([result]), output, "kruskal_wallis")
    # Критерий Данна использует общие ранги и поправку на совпадающие цены.
    rank_variance = n * (n + 1) / 12 - tie_sum / (12 * (n - 1))
    rows = []
    for i, j in combinations(range(k), 2):
        a, b = rank_rows[i], rank_rows[j]
        z = (a["mean_rank"] - b["mean_rank"]) / np.sqrt(rank_variance * (1 / a["count"] + 1 / b["count"]))
        rows.append({"group1": a["group"], "group2": b["group"], "z": z,
                     "p_value": 2 * stats.norm.sf(abs(z))})
    pairs = pd.DataFrame(rows)
    pairs["reject_holm"], pairs["p_holm"], _, _ = multipletests(pairs["p_value"], alpha=ALPHA, method="holm")
    save_csv(pairs, output, "pairwise_dunn_holm")
    return result


def two_way_analysis(clean, output):
    counts = pd.crosstab(clean["drive-wheels"], clean["aspiration"]).reindex(
        index=DRIVE_LEVELS, columns=ASPIRATION_LEVELS, fill_value=0)
    if (counts < 2).any().any():
        raise ValueError("Для модели с взаимодействием требуется не менее двух наблюдений в каждой ячейке")
    save_csv(counts, output, "two_way_cell_counts", index=True)
    cells = []
    cell_groups = []
    for drive in DRIVE_LEVELS:
        for aspiration in ASPIRATION_LEVELS:
            values = clean.loc[(clean["drive-wheels"] == drive) & (clean["aspiration"] == aspiration), "price"]
            cell_groups.append(values.to_numpy())
            w, p = stats.shapiro(values) if len(values) >= 3 else (np.nan, np.nan)
            cells.append({"drive": drive, "aspiration": aspiration, **price_summary(values),
                          "shapiro_W": w, "shapiro_p_value": p})
    cell_table = pd.DataFrame(cells)
    save_csv(cell_table, output, "two_way_cell_statistics")
    model_data = clean.rename(columns={"drive-wheels": "drive"})
    model = ols("price ~ C(drive, Sum) * C(aspiration, Sum)", data=model_data).fit()
    if np.linalg.matrix_rank(model.model.exog) != model.model.exog.shape[1]:
        raise ValueError("Матрица двухфакторной модели имеет неполный ранг")
    terms = {"C(drive, Sum)": "drive", "C(aspiration, Sum)": "aspiration",
             "C(drive, Sum):C(aspiration, Sum)": "interaction", "Residual": "residual"}
    classic = anova_lm(model, typ=3)
    robust = anova_lm(model, typ=3, robust="hc3")
    classic = classic.loc[list(terms)].rename(index=terms).rename_axis("effect").reset_index()
    classic = classic.rename(columns={"sum_sq": "SS_partial", "PR(>F)": "p_value"})
    classic["MS"] = classic["SS_partial"] / classic["df"]
    classic["F_critical"] = [stats.f.ppf(1 - ALPHA, df, model.df_resid) if effect != "residual" else np.nan
                             for effect, df in zip(classic["effect"], classic["df"])]
    # При HC3 сохраняем только F и p: robust sum_sq не является разложением разброса.
    robust = robust.loc[list(terms)[:-1], ["df", "F", "PR(>F)"]].rename(index=terms)
    robust = robust.rename_axis("effect").reset_index().rename(columns={"PR(>F)": "p_value"})
    robust["df_residual"] = model.df_resid
    save_csv(classic, output, "two_way_anova_type3")
    save_csv(robust, output, "two_way_anova_hc3")
    coefficients = pd.DataFrame({"coefficient": model.params, "standard_error": model.bse,
                                 "t": model.tvalues, "p_value": model.pvalues})
    save_csv(coefficients, output, "two_way_coefficients", index=True)
    save_csv(clean.assign(fitted=model.fittedvalues.to_numpy(), residual=model.resid.to_numpy()),
             output, "two_way_residuals")
    w, normal_p = stats.shapiro(model.resid)
    lev_f, lev_p = stats.levene(*cell_groups, center="median")
    metrics = {"n": len(clean), "parameters": len(model.params), "df_residual": int(model.df_resid),
               "R_squared": float(model.rsquared), "adjusted_R_squared": float(model.rsquared_adj),
               "SSE": float(model.ssr), "MSE": float(model.mse_resid),
               "residual_shapiro_W": float(w), "residual_shapiro_p_value": float(normal_p),
               "cell_levene_F": float(lev_f), "cell_levene_p_value": float(lev_p),
               "minimum_cell_count": int(counts.to_numpy().min())}
    save_csv(pd.DataFrame([metrics]), output, "two_way_metrics")
    # Независимая сверка частных F: суммы квадратов вложенных моделей NumPy.
    x, y = model.model.exog, model.model.endog
    checks = []
    for term, effect in list(terms.items())[:-1]:
        indices = [i for i, name in enumerate(model.model.exog_names)
                   if (":" in name if effect == "interaction"
                       else term in name and ":" not in name)]
        reduced = np.delete(x, indices, axis=1)
        errors = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        nested_f = ((errors @ errors - model.ssr) / len(indices)) / model.mse_resid
        table_f = classic.loc[classic["effect"] == effect, "F"].iloc[0]
        np.testing.assert_allclose(nested_f, table_f, rtol=1e-9)
        checks.append({"effect": effect, "nested_F": float(nested_f), "type3_F": float(table_f)})
    save_csv(pd.DataFrame(checks), output, "two_way_validation")
    return metrics, cell_table, model


def create_plots(clean, cells, model, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), layout="constrained")
    histogram_rows = []
    for axis, level in zip(axes.flat, ["all", *DRIVE_LEVELS]):
        values = clean["price"] if level == "all" else clean.loc[clean["drive-wheels"] == level, "price"]
        k = int(np.ceil(1 + np.log2(len(values))))
        edges = np.linspace(values.min(), values.max(), k + 1)
        counts, _ = np.histogram(values, edges)
        h = (values.max() - values.min()) / k
        axis.hist(values, bins=edges, edgecolor="white", color="#3576a8")
        title = "Вся выборка" if level == "all" else f"Привод {level}"
        axis.set(title=f"{title}: n = {len(values)}, k = {k}, h = {h:.2f}",
                 xlabel="Цена, долл. США", ylabel="Частота")
        axis.ticklabel_format(axis="x", style="plain")
        axis.grid(axis="y", alpha=0.18)
        histogram_rows.extend({"group": level, "n": len(values), "k": k, "h": h,
                               "bin": i + 1, "left": edges[i], "right": edges[i + 1],
                               "count": int(counts[i]), "right_inclusive": i == k - 1} for i in range(k))
    fig.savefig(output / "price_histograms.png", dpi=180)
    plt.close(fig)
    save_csv(pd.DataFrame(histogram_rows), output, "histogram_bins")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")
    for axis, column, levels, title in zip(axes, ["drive-wheels", "aspiration"],
                                          [DRIVE_LEVELS, ASPIRATION_LEVELS], ["Тип привода", "Тип наддува"]):
        counts = clean[column].value_counts().reindex(levels)
        bars = axis.bar(levels, counts / len(clean) * 100, color="#3576a8")
        axis.bar_label(bars, labels=[f"{c} ({c / len(clean) * 100:.2f}%)" for c in counts], padding=3)
        axis.set(title=title, ylabel="Доля автомобилей, %", ylim=(0, 100))
    fig.savefig(output / "category_frequencies.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), layout="constrained")
    for axis, level in zip(axes, DRIVE_LEVELS):
        values = clean.loc[clean["drive-wheels"] == level, "price"]
        stats.probplot(values, dist="norm", plot=axis)
        axis.set(title=f"Q–Q: {level}", xlabel="Теоретические квантили", ylabel="Цена, долл. США")
    fig.savefig(output / "price_qq_by_drive.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    groups = [clean.loc[clean["drive-wheels"] == level, "price"].to_numpy() for level in DRIVE_LEVELS]
    boxes = axis.boxplot(groups, tick_labels=DRIVE_LEVELS, patch_artist=True, showmeans=True)
    for patch in boxes["boxes"]:
        patch.set_facecolor("#d6e6f1")
    axis.set(title="Цена по типу привода", xlabel="Тип привода", ylabel="Цена, долл. США")
    axis.grid(axis="y", alpha=0.18)
    fig.savefig(output / "price_boxplot.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for aspiration, color in zip(ASPIRATION_LEVELS, ["#3576a8", "#d77b33"]):
        selected = cells.loc[cells["aspiration"] == aspiration].set_index("drive").loc[DRIVE_LEVELS]
        axis.plot(DRIVE_LEVELS, selected["mean"], marker="o", label=aspiration, color=color)
        for i, (_, row) in enumerate(selected.iterrows()):
            axis.annotate(f"n={int(row['count'])}", (i, row["mean"]), xytext=(5, 8), textcoords="offset points", fontsize=9)
    axis.set(title="Средняя цена: привод × наддув", xlabel="Тип привода", ylabel="Средняя цена, долл. США")
    axis.legend(title="Наддув")
    axis.grid(alpha=0.18)
    fig.savefig(output / "two_way_interaction.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout="constrained")
    axes[0].scatter(model.fittedvalues, model.resid, alpha=0.6, color="#3576a8")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Остатки двухфакторной модели", xlabel="Предсказанная цена", ylabel="Остаток")
    stats.probplot(model.resid, dist="norm", plot=axes[1])
    axes[1].set(title="Q–Q остатков", xlabel="Теоретические квантили", ylabel="Остаток")
    fig.savefig(output / "two_way_diagnostics.png", dpi=180)
    plt.close(fig)


def run_analysis(data_path=BASE.parent / "lab1/data/imports-85.data", output=BASE / "results"):
    data_path, output = Path(data_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw, clean, excluded = load_data(data_path)
    descriptive_analysis(raw, clean, excluded, output)
    one_way = one_way_analysis(clean, output)
    kruskal = kruskal_analysis(clean, output)
    two_way, cells, model = two_way_analysis(clean, output)
    create_plots(clean, cells, model, output)
    summary = {"dataset": "1985 Auto Imports (Automobile)",
               "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "alpha": ALPHA, "total_rows": len(raw), "analysis_rows": len(clean),
               "excluded_rows": len(excluded), "dependent_variable": "price",
               "factor1": "drive-wheels", "factor2": "aspiration", "one_way": one_way,
               "kruskal": kruskal, "two_way": two_way,
               "verification": {"one_way_manual_vs_scipy": "passed",
                                "welch_manual_vs_scipy": "passed",
                                "kruskal_manual_vs_scipy": "passed",
                                "two_way_nested_models_vs_type3": "passed"},
               "packages": {name: importlib.metadata.version(name)
                            for name in ["numpy", "pandas", "scipy", "matplotlib", "statsmodels", "python-docx"]}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE.parent / "lab1/data/imports-85.data")
    parser.add_argument("--output", type=Path, default=BASE / "results")
    args = parser.parse_args()
    summary = run_analysis(args.data, args.output)
    print(f"Обработано {summary['analysis_rows']} из {summary['total_rows']} строк. Результаты: {args.output}")
    print(f"ANOVA: F={summary['one_way']['F']:.6f}; p={summary['one_way']['p_value']:.6g}")
    print(f"Краскел–Уоллис: H={summary['kruskal']['H']:.6f}; p={summary['kruskal']['p_value']:.6g}")


if __name__ == "__main__":
    main()
