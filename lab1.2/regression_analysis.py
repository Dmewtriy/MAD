"""Расчёты для лабораторной работы 1.2 «Регрессионный анализ»."""

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
from scipy.special import gammaln, xlogy


COLUMNS = (
    "symboling normalized-losses make fuel-type aspiration num-of-doors "
    "body-style drive-wheels engine-location wheel-base length width height "
    "curb-weight engine-type num-of-cylinders engine-size fuel-system bore "
    "stroke compression-ratio horsepower peak-rpm city-mpg highway-mpg price"
).split()
SELECTED = ["symboling", "drive-wheels", "curb-weight"]
DRIVE_LEVELS = ["4wd", "fwd", "rwd"]
ALPHA = 0.05


def save_csv(frame, output, name, index=True):
    frame.to_csv(output / f"{name}.csv", index=index, encoding="utf-8-sig")


def load_data(path):
    raw = pd.read_csv(path, header=None, na_values="?")
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"Ожидалось 26 столбцов, получено {raw.shape[1]}")
    raw.columns = COLUMNS
    data = raw[SELECTED].copy()
    data["symboling"] = pd.to_numeric(data["symboling"], errors="raise")
    data["curb-weight"] = pd.to_numeric(data["curb-weight"], errors="raise")
    if data.isna().any().any():
        raise ValueError("В выбранных переменных обнаружены пропуски")
    if not data["symboling"].isin(range(-3, 4)).all():
        raise ValueError("Недопустимые значения symboling")
    if not data["drive-wheels"].isin(DRIVE_LEVELS).all():
        raise ValueError("Недопустимые значения drive-wheels")
    return raw, data


def design_matrix(data, nonlinear=False):
    weight_100 = data["curb-weight"].to_numpy(dtype=float) / 100
    columns = [np.ones(len(data)), weight_100]
    names = ["intercept", "curb_weight_100"]
    if nonlinear:
        columns.append(weight_100**2)
        names.append("curb_weight_100_squared")
    columns.extend(
        [
            (data["drive-wheels"] == "fwd").to_numpy(dtype=float),
            (data["drive-wheels"] == "rwd").to_numpy(dtype=float),
        ]
    )
    names.extend(["drive_fwd", "drive_rwd"])
    return np.column_stack(columns), names


def ols_model(data, nonlinear=False):
    response = data["symboling"].to_numpy(dtype=float)
    matrix, names = design_matrix(data, nonlinear)
    coefficients = np.linalg.lstsq(matrix, response, rcond=None)[0]
    fitted = matrix @ coefficients
    residuals = response - fitted
    observations, parameters = matrix.shape
    residual_df = observations - parameters
    sse = float(residuals @ residuals)
    tss = float(((response - response.mean()) ** 2).sum())
    mse = sse / residual_df
    covariance = mse * np.linalg.pinv(matrix.T @ matrix)
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = coefficients / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), residual_df)
    leverage = np.diag(matrix @ np.linalg.pinv(matrix.T @ matrix) @ matrix.T)
    standardized = residuals / np.sqrt(mse * np.maximum(1 - leverage, 1e-12))
    r_squared = 1 - sse / tss
    adjusted_r_squared = 1 - (1 - r_squared) * (observations - 1) / residual_df
    model_df = parameters - 1
    f_statistic = ((tss - sse) / model_df) / mse
    model_p_value = stats.f.sf(f_statistic, model_df, residual_df)
    coefficient_table = pd.DataFrame(
        {
            "coefficient": coefficients,
            "standard_error": standard_errors,
            "statistic": t_values,
            "p_value": p_values,
            "significant_0.05": p_values < ALPHA,
        },
        index=names,
    )
    metrics = {
        "observations": observations,
        "parameters": parameters,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "rmse": np.sqrt(np.mean(residuals**2)),
        "mae": np.mean(np.abs(residuals)),
        "f_statistic": f_statistic,
        "model_p_value": model_p_value,
        "residual_mean": residuals.mean(),
        "residual_variance": residuals.var(ddof=1),
    }
    diagnostics = pd.DataFrame(
        {"observed": response, "fitted": fitted, "residual": residuals, "standardized_residual": standardized}
    )
    return coefficient_table, metrics, diagnostics


def poisson_deviance(response, fitted):
    logarithmic = xlogy(response, response / fitted)
    return float(2 * np.sum(logarithmic - (response - fitted)))


def poisson_model(data, tolerance=1e-10, max_iterations=200):
    response = (data["symboling"] + 2).to_numpy(dtype=float)
    matrix, names = design_matrix(data)
    coefficients = np.zeros(matrix.shape[1])
    coefficients[0] = np.log(response.mean())
    converged = False
    for iteration in range(1, max_iterations + 1):
        linear_predictor = np.clip(matrix @ coefficients, -20, 20)
        fitted = np.exp(linear_predictor)
        adjusted = linear_predictor + (response - fitted) / fitted
        weighted_matrix = matrix * np.sqrt(fitted)[:, None]
        weighted_response = adjusted * np.sqrt(fitted)
        updated = np.linalg.lstsq(weighted_matrix, weighted_response, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < tolerance:
            coefficients = updated
            converged = True
            break
        coefficients = updated
    fitted = np.exp(np.clip(matrix @ coefficients, -20, 20))
    covariance = np.linalg.pinv(matrix.T @ (matrix * fitted[:, None]))
    standard_errors = np.sqrt(np.diag(covariance))
    z_values = coefficients / standard_errors
    p_values = 2 * stats.norm.sf(np.abs(z_values))
    log_likelihood = float(np.sum(response * np.log(fitted) - fitted - gammaln(response + 1)))
    null_mean = response.mean()
    null_log_likelihood = float(np.sum(response * np.log(null_mean) - null_mean - gammaln(response + 1)))
    likelihood_ratio = 2 * (log_likelihood - null_log_likelihood)
    model_df = matrix.shape[1] - 1
    model_p_value = stats.chi2.sf(likelihood_ratio, model_df)
    pearson_residual = (response - fitted) / np.sqrt(fitted)
    coefficient_table = pd.DataFrame(
        {
            "coefficient": coefficients,
            "standard_error": standard_errors,
            "statistic": z_values,
            "p_value": p_values,
            "significant_0.05": p_values < ALPHA,
            "exp_coefficient": np.exp(coefficients),
        },
        index=names,
    )
    metrics = {
        "observations": len(response),
        "parameters": matrix.shape[1],
        "converged": converged,
        "iterations": iteration,
        "log_likelihood": log_likelihood,
        "deviance": poisson_deviance(response, fitted),
        "aic": 2 * matrix.shape[1] - 2 * log_likelihood,
        "mcfadden_pseudo_r_squared": 1 - log_likelihood / null_log_likelihood,
        "likelihood_ratio": likelihood_ratio,
        "model_p_value": model_p_value,
        "rmse_shifted_scale": np.sqrt(np.mean((response - fitted) ** 2)),
        "residual_mean": pearson_residual.mean(),
        "residual_variance": pearson_residual.var(ddof=1),
    }
    diagnostics = pd.DataFrame(
        {
            "observed_shifted": response,
            "fitted_shifted": fitted,
            "observed_symboling": response - 2,
            "fitted_symboling": fitted - 2,
            "residual": response - fitted,
            "standardized_residual": pearson_residual,
        }
    )
    return coefficient_table, metrics, diagnostics


def save_model(name, coefficients, metrics, diagnostics, output):
    save_csv(coefficients, output, f"{name}_coefficients")
    save_csv(pd.DataFrame([metrics], index=[name]), output, f"{name}_metrics")
    save_csv(diagnostics, output, f"{name}_diagnostics", index=False)
    fitted_column = "fitted_shifted" if "fitted_shifted" in diagnostics else "fitted"
    groups = pd.qcut(diagnostics[fitted_column], q=4, duplicates="drop")
    residual_groups = diagnostics.assign(fitted_group=groups).groupby(
        "fitted_group", observed=True
    )["residual"].agg(["count", "mean", "var"])
    residual_groups.index = residual_groups.index.astype(str)
    save_csv(residual_groups, output, f"{name}_residual_groups")


def plot_model_diagnostics(models, output):
    for name, title, diagnostics in models:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
        axes[0].scatter(diagnostics["fitted"], diagnostics["residual"], alpha=0.7)
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set(title=f"{title}: остатки", xlabel="Предсказанные значения", ylabel="Остаток")
        axes[0].grid(alpha=0.2)
        bins = int(np.ceil(1 + np.log2(len(diagnostics))))
        axes[1].hist(diagnostics["standardized_residual"], bins=bins, edgecolor="white")
        axes[1].set(title=f"{title}: стандартизированные остатки", xlabel="Стандартизированный остаток", ylabel="Частота")
        fig.savefig(output / f"{name}_residuals.png", dpi=170)
        plt.close(fig)


def descriptive_analysis(data, output):
    quality = pd.DataFrame(
        {
            "type": ["ordinal", "nominal", "quantitative"],
            "count": data.count(), "missing": data.isna().sum(), "unique": data.nunique(),
        },
        index=SELECTED,
    )
    weight = data["curb-weight"]
    descriptive = pd.DataFrame(
        {
            "count": [len(weight)], "mean": [weight.mean()], "median": [weight.median()],
            "mode": [", ".join(map(str, weight.mode()))], "minimum": [weight.min()],
            "maximum": [weight.max()], "range": [weight.max() - weight.min()],
            "variance": [weight.var(ddof=1)], "standard_deviation": [weight.std(ddof=1)],
        },
        index=["curb-weight"],
    )
    save_csv(quality, output, "data_quality")
    save_csv(descriptive, output, "descriptive_statistics")
    for column, levels in (("symboling", range(-3, 4)), ("drive-wheels", DRIVE_LEVELS)):
        counts = data[column].value_counts().reindex(levels, fill_value=0)
        save_csv(pd.DataFrame({"count": counts, "percent": counts / len(data) * 100}), output, f"frequencies_{column}")

    intervals = int(np.ceil(1 + np.log2(len(data))))
    edges = np.linspace(weight.min(), weight.max(), intervals + 1)
    counts, _ = np.histogram(weight, bins=edges)
    save_csv(pd.DataFrame({"interval": range(1, intervals + 1), "left": edges[:-1], "right": edges[1:], "count": counts}), output, "histogram_bins", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    axes[0].hist(weight, bins=edges, edgecolor="white")
    axes[0].set(title=f"curb-weight: k={intervals}", xlabel="curb-weight", ylabel="Частота")
    symbol_counts = data["symboling"].value_counts().reindex(range(-3, 4), fill_value=0)
    axes[1].bar(symbol_counts.index.astype(str), symbol_counts / len(data) * 100)
    axes[1].set(title="symboling", xlabel="Категория", ylabel="Доля, %")
    drive_counts = data["drive-wheels"].value_counts().reindex(DRIVE_LEVELS, fill_value=0)
    axes[2].bar(drive_counts.index, drive_counts / len(data) * 100)
    axes[2].set(title="drive-wheels", xlabel="Тип привода", ylabel="Доля, %")
    fig.savefig(output / "selected_variables_histograms.png", dpi=170)
    plt.close(fig)


def run_analysis(data_path, output):
    output.mkdir(parents=True, exist_ok=True)
    _, data = load_data(data_path)
    descriptive_analysis(data, output)
    linear = ols_model(data)
    nonlinear = ols_model(data, nonlinear=True)
    poisson = poisson_model(data)
    save_model("linear", *linear, output)
    save_model("nonlinear", *nonlinear, output)
    save_model("poisson", *poisson, output)
    poisson_plot = poisson[2].rename(columns={"fitted_shifted": "fitted"})
    comparison = pd.DataFrame(
        [
            {
                "model": "linear", "quality_measure": "R-squared",
                "quality_value": linear[1]["r_squared"],
                "adjusted_r_squared": linear[1]["adjusted_r_squared"],
                "rmse": linear[1]["rmse"], "aic": np.nan,
                "model_p_value": linear[1]["model_p_value"],
                "significant_0.05": linear[1]["model_p_value"] < ALPHA,
            },
            {
                "model": "nonlinear", "quality_measure": "R-squared",
                "quality_value": nonlinear[1]["r_squared"],
                "adjusted_r_squared": nonlinear[1]["adjusted_r_squared"],
                "rmse": nonlinear[1]["rmse"], "aic": np.nan,
                "model_p_value": nonlinear[1]["model_p_value"],
                "significant_0.05": nonlinear[1]["model_p_value"] < ALPHA,
            },
            {
                "model": "poisson", "quality_measure": "McFadden pseudo-R-squared",
                "quality_value": poisson[1]["mcfadden_pseudo_r_squared"],
                "adjusted_r_squared": np.nan,
                "rmse": poisson[1]["rmse_shifted_scale"], "aic": poisson[1]["aic"],
                "model_p_value": poisson[1]["model_p_value"],
                "significant_0.05": poisson[1]["model_p_value"] < ALPHA,
            },
        ]
    ).set_index("model")
    save_csv(comparison, output, "model_comparison")
    plot_model_diagnostics(
        [
            ("linear", "Линейная модель", linear[2]),
            ("nonlinear", "Нелинейная модель", nonlinear[2]),
            ("poisson", "Пуассоновская модель", poisson_plot),
        ],
        output,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE.parent / "lab1/data/imports-85.data")
    parser.add_argument("--output", type=Path, default=BASE / "results")
    args = parser.parse_args()
    print(f"Таблицы и графики: {run_analysis(args.data, args.output)}")


if __name__ == "__main__":
    main()
