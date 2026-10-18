"""Два дополнительных метода: порядковые logit- и probit-регрессии.

Запуск: python lab1.2_opt/ordinal_analysis.py
Данные, результаты и случайные числа фиксированы для воспроизводимости.
"""

from __future__ import annotations

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
from statsmodels.miscmodels.ordinal_model import OrderedModel

RESULTS = BASE / "results"
LEVELS = np.array([-2, -1, 0, 1, 2, 3])
LINKS = ("logit", "probit")
SEED = 12026
COLUMNS = (
    "symboling normalized-losses make fuel-type aspiration num-of-doors "
    "body-style drive-wheels engine-location wheel-base length width height "
    "curb-weight engine-type num-of-cylinders engine-size fuel-system bore "
    "stroke compression-ratio horsepower peak-rpm city-mpg highway-mpg price"
).split()


def csv(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False, encoding="utf-8-sig")


def load_data():
    raw = pd.read_csv(BASE / "data/imports-85.data", header=None, na_values="?")
    assert raw.shape == (205, 26)
    raw.columns = COLUMNS
    data = raw[["symboling", "curb-weight", "drive-wheels"]].copy()
    assert not data.isna().any().any()
    assert np.array_equal(np.sort(data.symboling.unique()), LEVELS)
    assert set(data["drive-wheels"]) == {"4wd", "fwd", "rwd"}
    data.insert(0, "source_row", np.arange(1, len(data) + 1))
    x = pd.DataFrame({
        "weight_100": data["curb-weight"] / 100,
        "drive_fwd": (data["drive-wheels"] == "fwd").astype(float),
        "drive_rwd": (data["drive-wheels"] == "rwd").astype(float),
    })
    return data, x


def fit(y, x, link):
    # Константу нельзя включать: её роль выполняют пять порогов.
    model = OrderedModel(y, x, distr=link)
    result = model.fit(method="bfgs", disp=False, maxiter=1000, gtol=1e-6)
    if not result.mle_retvals["converged"]:
        result = model.fit(start_params=result.params, method="lbfgs", disp=False,
                           maxiter=2000, pgtol=1e-7)
    if not result.mle_retvals["converged"]:
        raise RuntimeError(f"{link}: модель не сошлась")
    if not np.isfinite(result.bse).all():
        raise RuntimeError(f"{link}: некорректная ковариационная матрица")
    return result


def check_probabilities(probabilities):
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-10)


def diagnostics(y, probabilities, eta, random_uniform):
    codes = np.searchsorted(LEVELS, np.asarray(y))
    cdf = np.column_stack([np.zeros(len(y)), probabilities.cumsum(axis=1)])
    lower, upper = cdf[np.arange(len(y)), codes], cdf[np.arange(len(y)), codes + 1]
    quantile = lower + random_uniform * (upper - lower)
    residual = stats.norm.ppf(np.clip(quantile, 1e-12, 1 - 1e-12))
    frame = pd.DataFrame({"source_row": np.arange(1, len(y) + 1), "observed": y,
                          "eta": eta, "predicted_category": LEVELS[probabilities.argmax(axis=1)],
                          "quantile_residual": residual})
    for i, level in enumerate(LEVELS):
        frame[f"p_{level}"] = probabilities[:, i]
    frame["fitted_quartile"] = pd.qcut(frame.eta, 4, labels=[1, 2, 3, 4])
    groups = frame.groupby("fitted_quartile", observed=True).quantile_residual.agg(
        ["count", "mean", "var"]).reset_index()
    return frame, groups


def plot_diagnostics(frame, groups, link):
    title = "Порядковая логистическая регрессия" if link == "logit" else "Порядковая пробит-регрессия"
    r = frame.quantile_residual.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    axes[0].scatter(frame.eta, r, alpha=.65, s=22)
    axes[0].axhline(0, color="black", linewidth=.8)
    for group in groups.itertuples():
        selected = frame[frame.fitted_quartile == group.fitted_quartile]
        axes[0].scatter(selected.eta.mean(), group.mean, marker="D", c="#b63c31", s=45)
    axes[0].set(xlabel="Линейный предиктор η", ylabel="Квантильный остаток",
                title="Остатки и средние по квартилям")
    k = int(np.ceil(1 + np.log2(len(r))))
    axes[1].hist(r, bins=k, density=True, color="#5486aa", edgecolor="white")
    grid = np.linspace(-3.5, 3.5, 300)
    axes[1].plot(grid, stats.norm.pdf(grid), color="#b63c31", label="N(0, 1)")
    axes[1].set(xlabel="Квантильный остаток", ylabel="Плотность", title=f"Гистограмма: {k} интервалов")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle(title, fontsize=14)
    fig.savefig(RESULTS / f"{link}_residuals.png", dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    data, x = load_data()
    y = data.symboling
    csv(data, "analysis_data")
    frequencies = np.array([(y == level).mean() for level in LEVELS])
    null_log_likelihood = float(np.sum(np.log(frequencies[np.searchsorted(LEVELS, y)])))
    common_uniform = np.random.default_rng(SEED).uniform(size=len(y))
    summary = {"n": len(y), "levels": LEVELS.tolist(), "seed": SEED,
               "null_log_likelihood": null_log_likelihood,
               "models": {}}
    for link in LINKS:
        result = fit(y, x, link)
        p = np.asarray(result.model.predict(result.params))
        check_probabilities(p)
        thresholds = result.model.transform_threshold_params(result.params)[1:-1]
        assert np.all(np.diff(thresholds) > 0)
        coefficient_rows = []
        for j, name in enumerate(x.columns):
            b, se = float(result.params.iloc[j]), float(result.bse.iloc[j])
            coefficient_rows.append({"parameter": name, "coefficient": b, "standard_error": se,
                                     "z": b / se, "p_value": float(result.pvalues.iloc[j]),
                                     "odds_ratio": np.exp(b) if link == "logit" else np.nan})
        csv(pd.DataFrame(coefficient_rows), f"{link}_coefficients")
        csv(pd.DataFrame({"boundary_after": LEVELS[:-1], "threshold": thresholds}), f"{link}_thresholds")
        lr = float(2 * (result.llf - null_log_likelihood))
        eta = x.to_numpy() @ np.asarray(result.params[:3])
        frame, groups = diagnostics(y, p, eta, common_uniform)
        csv(frame, f"{link}_diagnostics")
        csv(groups, f"{link}_residual_groups")
        residual = frame.quantile_residual.to_numpy()
        metrics = {
            "observations": len(y), "parameters": len(result.params),
            "converged": bool(result.mle_retvals["converged"]), "log_likelihood": float(result.llf),
            "aic": float(result.aic),
            "mcfadden_pseudo_r_squared": float(1 - result.llf / null_log_likelihood),
            "likelihood_ratio": lr, "lr_df": 3, "model_p_value": float(stats.chi2.sf(lr, 3)),
            "residual_mean": float(residual.mean()), "residual_variance": float(residual.var(ddof=1)),
            "max_group_abs_mean": float(groups["mean"].abs().max()),
            "group_variance_ratio": float(groups["var"].max() / groups["var"].min()),
        }
        summary["models"][link] = metrics
        csv(pd.DataFrame([metrics]), f"{link}_metrics")
        plot_diagnostics(frame, groups, link)
        print(f"{link}: LL={result.llf:.6f}; AIC={result.aic:.6f}; LR p={metrics['model_p_value']:.6g}", flush=True)
    csv(pd.DataFrame(summary["models"]).T.reset_index(names="model"), "model_comparison")
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Расчёты двух моделей завершены.", flush=True)


if __name__ == "__main__":
    main()
