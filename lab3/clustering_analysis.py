"""Анализ Bank Marketing: lab3/.venv/bin/python lab3/clustering_analysis.py."""

from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import time

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".venv/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from kmodes.kmodes import KModes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats.contingency import association
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix

from vendor.pyclope import CLOPE

SEED = 42
RESULTS = BASE / "results"

NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
NAMES = {
    "age": "Возраст", "job": "Профессия", "marital": "Семейное положение",
    "education": "Образование", "default": "Просроченный кредит", "balance": "Баланс, евро",
    "housing": "Ипотечный кредит", "loan": "Потребительский кредит", "contact": "Способ контакта",
    "day": "День месяца", "month": "Месяц", "duration": "Длительность, с",
    "campaign": "Контакты в кампании", "pdays": "Дней с прошлого контакта",
    "previous": "Предыдущие контакты", "poutcome": "Исход прошлой кампании",
}


def load_raw_data():
    """Load the saved UCI data, downloading it only if the local CSV is absent."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "bank-full.csv"
    if not path.exists():
        url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
        with urlopen(url, timeout=90) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as outer:
            member = next(name for name in outer.namelist() if name.endswith("bank.zip"))
            with ZipFile(BytesIO(outer.read(member))) as inner:
                for name in ["bank-full.csv", "bank-names.txt"]:
                    (RESULTS / name).write_bytes(inner.read(name))
    return pd.read_csv(path, sep=";")


def configure_plots():
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.titleweight": "bold"})


def load_prepared():
    raw = load_raw_data()
    frame = pd.read_csv(RESULTS / "prepared_categorical.csv", dtype=str)
    codes = np.column_stack([pd.Categorical(frame[c]).codes for c in frame]).astype(np.int16)
    return raw, frame, codes


def prepare_data():
    RESULTS.mkdir(exist_ok=True)
    raw = load_raw_data()
    assert raw.shape == (45211, 17)
    assert raw.isna().sum().sum() == 0
    frame = raw.drop(columns="y").copy()
    q = raw.duration.quantile([0.25, 0.5, 0.75]).to_numpy().tolist()
    rules = {
        "age": ([-np.inf, 30, 45, 60, np.inf], ["18-30", "31-45", "46-60", "61+"]),
        "balance": ([-np.inf, -1, 0, 1000, 5000, np.inf], ["negative", "zero", "1-1000", "1001-5000", "5001+"]),
        "day": ([-np.inf, 10, 20, np.inf], ["1-10", "11-20", "21-31"]),
        "duration": ([-np.inf, *q, np.inf], ["Q1", "Q2", "Q3", "Q4"]),
        "campaign": ([-np.inf, 1, 2, 5, np.inf], ["1", "2", "3-5", "6+"]),
        "pdays": ([-np.inf, -1, 90, 180, 365, np.inf], ["not_contacted", "0-90", "91-180", "181-365", "366+"]),
        "previous": ([-np.inf, 0, 1, 3, np.inf], ["0", "1", "2-3", "4+"]),
    }
    bins = []
    for col, (edges, labels) in rules.items():
        frame[col] = pd.cut(raw[col], bins=edges, labels=labels, right=True).astype(str)
        for i, label in enumerate(labels):
            bins.append({"feature": col, "category": label, "left_exclusive": str(edges[i]),
                         "right_inclusive": str(edges[i+1]), "count": int((frame[col] == label).sum())})
    assert not frame.isna().any().any()
    frame.to_csv(RESULTS / "prepared_categorical.csv", index=False)
    pd.DataFrame(bins).to_csv(RESULTS / "binning_rules.csv", index=False)
    raw[NUMERIC].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T.to_csv(RESULTS / "numeric_statistics.csv", index_label="feature")
    freq = []
    for col in frame:
        for level, count in frame[col].value_counts().items():
            freq.append({"feature": col, "category": level, "count": int(count), "percent": 100*count/len(frame)})
    pd.DataFrame(freq).to_csv(RESULTS / "category_frequencies.csv", index=False)
    quality = pd.DataFrame([{"feature": c, "type": "numeric" if c in NUMERIC else "categorical",
                             "raw_levels": raw[c].nunique(), "prepared_levels": frame[c].nunique(),
                             "missing": int(raw[c].isna().sum()), "unknown": int((raw[c] == "unknown").sum())}
                            for c in frame])
    quality.to_csv(RESULTS / "feature_quality.csv", index=False)
    target = raw.y.value_counts().rename_axis("class").reset_index(name="count")
    target["percent"] = target["count"] / len(raw) * 100
    target.to_csv(RESULTS / "class_balance.csv", index=False)
    assoc = pd.DataFrame(np.eye(len(frame.columns)), index=frame.columns, columns=frame.columns)
    pairs = []
    for i, col in enumerate(frame.columns):
        for other in frame.columns[i+1:]:
            v = float(association(pd.crosstab(frame[col], frame[other]).to_numpy(), method="cramer", correction=False))
            assoc.loc[col, other] = assoc.loc[other, col] = v
            pairs.append({"feature_1": col, "feature_2": other, "cramers_v": v})
    assoc.to_csv(RESULTS / "feature_associations.csv", index_label="feature")
    pd.DataFrame(pairs).sort_values("cramers_v", ascending=False).to_csv(RESULTS / "association_pairs.csv", index=False)
    metadata = {"n": len(raw), "p": len(frame.columns), "numeric": NUMERIC,
                "categorical": [c for c in frame if c not in NUMERIC], "target": "y",
                "duplicates_raw": int(raw.duplicated().sum()), "duplicates_features": int(raw.drop(columns="y").duplicated().sum()),
                "duplicates_after_binning": int(frame.duplicated().sum()), "missing_cells": int(raw.isna().sum().sum()),
                "unknown_rows": int((raw == "unknown").any(axis=1).sum()),
                "duration_quartiles": q, "positive_rate": float((raw.y == "yes").mean()),
                "prepared_levels": {c: sorted(frame[c].unique().tolist()) for c in frame},
                "preparation": "all 16 features, seven numeric features discretized; target excluded; unknown retained"}
    (RESULTS / "data_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")




def normalize_labels(labels):
    counts = Counter(labels)
    order = sorted(counts, key=lambda c: (-counts[c], int(c)))
    mapping = {label: idx for idx, label in enumerate(order, 1)}
    return np.array([mapping[label] for label in labels], dtype=np.int32), mapping


def fit_clope(X, r, seed=SEED):
    """Supply the callback interface expected by HECTOPK/pyclope.

    All add/remove, profit-change and cluster-assignment decisions belong to
    the third-party module. This adapter only delivers rows and stores labels.
    """
    if len(X) == 0 or r <= 1:
        raise ValueError("CLOPE needs data and repulsion > 1")
    order = np.random.default_rng(seed).permutation(len(X))
    transactions = [tuple(f"{j}={int(v)}" for j, v in enumerate(row)) for row in X]
    labels = np.full(len(X), -1, dtype=np.int32)
    cursor = 0

    def next_transaction():
        nonlocal cursor
        idx = int(order[cursor % len(X)])
        cursor += 1
        return idx, transactions[idx], int(labels[idx])

    def save_assignment(idx, cluster):
        labels[idx] = cluster

    start = time.perf_counter()
    model = CLOPE(len(X), next_transaction, save_assignment, r)
    model.exec()
    elapsed = time.perf_counter() - start
    if np.any(labels < 0):
        raise AssertionError("Unassigned transactions")
    normalized, mapping = normalize_labels(labels)
    clusters = []
    for old, new in mapping.items():
        c = model.clusters[int(old)]
        # Independently reconcile every item counter with the returned labels.
        actual = Counter(item for idx in np.flatnonzero(labels == old) for item in transactions[idx])
        assert actual == c.Occ
        assert c.N == int(np.sum(labels == old))
        assert c.S == sum(actual.values()) and c.W == len(actual)
        clusters.append({"cluster": new, "n": c.N, "S": c.S, "W": c.W,
                         "H": c.S/c.W, "profit_contribution": c.S*c.N/(c.W**r)/len(X)})
    assert sum(c.N for c in model.clusters) == len(X)
    profit = sum(row["profit_contribution"] for row in clusters)
    return normalized, {"r": r, "seed": seed, "fit_seconds": elapsed,
                        "iterations": model.n_iter_, "objective": profit,
                        "clusters": sorted(clusters, key=lambda row: row["cluster"])}


def fit_kmodes(X, k, seed=SEED, n_init=3):
    start = time.perf_counter()
    model = KModes(n_clusters=k, init="Huang", n_init=n_init, max_iter=50,
                   random_state=seed, n_jobs=1, verbose=0)
    predicted = model.fit_predict(X)
    labels, mapping = normalize_labels(predicted)
    elapsed = time.perf_counter()-start
    # Keep fitted modes: a componentwise mode with tied frequencies need not
    # equal a newly computed mode selected using a different tie convention.
    centers = {mapping[i]: np.asarray(center, dtype=int).tolist()
               for i, center in enumerate(model.cluster_centroids_) if i in mapping}
    cost = sum(np.sum(X[labels == c] != np.array(mode)) for c, mode in centers.items())
    assert np.isclose(cost, model.cost_)
    return labels, {"k": k, "seed": seed, "n_init": n_init, "init": "Huang",
                    "fit_seconds": elapsed, "iterations": model.n_iter_,
                    "objective": float(model.cost_), "modes_encoded": centers}, model


def score_partition(labels, idx, distances):
    counts = np.unique(labels, return_counts=True)[1]
    n_clusters = len(counts)
    sampled_clusters = len(np.unique(labels[idx]))
    sil = float(silhouette_score(distances, labels[idx], metric="precomputed")) if 1 < sampled_clusters < len(idx) else np.nan
    return {"n_clusters": n_clusters, "silhouette_selection": sil, "min_size": int(counts.min()),
            "max_size": int(counts.max()), "small_cluster_fraction": float(counts[counts < 0.01*len(labels)].sum()/len(labels)),
            "sampled_clusters": sampled_clusters}


def purity(y, labels):
    table = contingency_matrix(y, labels)
    return float(table.max(axis=0).sum()/table.sum())


def run_method(method, X, parameter, seed=SEED):
    result = fit_clope(X, parameter, seed) if method == "CLOPE" else fit_kmodes(X, int(parameter), seed)
    print(method, parameter, "seed", seed, "clusters", len(np.unique(result[0])),
          "seconds", round(result[1]["fit_seconds"], 2), flush=True)
    return result


def cluster_profiles(raw, frame, X, labels, method, info):
    summaries, proportions, modes, importance = [], [], [], []
    overall = {c: frame[c].value_counts(normalize=True) for c in frame}
    distance_to_mode = np.zeros(len(X), dtype=float)
    for cluster in np.unique(labels):
        mask = labels == cluster
        cluster_frame = frame.loc[mask]
        center = []
        for j, col in enumerate(frame):
            values, counts = np.unique(X[mask, j], return_counts=True)
            center.append(int(values[counts.argmax()]))
            frequencies = cluster_frame[col].value_counts(normalize=True)
            modes.append({"method": method, "cluster": int(cluster), "feature": col,
                          "mode": frequencies.index[0], "mode_percent": float(100*frequencies.iloc[0])})
            for category, fraction in frequencies.items():
                proportions.append({"method": method, "cluster": int(cluster), "feature": col,
                                    "category": category, "fraction": float(fraction),
                                    "overall_fraction": float(overall[col][category]),
                                    "lift": float(fraction/overall[col][category]),
                                    "difference_pp": float(100*(fraction-overall[col][category]))})
        if method == "k-modes":
            saved = info["modes_encoded"]
            center = saved.get(int(cluster), saved.get(str(cluster)))
        distances = np.mean(X[mask] != np.asarray(center), axis=1)
        distance_to_mode[mask] = distances
        summary = {"method": method, "cluster": int(cluster), "n": int(mask.sum()),
                   "percent": float(100*mask.mean()), "yes_count": int((raw.loc[mask, "y"] == "yes").sum()),
                   "yes_percent": float(100*(raw.loc[mask, "y"] == "yes").mean()),
                   "mean_distance_to_mode": float(distances.mean()),
                   "age_median": float(raw.loc[mask, "age"].median()),
                   "balance_median": float(raw.loc[mask, "balance"].median()),
                   "duration_median": float(raw.loc[mask, "duration"].median()),
                   "housing_yes_percent": float(100*(raw.loc[mask, "housing"] == "yes").mean()),
                   "previous_zero_percent": float(100*(raw.loc[mask, "previous"] == 0).mean())}
        summaries.append(summary)
    for col in frame:
        v = association(pd.crosstab(frame[col], labels).to_numpy(), method="cramer", correction=False)
        importance.append({"method": method, "feature": col, "cramers_v": float(v)})
    pd.DataFrame(summaries).to_csv(RESULTS/f"{method}_cluster_profiles.csv", index=False)
    pd.DataFrame(proportions).to_csv(RESULTS/f"{method}_category_profiles.csv", index=False)
    pd.DataFrame(modes).to_csv(RESULTS/f"{method}_modes.csv", index=False)
    pd.DataFrame(importance).sort_values("cramers_v", ascending=False).to_csv(RESULTS/f"{method}_feature_importance.csv", index=False)
    return distance_to_mode


def main():
    prepare_data()
    raw, frame, X = load_prepared()
    configure_plots()
    rng = np.random.default_rng(20261114)
    all_sample = rng.choice(len(X), 6000, replace=False)
    selection_idx, validation_idx = all_sample[:3000], all_sample[3000:]
    pd.DataFrame({"row_id": all_sample + 1, "role": ["selection"]*3000 + ["validation"]*3000}).to_csv(RESULTS/"evaluation_sample.csv", index=False)
    D_selection = cdist(X[selection_idx], X[selection_idx], metric="hamming").astype(np.float32)
    D_validation = cdist(X[validation_idx], X[validation_idx], metric="hamming").astype(np.float32)
    grid, fitted = [], {}
    params = {"CLOPE": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "k-modes": [2, 3, 4, 5, 6, 8, 10]}
    for method, values in params.items():
        for value in values:
            result = run_method(method, X, value)
            fitted[method, value] = result
            scored = score_partition(result[0], selection_idx, D_selection)
            row = {"method": method, "parameter": value, "seed": SEED, **scored,
                   "fit_seconds": result[1]["fit_seconds"], "iterations": result[1]["iterations"],
                   "objective": result[1]["objective"]}
            grid.append(row)
            pd.DataFrame(grid).to_csv(RESULTS/"parameter_search.csv", index=False)
            print("SCORE", method, value, scored, flush=True)
    selected, final, assignments = {}, [], pd.DataFrame({"row_id": np.arange(1, len(raw)+1), "y": raw.y})
    silhouettes, stability, stability_labels = [], [], {}
    for method in params:
        candidates = [row for row in grid if row["method"] == method and np.isfinite(row["silhouette_selection"])]
        best = max(candidates, key=lambda row: row["silhouette_selection"])
        result = fitted[method, best["parameter"]]
        labels, info = result[:2]
        selected[method] = {"parameter": best["parameter"], **info}
        assignments[method] = labels
        assignments[method+"_distance_to_mode"] = cluster_profiles(raw, frame, X, labels, method, info)
        sil_values = silhouette_samples(D_validation, labels[validation_idx], metric="precomputed")
        silhouettes.append(pd.DataFrame({"method": method, "row_id": validation_idx+1, "cluster": labels[validation_idx], "silhouette": sil_values}))
        metrics = {**best, "silhouette_validation": float(np.mean(sil_values)),
                   "negative_silhouette_percent": float(100*np.mean(sil_values < 0)),
                   "ARI_y": float(adjusted_rand_score(raw.y, labels)),
                   "AMI_y": float(adjusted_mutual_info_score(raw.y, labels)),
                   "NMI_y": float(normalized_mutual_info_score(raw.y, labels)),
                   "purity": purity(raw.y, labels)}
        final.append(metrics)
        if method == "CLOPE":
            pd.DataFrame(info["clusters"]).to_csv(RESULTS/"CLOPE_cluster_parameters.csv", index=False)
        runs = {SEED: labels}
        for seed in [43, 44]:
            runs[seed] = run_method(method, X, best["parameter"], seed)[0]
        for (s1, l1), (s2, l2) in combinations(runs.items(), 2):
            stability.append({"method": method, "seed_1": s1, "seed_2": s2, "ARI": adjusted_rand_score(l1, l2),
                              "clusters_1": len(np.unique(l1)), "clusters_2": len(np.unique(l2))})
        for seed, labels_run in runs.items():
            stability_labels[f"{method}_{seed}"] = labels_run
    np.savez_compressed(RESULTS/"stability_labels.npz", **stability_labels)
    pd.DataFrame(stability).to_csv(RESULTS/"stability.csv", index=False)
    pd.concat(silhouettes).to_csv(RESULTS/"silhouette_values.csv", index=False)
    assignments.to_csv(RESULTS/"assignments.csv", index=False)
    metrics_frame = pd.DataFrame(final)
    for method in params:
        values = [row["ARI"] for row in stability if row["method"] == method]
        metrics_frame.loc[metrics_frame.method == method, "stability_ARI_mean"] = np.mean(values)
        metrics_frame.loc[metrics_frame.method == method, "stability_ARI_min"] = np.min(values)
    metrics_frame.to_csv(RESULTS/"metrics.csv", index=False)
    pd.crosstab(assignments.CLOPE, assignments["k-modes"]).to_csv(RESULTS/"cross_method_contingency.csv")
    comparison_ari = float(adjusted_rand_score(assignments.CLOPE, assignments["k-modes"]))
    summary = {"n": len(X), "p": X.shape[1], "selection_sample_size": 3000, "validation_sample_size": 3000,
               "sample_seed": 20261114, "distance": "normalized Hamming; Gower for complete nominal data with equal weights",
               "selection_rule": "maximum selection silhouette among valid partitions; no target used",
               "selected": selected, "cross_method_ARI": comparison_ari}
    (RESULTS/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # Compare at the same cluster count and check two correlated history fields.
    common_k = assignments.CLOPE.nunique()
    if ("k-modes", common_k) in fitted:
        matched_labels = fitted["k-modes", common_k][0]
    else:
        matched_labels = run_method("k-modes", X, common_k)[0]
    matched = {"n_clusters": int(common_k),
               "kmodes_silhouette_validation": float(silhouette_score(D_validation, matched_labels[validation_idx], metric="precomputed")),
               "ARI_between_methods": float(adjusted_rand_score(assignments.CLOPE, matched_labels)),
               "kmodes_ARI_y": float(adjusted_rand_score(raw.y, matched_labels))}
    (RESULTS/"matched_k_comparison.json").write_text(json.dumps(matched, indent=2)+"\n")
    reduced = X[:, [col not in ["pdays", "previous"] for col in frame]]
    sensitivity, reduced_assignments = [], pd.DataFrame({"row_id": assignments.row_id})
    for method in params:
        labels_reduced = run_method(method, reduced, selected[method]["parameter"])[0]
        reduced_assignments[method] = labels_reduced
        sensitivity.append({"method": method, "removed": "pdays, previous", "n_features": reduced.shape[1],
                            "n_clusters": len(np.unique(labels_reduced)),
                            "ARI_with_full_feature_solution": float(adjusted_rand_score(assignments[method], labels_reduced))})
    pd.DataFrame(sensitivity).to_csv(RESULTS/"feature_sensitivity.csv", index=False)
    reduced_assignments.to_csv(RESULTS/"sensitivity_assignments.csv", index=False)
    make_figures(grid, assignments, silhouettes)
    print(metrics_frame.to_string(index=False), flush=True)
    print("Cross-method ARI:", comparison_ari, flush=True)


def make_figures(grid, assignments, silhouettes):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    search = pd.DataFrame(grid)
    for ax, method in zip(axes, ["CLOPE", "k-modes"]):
        part = search[search.method == method]
        ax.plot(part.parameter, part.silhouette_selection, "o-", color="#527a96")
        for row in part.itertuples():
            if np.isfinite(row.silhouette_selection):
                ax.annotate(f"k={row.n_clusters}", (row.parameter, row.silhouette_selection), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=method, xlabel="Отталкивание r" if method == "CLOPE" else "Число кластеров k", ylabel="Силуэт: выбор параметров")
        ax.margins(y=0.20)
    fig.tight_layout(); fig.savefig(RESULTS/"05_parameter_selection.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for ax, method in zip(axes, ["CLOPE", "k-modes"]):
        counts = assignments[method].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#527a96")
        ax.set(title=method, xlabel="Кластер", ylabel="Число записей")
    fig.tight_layout(); fig.savefig(RESULTS/"06_cluster_sizes.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    tables = [pd.read_csv(RESULTS/f"{method}_feature_importance.csv") for method in ["CLOPE", "k-modes"]]
    importance = pd.concat(tables).pivot(index="feature", columns="method", values="cramers_v")
    importance = importance.loc[importance.max(axis=1).sort_values().index]
    y_positions = np.arange(len(importance))
    ax.barh(y_positions-0.2, importance.CLOPE, height=0.38, color="#527a96", label="CLOPE")
    ax.barh(y_positions+0.2, importance["k-modes"], height=0.38, color="#cc8844", label="k-modes")
    ax.set_yticks(y_positions, [NAMES[c] for c in importance.index], fontsize=9)
    ax.set(xlabel="V Крамера: признак × кластер", xlim=(0, 0.8))
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS/"07_feature_importance.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.3))
    for ax, method in zip(axes, ["CLOPE", "k-modes"]):
        part = pd.concat(silhouettes).query("method == @method")
        bottom = 0
        for cluster, group in part.groupby("cluster"):
            values = np.sort(group.silhouette.to_numpy())
            top = bottom + len(values)
            ax.fill_betweenx(np.arange(bottom, top), 0, values, alpha=0.75)
            ax.text(-0.10, (bottom+top)/2, str(cluster), fontsize=8)
            bottom = top + 15
        mean = part.silhouette.mean()
        ax.axvline(mean, color="#ba5a3a", linestyle="--", label=f"Среднее {mean:.3f}")
        ax.set(title=method, xlabel="Силуэт: контрольная подвыборка", yticks=[])
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS/"08_silhouettes.png"); plt.close(fig)
    table = pd.crosstab(assignments.CLOPE, assignments["k-modes"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.heatmap(table.div(table.sum(axis=1), axis=0), cmap="Blues", vmin=0, vmax=1,
                annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Доля внутри кластера CLOPE"})
    ax.set(title="Соответствие двух разбиений", xlabel="Кластер k-modes", ylabel="Кластер CLOPE")
    fig.tight_layout(); fig.savefig(RESULTS/"09_cluster_overlap.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for ax, method in zip(axes, ["CLOPE", "k-modes"]):
        profiles = pd.read_csv(RESULTS/f"{method}_cluster_profiles.csv")
        ax.bar(profiles.cluster.astype(str), profiles.yes_percent, color="#527a96")
        ax.axhline(100*(assignments.y == "yes").mean(), color="#ba5a3a", linestyle="--", label="Весь набор")
        ax.set(title=method, xlabel="Кластер", ylabel="Оформлен депозит, %")
        ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS/"10_deposit_rates.png"); plt.close(fig)


if __name__ == "__main__":
    main()
