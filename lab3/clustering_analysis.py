"""Кластеризация Bank Marketing методами CLOPE и k-modes.

Запуск: lab3/.venv/bin/python lab3/clustering_analysis.py
Данные, таблицы и рисунки сохраняются в lab3/results.
"""

from collections import Counter
from io import BytesIO
from itertools import combinations
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".venv/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmodes.kmodes import KModes
from scipy.spatial.distance import cdist
from scipy.stats.contingency import association
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix

from vendor.pyclope import CLOPE

RANDOM_SEED = 42
SAMPLE_SEED = 20261114
SAMPLE_SIZE = 3000
STABILITY_SEEDS = (43, 44)
CLOPE_N_INIT = 3
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
PARAMETER_GRID = {
    "CLOPE": [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    "k-modes": [2, 3, 4, 5, 6, 8, 10],
}
NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
FEATURE_NAMES = {
    "age": "Возраст",
    "job": "Профессия",
    "marital": "Семейное положение",
    "education": "Образование",
    "default": "Просроченный кредит",
    "balance": "Баланс, евро",
    "housing": "Ипотечный кредит",
    "loan": "Потребительский кредит",
    "contact": "Способ контакта",
    "day": "День месяца",
    "month": "Месяц",
    "duration": "Длительность, с",
    "campaign": "Контакты в кампании",
    "pdays": "Дней с прошлого контакта",
    "previous": "Предыдущие контакты",
    "poutcome": "Исход прошлой кампании",
}


def save_json(filename, data):
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_raw_data():
    """Загрузить локальный CSV; при его отсутствии скачать исходный архив UCI."""
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "bank-full.csv"
    if not csv_path.exists():
        with urlopen(DATASET_URL, timeout=90) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as outer_archive:
            bank_archive_name = next(
                name for name in outer_archive.namelist() if name.endswith("bank.zip")
            )
            with ZipFile(BytesIO(outer_archive.read(bank_archive_name))) as bank_archive:
                for filename in ("bank-full.csv", "bank-names.txt"):
                    (RESULTS_DIR / filename).write_bytes(bank_archive.read(filename))
    return pd.read_csv(csv_path, sep=";")


def categorize_features(raw_data):
    """Исключить ответ Y и заменить семь числовых признаков интервалами."""
    features = raw_data.drop(columns="y").copy()
    duration_quartiles = raw_data["duration"].quantile([0.25, 0.5, 0.75]).tolist()
    rules = {
        "age": ([-np.inf, 30, 45, 60, np.inf], ["18-30", "31-45", "46-60", "61+"]),
        "balance": (
            [-np.inf, -1, 0, 1000, 5000, np.inf],
            ["negative", "zero", "1-1000", "1001-5000", "5001+"],
        ),
        "day": ([-np.inf, 10, 20, np.inf], ["1-10", "11-20", "21-31"]),
        "duration": ([-np.inf, *duration_quartiles, np.inf], ["Q1", "Q2", "Q3", "Q4"]),
        "campaign": ([-np.inf, 1, 2, 5, np.inf], ["1", "2", "3-5", "6+"]),
        "pdays": (
            [-np.inf, -1, 90, 180, 365, np.inf],
            ["not_contacted", "0-90", "91-180", "181-365", "366+"],
        ),
        "previous": ([-np.inf, 0, 1, 3, np.inf], ["0", "1", "2-3", "4+"]),
    }

    binning_rows = []
    for feature, (edges, categories) in rules.items():
        features[feature] = pd.cut(
            raw_data[feature], bins=edges, labels=categories, right=True
        ).astype(str)
        for index, category in enumerate(categories):
            binning_rows.append({
                "feature": feature,
                "category": category,
                "left_exclusive": str(edges[index]),
                "right_inclusive": str(edges[index + 1]),
                "count": int((features[feature] == category).sum()),
            })
    return features, pd.DataFrame(binning_rows), duration_quartiles


def save_feature_statistics(raw_data, features):
    """Сохранить описательные статистики, частоты категорий и качество полей."""
    statistics = raw_data[NUMERIC_FEATURES].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T
    statistics.to_csv(RESULTS_DIR / "numeric_statistics.csv", index_label="feature")
    frequency_rows = []
    quality_rows = []
    for feature in features:
        for category, count in features[feature].value_counts().items():
            frequency_rows.append({
                "feature": feature,
                "category": category,
                "count": int(count),
                "percent": 100 * count / len(features),
            })
        quality_rows.append({
            "feature": feature,
            "type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
            "raw_levels": raw_data[feature].nunique(),
            "prepared_levels": features[feature].nunique(),
            "missing": int(raw_data[feature].isna().sum()),
            "unknown": int((raw_data[feature] == "unknown").sum()),
        })
    pd.DataFrame(frequency_rows).to_csv(RESULTS_DIR / "category_frequencies.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(RESULTS_DIR / "feature_quality.csv", index=False)

    class_balance = raw_data["y"].value_counts().rename_axis("class").reset_index(name="count")
    class_balance["percent"] = class_balance["count"] / len(raw_data) * 100
    class_balance.to_csv(RESULTS_DIR / "class_balance.csv", index=False)


def save_feature_associations(features):
    """Рассчитать V Крамера для каждой пары подготовленных признаков."""
    associations = pd.DataFrame(
        np.eye(len(features.columns)), index=features.columns, columns=features.columns
    )
    pair_rows = []
    for feature, other_feature in combinations(features.columns, 2):
        frequency_table = pd.crosstab(features[feature], features[other_feature])
        cramers_v = float(association(frequency_table.to_numpy(), method="cramer", correction=False))
        associations.loc[feature, other_feature] = cramers_v
        associations.loc[other_feature, feature] = cramers_v
        pair_rows.append({
            "feature_1": feature,
            "feature_2": other_feature,
            "cramers_v": cramers_v,
        })
    associations.to_csv(RESULTS_DIR / "feature_associations.csv", index_label="feature")
    pairs = pd.DataFrame(pair_rows).sort_values("cramers_v", ascending=False)
    pairs.to_csv(RESULTS_DIR / "association_pairs.csv", index=False)


def prepare_data():
    """Подготовить общие для двух методов признаки и сохранить их описание."""
    raw_data = load_raw_data()
    assert raw_data.shape == (45211, 17)
    assert not raw_data.isna().any().any()
    features, binning_rules, duration_quartiles = categorize_features(raw_data)
    assert not features.isna().any().any()

    features.to_csv(RESULTS_DIR / "prepared_categorical.csv", index=False)
    binning_rules.to_csv(RESULTS_DIR / "binning_rules.csv", index=False)
    save_feature_statistics(raw_data, features)
    save_feature_associations(features)
    save_json("data_summary.json", {
        "n": len(raw_data),
        "p": len(features.columns),
        "numeric": NUMERIC_FEATURES,
        "categorical": [feature for feature in features if feature not in NUMERIC_FEATURES],
        "target": "y",
        "duplicates_raw": int(raw_data.duplicated().sum()),
        "duplicates_features": int(raw_data.drop(columns="y").duplicated().sum()),
        "duplicates_after_binning": int(features.duplicated().sum()),
        "missing_cells": int(raw_data.isna().sum().sum()),
        "unknown_rows": int((raw_data == "unknown").any(axis=1).sum()),
        "duration_quartiles": duration_quartiles,
        "positive_rate": float((raw_data["y"] == "yes").mean()),
        "prepared_levels": {feature: sorted(features[feature].unique().tolist()) for feature in features},
        "preparation": "all 16 features, seven numeric features discretized; target excluded; unknown retained",
    })

    # Числа обозначают категории; расстояние использует только совпадение кодов.
    encoded_data = np.column_stack([
        pd.Categorical(features[feature]).codes for feature in features
    ]).astype(np.int16)
    return raw_data, features, encoded_data


def normalize_labels(labels):
    """Пронумеровать непустые кластеры с единицы по убыванию размера."""
    counts = Counter(labels)
    ordered_labels = sorted(counts, key=lambda label: (-counts[label], int(label)))
    label_mapping = {label: number for number, label in enumerate(ordered_labels, start=1)}
    normalized = np.array([label_mapping[label] for label in labels], dtype=np.int32)
    return normalized, label_mapping


def fit_clope(encoded_data, repulsion, seed=RANDOM_SEED):
    """Передать транзакции готовому CLOPE через его функции чтения и записи."""
    if len(encoded_data) == 0 or repulsion <= 1:
        raise ValueError("CLOPE needs data and repulsion > 1")

    row_order = np.random.default_rng(seed).permutation(len(encoded_data))
    transactions = [
        tuple(f"{feature_index}={int(value)}" for feature_index, value in enumerate(row))
        for row in encoded_data
    ]
    labels = np.full(len(encoded_data), -1, dtype=np.int32)
    cursor = 0

    def next_transaction():
        nonlocal cursor
        row_index = int(row_order[cursor % len(encoded_data)])
        cursor += 1
        return row_index, transactions[row_index], int(labels[row_index])

    def save_assignment(row_index, cluster):
        labels[row_index] = cluster

    started_at = time.perf_counter()
    model = CLOPE(len(encoded_data), next_transaction, save_assignment, repulsion)
    model.exec()
    elapsed_seconds = time.perf_counter() - started_at
    assert np.all(labels >= 0), "Unassigned transactions"
    normalized_labels, label_mapping = normalize_labels(labels)

    cluster_parameters = []
    for original_label, normalized_label in label_mapping.items():
        cluster = model.clusters[int(original_label)]
        cluster_indices = np.flatnonzero(labels == original_label)
        item_counts = Counter(
            item for row_index in cluster_indices for item in transactions[row_index]
        )
        # Сверяем внутренние счётчики CLOPE с фактически назначенными строками.
        assert item_counts == cluster.Occ
        assert cluster.N == len(cluster_indices)
        assert cluster.S == sum(item_counts.values())
        assert cluster.W == len(item_counts)
        cluster_parameters.append({
            "cluster": normalized_label,
            "n": cluster.N,
            "S": cluster.S,
            "W": cluster.W,
            "H": cluster.S / cluster.W,
            "profit_contribution": cluster.S * cluster.N / cluster.W**repulsion / len(encoded_data),
        })
    assert sum(cluster.N for cluster in model.clusters) == len(encoded_data)

    return normalized_labels, {
        "r": repulsion,
        "seed": seed,
        "fit_seconds": elapsed_seconds,
        "iterations": model.n_iter_,
        "objective": sum(cluster["profit_contribution"] for cluster in cluster_parameters),
        "clusters": sorted(cluster_parameters, key=lambda cluster: cluster["cluster"]),
    }


def fit_kmodes(encoded_data, n_clusters, seed=RANDOM_SEED, n_init=3):
    """Обучить k-modes и сохранить именно те моды, которые выбрала модель."""
    started_at = time.perf_counter()
    model = KModes(
        n_clusters=n_clusters, init="Huang", n_init=n_init, max_iter=50,
        random_state=seed, n_jobs=1, verbose=0,
    )
    predictions = model.fit_predict(encoded_data)
    labels, label_mapping = normalize_labels(predictions)
    elapsed_seconds = time.perf_counter() - started_at

    # При равных частотах повторно вычисленная мода может отличаться от моды модели.
    centers = {
        label_mapping[label]: np.asarray(center, dtype=int).tolist()
        for label, center in enumerate(model.cluster_centroids_)
        if label in label_mapping
    }
    cost = sum(
        np.sum(encoded_data[labels == cluster] != np.array(center))
        for cluster, center in centers.items()
    )
    assert np.isclose(cost, model.cost_)
    return labels, {
        "k": n_clusters,
        "seed": seed,
        "n_init": n_init,
        "init": "Huang",
        "fit_seconds": elapsed_seconds,
        "iterations": model.n_iter_,
        "objective": float(model.cost_),
        "modes_encoded": centers,
    }


def fit_clope_multistart(encoded_data, repulsion, seed=RANDOM_SEED):
    """При одном r выбрать максимальный Profit из трёх начальных порядков."""
    runs = [
        fit_clope(encoded_data, repulsion, current_seed)
        for current_seed in range(seed, seed + CLOPE_N_INIT)
    ]
    labels, best_details = max(runs, key=lambda run: run[1]["objective"])
    details = {
        **best_details,
        "n_init": CLOPE_N_INIT,
        "start_seed": seed,
        "fit_seconds": sum(run_details["fit_seconds"] for _, run_details in runs),
        "initializations": [
            {
                "seed": run_details["seed"],
                "n_clusters": len(np.unique(run_labels)),
                "objective": run_details["objective"],
            }
            for run_labels, run_details in runs
        ],
    }
    return labels, details


def run_method(method, encoded_data, parameter, seed=RANDOM_SEED):
    if method == "CLOPE":
        labels, details = fit_clope_multistart(encoded_data, parameter, seed)
    elif method == "k-modes":
        labels, details = fit_kmodes(encoded_data, int(parameter), seed)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    print(
        method, parameter, "seed", details["seed"], "clusters", len(np.unique(labels)),
        "seconds", round(details["fit_seconds"], 2), flush=True,
    )
    return labels, details


def create_evaluation_samples(n_rows):
    """Выбрать две непересекающиеся подвыборки для настройки и контроля."""
    generator = np.random.default_rng(SAMPLE_SEED)
    sampled_indices = generator.choice(n_rows, 2 * SAMPLE_SIZE, replace=False)
    selection_indices = sampled_indices[:SAMPLE_SIZE]
    validation_indices = sampled_indices[SAMPLE_SIZE:]
    pd.DataFrame({
        "row_id": sampled_indices + 1,
        "role": ["selection"] * SAMPLE_SIZE + ["validation"] * SAMPLE_SIZE,
    }).to_csv(RESULTS_DIR / "evaluation_sample.csv", index=False)
    return selection_indices, validation_indices


def evaluate_selection(labels, sample_indices, distances):
    cluster_sizes = np.unique(labels, return_counts=True)[1]
    sampled_labels = labels[sample_indices]
    sampled_clusters = len(np.unique(sampled_labels))
    silhouette = np.nan
    if 1 < sampled_clusters < len(sample_indices):
        silhouette = float(silhouette_score(distances, sampled_labels, metric="precomputed"))

    small_cluster_sizes = cluster_sizes[cluster_sizes < 0.01 * len(labels)]
    return {
        "n_clusters": len(cluster_sizes),
        "silhouette_selection": silhouette,
        "min_size": int(cluster_sizes.min()),
        "max_size": int(cluster_sizes.max()),
        "small_cluster_fraction": float(small_cluster_sizes.sum() / len(labels)),
        "sampled_clusters": sampled_clusters,
    }


def search_parameters(encoded_data, selection_indices, selection_distances):
    """Проверить сетку параметров, сохраняя метки для последующего сравнения."""
    search_rows = []
    fitted_results = {}
    for method, parameters in PARAMETER_GRID.items():
        for parameter in parameters:
            labels, details = run_method(method, encoded_data, parameter)
            fitted_results[method, parameter] = (labels, details)
            scores = evaluate_selection(labels, selection_indices, selection_distances)
            search_rows.append({
                "method": method,
                "parameter": parameter,
                "seed": details["seed"],
                **scores,
                "fit_seconds": details["fit_seconds"],
                "iterations": details["iterations"],
                "objective": details["objective"],
            })
            pd.DataFrame(search_rows).to_csv(RESULTS_DIR / "parameter_search.csv", index=False)
            print("SCORE", method, parameter, scores, flush=True)
    return search_rows, fitted_results


def purity(target, labels):
    frequency_table = contingency_matrix(target, labels)
    return float(frequency_table.max(axis=0).sum() / frequency_table.sum())


def evaluate_cluster_geometry(encoded_data, labels, fitted_centers=None):
    """Точные расстояния Хэмминга и силуэты по всем строкам через частоты категорий.

    Сумма расстояний от x до кластера равна сумме несовпадений по признакам.
    Это позволяет учитывать все пары, не создавая матрицу 45 211 × 45 211.
    """
    clusters, cluster_sizes = np.unique(labels, return_counts=True)
    n_rows, n_features = encoded_data.shape
    distance_sums = np.zeros((n_rows, len(clusters)), dtype=float)
    centers = {int(cluster): [] for cluster in clusters}

    for feature_index in range(n_features):
        values = encoded_data[:, feature_index]
        category_counts = np.vstack([
            np.bincount(values[labels == cluster], minlength=int(values.max()) + 1)
            for cluster in clusters
        ])
        distance_sums += cluster_sizes - category_counts[:, values].T
        for cluster_index, cluster in enumerate(clusters):
            centers[int(cluster)].append(int(category_counts[cluster_index].argmax()))
    distance_sums /= n_features
    if fitted_centers is not None:
        centers = fitted_centers

    row_indices = np.arange(n_rows)
    own_clusters = np.searchsorted(clusters, labels)
    own_sizes = cluster_sizes[own_clusters]
    own_distance_sums = distance_sums[row_indices, own_clusters]
    intra_distances = own_distance_sums / np.maximum(own_sizes - 1, 1)
    mean_distances = distance_sums / cluster_sizes
    mean_distances[row_indices, own_clusters] = np.inf
    nearest_distances = mean_distances.min(axis=1)
    silhouettes = np.full(n_rows, np.nan)
    if 1 < len(clusters) < n_rows:
        denominator = np.maximum(intra_distances, nearest_distances)
        silhouettes = np.divide(
            nearest_distances - intra_distances, denominator,
            out=np.zeros(n_rows), where=denominator > 0,
        )
        silhouettes[own_sizes == 1] = 0

    distance_to_mode = np.zeros(n_rows)
    cluster_rows = []
    for cluster in clusters:
        mask = labels == cluster
        distances = np.mean(encoded_data[mask] != np.asarray(centers[int(cluster)]), axis=1)
        distance_to_mode[mask] = distances
        cluster_rows.append({
            "cluster": int(cluster),
            "mean_within_distance": float(intra_distances[mask].mean()) if mask.sum() > 1 else np.nan,
            "mean_distance_to_mode": float(distances.mean()),
            "max_distance_to_mode": float(distances.max()),
            "silhouette_full": float(silhouettes[mask].mean()),
            "negative_silhouette_percent_full": float(100 * (silhouettes[mask] < 0).mean()),
        })

    center_distances = cdist(
        np.array([centers[int(cluster)] for cluster in clusters]),
        np.array([centers[int(cluster)] for cluster in clusters]),
        metric="hamming",
    )
    between_rows = []
    for first_index, second_index in combinations(range(len(clusters)), 2):
        first_cluster, second_cluster = clusters[first_index], clusters[second_index]
        n_pairs = int(cluster_sizes[first_index] * cluster_sizes[second_index])
        mean_distance = distance_sums[labels == first_cluster, second_index].sum() / n_pairs
        between_rows.append({
            "cluster_1": int(first_cluster),
            "cluster_2": int(second_cluster),
            "n_pairs": n_pairs,
            "mean_between_distance": float(mean_distance),
            "center_distance": float(center_distances[first_index, second_index]),
        })

    within_pairs = np.sum(cluster_sizes * (cluster_sizes - 1))
    between_pairs = sum(row["n_pairs"] for row in between_rows)
    metrics = {
        "mean_within_distance": float(own_distance_sums.sum() / within_pairs) if within_pairs else np.nan,
        "mean_between_distance": float(sum(
            row["mean_between_distance"] * row["n_pairs"] for row in between_rows
        ) / between_pairs) if between_pairs else np.nan,
        "mean_distance_to_mode": float(distance_to_mode.mean()),
        "min_center_distance": min((row["center_distance"] for row in between_rows), default=np.nan),
        "silhouette_full": float(silhouettes.mean()),
        "negative_silhouette_percent_full": float(100 * (silhouettes < 0).mean()),
    }
    return {
        "metrics": metrics,
        "cluster_metrics": pd.DataFrame(cluster_rows).set_index("cluster"),
        "between_clusters": pd.DataFrame(between_rows),
        "centers": centers,
        "distance_to_mode": distance_to_mode,
        "silhouettes": silhouettes,
    }


def save_cluster_profiles(raw_data, features, labels, method, geometry):
    """Сохранить размеры, моды, частоты и связь признаков с кластерами."""
    summaries, category_profiles, modes = [], [], []
    overall_frequencies = {
        feature: features[feature].value_counts(normalize=True) for feature in features
    }
    feature_categories = {
        feature: pd.Categorical(features[feature]).categories for feature in features
    }

    for cluster in np.unique(labels):
        cluster_mask = labels == cluster
        cluster_features = features.loc[cluster_mask]
        cluster_raw = raw_data.loc[cluster_mask]
        center = geometry["centers"][int(cluster)]
        for feature_index, feature in enumerate(features):
            frequencies = cluster_features[feature].value_counts(normalize=True)
            mode = feature_categories[feature][center[feature_index]]
            modes.append({
                "method": method,
                "cluster": int(cluster),
                "feature": feature,
                "mode": mode,
                "mode_percent": float(100 * frequencies[mode]),
            })
            for category, fraction in frequencies.items():
                overall_fraction = overall_frequencies[feature][category]
                category_profiles.append({
                    "method": method,
                    "cluster": int(cluster),
                    "feature": feature,
                    "category": category,
                    "fraction": float(fraction),
                    "overall_fraction": float(overall_fraction),
                    "lift": float(fraction / overall_fraction),
                    "difference_pp": float(100 * (fraction - overall_fraction)),
                })
        summaries.append({
            "method": method,
            "cluster": int(cluster),
            "n": int(cluster_mask.sum()),
            "percent": float(100 * cluster_mask.mean()),
            "yes_count": int((cluster_raw["y"] == "yes").sum()),
            "yes_percent": float(100 * (cluster_raw["y"] == "yes").mean()),
            **geometry["cluster_metrics"].loc[cluster].to_dict(),
            "age_median": float(cluster_raw["age"].median()),
            "balance_median": float(cluster_raw["balance"].median()),
            "duration_median": float(cluster_raw["duration"].median()),
            "housing_yes_percent": float(100 * (cluster_raw["housing"] == "yes").mean()),
            "previous_zero_percent": float(100 * (cluster_raw["previous"] == 0).mean()),
        })

    importance_rows = []
    for feature in features:
        frequency_table = pd.crosstab(features[feature], labels)
        cramers_v = association(frequency_table.to_numpy(), method="cramer", correction=False)
        importance_rows.append({"method": method, "feature": feature, "cramers_v": float(cramers_v)})

    pd.DataFrame(summaries).to_csv(RESULTS_DIR / f"{method}_cluster_profiles.csv", index=False)
    pd.DataFrame(category_profiles).to_csv(RESULTS_DIR / f"{method}_category_profiles.csv", index=False)
    pd.DataFrame(modes).to_csv(RESULTS_DIR / f"{method}_modes.csv", index=False)
    importance = pd.DataFrame(importance_rows).sort_values("cramers_v", ascending=False)
    importance.to_csv(RESULTS_DIR / f"{method}_feature_importance.csv", index=False)
    return geometry["distance_to_mode"]


def evaluate_stability(method, encoded_data, parameter, base_labels, base_seed):
    """Сравнить попарно три запуска с разными seed при одном параметре."""
    labels_by_seed = {base_seed: base_labels}
    for seed in (RANDOM_SEED, *STABILITY_SEEDS):
        if seed not in labels_by_seed:
            result = (fit_clope(encoded_data, parameter, seed) if method == "CLOPE"
                      else run_method(method, encoded_data, parameter, seed))
            labels_by_seed[seed] = result[0]

    comparison_rows = []
    for (first_seed, first_labels), (second_seed, second_labels) in combinations(labels_by_seed.items(), 2):
        comparison_rows.append({
            "method": method,
            "seed_1": first_seed,
            "seed_2": second_seed,
            "ARI": adjusted_rand_score(first_labels, second_labels),
            "clusters_1": len(np.unique(first_labels)),
            "clusters_2": len(np.unique(second_labels)),
        })
    return comparison_rows, labels_by_seed


def evaluate_best_models(
    raw_data, features, encoded_data, search_rows, fitted_results,
    validation_indices, validation_distances,
):
    """Выбрать лучшие параметры по силуэту и оценить полученные разбиения."""
    selected_models = {}
    metric_rows, silhouette_tables, stability_rows, between_tables = [], [], [], []
    stability_labels = {}
    assignments = pd.DataFrame({"row_id": np.arange(1, len(raw_data) + 1), "y": raw_data["y"]})

    for method in PARAMETER_GRID:
        candidates = [
            row for row in search_rows
            if row["method"] == method and np.isfinite(row["silhouette_selection"])
        ]
        best = max(candidates, key=lambda row: row["silhouette_selection"])
        labels, details = fitted_results[method, best["parameter"]]
        selected_models[method] = {"parameter": best["parameter"], **details}
        geometry = evaluate_cluster_geometry(encoded_data, labels, details.get("modes_encoded"))
        assignments[method] = labels
        assignments[f"{method}_distance_to_mode"] = save_cluster_profiles(
            raw_data, features, labels, method, geometry
        )
        between_tables.append(geometry["between_clusters"].assign(method=method))

        validation_labels = labels[validation_indices]
        silhouettes = silhouette_samples(validation_distances, validation_labels, metric="precomputed")
        silhouette_tables.append(pd.DataFrame({
            "method": method,
            "row_id": np.arange(1, len(labels) + 1),
            "cluster": labels,
            "silhouette": geometry["silhouettes"],
        }))
        metric_rows.append({
            **best,
            "silhouette_validation": float(silhouettes.mean()),
            "negative_silhouette_percent": float(100 * np.mean(silhouettes < 0)),
            "ARI_y": float(adjusted_rand_score(raw_data["y"], labels)),
            "AMI_y": float(adjusted_mutual_info_score(raw_data["y"], labels)),
            "NMI_y": float(normalized_mutual_info_score(raw_data["y"], labels)),
            "purity": purity(raw_data["y"], labels),
            **geometry["metrics"],
        })
        if method == "CLOPE":
            pd.DataFrame(details["clusters"]).to_csv(RESULTS_DIR / "CLOPE_cluster_parameters.csv", index=False)

        comparisons, labels_by_seed = evaluate_stability(
            method, encoded_data, best["parameter"], labels, details["seed"]
        )
        stability_rows.extend(comparisons)
        for seed, run_labels in labels_by_seed.items():
            stability_labels[f"{method}_{seed}"] = run_labels

    metrics = pd.DataFrame(metric_rows)
    for method in PARAMETER_GRID:
        ari_values = [row["ARI"] for row in stability_rows if row["method"] == method]
        metrics.loc[metrics["method"] == method, "stability_ARI_mean"] = np.mean(ari_values)
        metrics.loc[metrics["method"] == method, "stability_ARI_min"] = np.min(ari_values)

    silhouettes = pd.concat(silhouette_tables)
    np.savez_compressed(RESULTS_DIR / "stability_labels.npz", **stability_labels)
    pd.DataFrame(stability_rows).to_csv(RESULTS_DIR / "stability.csv", index=False)
    silhouettes.to_csv(RESULTS_DIR / "silhouette_values.csv", index=False)
    assignments.to_csv(RESULTS_DIR / "assignments.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pd.concat(between_tables).to_csv(RESULTS_DIR / "intercluster_distances.csv", index=False)
    return selected_models, assignments, metrics, silhouettes


def compare_clope_cluster_counts(encoded_data, fitted_results):
    """Сравнить два запуска при одном r и отдельно проверить слияние малой группы."""
    repulsion = 1.1
    two_cluster_result = fitted_results["CLOPE", repulsion]
    three_cluster_result = fit_clope(encoded_data, repulsion, seed=42)
    original_labels, original_details = three_cluster_result
    original_geometry = evaluate_cluster_geometry(encoded_data, original_labels)
    clusters, sizes = np.unique(original_labels, return_counts=True)
    smallest_cluster = int(clusters[sizes.argmin()])
    between = original_geometry["between_clusters"]
    neighbours = between[
        (between["cluster_1"] == smallest_cluster) | (between["cluster_2"] == smallest_cluster)
    ]
    closest_pair = neighbours.loc[neighbours["mean_between_distance"].idxmin()]
    nearest_cluster = int(
        closest_pair["cluster_2"] if closest_pair["cluster_1"] == smallest_cluster
        else closest_pair["cluster_1"]
    )
    merged_labels = original_labels.copy()
    merged_labels[merged_labels == smallest_cluster] = nearest_cluster
    merged_labels, _ = normalize_labels(merged_labels)
    merged_profit = 0.0
    for cluster in np.unique(merged_labels):
        cluster_data = encoded_data[merged_labels == cluster]
        width = sum(np.unique(cluster_data[:, column]).size for column in range(encoded_data.shape[1]))
        merged_profit += len(cluster_data)**2 * encoded_data.shape[1] / width**repulsion / len(encoded_data)
    variants = [
        ("two_clusters", *two_cluster_result),
        ("three_clusters", *three_cluster_result),
        ("merged_smallest", merged_labels, {"seed": original_details["seed"], "objective": merged_profit}),
    ]
    comparison_rows = []
    for variant, labels, details in variants:
        geometry = evaluate_cluster_geometry(encoded_data, labels)
        clusters, sizes = np.unique(labels, return_counts=True)
        smallest_cluster = int(clusters[sizes.argmin()])
        comparison_rows.append({
            "variant": variant,
            "r": repulsion,
            "seed": details["seed"],
            "n_clusters": len(clusters),
            "sizes": "; ".join(str(size) for size in sizes),
            "objective": details["objective"],
            **geometry["metrics"],
            "smallest_cluster_silhouette": float(
                geometry["cluster_metrics"].loc[smallest_cluster, "silhouette_full"]
            ),
        })
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "CLOPE_cluster_count_comparison.csv", index=False)


def compare_same_cluster_count(
    encoded_data, assignments, fitted_results, validation_indices, validation_distances,
):
    """Дополнительно сравнить методы при числе групп, полученном CLOPE."""
    n_clusters = assignments["CLOPE"].nunique()
    if ("k-modes", n_clusters) in fitted_results:
        matched_labels = fitted_results["k-modes", n_clusters][0]
    else:
        matched_labels = run_method("k-modes", encoded_data, n_clusters)[0]
    save_json("matched_k_comparison.json", {
        "n_clusters": int(n_clusters),
        "kmodes_silhouette_validation": float(silhouette_score(
            validation_distances, matched_labels[validation_indices], metric="precomputed"
        )),
        "ARI_between_methods": float(adjusted_rand_score(assignments["CLOPE"], matched_labels)),
        "kmodes_ARI_y": float(adjusted_rand_score(assignments["y"], matched_labels)),
    })


def evaluate_feature_sensitivity(features, encoded_data, assignments, selected_models):
    """Проверить влияние двух связанных признаков без изменения параметров."""
    removed_features = ["pdays", "previous"]
    retained_columns = [feature not in removed_features for feature in features]
    reduced_data = encoded_data[:, retained_columns]
    sensitivity_rows = []
    reduced_assignments = pd.DataFrame({"row_id": assignments["row_id"]})
    for method in PARAMETER_GRID:
        parameter = selected_models[method]["parameter"]
        reduced_labels = run_method(method, reduced_data, parameter)[0]
        reduced_assignments[method] = reduced_labels
        sensitivity_rows.append({
            "method": method,
            "removed": ", ".join(removed_features),
            "n_features": reduced_data.shape[1],
            "n_clusters": len(np.unique(reduced_labels)),
            "ARI_with_full_feature_solution": float(adjusted_rand_score(assignments[method], reduced_labels)),
        })
    pd.DataFrame(sensitivity_rows).to_csv(RESULTS_DIR / "feature_sensitivity.csv", index=False)
    reduced_assignments.to_csv(RESULTS_DIR / "sensitivity_assignments.csv", index=False)


def save_figure(figure, filename):
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename)
    plt.close(figure)


def plot_parameter_selection(search_results):
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for axis, method in zip(axes, PARAMETER_GRID):
        method_results = search_results[search_results["method"] == method]
        axis.plot(method_results["parameter"], method_results["silhouette_selection"], "o-", color="#527a96")
        previous_cluster_count = None
        for position, row in enumerate(method_results.itertuples()):
            if np.isfinite(row.silhouette_selection) and row.n_clusters != previous_cluster_count:
                axis.annotate(
                    f"k={row.n_clusters}", (row.parameter, row.silhouette_selection),
                    xytext=(0, 8 + 10 * (position % 2)),
                    textcoords="offset points", ha="center", fontsize=8,
                )
            previous_cluster_count = row.n_clusters
        axis.set(
            title=method,
            xlabel="Отталкивание r" if method == "CLOPE" else "Число кластеров k",
            ylabel="Силуэт: выбор параметров",
        )
        axis.margins(y=0.20)
    save_figure(figure, "05_parameter_selection.png")


def plot_cluster_sizes(assignments):
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for axis, method in zip(axes, PARAMETER_GRID):
        counts = assignments[method].value_counts().sort_index()
        axis.bar(counts.index.astype(str), counts.values, color="#527a96")
        axis.set(title=method, xlabel="Кластер", ylabel="Число записей")
    save_figure(figure, "06_cluster_sizes.png")


def plot_feature_importance():
    tables = [
        pd.read_csv(RESULTS_DIR / f"{method}_feature_importance.csv") for method in PARAMETER_GRID
    ]
    importance = pd.concat(tables).pivot(index="feature", columns="method", values="cramers_v")
    importance = importance.loc[importance.max(axis=1).sort_values().index]
    positions = np.arange(len(importance))

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.barh(positions - 0.2, importance["CLOPE"], height=0.38, color="#527a96", label="CLOPE")
    axis.barh(positions + 0.2, importance["k-modes"], height=0.38, color="#cc8844", label="k-modes")
    axis.set_yticks(positions, [FEATURE_NAMES[feature] for feature in importance.index], fontsize=9)
    axis.set(xlabel="V Крамера: признак × кластер", xlim=(0, 0.8))
    axis.legend(fontsize=9)
    save_figure(figure, "07_feature_importance.png")


def plot_silhouettes(silhouettes):
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 4.3), sharex=True)
    for axis, method in zip(axes, PARAMETER_GRID):
        method_values = silhouettes[silhouettes["method"] == method]
        bottom = 0
        for cluster, cluster_values in method_values.groupby("cluster"):
            values = np.sort(cluster_values["silhouette"].to_numpy())
            top = bottom + len(values)
            axis.fill_betweenx(np.arange(bottom, top), 0, values, alpha=0.75)
            axis.text(-0.10, (bottom + top) / 2, str(cluster), fontsize=8)
            bottom = top + 15
        mean_silhouette = method_values["silhouette"].mean()
        axis.axvline(
            mean_silhouette, color="#ba5a3a", linestyle="--", label=f"Среднее {mean_silhouette:.3f}"
        )
        axis.set(title=method, xlabel="Силуэт: все 45 211 записей", yticks=[])
        axis.legend(fontsize=8, loc="lower right")
    save_figure(figure, "08_silhouettes.png")


def plot_cluster_overlap(assignments):
    frequency_table = pd.crosstab(assignments["CLOPE"], assignments["k-modes"])
    row_proportions = frequency_table.div(frequency_table.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(6.5, 4))
    sns.heatmap(
        row_proportions, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", ax=axis,
        cbar_kws={"label": "Доля внутри кластера CLOPE"},
    )
    axis.set(title="Соответствие двух разбиений", xlabel="Кластер k-modes", ylabel="Кластер CLOPE")
    save_figure(figure, "09_cluster_overlap.png")


def plot_deposit_rates(assignments):
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    overall_rate = 100 * (assignments["y"] == "yes").mean()
    for axis, method in zip(axes, PARAMETER_GRID):
        profiles = pd.read_csv(RESULTS_DIR / f"{method}_cluster_profiles.csv")
        axis.bar(profiles["cluster"].astype(str), profiles["yes_percent"], color="#527a96")
        axis.axhline(overall_rate, color="#ba5a3a", linestyle="--", label="Весь набор")
        axis.set(title=method, xlabel="Кластер", ylabel="Оформлен депозит, %")
        axis.legend()
    save_figure(figure, "10_deposit_rates.png")


def make_figures(search_results, assignments, silhouettes):
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.titleweight": "bold"})
    plot_parameter_selection(search_results)
    plot_cluster_sizes(assignments)
    plot_feature_importance()
    plot_silhouettes(silhouettes)
    plot_cluster_overlap(assignments)
    plot_deposit_rates(assignments)


def main():
    # 1. Общая подготовка данных и две подвыборки для оценки силуэта.
    raw_data, features, encoded_data = prepare_data()
    selection_indices, validation_indices = create_evaluation_samples(len(encoded_data))
    selection_distances = cdist(
        encoded_data[selection_indices], encoded_data[selection_indices], metric="hamming"
    ).astype(np.float32)
    validation_distances = cdist(
        encoded_data[validation_indices], encoded_data[validation_indices], metric="hamming"
    ).astype(np.float32)

    # 2. Настройка параметров и оценка лучших разбиений; Y не участвует в выборе.
    search_rows, fitted_results = search_parameters(encoded_data, selection_indices, selection_distances)
    selected_models, assignments, metrics, silhouettes = evaluate_best_models(
        raw_data, features, encoded_data, search_rows, fitted_results,
        validation_indices, validation_distances,
    )

    # 3. Сравнение методов, одинакового числа групп и сокращённого набора признаков.
    pd.crosstab(assignments["CLOPE"], assignments["k-modes"]).to_csv(
        RESULTS_DIR / "cross_method_contingency.csv"
    )
    comparison_ari = float(adjusted_rand_score(assignments["CLOPE"], assignments["k-modes"]))
    save_json("summary.json", {
        "n": len(encoded_data),
        "p": encoded_data.shape[1],
        "selection_sample_size": SAMPLE_SIZE,
        "validation_sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "distance": "normalized Hamming; Gower for complete nominal data with equal weights",
        "selection_rule": "CLOPE: maximum Profit over three orders at each r; then maximum selection silhouette; no target used",
        "quality_evaluation": "exact full-data Hamming distances, modal compactness and silhouette via category counts",
        "selected": selected_models,
        "cross_method_ARI": comparison_ari,
    })
    compare_same_cluster_count(
        encoded_data, assignments, fitted_results, validation_indices, validation_distances
    )
    compare_clope_cluster_counts(encoded_data, fitted_results)
    evaluate_feature_sensitivity(features, encoded_data, assignments, selected_models)

    # 4. Рисунки для отчёта и итоговые метрики в консоли.
    make_figures(pd.DataFrame(search_rows), assignments, silhouettes)
    print(metrics.to_string(index=False), flush=True)
    print("Cross-method ARI:", comparison_ari, flush=True)


if __name__ == "__main__":
    main()
