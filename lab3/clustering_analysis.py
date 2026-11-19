"""ЛР 3. Кластеризация Bank Marketing методами CLOPE и k-modes.

Запуск из MAD: lab3/.venv/bin/python lab3/clustering_analysis.py
Чтение, категоризация и дескриптивный анализ находятся в data_preparation.py.
Таблицы и рисунки сохраняются в results; Word создаётся отдельно build_report.py.

Порядок работы:
    1. Чтение и подготовка общих категориальных признаков.
    2. Подвыборки для настройки и контрольной оценки силуэта.
    3. Подбор r для CLOPE и k для k-modes.
    4. Оценка лучших разбиений: расстояния, компактность, центры и силуэты.
    5. Проверка необходимости третьего кластера CLOPE.
    6. Сравнение методов и сохранение параметров.
    7. Графики для отчёта.

Оба метода обучаются на всех строках; подвыборки нужны только для оценки.
Ответ Y не используется при обучении. Стандартизация категорий не применяется.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
import os
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".venv/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")  # сохраняем рисунки в файлы, без окна
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmodes.kmodes import KModes
from scipy.spatial.distance import cdist
from scipy.stats.contingency import association
from sklearn.metrics import adjusted_rand_score, silhouette_score

from data_preparation import FEATURE_NAMES, prepare_data
from vendor.pyclope import CLOPE

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
SAMPLE_SEED = 20261114
SAMPLE_SIZE = 3000
CLOPE_N_INIT = 3
PARAMETER_GRID = {
    "CLOPE": [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    "k-modes": [2, 3, 4, 5, 6, 8, 10],
}


@dataclass
class Dataset:
    """Исходные данные, категории и две непересекающиеся подвыборки для оценки."""

    df: pd.DataFrame                   # исходная таблица вместе с ответом Y
    features: pd.DataFrame             # 16 категориальных признаков без Y
    X: np.ndarray                      # коды категорий; значимы только их совпадения
    selection_rows: np.ndarray         # строки для выбора r и k
    validation_rows: np.ndarray        # строки для контрольной оценки
    selection_distances: np.ndarray    # матрица Хэмминга для настройки
    validation_distances: np.ndarray   # матрица Хэмминга для контроля


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def save_json(filename, data):
    """Сохраняет параметры или итоговые показатели в results."""
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_figure(figure, filename):
    """Сохраняет рисунок в results и закрывает его."""
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename)
    plt.close(figure)


def normalize_labels(labels):
    """Пронумеровать непустые кластеры с единицы по убыванию размера."""
    counts = Counter(labels)
    ordered_labels = sorted(counts, key=lambda label: (-counts[label], int(label)))
    label_mapping = {label: number for number, label in enumerate(ordered_labels, start=1)}
    normalized = np.array([label_mapping[label] for label in labels], dtype=np.int32)
    return normalized, label_mapping


# ---------------------------------------------------------------------------
# 2. Подвыборки для настройки и контроля
# ---------------------------------------------------------------------------

def prepare_evaluation(df, features, X):
    """Фиксирует подвыборки и общие матрицы расстояний для сравнения методов.

    Модели получают все строки X.
    Матрицы 3000 × 3000 нужны только для выбора параметров и контроля.
    """
    generator = np.random.default_rng(SAMPLE_SEED)
    sampled_indices = generator.choice(len(X), 2 * SAMPLE_SIZE, replace=False)
    selection_indices = sampled_indices[:SAMPLE_SIZE]
    validation_indices = sampled_indices[SAMPLE_SIZE:]
    pd.DataFrame({
        "row_id": sampled_indices + 1,
        "role": ["selection"] * SAMPLE_SIZE + ["validation"] * SAMPLE_SIZE,
    }).to_csv(RESULTS_DIR / "evaluation_sample.csv", index=False)
    selection_distances = cdist(
        X[selection_indices], X[selection_indices], metric="hamming"
    ).astype(np.float32)
    validation_distances = cdist(
        X[validation_indices], X[validation_indices], metric="hamming"
    ).astype(np.float32)
    return Dataset(
        df=df, features=features, X=X,
        selection_rows=selection_indices, validation_rows=validation_indices,
        selection_distances=selection_distances, validation_distances=validation_distances,
    )


def evaluate_selection(labels, sample_indices, distances):
    """Силуэт настройки и размеры групп для таблиц подбора параметров."""
    cluster_sizes = np.unique(labels, return_counts=True)[1]
    sampled_labels = labels[sample_indices]
    sampled_clusters = len(np.unique(sampled_labels))
    silhouette = np.nan
    if 1 < sampled_clusters < len(sample_indices):
        silhouette = float(silhouette_score(distances, sampled_labels, metric="precomputed"))

    return {
        "n_clusters": len(cluster_sizes),
        "silhouette_selection": silhouette,
        "min_size": int(cluster_sizes.min()),
        "max_size": int(cluster_sizes.max()),
    }


# ---------------------------------------------------------------------------
# 3.1. CLOPE: один порядок строк и лучший из трёх запусков
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3.2. K-modes: модальные центры и стоимость разбиения
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3.3. Подбор параметров двух методов
# ---------------------------------------------------------------------------

def run_method(method, encoded_data, parameter, seed=RANDOM_SEED):
    """Обучает выбранный метод и выводит краткий результат запуска."""
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


def tune_models(data):
    """Проверить сетку параметров, сохраняя метки для последующего сравнения."""
    search_rows = []
    fitted_results = {}
    for method, parameters in PARAMETER_GRID.items():
        for parameter in parameters:
            labels, details = run_method(method, data.X, parameter)
            fitted_results[method, parameter] = (labels, details)
            scores = evaluate_selection(labels, data.selection_rows, data.selection_distances)
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


# ---------------------------------------------------------------------------
# 4. Расстояния, компактность, центры и профили кластеров
# ---------------------------------------------------------------------------

def evaluate_cluster_geometry(encoded_data, labels, fitted_centers=None):
    """Точные расстояния Хэмминга и силуэты по всем строкам через частоты категорий.

    Сумма расстояний от x до кластера равна сумме несовпадений по признакам.
    Это позволяет учитывать все пары, не создавая матрицу 45 211 × 45 211.
    """
    clusters, cluster_sizes = np.unique(labels, return_counts=True)
    n_rows, n_features = encoded_data.shape
    distance_sums = np.zeros((n_rows, len(clusters)), dtype=float)
    centers = {int(cluster): [] for cluster in clusters}

    # 1. Частоты категорий дают модальные центры и суммы расстояний до групп.
    # Для одного признака: сумма несовпадений = размер группы − число совпадений.
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

    # 2. Силуэт: a – среднее расстояние до своей группы без самого объекта;
    # b – минимальное среднее расстояние до остальных групп.
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

    # 3. Компактность: среднее расстояние до моды; радиус – максимальное.
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

    # 4. Межкластерные расстояния: между модами и между всеми парами объектов.
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

    # Общие расстояния взвешены числом пар, компактность и силуэт – числом строк.
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


def save_cluster_profiles(data, labels, method, geometry):
    """Сохранить размеры, моды, частоты и связь признаков с кластерами."""
    raw_data, features = data.df, data.features
    summaries, category_profiles, modes = [], [], []
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
            })
            for category, fraction in frequencies.items():
                category_profiles.append({
                    "method": method,
                    "cluster": int(cluster),
                    "feature": feature,
                    "category": category,
                    "fraction": float(fraction),
                })
        summaries.append({
            "method": method,
            "cluster": int(cluster),
            "n": int(cluster_mask.sum()),
            "percent": float(100 * cluster_mask.mean()),
            "yes_percent": float(100 * (cluster_raw["y"] == "yes").mean()),
            **geometry["cluster_metrics"].loc[cluster].to_dict(),
            "age_median": float(cluster_raw["age"].median()),
            "balance_median": float(cluster_raw["balance"].median()),
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


def evaluate_models(data, search_rows, fitted_results):
    """Выбрать лучшие параметры по силуэту и оценить полученные разбиения."""
    selected_models = {}
    metric_rows, silhouette_tables, between_tables = [], [], []
    assignments = pd.DataFrame({"row_id": np.arange(1, len(data.df) + 1), "y": data.df["y"]})

    for method in PARAMETER_GRID:
        candidates = [
            row for row in search_rows
            if row["method"] == method and np.isfinite(row["silhouette_selection"])
        ]
        best = max(candidates, key=lambda row: row["silhouette_selection"])
        labels, details = fitted_results[method, best["parameter"]]
        selected_models[method] = {"parameter": best["parameter"], **details}
        geometry = evaluate_cluster_geometry(data.X, labels, details.get("modes_encoded"))
        assignments[method] = labels
        assignments[f"{method}_distance_to_mode"] = save_cluster_profiles(data, labels, method, geometry)
        between_tables.append(geometry["between_clusters"].assign(method=method))

        validation_labels = labels[data.validation_rows]
        validation_silhouette = silhouette_score(
            data.validation_distances, validation_labels, metric="precomputed"
        )
        silhouette_tables.append(pd.DataFrame({
            "method": method,
            "row_id": np.arange(1, len(labels) + 1),
            "cluster": labels,
            "silhouette": geometry["silhouettes"],
        }))
        metric_rows.append({
            **best,
            "silhouette_validation": float(validation_silhouette),
            "ARI_y": float(adjusted_rand_score(data.df["y"], labels)),
            **geometry["metrics"],
        })
    metrics = pd.DataFrame(metric_rows)
    silhouettes = pd.concat(silhouette_tables)
    silhouettes.to_csv(RESULTS_DIR / "silhouette_values.csv", index=False)
    assignments.to_csv(RESULTS_DIR / "assignments.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pd.concat(between_tables).to_csv(RESULTS_DIR / "intercluster_distances.csv", index=False)
    return selected_models, assignments, metrics, silhouettes


# ---------------------------------------------------------------------------
# 5. Необходимость третьего кластера CLOPE
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. Сравнение методов и сохранение параметров
# ---------------------------------------------------------------------------

def save_model_comparison(data, selected_models, assignments):
    """Сохраняет параметры лучших запусков и ARI между двумя методами."""
    comparison_ari = float(adjusted_rand_score(assignments["CLOPE"], assignments["k-modes"]))
    save_json("summary.json", {
        "n": len(data.X),
        "p": data.X.shape[1],
        "selection_sample_size": SAMPLE_SIZE,
        "validation_sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "distance": "normalized Hamming; Gower for complete nominal data with equal weights",
        "selection_rule": "CLOPE: maximum Profit over three orders at each r; then maximum selection silhouette; no target used",
        "quality_evaluation": "exact full-data Hamming distances, modal compactness and silhouette via category counts",
        "selected": selected_models,
        "cross_method_ARI": comparison_ari,
    })
    return comparison_ari


# ---------------------------------------------------------------------------
# 7. Графики для отчёта
# ---------------------------------------------------------------------------

def configure_plots():
    """Общий шрифт, стиль и разрешение всех рисунков."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.titleweight": "bold"})


def plot_parameter_selection(search_results):
    """Силуэт при разных r и k; подписи показывают число непустых групп."""
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
    """Размеры групп в двух лучших разбиениях."""
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for axis, method in zip(axes, PARAMETER_GRID):
        counts = assignments[method].value_counts().sort_index()
        axis.bar(counts.index.astype(str), counts.values, color="#527a96")
        axis.set(title=method, xlabel="Кластер", ylabel="Число записей")
    save_figure(figure, "06_cluster_sizes.png")


def plot_feature_importance():
    """V Крамера для связи каждого признака с номером кластера."""
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
    """Индивидуальные силуэты всех строк, упорядоченные внутри каждой группы."""
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


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Чтение, категоризация и дескриптивная статистика
    df, features, X = prepare_data(RESULTS_DIR)

    # 2. Общие для двух методов подвыборки и матрицы Хэмминга.
    data = prepare_evaluation(df, features, X)

    # 3. Подбор r для CLOPE и k для k-modes без использования Y.
    search_rows, fitted_results = tune_models(data)

    # 4. Лучшие разбиения: качество, центры, профили и важность признаков.
    selected_models, assignments, metrics, silhouettes = evaluate_models(data, search_rows, fitted_results)

    # 5. Два и три кластера CLOPE; отдельно – объединение малой группы.
    compare_clope_cluster_counts(data.X, fitted_results)

    # 6. Согласие двух разбиений и параметры выбранных моделей.
    comparison_ari = save_model_comparison(data, selected_models, assignments)

    # 7. Рисунки строятся по тем же данным, которые сохранены в таблицах.
    configure_plots()
    plot_parameter_selection(pd.DataFrame(search_rows))
    plot_cluster_sizes(assignments)
    plot_feature_importance()
    plot_silhouettes(silhouettes)

    print(metrics.to_string(index=False), flush=True)
    print("Cross-method ARI:", comparison_ari, flush=True)


if __name__ == "__main__":
    main()
