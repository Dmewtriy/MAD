"""Чтение, категоризация и дескриптивный анализ Bank Marketing.

prepare_data(results_dir) возвращает исходные данные, категориальные признаки
и их числовые коды для CLOPE и k-modes. Ответ Y не включается в признаки.

Порядок подготовки: чтение CSV → интервалы числовых признаков → статистика → коды категорий.
"""

from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats.contingency import association

# ---------------------------------------------------------------------------
# Константы набора данных
# ---------------------------------------------------------------------------

DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
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


# ---------------------------------------------------------------------------
# 1. Чтение локального CSV или загрузка архива UCI
# ---------------------------------------------------------------------------

def load_raw_data(results_dir):
    """Загрузить локальный CSV; при его отсутствии скачать исходный архив UCI."""
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "bank-full.csv"
    if not csv_path.exists():
        with urlopen(DATASET_URL, timeout=90) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as outer_archive:
            bank_archive_name = next(
                name for name in outer_archive.namelist() if name.endswith("bank.zip")
            )
            with ZipFile(BytesIO(outer_archive.read(bank_archive_name))) as bank_archive:
                for filename in ("bank-full.csv", "bank-names.txt"):
                    (results_dir / filename).write_bytes(bank_archive.read(filename))
    return pd.read_csv(csv_path, sep=";")


# ---------------------------------------------------------------------------
# 2. Подготовка общих категориальных признаков
# ---------------------------------------------------------------------------

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
            })
    return features, pd.DataFrame(binning_rows), duration_quartiles


# ---------------------------------------------------------------------------
# 3. Дескриптивный анализ
# ---------------------------------------------------------------------------

def save_feature_statistics(raw_data, features, results_dir):
    """Сохранить статистики таблицы 2, пропуски, unknown и баланс ответа Y."""
    statistics = raw_data[NUMERIC_FEATURES].agg(["min", "median", "mean", "max"]).T
    statistics = statistics.rename(columns={"median": "50%"})
    statistics.to_csv(results_dir / "numeric_statistics.csv", index_label="feature")
    quality_rows = []
    for feature in features:
        quality_rows.append({
            "feature": feature,
            "type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
            "missing": int(raw_data[feature].isna().sum()),
            "unknown": int((raw_data[feature] == "unknown").sum()),
        })
    pd.DataFrame(quality_rows).to_csv(results_dir / "feature_quality.csv", index=False)

    class_balance = raw_data["y"].value_counts().rename_axis("class").reset_index(name="count")
    class_balance["percent"] = class_balance["count"] / len(raw_data) * 100
    class_balance.to_csv(results_dir / "class_balance.csv", index=False)


def save_feature_associations(features, results_dir):
    """Рассчитать V Крамера для каждой пары подготовленных признаков."""
    pair_rows = []
    for feature, other_feature in combinations(features.columns, 2):
        frequency_table = pd.crosstab(features[feature], features[other_feature])
        cramers_v = float(association(frequency_table.to_numpy(), method="cramer", correction=False))
        pair_rows.append({
            "feature_1": feature,
            "feature_2": other_feature,
            "cramers_v": cramers_v,
        })
    pairs = pd.DataFrame(pair_rows).sort_values("cramers_v", ascending=False)
    pairs.to_csv(results_dir / "association_pairs.csv", index=False)


# ---------------------------------------------------------------------------
# Общая последовательность подготовки
# ---------------------------------------------------------------------------

def prepare_data(results_dir):
    """Подготовить общие для двух методов признаки и сохранить их описание."""
    results_dir = Path(results_dir)
    raw_data = load_raw_data(results_dir)
    assert raw_data.shape == (45211, 17)
    assert not raw_data.isna().any().any()
    features, binning_rules, duration_quartiles = categorize_features(raw_data)
    assert not features.isna().any().any()

    features.to_csv(results_dir / "prepared_categorical.csv", index=False)
    binning_rules.to_csv(results_dir / "binning_rules.csv", index=False)
    save_feature_statistics(raw_data, features, results_dir)
    save_feature_associations(features, results_dir)
    metadata = {
        "n": len(raw_data),
        "p": len(features.columns),
        "numeric": NUMERIC_FEATURES,
        "categorical": [feature for feature in features if feature not in NUMERIC_FEATURES],
        "target": "y",
        "duplicates_raw": int(raw_data.duplicated().sum()),
        "duplicates_after_binning": int(features.duplicated().sum()),
        "missing_cells": int(raw_data.isna().sum().sum()),
        "duration_quartiles": duration_quartiles,
        "prepared_levels": {feature: sorted(features[feature].unique().tolist()) for feature in features},
        "preparation": "all 16 features, seven numeric features discretized; target excluded; unknown retained",
    }
    (results_dir / "data_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Числа обозначают категории; расстояние использует только совпадение кодов.
    encoded_data = np.column_stack([
        pd.Categorical(features[feature]).codes for feature in features
    ]).astype(np.int16)
    return raw_data, features, encoded_data
