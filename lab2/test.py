"""ЛР 2. Классификация сортов риса деревом решений и методом SVM.

Запуск из MAD: lab1.3/.venv/bin/python lab2/classification_analysis.py
Библиотеки: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn.
Исходный ARFF, таблицы, рисунки и параметры моделей находятся в results.
Для создания отчёта отдельно запустите build_report.py.

Порядок работы скрипта (соответствует пунктам задания):
    1. Загрузка данных и дескриптивный анализ (в том числе тест Шапиро–Уилка).
    2. Разбиение на обучающую и тестовую выборки.
    3. Отбор информативных признаков и диаграммы рассеяния.
    4. Подбор параметров двух моделей: дерево решений и SVM.
    5. Важность признаков.
    6. Дерево на отобранных признаках (исключение малозначимых).
    7. Сравнение моделей: метрики, матрицы ошибок, ROC-кривые.
    8. Влияние глубины дерева, визуализация дерева, сохранение параметров.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mad_lab2_matplotlib")

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, без окна
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
from scipy.stats import shapiro
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
ARFF_PATH = RESULTS / "Rice_Cammeo_Osmancik.arff"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

CLASS_NAMES = ["Osmancik", "Cammeo"]            # 0 и 1; положительный класс — Cammeo
CLASS_CODES = {"Osmancik": 0, "Cammeo": 1}

SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "auc": "roc_auc"}
REFIT_METRIC = "f1_macro"                       # по ней выбираются лучшие параметры

TREE_NAME, SVM_NAME, REDUCED_TREE_NAME = "Дерево", "SVM", "Дерево, отбор"

# Отбор признаков: порог важности по Gini для SelectFromModel.
SELECTION_THRESHOLD = 0.02

TREE_GRID = {
    "max_depth": [5],
    "min_samples_leaf": [5],
}
SVM_GRID = [
    {"model__kernel": ["linear"], "model__C": [0.1, 1, 10, 100]},
    {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100],
     "model__gamma": ["scale", 0.01, 0.1, 1]},
]

SCATTER_FEATURES = ["Major_Axis_Length", "Perimeter", "Eccentricity", "Extent"]
DEPTHS_TO_TEST = [1, 2, 3, 4, 5, 6, 8, None]
PERMUTATION_REPEATS = 10

sns.set_theme(style="whitegrid", font="DejaVu Sans")


@dataclass
class Dataset:
    """Данные, разбиение train/test и схема кросс-валидации."""

    df: pd.DataFrame            # исходная таблица вместе с колонкой Class
    X: pd.DataFrame             # признаки
    y: pd.Series                # метки: 0 — Osmancik, 1 — Cammeo
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    cv: StratifiedKFold


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def save_figure(name):
    """Сохраняет текущий рисунок в results и закрывает его."""
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=180, bbox_inches="tight")
    plt.close()


def get_scores(model, X):
    """Оценка принадлежности к положительному классу (для ROC и AUC)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def calculate_metrics(model, X, y):
    """Accuracy, AUC, а также precision/recall/F1 с тремя видами усреднения."""
    prediction = model.predict(X)
    result = {
        "Accuracy": accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, get_scores(model, X)),
    }
    for average in ["binary", "macro", "micro"]:
        result[f"Precision_{average}"] = precision_score(y, prediction, average=average)
        result[f"Recall_{average}"] = recall_score(y, prediction, average=average)
        result[f"F1_{average}"] = f1_score(y, prediction, average=average)
    return result


def model_label(name):
    """Префикс колонок важности: 'Tree' для дерева, 'SVM' для SVM."""
    return "Tree" if name == TREE_NAME else "SVM"


# ---------------------------------------------------------------------------
# 1. Данные и дескриптивный анализ
# ---------------------------------------------------------------------------

def load_data():
    """Читает ARFF; Class — ответ, остальные столбцы — признаки."""
    records, _ = arff.loadarff(ARFF_PATH)
    df = pd.DataFrame(records)
    df["Class"] = df["Class"].str.decode("utf-8")
    X = df.drop(columns="Class")
    y = df["Class"].map(CLASS_CODES)
    return df, X, y


def describe_data(df, X):
    """Пропуски, дубликаты, описательные статистики, баланс классов."""
    df.info()
    print("Пропуски:", df.isna().sum().sum(), "Дубликаты:", df.duplicated().sum())

    descriptive = X.describe().T
    descriptive["variance"] = X.var()
    descriptive["skew"] = X.skew()
    descriptive.to_csv(RESULTS / "descriptive_statistics.csv", index_label="Feature")
    df.groupby("Class")[X.columns].describe().to_csv(RESULTS / "class_statistics.csv")
    print(descriptive.round(3).to_string())

    balance = df["Class"].value_counts().reindex(CLASS_NAMES).to_frame("Count")
    balance["Percent"] = 100 * balance["Count"] / len(df)
    balance.to_csv(RESULTS / "class_balance.csv", index_label="Class")


def test_normality(df, X):
    """Тест Шапиро–Уилка для каждого признака: по всей выборке и по классам."""
    rows = []
    for group in ["Все", *CLASS_NAMES]:
        part = X if group == "Все" else X.loc[df["Class"] == group]
        for feature in X.columns:
            statistic, p_value = shapiro(part[feature])
            rows.append({"Group": group, "Feature": feature, "n": len(part),
                         "W": statistic, "p_value": p_value})
    pd.DataFrame(rows).to_csv(RESULTS / "normality_shapiro.csv", index=False)


def plot_distributions(df, X):
    """Гистограммы признаков по классам и число объектов в классах."""
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    for ax, feature in zip(axes.flat, X.columns):
        sns.histplot(data=df, x=feature, hue="Class", hue_order=CLASS_NAMES,
                     bins="sturges", stat="density", common_norm=False,
                     element="step", ax=ax)
    last_ax = axes.flat[-1]
    sns.countplot(data=df, x="Class", order=CLASS_NAMES, ax=last_ax, color="#6699bb")
    last_ax.set_ylabel("Число объектов")
    save_figure("01_distributions.png")


# ---------------------------------------------------------------------------
# 2. Разбиение на обучающую и тестовую выборки
# ---------------------------------------------------------------------------

def split_dataset(df, X, y):
    """Стратифицированное разбиение. Тест не участвует ни в отборе признаков,
    
    ни в настройке моделей."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    subset = pd.Series("train", index=df.index, name="Subset")
    subset.loc[X_test.index] = "test"
    subset.to_csv(RESULTS / "split_assignment.csv", index_label="Row")
    pd.crosstab(subset, df["Class"]).reindex(columns=CLASS_NAMES).to_csv(
        RESULTS / "split_counts.csv")

    return Dataset(df, X, y, X_train, X_test, y_train, y_test, cv)


# ---------------------------------------------------------------------------
# 3. Отбор признаков, корреляции, диаграммы рассеяния
# ---------------------------------------------------------------------------

def make_feature_selector():
    """Селектор: признаки с важностью по Gini выше порога в неглубоком дереве."""
    return SelectFromModel(
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED),
        threshold=SELECTION_THRESHOLD,
    )


def select_features(data):
    """Отбор по обучающим данным. В уменьшенной модели этот же отбор
    повторяется внутри каждого CV-фолда (см. build_reduced_model)."""
    X, X_train, y_train = data.X, data.X_train, data.y_train
    selector = make_feature_selector().fit(X_train, y_train)
    selected = X.columns[selector.get_support()].tolist()

    table = pd.DataFrame({
        "Feature": X.columns,
        "Mutual_information": mutual_info_classif(X_train, y_train, random_state=SEED),
        "Selection_Gini": selector.estimator_.feature_importances_,
        "Selected": selector.get_support(),
    }).sort_values("Mutual_information", ascending=False)
    table.to_csv(RESULTS / "feature_selection.csv", index=False)
    print("Отобранные признаки:", selected)
    return selector


def plot_correlations(X_train):
    """Матрица корреляций признаков (только обучающая выборка)."""
    correlation = X_train.corr()
    correlation.to_csv(RESULTS / "train_correlations.csv", index_label="Feature")
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Корреляции признаков на обучающей выборке")
    save_figure("02_correlations.png")


def plot_scatterplots(df):
    """Категоризованные диаграммы рассеяния по выбранным признакам."""
    grid = sns.pairplot(df, vars=SCATTER_FEATURES, hue="Class", hue_order=CLASS_NAMES,
                        diag_kind="hist", plot_kws={"s": 9, "alpha": 0.35},
                        diag_kws={"bins": 20, "alpha": 0.45}, height=2.5)
    grid.savefig(RESULTS / "03_scatterplots.png", dpi=170)
    plt.close(grid.figure)


# ---------------------------------------------------------------------------
# 4. Подбор параметров моделей
# ---------------------------------------------------------------------------

def tune_models(data):
    """Подбирает параметры по сетке с кросс-валидацией на обучающей выборке.

    SVM масштабируется внутри Pipeline (чтобы не было утечки данных),
    дерево масштабирования не требует.
    Возвращает (лучшие модели, объекты GridSearchCV).
    """
    candidates = {
        TREE_NAME: (DecisionTreeClassifier(random_state=SEED), TREE_GRID, "tree_cv.csv"),
        SVM_NAME: (Pipeline([("scale", StandardScaler()), ("model", SVC())]),
                   SVM_GRID, "svm_cv.csv"),
    }
    models, searches = {}, {}
    for name, (model, grid, cv_file) in candidates.items():
        search = GridSearchCV(model, grid, scoring=SCORING, refit=REFIT_METRIC,
                              cv=data.cv, return_train_score=True)
        search.fit(data.X_train, data.y_train)
        pd.DataFrame(search.cv_results_).to_csv(RESULTS / cv_file, index=False)
        print(name, search.best_params_,
              "CV F1-macro:", round(search.best_score_, 5), flush=True)
        models[name] = search.best_estimator_
        searches[name] = search
    return models, searches


def save_scaling(svm_pipeline, feature_names):
    """Среднее и масштаб, по которым SVM стандартизирует признаки."""
    scaler = svm_pipeline.named_steps["scale"]
    pd.DataFrame({"Feature": feature_names, "Train_mean": scaler.mean_,
                  "Train_scale": scaler.scale_}).to_csv(RESULTS / "scaling.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Важность признаков
# ---------------------------------------------------------------------------

def cv_permutation_importance(model, data):
    """Перестановочная важность на валидационных фолдах обучающей выборки.

    Для каждого фолда модель обучается заново; важность — падение F1-macro
    после перемешивания признака. Возвращает массив (признаки × повторы).
    """
    per_fold = []
    for fit_rows, validation_rows in data.cv.split(data.X_train, data.y_train):
        fitted = clone(model).fit(data.X_train.iloc[fit_rows], data.y_train.iloc[fit_rows])
        result = permutation_importance(
            fitted, data.X_train.iloc[validation_rows], data.y_train.iloc[validation_rows],
            scoring="f1_macro", n_repeats=PERMUTATION_REPEATS, random_state=SEED,
        )
        per_fold.append(result.importances)
    return np.concatenate(per_fold, axis=1)


def compute_feature_importance(models, data):
    """Важность по Gini (дерево) и перестановочная важность обеих моделей."""
    importance = pd.DataFrame(index=data.X.columns)
    importance["Tree_Gini"] = models[TREE_NAME].feature_importances_
    for name, model in models.items():
        values = cv_permutation_importance(model, data)
        prefix = model_label(name)
        importance[prefix + "_permutation_mean"] = values.mean(axis=1)
        importance[prefix + "_permutation_std"] = values.std(axis=1)
    importance.to_csv(RESULTS / "feature_importance.csv", index_label="Feature")
    return importance


def plot_feature_importance(importance):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    importance["Tree_Gini"].plot.barh(ax=axes[0], title="Дерево: уменьшение Gini")
    for ax, prefix in zip(axes[1:], ["Tree", "SVM"]):
        ax.barh(importance.index, importance[prefix + "_permutation_mean"],
                xerr=importance[prefix + "_permutation_std"], capsize=3)
        ax.set_title(prefix + ": перестановочная важность")
        ax.set_xlabel("Падение F1-macro на CV")
    save_figure("04_feature_importance.png")


# ---------------------------------------------------------------------------
# 6. Исключение малозначимых признаков
# ---------------------------------------------------------------------------

def build_reduced_model(selector, tree, data):
    """Дерево с теми же параметрами, но обученное только на отобранных признаках.

    Отбор встроен в Pipeline, поэтому в кросс-валидации он повторяется
    в каждом фолде. Возвращает (обученную модель, результаты CV).
    """
    reduced = Pipeline([("select", clone(selector)), ("model", clone(tree))])
    reduced_cv = cross_validate(reduced, data.X_train, data.y_train,
                                cv=data.cv, scoring=SCORING)
    reduced.fit(data.X_train, data.y_train)
    return reduced, reduced_cv


# ---------------------------------------------------------------------------
# 7. Сравнение моделей на train и test
# ---------------------------------------------------------------------------

def evaluate_models(all_models, data):
    """Метрики, метрики по классам, матрицы ошибок и предсказания на тесте."""
    metric_rows, class_rows, confusion_rows = [], [], []
    predictions = pd.DataFrame({"Row": data.y_test.index,
                                "True_class": data.y_test.to_numpy()})
    subsets = [("train", data.X_train, data.y_train), ("test", data.X_test, data.y_test)]

    for name, model in all_models.items():
        for subset, part_X, part_y in subsets:
            metric_rows.append({"Model": name, "Subset": subset,
                                **calculate_metrics(model, part_X, part_y)})

            prediction = model.predict(part_X)
            report = classification_report(part_y, prediction, target_names=CLASS_NAMES,
                                           output_dict=True)
            for class_name in CLASS_NAMES:
                class_rows.append({"Model": name, "Subset": subset, "Class": class_name,
                                   **report[class_name]})

            tn, fp, fn, tp = confusion_matrix(part_y, prediction, labels=[0, 1]).ravel()
            confusion_rows.append({"Model": name, "Subset": subset,
                                   "TN": tn, "FP": fp, "FN": fn, "TP": tp})

        predictions[name + "_prediction"] = model.predict(data.X_test)
        predictions[name + "_score"] = get_scores(model, data.X_test)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / "class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(RESULTS / "confusion_matrices.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    return predictions


def summarize_cv(searches, reduced_cv):
    """Среднее и std метрик кросс-валидации для лучших параметров каждой модели."""
    rows = []
    for name, search in searches.items():
        row = {"Model": name}
        for metric in SCORING:
            row[metric + "_mean"] = search.cv_results_["mean_test_" + metric][search.best_index_]
            row[metric + "_std"] = search.cv_results_["std_test_" + metric][search.best_index_]
        rows.append(row)

    reduced_row = {"Model": REDUCED_TREE_NAME}
    for metric in SCORING:
        fold_scores = reduced_cv["test_" + metric]
        reduced_row[metric + "_mean"] = np.mean(fold_scores)
        reduced_row[metric + "_std"] = np.std(fold_scores)
    rows.append(reduced_row)

    pd.DataFrame(rows).to_csv(RESULTS / "cv_summary.csv", index=False)


def plot_confusion_matrices(all_models, data):
    """Строки — train/test, столбцы — модели."""
    fig, axes = plt.subplots(2, len(all_models), figsize=(10, 7))
    subsets = [("train", data.X_train, data.y_train), ("test", data.X_test, data.y_test)]
    for column, (name, model) in enumerate(all_models.items()):
        for row, (subset, part_X, part_y) in enumerate(subsets):
            ax = axes[row, column]
            ConfusionMatrixDisplay.from_estimator(
                model, part_X, part_y, display_labels=CLASS_NAMES, cmap="Blues",
                colorbar=False, values_format="d", ax=ax,
            )
            ax.set(title=f"{name}, {subset}",
                   xlabel="Предсказанный класс", ylabel="Истинный класс")
            ax.grid(False)
    save_figure("05_confusion_matrices.png")


def plot_roc_curves(all_models, predictions, y_test):
    plt.figure(figsize=(8, 6))
    for name in all_models:
        scores = predictions[name + "_score"]
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name}: AUC = {roc_auc_score(y_test, scores):.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR = FP / (FP + TN)")
    plt.ylabel("TPR = Recall")
    plt.title("ROC-кривые на тестовой выборке, положительный класс Cammeo")
    plt.legend()
    save_figure("06_roc.png")


# ---------------------------------------------------------------------------
# 8. Влияние глубины дерева, визуализация, параметры моделей
# ---------------------------------------------------------------------------

def analyze_tree_depth(tree, data):
    """Меняем max_depth при постоянном min_samples_leaf выбранного дерева."""
    rows = []
    for depth in DEPTHS_TO_TEST:
        model = clone(tree).set_params(max_depth=depth)
        scores = cross_validate(model, data.X_train, data.y_train, cv=data.cv,
                                scoring=SCORING, return_train_score=True)
        model.fit(data.X_train, data.y_train)
        test_prediction = model.predict(data.X_test)
        rows.append({
            "Depth": str(depth),
            "Train_accuracy": accuracy_score(data.y_train, model.predict(data.X_train)),
            "CV_accuracy": scores["test_accuracy"].mean(),
            "CV_F1_macro": scores["test_f1_macro"].mean(),
            "CV_F1_std": scores["test_f1_macro"].std(),
            "CV_AUC": scores["test_auc"].mean(),
            "Train_CV_F1_macro": scores["train_f1_macro"].mean(),
            "Test_accuracy": accuracy_score(data.y_test, test_prediction),
            "Test_F1_macro": f1_score(data.y_test, test_prediction, average="macro"),
        })
    depth_effect = pd.DataFrame(rows)
    depth_effect.to_csv(RESULTS / "tree_depth_effect.csv", index=False)
    print("Влияние глубины:\n", depth_effect.round(4).to_string(index=False))
    return depth_effect


def plot_depth_effect(depth_effect):
    plt.figure(figsize=(9, 5))
    plt.plot(depth_effect["Depth"], depth_effect["Train_CV_F1_macro"], "o-",
             label="Train, среднее по CV")
    plt.errorbar(depth_effect["Depth"], depth_effect["CV_F1_macro"],
                 yerr=depth_effect["CV_F1_std"], fmt="o-", capsize=4,
                 label="Validation, CV ± std")
    plt.xlabel("max_depth (None – без ограничения глубины)")
    plt.ylabel("F1-macro")
    plt.title("Влияние глубины дерева при постоянном min_samples_leaf")
    plt.legend()
    save_figure("07_depth_effect.png")


def save_tree_visualization(tree, feature_names):
    """Рисунок дерева с индексами Gini и текстовые правила.
    Размер рисунка зависит от числа листьев и глубины."""
    plt.figure(figsize=(max(10, tree.get_n_leaves() * 1.6),
                        max(5, tree.get_depth() * 1.5)))
    plot_tree(tree, feature_names=feature_names, class_names=CLASS_NAMES,
              filled=True, rounded=True, impurity=True, precision=3, fontsize=13)
    save_figure("08_decision_tree.png")
    (RESULTS / "tree_rules.txt").write_text(
        export_text(tree, feature_names=feature_names, decimals=3), encoding="utf-8",
    )


def save_model_parameters(tree, svm_pipeline):
    """Параметры для отчёта (сохраняются отдельно от обученных моделей)."""
    svm = svm_pipeline.named_steps["model"]
    parameters = {
        TREE_NAME: {
            "max_depth": tree.max_depth,
            "min_samples_leaf": tree.min_samples_leaf,
            "node_count": int(tree.tree_.node_count),
            "n_leaves": int(tree.get_n_leaves()),
            "root_gini": float(tree.tree_.impurity[0]),
        },
        SVM_NAME: {
            "kernel": svm.kernel,
            "C": svm.C,
            "n_support": svm.n_support_.tolist(),
            "n_support_total": int(svm.n_support_.sum()),
        },
    }
    (RESULTS / "model_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8",
    )

def main():
    RESULTS.mkdir(exist_ok=True)

    # 1. Данные и дескриптивный анализ
    df, X, y = load_data()
    describe_data(df, X)
    test_normality(df, X)
    plot_distributions(df, X)

    # 2. Разбиение
    data = split_dataset(df, X, y)

    # 3. Отбор признаков и диаграммы
    selector = select_features(data)
    plot_correlations(data.X_train)
    plot_scatterplots(df)

    # 4. Подбор параметров двух моделей
    models, searches = tune_models(data)
    save_scaling(models[SVM_NAME], X.columns)

    # 5. Важность признаков
    importance = compute_feature_importance(models, data)
    plot_feature_importance(importance)

    # 6. Модель без малозначимых признаков
    reduced, reduced_cv = build_reduced_model(selector, models[TREE_NAME], data)
    all_models = {**models, REDUCED_TREE_NAME: reduced}

    # 7. Сравнение моделей
    predictions = evaluate_models(all_models, data)
    summarize_cv(searches, reduced_cv)
    plot_confusion_matrices(all_models, data)
    plot_roc_curves(all_models, predictions, data.y_test)

    # 8. Глубина дерева, визуализация, параметры
    tree = models[TREE_NAME]
    plot_depth_effect(analyze_tree_depth(tree, data))
    save_tree_visualization(tree, X.columns.tolist())
    print("Важность признаков:\n", importance.round(4).to_string())
    save_model_parameters(tree, models[SVM_NAME])


if __name__ == "__main__":
    main()