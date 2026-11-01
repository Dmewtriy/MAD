"""ЛР 2. Классификация сортов риса деревом решений и методом SVM.

Запуск из MAD: lab1.3/.venv/bin/python lab2/classification_analysis.py
Библиотеки: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn.
Исходный ARFF, таблицы, рисунки и параметры моделей находятся в results.
Для создания отчёта отдельно запустите build_report.py.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mad_lab2_matplotlib")

import matplotlib

matplotlib.use("Agg")
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


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
SEED = 42
CLASS_NAMES = ["Osmancik", "Cammeo"]
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "auc": "roc_auc"}
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def save_figure(name):
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=180, bbox_inches="tight")
    plt.close()


def calculate_metrics(model, X, y):
    prediction = model.predict(X)
    scores = (model.decision_function(X) if hasattr(model, "decision_function")
              else model.predict_proba(X)[:, 1])
    result = {"Accuracy": accuracy_score(y, prediction),
              "AUC": roc_auc_score(y, scores)}
    for average in ["binary", "macro", "micro"]:
        result[f"Precision_{average}"] = precision_score(y, prediction, average=average)
        result[f"Recall_{average}"] = recall_score(y, prediction, average=average)
        result[f"F1_{average}"] = f1_score(y, prediction, average=average)
    return result


def main():
    RESULTS.mkdir(exist_ok=True)

    # 1. Данные и дескриптивный анализ. Class — ответ, остальные столбцы — признаки.
    records, _ = arff.loadarff(RESULTS / "Rice_Cammeo_Osmancik.arff")
    df = pd.DataFrame(records)
    df["Class"] = df["Class"].str.decode("utf-8")
    X = df.drop(columns="Class")
    y = df["Class"].map({"Osmancik": 0, "Cammeo": 1})
    df.info()
    print("Пропуски:", df.isna().sum().sum(), "Дубликаты:", df.duplicated().sum())
    descriptive = X.describe().T
    descriptive["variance"] = X.var()
    descriptive["skew"] = X.skew()
    descriptive.to_csv(RESULTS / "descriptive_statistics.csv", index_label="Feature")
    df.groupby("Class")[X.columns].describe().to_csv(RESULTS / "class_statistics.csv")
    print(descriptive.round(3).to_string())

    class_balance = df["Class"].value_counts().reindex(CLASS_NAMES).to_frame("Count")
    class_balance["Percent"] = 100 * class_balance["Count"] / len(df)
    class_balance.to_csv(RESULTS / "class_balance.csv", index_label="Class")

    normality_rows = []
    for group in ["Все", *CLASS_NAMES]:
        part = X if group == "Все" else X.loc[df["Class"] == group]
        for feature in X.columns:
            statistic, p_value = shapiro(part[feature])
            normality_rows.append({"Group": group, "Feature": feature, "n": len(part),
                                   "W": statistic, "p_value": p_value})
    normality = pd.DataFrame(normality_rows)
    normality.to_csv(RESULTS / "normality_shapiro.csv", index=False)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    for ax, feature in zip(axes.flat, X.columns):
        sns.histplot(data=df, x=feature, hue="Class", hue_order=CLASS_NAMES,
                     bins="sturges", stat="density", common_norm=False,
                     element="step", ax=ax)
    sns.countplot(data=df, x="Class", order=CLASS_NAMES, ax=axes.flat[-1], color="#6699bb")
    axes.flat[-1].set_ylabel("Число объектов")
    save_figure("01_distributions.png")

    # 2. Разбиение. Тест не участвует в отборе признаков и настройке моделей.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    split = pd.Series("train", index=df.index, name="Subset")
    split.loc[X_test.index] = "test"
    split.to_csv(RESULTS / "split_assignment.csv", index_label="Row")
    split_counts = pd.crosstab(split, df["Class"]).reindex(columns=CLASS_NAMES)
    split_counts.to_csv(RESULTS / "split_counts.csv")

    # 3. Информативность и отбор признаков по обучающим данным.
    # В уменьшенной модели этот же отбор повторяется внутри каждого CV-фолда.
    selector = SelectFromModel(
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED),
        threshold=0.02,
    ).fit(X_train, y_train)
    selected = X.columns[selector.get_support()].tolist()
    feature_selection = pd.DataFrame({
        "Feature": X.columns,
        "Mutual_information": mutual_info_classif(X_train, y_train, random_state=SEED),
        "Selection_Gini": selector.estimator_.feature_importances_,
        "Selected": selector.get_support(),
    }).sort_values("Mutual_information", ascending=False)
    feature_selection.to_csv(RESULTS / "feature_selection.csv", index=False)
    print("Отобранные признаки:", selected)

    correlation = X_train.corr()
    correlation.to_csv(RESULTS / "train_correlations.csv", index_label="Feature")
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Корреляции признаков на обучающей выборке")
    save_figure("02_correlations.png")

    scatter_features = ["Major_Axis_Length", "Perimeter", "Eccentricity", "Extent"]
    grid = sns.pairplot(df, vars=scatter_features, hue="Class", hue_order=CLASS_NAMES,
                        diag_kind="hist", plot_kws={"s": 9, "alpha": 0.35},
                        diag_kws={"bins": 20, "alpha": 0.45}, height=2.5)
    grid.savefig(RESULTS / "03_scatterplots.png", dpi=170)
    plt.close(grid.figure)

    # 4. Два классификатора. SVM масштабируется внутри Pipeline, дерево — нет.
    models = {
        "Дерево": DecisionTreeClassifier(random_state=SEED),
        "SVM": Pipeline([("scale", StandardScaler()), ("model", SVC())]),
    }
    grids = {
        "Дерево": {"max_depth": [2, 3, 4, 5, 6, 8, None],
                   "min_samples_leaf": [1, 5, 10, 20]},
        "SVM": [{"model__kernel": ["linear"], "model__C": [0.1, 1, 10, 100]},
                {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100],
                 "model__gamma": ["scale", 0.01, 0.1, 1]}],
    }
    searches = {}
    for name, model in models.items():
        search = GridSearchCV(model, grids[name], scoring=SCORING, refit="f1_macro",
                              cv=cv, return_train_score=True)
        search.fit(X_train, y_train)
        searches[name] = search
        models[name] = search.best_estimator_
        pd.DataFrame(search.cv_results_).to_csv(
            RESULTS / ("tree_cv.csv" if name == "Дерево" else "svm_cv.csv"), index=False,
        )
        print(name, search.best_params_, "CV F1-macro:", round(search.best_score_, 5), flush=True)

    scaler = models["SVM"].named_steps["scale"]
    pd.DataFrame({"Feature": X.columns, "Train_mean": scaler.mean_,
                  "Train_scale": scaler.scale_}).to_csv(RESULTS / "scaling.csv", index=False)

    # 5. Важность признаков. Перестановки выполняются на валидационных фолдах train.
    importance = pd.DataFrame(index=X.columns)
    importance["Tree_Gini"] = models["Дерево"].feature_importances_
    for name, model in models.items():
        fold_importances = []
        for fit_rows, validation_rows in cv.split(X_train, y_train):
            fitted = clone(model).fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
            result = permutation_importance(
                fitted, X_train.iloc[validation_rows], y_train.iloc[validation_rows],
                scoring="f1_macro", n_repeats=10, random_state=SEED,
            )
            fold_importances.append(result.importances)
        values = np.concatenate(fold_importances, axis=1)
        prefix = "Tree" if name == "Дерево" else "SVM"
        importance[prefix + "_permutation_mean"] = values.mean(axis=1)
        importance[prefix + "_permutation_std"] = values.std(axis=1)
    importance.to_csv(RESULTS / "feature_importance.csv", index_label="Feature")

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    importance["Tree_Gini"].plot.barh(ax=axes[0], title="Дерево: уменьшение Gini")
    for ax, prefix in zip(axes[1:], ["Tree", "SVM"]):
        ax.barh(importance.index, importance[prefix + "_permutation_mean"],
                xerr=importance[prefix + "_permutation_std"], capsize=3)
        ax.set_title(prefix + ": перестановочная важность")
        ax.set_xlabel("Падение F1-macro на CV")
    save_figure("04_feature_importance.png")

    # 6. Исключение малозначимых признаков с сохранением параметров дерева.
    reduced = Pipeline([("select", clone(selector)), ("model", clone(models["Дерево"]))])
    reduced_cv = cross_validate(reduced, X_train, y_train, cv=cv, scoring=SCORING)
    reduced.fit(X_train, y_train)
    all_models = {**models, "Дерево, отбор": reduced}

    # 7. Сравнение фиксированных моделей на train и ранее отложенном test.
    metric_rows, class_rows, confusion_rows = [], [], []
    predictions = pd.DataFrame({"Row": y_test.index, "True_class": y_test.to_numpy()})
    for name, model in all_models.items():
        for subset, part_X, part_y in [("train", X_train, y_train), ("test", X_test, y_test)]:
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
        predictions[name + "_prediction"] = model.predict(X_test)
        predictions[name + "_score"] = (
            model.decision_function(X_test) if hasattr(model, "decision_function")
            else model.predict_proba(X_test)[:, 1]
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / "class_metrics.csv", index=False)
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(RESULTS / "confusion_matrices.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

    cv_summary = []
    for name, search in searches.items():
        row = {"Model": name}
        for metric in SCORING:
            row[metric + "_mean"] = search.cv_results_["mean_test_" + metric][search.best_index_]
            row[metric + "_std"] = search.cv_results_["std_test_" + metric][search.best_index_]
        cv_summary.append(row)
    cv_summary.append({"Model": "Дерево, отбор", **{
        metric + suffix: function(reduced_cv["test_" + metric])
        for metric in SCORING for suffix, function in [("_mean", np.mean), ("_std", np.std)]
    }})
    cv_summary = pd.DataFrame(cv_summary)
    cv_summary.to_csv(RESULTS / "cv_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for column, (name, model) in enumerate(all_models.items()):
        for row, (subset, part_X, part_y) in enumerate(
            [("train", X_train, y_train), ("test", X_test, y_test)]
        ):
            ax = axes[row, column]
            ConfusionMatrixDisplay.from_estimator(
                model, part_X, part_y, display_labels=CLASS_NAMES, cmap="Blues",
                colorbar=False, values_format="d", ax=ax,
            )
            ax.set(title=f"{name}, {subset}", xlabel="Предсказанный класс", ylabel="Истинный класс")
            ax.grid(False)
    save_figure("05_confusion_matrices.png")

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

    # 8. Изменение глубины при постоянном min_samples_leaf выбранного дерева.
    depth_rows = []
    for depth in [1, 2, 3, 4, 5, 6, 8, None]:
        model = clone(models["Дерево"]).set_params(max_depth=depth)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING,
                                return_train_score=True)
        model.fit(X_train, y_train)
        depth_rows.append({
            "Depth": str(depth), "Train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "CV_accuracy": scores["test_accuracy"].mean(),
            "CV_F1_macro": scores["test_f1_macro"].mean(),
            "CV_F1_std": scores["test_f1_macro"].std(),
            "CV_AUC": scores["test_auc"].mean(),
            "Train_CV_F1_macro": scores["train_f1_macro"].mean(),
            "Test_accuracy": accuracy_score(y_test, model.predict(X_test)),
            "Test_F1_macro": f1_score(y_test, model.predict(X_test), average="macro"),
        })
    depth_effect = pd.DataFrame(depth_rows)
    depth_effect.to_csv(RESULTS / "tree_depth_effect.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.plot(depth_effect["Depth"], depth_effect["Train_CV_F1_macro"], "o-", label="Train, среднее по CV")
    plt.errorbar(depth_effect["Depth"], depth_effect["CV_F1_macro"],
                 yerr=depth_effect["CV_F1_std"], fmt="o-", capsize=4, label="Validation, CV ± std")
    plt.xlabel("max_depth (None – без ограничения глубины)")
    plt.ylabel("F1-macro")
    plt.title("Влияние глубины дерева при постоянном min_samples_leaf")
    plt.legend()
    save_figure("07_depth_effect.png")

    # Полное дерево с индексами Gini; размеры рисунка зависят от числа листьев.
    tree = models["Дерево"]
    plt.figure(figsize=(max(10, tree.get_n_leaves() * 1.6), max(5, tree.get_depth() * 1.5)))
    plot_tree(tree, feature_names=X.columns.tolist(), class_names=CLASS_NAMES,
              filled=True, rounded=True, impurity=True, precision=3, fontsize=13)
    save_figure("08_decision_tree.png")
    (RESULTS / "tree_rules.txt").write_text(
        export_text(tree, feature_names=X.columns.tolist(), decimals=3), encoding="utf-8",
    )

    print(metrics.round(4).to_string(index=False))
    print("Влияние глубины:\n", depth_effect.round(4).to_string(index=False))
    print("Важность признаков:\n", importance.round(4).to_string())

    # Параметры для отчёта сохраняются отдельно от обученных моделей.
    svm = models["SVM"].named_steps["model"]
    model_parameters = {
        "Дерево": {
            "max_depth": tree.max_depth,
            "min_samples_leaf": tree.min_samples_leaf,
            "node_count": int(tree.tree_.node_count),
            "n_leaves": int(tree.get_n_leaves()),
            "root_gini": float(tree.tree_.impurity[0]),
        },
        "SVM": {
            "kernel": svm.kernel,
            "C": svm.C,
            "n_support": svm.n_support_.tolist(),
            "n_support_total": int(svm.n_support_.sum()),
        },
    }
    (RESULTS / "model_parameters.json").write_text(
        json.dumps(model_parameters, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
