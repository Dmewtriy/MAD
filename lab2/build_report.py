"""Создание Word-отчёта ЛР 2 из готовых результатов расчётов.

Сначала выполните classification_analysis.py.
Запуск из MAD: lab1.3/.venv/bin/python lab2/build_report.py
"""

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"


def load_results():
    """Прочитать таблицы и параметры, сохранённые classification_analysis.py."""
    class_statistics = pd.read_csv(RESULTS / "class_statistics.csv", header=[0, 1], index_col=0)
    return {
        "descriptive": pd.read_csv(RESULTS / "descriptive_statistics.csv", index_col="Feature"),
        "normality": pd.read_csv(RESULTS / "normality_shapiro.csv"),
        "balance": pd.read_csv(RESULTS / "class_balance.csv", index_col="Class"),
        "split_counts": pd.read_csv(RESULTS / "split_counts.csv", index_col=0),
        "selection": pd.read_csv(RESULTS / "feature_selection.csv"),
        "importance": pd.read_csv(RESULTS / "feature_importance.csv", index_col="Feature"),
        "metrics": pd.read_csv(RESULTS / "metrics.csv"),
        "confusion": pd.read_csv(RESULTS / "confusion_matrices.csv"),
        "cv_summary": pd.read_csv(RESULTS / "cv_summary.csv"),
        "depth": pd.read_csv(RESULTS / "tree_depth_effect.csv", keep_default_na=False),
        "correlation": pd.read_csv(RESULTS / "train_correlations.csv", index_col="Feature"),
        "class_means": class_statistics.xs("mean", axis=1, level=1),
        "model_parameters": json.loads((RESULTS / "model_parameters.json").read_text(encoding="utf-8")),
    }


def create_report(data):
    """Оформление отчёта по сохранённым таблицам, параметрам и рисункам."""
    reference = BASE.parent / "lab1.3" / "Отчёт_ЛР1.3_Дисперсионный_анализ.docx"
    doc = Document(reference)

    # Сохраняем титульный лист образца, включая авторов, преподавателя и группу.
    body = doc._element.body
    title_end = next(child for child in body if child.tag == qn("w:p")
                     and child.find(".//" + qn("w:sectPr")) is not None)
    after_title = False
    for child in list(body):
        if after_title and child.tag != qn("w:sectPr"):
            body.remove(child)
        if child is title_end:
            after_title = True
    used_images = set(doc._element.xpath("//@r:embed"))
    for relation_id, relation in list(doc.part.rels.items()):
        if relation.reltype.endswith("/image") and relation_id not in used_images:
            doc.part.drop_rel(relation_id)

    def font(run, size=14, bold=False):
        run.font.name, run.font.size, run.font.bold = "Times New Roman", Pt(size), bold
        run.font.color.rgb = RGBColor(0, 0, 0)
        for name in ["ascii", "hAnsi", "eastAsia", "cs"]:
            run._element.get_or_add_rPr().rFonts.set(qn("w:" + name), "Times New Roman")

    for p in doc.paragraphs:
        if "ЛАБОРАТОРНАЯ РАБОТА №" in p.text:
            p.clear()
            font(p.add_run("ЛАБОРАТОРНАЯ РАБОТА №2"), bold=True)
        elif "Дисперсионный анализ" in p.text:
            p.clear()
            run = p.add_run("«Классификация»")
            font(run, bold=True)
            run.add_break()
            font(p.add_run("по дисциплине «Методы анализа данных»"), size=12)

    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(14)
    pf = normal.paragraph_format
    pf.alignment, pf.first_line_indent = WD_ALIGN_PARAGRAPH.JUSTIFY, Cm(1.25)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = pf.space_after = Pt(0)
    for name in ["Heading 1", "Heading 2"]:
        style = doc.styles[name]
        style.font.name, style.font.size = "Times New Roman", Pt(14)
        style.font.bold, style.font.color.rgb = True, RGBColor(0, 0, 0)
        pf = style.paragraph_format
        pf.first_line_indent, pf.left_indent = Cm(1.25), Cm(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.keep_with_next = True
        pf.page_break_before = False
    section = doc.sections[-1]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(1.5)
    section.top_margin = section.bottom_margin = Cm(2)
    doc.core_properties.title = "Лабораторная работа №2. Классификация"
    doc.core_properties.subject = "Rice (Cammeo and Osmancik): дерево решений и SVM"
    doc.core_properties.keywords = "классификация, рис, дерево решений, SVM"
    doc.core_properties.comments = ""

    table_number, figure_number = 0, 0

    def fit_table_width(t, widths):
        available = section.page_width.twips - section.left_margin.twips - section.right_margin.twips
        total = sum(widths)
        column_widths = [round(available * width / total) for width in widths]
        column_widths[-1] += available - sum(column_widths)
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        preferred_width = t._tbl.tblPr.find(qn("w:tblW"))
        preferred_width.set(qn("w:type"), "dxa")
        preferred_width.set(qn("w:w"), str(available))
        for column, width in zip(t.columns, column_widths):
            column.width = Twips(width)
        for row in t.rows:
            for cell, width in zip(row.cells, column_widths):
                cell.width = Twips(width)

    fit_table_width(doc.tables[0], [column.width for column in doc.tables[0].columns])

    def text(value):
        return doc.add_paragraph(value)

    def blank_line_after_heading():
        p = doc.add_paragraph()
        font(p.add_run(""))
        p.paragraph_format.keep_with_next = True

    def heading(value, new_page=True):
        p = doc.add_heading(value, level=1)
        p.paragraph_format.page_break_before = new_page
        blank_line_after_heading()

    def subheading(value):
        p = doc.add_heading(value, level=2)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.page_break_before = False
        blank_line_after_heading()

    def number(value, digits=4):
        return f"{value:.{digits}f}".replace(".", ",")

    def scientific(value):
        return f"{value:.2e}".replace(".", ",")

    def table(caption, headers, rows, widths=None):
        nonlocal table_number
        table_number += 1
        p = doc.add_paragraph(f"Таблица {table_number}. {caption}")
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pf = p.paragraph_format
        pf.first_line_indent, pf.line_spacing = Cm(0), 1.0
        pf.space_before, pf.space_after, pf.keep_with_next = Pt(7), Pt(4), True
        for run in p.runs:
            font(run, size=12)
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        for cell, value in zip(t.rows[0].cells, headers):
            cell.text = str(value)
        header_property = OxmlElement("w:tblHeader")
        t.rows[0]._tr.get_or_add_trPr().append(header_property)
        for row in rows:
            for cell, value in zip(t.add_row().cells, row):
                cell.text = str(value)
        for i, row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    pf = p.paragraph_format
                    pf.first_line_indent, pf.line_spacing = Cm(0), 1.0
                    pf.space_before = pf.space_after = Pt(3)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in p.runs:
                        font(run, size=11, bold=i == 0)
        fit_table_width(t, widths if widths is not None else [1] * len(headers))
        doc.add_paragraph().paragraph_format.line_spacing = 0.5

    def figure(filename, caption, width=16.3):
        nonlocal figure_number
        figure_number += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing, p.paragraph_format.keep_with_next = 1.0, True
        p.add_run().add_picture(str(RESULTS / filename), width=Cm(width))
        p = doc.add_paragraph(f"Рис. {figure_number} – {caption}")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent, p.paragraph_format.line_spacing = Cm(0), 1.0
        p.paragraph_format.space_after = Pt(21)
        for run in p.runs:
            font(run, size=12)

    feature_names = {
        "Area": "Площадь", "Perimeter": "Периметр",
        "Major_Axis_Length": "Большая ось", "Minor_Axis_Length": "Малая ось",
        "Eccentricity": "Эксцентриситет", "Convex_Area": "Площадь оболочки",
        "Extent": "Заполненность",
    }
    metrics = data["metrics"].set_index(["Model", "Subset"])
    tree_test, svm_test = metrics.loc[("Дерево", "test")], metrics.loc[("SVM", "test")]
    tree_train, svm_train = metrics.loc[("Дерево", "train")], metrics.loc[("SVM", "train")]
    cv_results = data["cv_summary"].set_index("Model")
    tree, svm = data["model_parameters"]["Дерево"], data["model_parameters"]["SVM"]

    heading("Цель работы", new_page=False)
    text("Изучение методов решения задачи классификации: построение дерева решений и модели "
         "опорных векторов, оценка их качества, информативности признаков и влияния параметров.")

    heading("1 Описание исходных данных")
    text("Использован набор Rice (Cammeo and Osmancik) из UCI Machine Learning Repository. "
         "Он содержит 3810 объектов: по изображению каждого зерна вычислены семь морфологических "
         "характеристик. Требуется определить один из двух сортов риса. Это бинарная классификация "
         "с учителем: сорт известен для каждого объекта и используется как целевая переменная Class.")
    table("Переменные набора Rice", ["Переменная", "Смысл и тип"], [
        ["Area", "Площадь зерна: число пикселей внутри его границы; количественная."],
        ["Perimeter", "Периметр границы зерна; количественная."],
        ["Major_Axis_Length", "Длина большой оси зерна; количественная."],
        ["Minor_Axis_Length", "Длина малой оси зерна; количественная."],
        ["Eccentricity", "Эксцентриситет эквивалентного эллипса; количественная."],
        ["Convex_Area", "Площадь выпуклой оболочки зерна в пикселях; количественная."],
        ["Extent", "Доля площади зерна в ограничивающем прямоугольнике; количественная."],
        ["Class", "Сорт Osmancik или Cammeo; категориальная целевая переменная."],
    ], [4.7, 11.8])
    text("Исходный файл Rice_Cammeo_Osmancik.arff взят из ранее загруженной копии этого же "
         "набора в AI/lab2 и сохранён в results. Пропусков и полностью совпадающих строк нет; "
         "используются все 3810 наблюдений. Масштабы количественных признаков различаются. "
         "При чтении ARFF они представлены float64; Area и Convex_Area содержат целочисленные значения.")
    text("Кодирование ответа: Osmancik = 0, Cammeo = 1. Положительным классом при расчёте "
         "бинарных Precision, Recall, F1 и ROC-AUC считается Cammeo. Названия признаков в таблицах "
         "далее сокращены по их смыслу; код сохраняет исходные имена столбцов.")

    heading("2 Результаты дескриптивного анализа")
    subheading("2.1 Распределение объектов по классам")
    table("Распределение объектов по классам", ["Класс", "Число", "Доля, %"], [
        [name, int(row["Count"]), number(row["Percent"], 2)]
        for name, row in data["balance"].iterrows()
    ], [6.5, 5, 5])
    text("Соотношение классов умеренно неравномерное: 57,22% и 42,78%. Если всегда предсказывать "
         "Osmancik, Accuracy составит всего 0,5722. Поэтому высокое качество должно подтверждаться "
         "также полнотой и F1 каждого класса. Основной критерий подбора параметров – F1-macro: "
         "оба сорта вносят в него одинаковый вклад. Искусственное выравнивание классов не применяется.")
    subheading("2.2 Описательные статистики и распределения признаков")
    table("Основные описательные статистики", ["Признак", "n", "Среднее", "s", "Min", "Медиана", "Max"], [
        [feature_names[name], int(row["count"]), *[
            number(row[key], 4 if name in ["Eccentricity", "Extent"] else 2)
            for key in ["mean", "std", "min", "50%", "max"]
        ]] for name, row in data["descriptive"].iterrows()
    ], [3.4, 1.0, 2.1, 1.8, 2.1, 2.1, 2.0])
    text("Статистики получены стандартным методом describe(); s – выборочное стандартное "
         "отклонение с делителем n − 1. Квартили, дисперсии и коэффициенты асимметрии приведены "
         "в descriptive_statistics.csv. У площади среднее 12667,73 выше медианы 12421,50; "
         "асимметрия положительна (0,325). Особенно заметно различие масштабов: площадь имеет "
         "порядок десятков тысяч, эксцентриситет – долей единицы.")
    means = data["class_means"]
    table("Средние значения признаков по сортам", ["Признак", "Osmancik", "Cammeo"], [
        [feature_names[name], number(means.loc["Osmancik", name], 4),
         number(means.loc["Cammeo", name], 4)] for name in feature_names
    ], [6.5, 5, 5])
    text("У Cammeo в среднем больше площадь, периметр и длина большой оси. Например, средние "
         "длины большой оси равны 205,48 против 176,29 у Osmancik. Различие существенно относительно "
         "разброса внутри сортов, поэтому геометрические размеры могут разделять классы. "
         "При этом диапазоны пересекаются, что допускает ошибки распознавания.")
    figure("01_distributions.png", "Распределения признаков по сортам и численность классов", width=15.3)

    subheading("2.3 Согласованность с нормальным распределением")
    text("Для каждого признака выполнен критерий Шапиро–Уилка: H₀ – распределение нормально, "
         "H₁ – отличается от нормального. Уровень значимости α = 0,05 применяется к каждой "
         "отдельной проверке. Проверены общая выборка и выборки внутри каждого сорта: смесь "
         "двух классов сама может отличаться от нормального распределения. Это описательная "
         "проверка отдельных признаков, а не доказательство многомерной нормальности.")
    n = data["normality"].set_index(["Feature", "Group"])
    table("Критерий Шапиро–Уилка", ["Признак", "W, все", "p, все", "W, Osm.", "p, Osm.", "W, Cam.", "p, Cam."], [
        [feature_names[name], *[value for group in ["Все", "Osmancik", "Cammeo"]
                               for value in [number(n.loc[(name, group), "W"]),
                                             scientific(n.loc[(name, group), "p_value"])]]]
        for name in feature_names
    ], [3.4, 1.8, 1.9, 1.8, 1.9, 1.8, 1.9])
    text("Для общей выборки все p < 0,05: гипотеза нормальности отвергается по всем семи "
         "признакам. Внутри Osmancik она не отвергается только для периметра (p = 0,2057); "
         "внутри Cammeo – для площади, большой оси и площади выпуклой оболочки "
         "(p = 0,6279; 0,5103; 0,6294). В остальных проверках p < 0,05. "
         "Неотклонение H₀ означает недостаток оснований её отвергнуть. При таких объёмах "
         "Шапиро–Уилк чувствителен и к небольшим отклонениям, поэтому учитываются также гистограммы.")
    subheading("2.4 Информативность признаков и расположение классов")
    text("Для изучения информативности использованы только обучающие данные. Рассчитана "
         "взаимная информация признака с сортом методом mutual_info_classif. Она выявляет "
         "в том числе нелинейную связь; большее значение означает больше информации об ответе "
         "при рассмотрении признака отдельно. Такая оценка не учитывает его избыточность "
         "относительно остальных признаков.")
    text("Дополнительно выполнен SelectFromModel с деревом глубины 3 и min_samples_leaf = 10. "
         "Фиксированные небольшая глубина и минимальный размер листа ограничивают подгонку "
         "случайных деталей при отборе. Порог нормированной важности Gini равен 0,02: "
         "сохраняются признаки с вкладом не менее 2% в суммарное уменьшение неоднородности "
         "этого дерева. Это рабочее правило отбора для эксперимента, а не статистический "
         "уровень значимости. Его результат далее проверяется повторным обучением.")
    table("Информативность и предварительный отбор", ["Признак", "Взаимная информация", "Важность Gini", "Оставлен"], [
        [feature_names[row["Feature"]], number(row["Mutual_information"]),
         number(row["Selection_Gini"]), "да" if row["Selected"] else "нет"]
        for _, row in data["selection"].iterrows()
    ], [5, 4.3, 4.2, 3])
    corr = data["correlation"]
    text(f"Наиболее информативны большая ось и периметр. Их отбор также согласуется "
         f"с различиями распределений по классам. Area и Convex_Area имеют высокую "
         f"взаимную информацию, но сильно дублируют друг друга: r = "
         f"{number(corr.loc['Area', 'Convex_Area'])}. Периметр и большая ось также тесно "
         f"связаны: r = {number(corr.loc['Perimeter', 'Major_Axis_Length'])}. "
         "Поэтому индивидуально информативный признак может почти не давать дополнительного "
         "выигрыша конкретному дереву. Предварительно оставлены Perimeter и Major_Axis_Length; "
         "основные модели сначала обучаются на всех семи признаках для последующего сравнения.")
    figure("02_correlations.png", "Корреляции Пирсона на обучающей выборке")
    figure("03_scatterplots.png", "Категоризованные диаграммы рассеивания")
    text("На диаграммах Cammeo смещён к большим значениям большой оси и периметра. "
         "В их совместном пространстве классы вытянуты вдоль общей зависимости и перекрываются "
         "в промежуточной области. По эксцентриситету различие слабее, а по Extent "
         "распределения заметно пересекаются. Следовательно, геометрические размеры позволяют "
         "ожидать хорошее разделение, но эти двумерные проекции не подтверждают идеальную "
         "разделимость. Поиск RBF-ядра позволяет проверить, нужна ли более сложная граница.")

    heading("3 Описание параметров метода")
    subheading("3.1 Обоснование выбора методов")
    text("Выбраны дерево решений и SVM. Оба метода работают с числовыми признаками "
         "и не требуют нормальности их распределений. Дерево даёт наглядные пороговые правила "
         "и собственную оценку важности. SVM позволяет сопоставить линейную и нелинейную границы. "
         "Размер выборки и семь признаков позволяют применить оба метода без предварительного "
         "понижения размерности. Их разные способы разделения классов полезны для сравнения.")
    subheading("3.2 Разбиение данных и стандартизация")
    table("Стратифицированное разбиение 80/20", ["Подмножество", "Osmancik", "Cammeo", "Всего"], [
        [subset, int(row["Osmancik"]), int(row["Cammeo"]), int(row.sum())]
        for subset, row in data["split_counts"].reindex(["train", "test"]).iterrows()
    ], [6, 3.5, 3.5, 3.5])
    text("Использован train_test_split(test_size=0.2, stratify=y, random_state=42): "
         "3048 объектов в train и 762 в test. Стратификация сохраняет соотношение сортов. "
         "Подбор параметров проводится пятикратной StratifiedKFold с перемешиванием и "
         "random_state=42 внутри train. Тестовая выборка не участвует в подборе параметров, "
         "масштабировании и выборе удаляемых признаков.")
    text("Для SVM применяется StandardScaler: zⱼ = (xⱼ − μⱼ,train) / σⱼ,train. "
         "Среднее и стандартное отклонение оцениваются на обучающей части; в StandardScaler "
         "используется делитель n. Преобразование находится внутри Pipeline, поэтому на "
         "каждом CV-фолде параметры масштаба заново оцениваются по его обучающей части. "
         "К test применяется уже обученный scaler. Стандартизация устраняет доминирование "
         "признаков из-за единиц измерения, но не делает распределения нормальными. "
         "Дереву масштабирование не требуется: оно сравнивает каждый признак с его порогами.")

    subheading("3.3 Параметры дерева решений и SVM")
    text("GridSearchCV перебирает заданную сетку, выбирает максимум среднего F1-macro "
         "на валидационных фолдах и переобучает выбранную модель на всём train. "
         "Одновременно сохраняются Accuracy и ROC-AUC. Средние CV-оценки используются "
         "для настройки; независимую итоговую оценку даёт test.")
    table("Сетки подбора параметров", ["Метод", "Параметр", "Значения"], [
        ["Дерево", "criterion", "gini (фиксирован)"],
        ["Дерево", "max_depth", "2, 3, 4, 5, 6, 8, None"],
        ["Дерево", "min_samples_leaf", "1, 5, 10, 20"],
        ["SVM", "kernel; C", "linear; C = 0,1; 1; 10; 100"],
        ["SVM", "kernel; C; gamma", "rbf; C = 0,1; 1; 10; 100; gamma = scale; 0,01; 0,1; 1"],
    ], [2.5, 4, 10])
    text("В дереве criterion='gini' задаёт индекс неоднородности G = 1 − Σpₖ². "
         "Нуль соответствует чистому узлу; для двух равновероятных классов G = 0,5. "
         "Ограничение глубины и числа объектов в листе контролирует сложность. "
         "Сетка включает и простые, и более глубокие деревья, чтобы проверить риск переобучения.")
    text("В SVM параметр C задаёт штраф за ошибки и нарушения зазора: рост C ослабляет "
         "регуляризацию. Линейное ядро проверяет достаточность гиперплоскости; "
         "RBF-ядро K(x, x′) = exp(−γ‖x − x′‖²) допускает нелинейную границу. "
         "Большое γ делает влияние объектов более локальным. Перебор нескольких порядков "
         "C и γ позволяет сравнить различные степени сложности после стандартизации.")
    subheading("3.4 Выбранные параметры и результаты кросс-валидации")
    table("Выбранные параметры", ["Модель", "Параметры и размер"], [
        ["Дерево", f"criterion = gini; max_depth = {tree['max_depth']}; "
                    f"min_samples_leaf = {tree['min_samples_leaf']}; "
                    f"узлов {tree['node_count']}, листьев {tree['n_leaves']}; random_state = 42."],
        ["SVM", f"StandardScaler → SVC(kernel = {svm['kernel']}, C = {svm['C']}); "
                 f"опорных векторов {svm['n_support_total']} "
                 f"({svm['n_support'][0]} Osmancik, {svm['n_support'][1]} Cammeo)."],
    ], [3, 13.5])
    table("Средние оценки на валидационных фолдах train", ["Модель", "Accuracy", "F1-macro ± std", "AUC"], [
        [name, number(row["accuracy_mean"]),
         number(row["f1_macro_mean"]) + " ± " + number(row["f1_macro_std"]),
         number(row["auc_mean"])] for name, row in cv_results.iterrows()
    ], [5, 3.5, 4.5, 3.5])
    text("Линейная SVM получила больший средний F1-macro, чем дерево: "
         f"{number(cv_results.loc['SVM', 'f1_macro_mean'])} против "
         f"{number(cv_results.loc['Дерево', 'f1_macro_mean'])}. По выбранному критерию "
         "усложнение границы до RBF не дало лучшего варианта в этой сетке. "
         "SVM выбрана предпочтительным из двух методов до сравнения тестовых результатов. "
         "Значения ± std описывают разброс между пятью фолдами и не являются доверительными интервалами.")

    heading("4 Результаты решения задачи классификации")
    subheading("4.1 Матрицы ошибок и определения метрик")
    text("Использованы готовые функции accuracy_score, precision_score, recall_score, "
         "f1_score, roc_auc_score и confusion_matrix. Строки матрицы ошибок – истинные "
         "классы, столбцы – предсказанные, порядок: Osmancik, Cammeo. TN – верно распознанный "
         "Osmancik, FP – Osmancik ошибочно отнесён к Cammeo, FN – пропущен Cammeo, "
         "TP – верно распознанный Cammeo.")
    text("Accuracy = (TP + TN) / N; Precision = TP / (TP + FP); "
         "Recall = TP / (TP + FN); F1 = 2·Precision·Recall / (Precision + Recall). "
         "Precision характеризует долю настоящих Cammeo среди предсказанных Cammeo, "
         "Recall – долю найденных Cammeo среди всех зёрен этого сорта.")
    table("Матрицы ошибок в форме TN, FP, FN, TP", ["Модель", "Выборка", "TN", "FP", "FN", "TP"], [
        [row["Model"], row["Subset"], *[int(row[key]) for key in ["TN", "FP", "FN", "TP"]]]
        for _, row in data["confusion"].iterrows()
    ], [4.8, 2.5, 2.3, 2.3, 2.3, 2.3])
    figure("05_confusion_matrices.png", "Матрицы ошибок на обучающем и тестовом множествах")
    subheading("4.2 Метрики на обучающем и тестовом множествах")
    table("Accuracy, ROC-AUC и бинарные метрики Cammeo", ["Модель", "Выборка", "Accuracy", "Precision", "Recall", "F1", "AUC"], [
        [name, subset, *[number(row[key]) for key in
                        ["Accuracy", "Precision_binary", "Recall_binary", "F1_binary", "AUC"]]]
        for (name, subset), row in metrics.iterrows()
    ], [3.8, 2.2, 2.1, 2.1, 2.1, 2.1, 2.1])
    text("Макроусреднение – среднее метрик двух классов с одинаковыми весами. "
         "F1-macro вычисляется как среднее двух F1, а не из уже усреднённых Precision и Recall. "
         "При микроусреднении сначала суммируются TP, FP и FN по классам. "
         "В данной задаче каждый объект имеет ровно один класс, поэтому "
         "Precision-micro = Recall-micro = F1-micro = Accuracy; эти три равных значения "
         "в таблице объединены в одну колонку. Все три отдельно сохранены в metrics.csv.")
    table("Макро- и микроусреднение", ["Модель", "Выборка", "Precision-macro", "Recall-macro", "F1-macro", "Micro: P = R = F1"], [
        [name, subset, *[number(row[key]) for key in
                        ["Precision_macro", "Recall_macro", "F1_macro", "F1_micro"]]]
        for (name, subset), row in metrics.iterrows()
    ], [3.8, 2.1, 2.65, 2.65, 2.65, 2.65])
    subheading("4.3 ROC-кривые и AUC")
    text("ROC-кривая показывает соотношение Recall и FPR при изменении порога. "
         "ROC-AUC вычисляется по непрерывной оценке: predict_proba для дерева и "
         "decision_function для SVM. Значения decision_function не интерпретируются как "
         "вероятности. AUC не заменяет метрики при фиксированном решающем правиле; "
         "для бинарного ROC достаточно одной кривой относительно Cammeo.")
    figure("06_roc.png", "ROC-кривые на test")
    subheading("4.4 Структура дерева с индексами Gini")
    figure("08_decision_tree.png", "Полное выбранное дерево: пороги, Gini и состав узлов")
    text(f"Корневой узел содержит 1744 Osmancik и 1304 Cammeo; "
         f"Gini = 1 − (1744/3048)² − (1304/3048)² = {number(tree['root_gini'])}. "
         "Первое правило – Major_Axis_Length ≤ 191,817. При выполнении условия "
         "преобладает Osmancik, при превышении порога – Cammeo. В узлах следующего "
         "уровня используются большая ось и периметр. samples – число объектов, "
         "value – состав [Osmancik, Cammeo], class – решение по большинству.")
    text("Дополнительные разбиения уточняют состав и оценки вероятностей в листьях, "
         "но в этом дереве не меняют класс большинства внутри каждой стороны корневого "
         "разбиения. Поэтому прогнозы классов совпадают с деревом глубины 1, "
         "хотя оценки вероятностей и AUC различаются. Низкий Gini в листе характеризует "
         "однородность обучающих объектов и сам по себе не гарантирует качество на новых данных.")
    subheading("4.5 Значимость признаков")
    importance = data["importance"]
    table("Важность в итоговых моделях", ["Признак", "Gini дерева", "Перестановки, дерево", "Перестановки, SVM"], [
        [feature_names[name], number(row["Tree_Gini"]),
         number(row["Tree_permutation_mean"]) + " ± " + number(row["Tree_permutation_std"]),
         number(row["SVM_permutation_mean"]) + " ± " + number(row["SVM_permutation_std"])]
        for name, row in importance.iterrows()
    ], [4.5, 3, 4.5, 4.5])
    text("Важность Gini – нормированный вклад признака в уменьшение неоднородности "
         "итогового дерева. Для обоих методов дополнительно рассчитана перестановочная "
         "важность: на валидационной части каждого фолда перемешивается один столбец, "
         "затем измеряется падение F1-macro. Выполнено 10 перестановок на каждом из пяти "
         "фолдов; в таблице среднее и стандартное отклонение 50 оценок. Тест для этого не используется.")
    text("В дереве почти весь вклад Gini приходится на большую ось (0,9772), "
         "остальное – на периметр (0,0228). Но перестановочная важность периметра "
         "для F1-macro равна нулю: его разбиение уточняет доли классов, не меняя предсказанный "
         "класс. Это объясняет различие двух оценок. Другие пять признаков дерево не использует.")
    text("У SVM наибольшее падение F1-macro вызывает перемешивание Convex_Area. "
         "Эту величину нельзя трактовать как независимую долю информации или причинный эффект. "
         "Area и Convex_Area почти совпадают по корреляции; независимая перестановка "
         "нарушает их естественную связь и создаёт нетипичные сочетания. Коррелирующие признаки "
         "могут заменять друг друга или совместно задавать решение, поэтому оценки важности "
         "зависят от модели. Extent даёт малое среднее падение при сопоставимом разбросе. "
         "Удаление по важности дерева далее проверяется именно для дерева; переносить этот "
         "вывод на SVM без нового эксперимента нельзя.")
    figure("04_feature_importance.png", "Важность признаков: Gini и падение F1-macro при перестановках", width=15)

    heading("5 Сравнительный анализ решений и интерпретация результатов")
    text(f"На test дерево допустило 35 ложных отнесений к Cammeo и 29 пропусков Cammeo, "
         f"всего 64 ошибки. SVM допустила 32 и 26 ошибок соответственно, всего 58. "
         f"Accuracy SVM = {number(svm_test['Accuracy'])}, у дерева – "
         f"{number(tree_test['Accuracy'])}: разница составляет "
         f"{number(100 * (svm_test['Accuracy'] - tree_test['Accuracy']), 2)} процентного пункта, "
         "или шесть дополнительных верных ответов из 762. Оба вида ошибок у SVM немного реже.")
    text(f"Разрыв Accuracy между train и test составляет "
         f"{number(100 * (tree_train['Accuracy'] - tree_test['Accuracy']), 2)} п.п. для дерева "
         f"и {number(100 * (svm_train['Accuracy'] - svm_test['Accuracy']), 2)} п.п. для SVM. "
         "Это умеренное снижение качества на новых объектах. Близость макро- и микрометрик "
         "показывает, что результат не сводится к распознаванию только более многочисленного сорта.")
    text(f"AUC SVM равна {number(svm_test['AUC'])}, дерева – {number(tree_test['AUC'])}. "
         "SVM лучше ранжирует объекты по принадлежности к Cammeo. Преимущество согласуется "
         "с CV, но невелико по числу верных ответов. Один тестовый разрез не устанавливает "
         "статистически значимое или универсальное превосходство метода.")
    text(f"На test SVM получила Accuracy = {number(svm_test['Accuracy'])}, "
         f"F1-macro = {number(svm_test['F1_macro'])}, AUC = {number(svm_test['AUC'])}; "
         f"дерево – {number(tree_test['Accuracy'])}, {number(tree_test['F1_macro'])}, "
         f"{number(tree_test['AUC'])}. Для приоритета качества предпочтительна линейная SVM: "
         "она также имела лучший F1-macro на CV и допустила на шесть ошибок меньше. "
         "Для наглядного объяснения решения удобно дерево глубины 2 с четырьмя листьями.")
    text("Полученные оценки характеризуют случайное стратифицированное разбиение этого набора. "
         "Качество на независимых партиях риса и в других условиях съёмки не проверялось.")

    heading("6 Результаты анализа влияния параметров одного из методов на точность классификатора")
    text(f"Для проверки влияния сложности изменён только max_depth: 1, 2, 3, 4, 5, 6, 8 "
         f"и None, при неизменном min_samples_leaf = {tree['min_samples_leaf']} и тех же "
         "данных и CV-разбиениях. None снимает ограничение глубины. Выбор основного дерева "
         "в разделе 3 уже сделан по CV; результаты test здесь показывают последствия "
         "изменения параметра и не используются для повторного выбора.")
    table("Изменение глубины при фиксированном размере листа", ["Глубина", "Accuracy, train", "Accuracy, CV", "F1-macro, CV", "Accuracy, test", "F1-macro, test"], [
        [row["Depth"], *[number(row[key]) for key in
                        ["Train_accuracy", "CV_accuracy", "CV_F1_macro", "Test_accuracy", "Test_F1_macro"]]]
        for _, row in data["depth"].iterrows()
    ], [2, 2.9, 2.9, 2.9, 2.9, 2.9])
    figure("07_depth_effect.png", "Влияние глубины на обучающий и валидационный F1-macro")
    depth = data["depth"].set_index("Depth")
    text("Деревья глубины 1 и 2 дают одинаковые метки классов и одинаковую Accuracy "
         "на train и test. При этом средний CV AUC составляет "
         f"{number(depth.loc['1', 'CV_AUC'])} и {number(depth.loc['2', 'CV_AUC'])} "
         "соответственно: дополнительные разбиения глубины 2 уточняют ранжирование. "
         "Дальнейшее увеличение глубины повышает качество на train, но в целом ухудшает "
         "CV. У неограниченного дерева Accuracy train = 1,0000, "
         f"Accuracy test = {number(depth.loc['None', 'Test_accuracy'])}; это характерное "
         "переобучение. Глубина 3 даёт чуть лучшую Accuracy test, чем 2, однако её "
         "F1-macro на CV ниже; выбирать её по уже просмотренному тесту было бы неверно.")

    heading("7 Результаты сравнительного анализа решений в случае исключения малозначимых признаков")
    text("По правилу отбора из подраздела 2.4 исключены Area, Minor_Axis_Length, Eccentricity, "
         "Convex_Area и Extent. Сохранены Perimeter и Major_Axis_Length. Для уменьшенной "
         "модели оставлены те же max_depth и min_samples_leaf, что у основного дерева, "
         "чтобы сравнить именно состав признаков. В CV SelectFromModel находится внутри "
         "Pipeline: на каждом фолде он заново обучается по его обучающей части. "
         "Окончательный набор из двух признаков получен после обучения на всём train.")
    table("Дерево до и после исключения пяти признаков", ["Признаки", "Accuracy, train", "Accuracy, test", "F1-macro, CV", "F1-macro, test", "AUC, test"], [
        [label, number(metrics.loc[(name, "train"), "Accuracy"]),
         number(metrics.loc[(name, "test"), "Accuracy"]),
         number(cv_results.loc[name, "f1_macro_mean"]),
         number(metrics.loc[(name, "test"), "F1_macro"]),
         number(metrics.loc[(name, "test"), "AUC"])]
        for label, name in [("Все 7", "Дерево"), ("Отобранные 2", "Дерево, отбор")]
    ], [2, 2.9, 2.9, 2.9, 2.9, 2.9])
    text("После исключения пяти признаков тестовая Accuracy, F1 и AUC не изменились; "
         "число ошибок осталось 64. Это ожидаемо: исходное итоговое дерево и так использует "
         "только большую ось и периметр. Средний CV AUC немного меняется, поскольку "
         "отбор повторяется на каждом фолде; Accuracy и F1-macro на CV совпали. "
         "Уменьшение набора входов упрощает эту модель без наблюдаемой потери качества, "
         "но не доказывает бесполезность удалённых признаков для любого классификатора. "
         "Нулевой вклад Gini относится к выбранной структуре дерева.")

    output = BASE / "Отчёт_ЛР2_Классификация.docx"
    doc.save(output)
    print("Отчёт:", output)



if __name__ == "__main__":
    create_report(load_results())
