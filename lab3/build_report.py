"""Создание Word-отчёта по сохранённым результатам кластеризации.

Запуск: lab3/.venv/bin/python lab3/build_report.py
"""

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TEMPLATE_PATH = BASE_DIR.parent / "lab2" / "Отчёт_ЛР2_Классификация.docx"
REPORT_PATH = BASE_DIR / "Отчёт_ЛР3_Кластеризация.docx"
FONT_NAME = "Times New Roman"


def format_decimal(value, digits=2):
    return f"{float(value):.{digits}f}".replace(".", ",")


def format_integer(value):
    return f"{int(value):,}".replace(",", " ")


def format_percent(value, digits=2):
    return format_decimal(value, digits) + "%"


def set_font(run, size=14, bold=False):
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    font_settings = run._element.get_or_add_rPr().rFonts
    for name in ("ascii", "hAnsi", "eastAsia", "cs"):
        font_settings.set(qn("w:" + name), FONT_NAME)


def fit_table_width(document, table, widths):
    section = document.sections[-1]
    available_width = section.page_width.twips - section.left_margin.twips - section.right_margin.twips
    total_weight = sum(widths)
    column_widths = [round(available_width * weight / total_weight) for weight in widths]
    column_widths[-1] += available_width - sum(column_widths)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    preferred_width = table._tbl.tblPr.find(qn("w:tblW"))
    preferred_width.set(qn("w:type"), "dxa")
    preferred_width.set(qn("w:w"), str(available_width))
    for column, width in zip(table.columns, column_widths):
        column.width = Twips(width)
    for row in table.rows:
        for cell, width in zip(row.cells, column_widths):
            cell.width = Twips(width)


def create_document():
    """Взять титульный лист из ЛР 2 и настроить общие стили ЛР 3."""
    document = Document(TEMPLATE_PATH)
    body = document._element.body
    title_end = next(
        child for child in body
        if child.tag == qn("w:p") and child.find(".//" + qn("w:sectPr")) is not None
    )
    after_title = False
    for child in list(body):
        if after_title and child.tag != qn("w:sectPr"):
            body.remove(child)
        if child is title_end:
            after_title = True

    # Удаляем связи с рисунками и ссылками из отброшенной части образца.
    used_images = set(document._element.xpath("//@r:embed"))
    for relation_id, relation in list(document.part.rels.items()):
        if relation.reltype.endswith("/image") and relation_id not in used_images:
            document.part.drop_rel(relation_id)
        elif relation.reltype.endswith("/hyperlink"):
            document.part.drop_rel(relation_id)

    for paragraph in document.paragraphs:
        if "ЛАБОРАТОРНАЯ РАБОТА №" in paragraph.text:
            paragraph.clear()
            set_font(paragraph.add_run("ЛАБОРАТОРНАЯ РАБОТА №3"), bold=True)
        elif "Классификация" in paragraph.text:
            paragraph.clear()
            title = paragraph.add_run("«Кластеризация категориальных и смешанных данных»")
            set_font(title, bold=True)
            title.add_break()
            set_font(paragraph.add_run("по дисциплине «Методы анализа данных»"), size=12)

    normal_style = document.styles["Normal"]
    normal_style.font.name = FONT_NAME
    normal_style.font.size = Pt(14)
    paragraph_format = normal_style.paragraph_format
    paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph_format.first_line_indent = Cm(1.25)
    paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    paragraph_format.space_before = Pt(0)
    paragraph_format.space_after = Pt(0)

    for name in ("Heading 1", "Heading 2"):
        style = document.styles[name]
        style.font.name = FONT_NAME
        style.font.size = Pt(14)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        paragraph_format = style.paragraph_format
        paragraph_format.first_line_indent = Cm(1.25)
        paragraph_format.left_indent = Cm(0)
        paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph_format.space_before = Pt(0)
        paragraph_format.space_after = Pt(0)
        paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        paragraph_format.keep_with_next = True
        paragraph_format.page_break_before = False

    section = document.sections[-1]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    document.core_properties.title = "Лабораторная работа №3. Кластеризация"
    document.core_properties.subject = "Bank Marketing: CLOPE и k-modes"
    document.core_properties.keywords = "кластеризация, Bank Marketing, CLOPE, k-modes"
    document.core_properties.comments = ""
    fit_table_width(document, document.tables[0], [column.width for column in document.tables[0].columns])
    return document


class Report:
    """Общие правила оформления; содержание разделов находится ниже."""

    def __init__(self):
        self.document = create_document()
        self.table_number = 0
        self.figure_number = 0

    def add_text(self, text):
        return self.document.add_paragraph(text)

    def add_blank_line(self):
        paragraphs = self.document.paragraphs
        previous = paragraphs[-1] if paragraphs else None
        if (previous is not None and not previous.text
                and not previous._element.xpath('.//w:drawing | .//w:sectPr')):
            paragraph = previous
        else:
            paragraph = self.document.add_paragraph()
        paragraph.style = self.document.styles["Normal"]
        set_font(paragraph.add_run(""))
        paragraph_format = paragraph.paragraph_format
        paragraph_format.first_line_indent = Cm(0)
        paragraph_format.space_before = Pt(0)
        paragraph_format.space_after = Pt(0)
        paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        paragraph_format.page_break_before = False
        paragraph_format.keep_with_next = True

    def add_goal(self, text):
        paragraph = self.document.add_paragraph("Цель работы")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.keep_with_next = True
        for run in paragraph.runs:
            set_font(run, bold=True)
        self.add_blank_line()
        self.add_text(text)

    def add_heading(self, title, level=1, new_page=True):
        # Интервал перед абзацем не создаёт пустую строку при переносе на новую страницу.
        previous = self.document.paragraphs[-1] if self.document.paragraphs else None
        if (previous is not None and not previous.text
                and not previous.paragraph_format.page_break_before
                and not previous._element.xpath('.//w:drawing | .//w:sectPr | .//w:br')):
            previous._element.getparent().remove(previous._element)
        paragraph = self.document.add_heading(title, level=level)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.page_break_before = new_page
        paragraph.paragraph_format.space_before = Pt(0 if new_page else 21)
        paragraph.paragraph_format.space_after = Pt(0)
        self.add_blank_line()

    def add_subheading(self, title):
        self.add_heading(title, level=2, new_page=False)

    def add_table(self, caption, headers, rows, widths=None):
        self.table_number += 1
        paragraph = self.document.add_paragraph(f"Таблица {self.table_number}. {caption}")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        paragraph_format = paragraph.paragraph_format
        paragraph_format.first_line_indent = Cm(0)
        paragraph_format.line_spacing = 1.0
        paragraph_format.space_before = Pt(7)
        paragraph_format.space_after = Pt(4)
        paragraph_format.keep_with_next = True
        for run in paragraph.runs:
            set_font(run, size=12)

        table = self.document.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        for cell, value in zip(table.rows[0].cells, headers):
            cell.text = str(value)
        table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for values in rows:
            for cell, value in zip(table.add_row().cells, values):
                cell.text = str(value)
        for row_index, row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for paragraph in cell.paragraphs:
                    paragraph_format = paragraph.paragraph_format
                    paragraph_format.first_line_indent = Cm(0)
                    paragraph_format.line_spacing = 1.0
                    paragraph_format.space_before = Pt(3)
                    paragraph_format.space_after = Pt(3)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in paragraph.runs:
                        set_font(run, size=11, bold=row_index == 0)
        fit_table_width(self.document, table, widths if widths is not None else [1] * len(headers))
        self.document.add_paragraph().paragraph_format.line_spacing = 0.5

    def add_figure(self, filename, caption, width=16.3):
        self.figure_number += 1
        paragraph = self.document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.keep_with_next = True
        paragraph.add_run().add_picture(str(RESULTS_DIR / filename), width=Cm(width))
        paragraph = self.document.add_paragraph(f"Рис. {self.figure_number} – {caption}")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(21)
        for run in paragraph.runs:
            set_font(run, size=12)

    def save(self):
        self.document.save(REPORT_PATH)
        print(REPORT_PATH)
        print("Tables:", self.table_number, "Figures:", self.figure_number)


def add_dataset_section(report):
    report.add_heading("1 Описание датасета")
    report.add_text(
        "Использован набор Bank Marketing из UCI Machine Learning Repository: 45 211 записей "
        "о телефонных маркетинговых контактах португальского банка. Таблица содержит 16 входных "
        "признаков: девять категориальных, включая бинарные, и семь числовых. Ответ Y показывает, "
        "оформлен ли срочный депозит; он исключён из кластеризации и используется только для оценки результатов.",
    )
    report.add_table(
        'Признаки набора Bank Marketing',
        ['Группа', 'Признаки', 'Содержание'],
        [
            [
                'Демографические',
                'age, job, marital, education',
                'Возраст, профессия, семейное положение и образование',
            ],
            ['Финансовые', 'default, balance, housing, loan', 'Просрочка, баланс и наличие кредитов'],
            ['Последний контакт', 'contact, day, month, duration', 'Способ, дата и длительность связи'],
            [
                'История кампаний',
                'campaign, pdays, previous, poutcome',
                'Число контактов, давность и исход прошлой кампании',
            ],
            ['Внешний ответ', 'Y', 'Оформление депозита: yes / no'],
        ],
        [3.7, 6, 6.8],
    )
    report.add_text(
        "Каждая строка характеризует запись о контакте; одному клиенту могут соответствовать "
        "несколько контактов. Требуется выделить группы похожих записей и сравнить их характеристики.",
    )


def add_descriptive_section(report):
    numeric = pd.read_csv(RESULTS_DIR / "numeric_statistics.csv", index_col="feature")

    report.add_heading("2 Дескриптивный анализ")
    report.add_subheading("2.1 Основные характеристики данных")
    report.add_text(
        "Пустых ячеек и дубликатов исходных строк нет; используются все 45 211 наблюдений. "
        "Отдельная категория unknown встречается в job – 288 раз, education – 1857, contact – 13 020, "
        "poutcome – 36 959. Она сохранена без замены. Исход no представлен 39 922 записями "
        "(88,30%), yes – 5289 записями (11,70%); этот дисбаланс учитывается при оценке чистоты кластеров.",
    )
    report.add_table(
        'Описательные статистики числовых признаков',
        ['Признак', 'Минимум', 'Медиана', 'Среднее', 'Максимум'],
        [
            [
                name,
                format_decimal(row['min'], 0),
                format_decimal(row['50%'], 0),
                format_decimal(row['mean']),
                format_decimal(row['max'], 0),
            ]
            for (name, row) in numeric.iterrows()
        ],
        [3, 3.2, 3.5, 3.5, 3.3],
    )
    report.add_text(
        "Баланс и длительность разговора имеют выраженные правые хвосты: средние превышают "
        "медианы. Значение pdays = −1 означает отсутствие предыдущего контакта, поэтому выделяется "
        "в отдельную категорию. Наиболее связаны признаки истории контактов: V Крамера для pdays, "
        "previous и poutcome составляет около 0,58; их влияние дополнительно проверяется в разделе 5.",
    )

    report.add_subheading("2.2 Подготовка признаков")
    report.add_table(
        'Категоризация числовых признаков',
        ['Признак', 'Категории'],
        [
            ['age', '18–30; 31–45; 46–60; 61 год и старше'],
            ['balance', 'Отрицательный; 0; 1–1000; 1001–5000; более 5000 евро'],
            ['day', '1–10; 11–20; 21–31 день месяца'],
            ['duration', '≤103; 104–180; 181–319; ≥320 секунд'],
            ['campaign', '1; 2; 3–5; 6 и более контактов'],
            ['pdays', 'Нет прошлого контакта; 0–90; 91–180; 181–365; более 365 дней'],
            ['previous', '0; 1; 2–3; 4 и более контактов'],
        ],
        [3.5, 13],
    )
    report.add_text(
        "Для duration использованы квартильные границы; остальные интервалы выбраны по смыслу "
        "признаков без обращения к Y. После преобразования 6939 строк повторяют ранее встречавшиеся "
        "профили; они сохранены для учёта исходных частот. Оба метода получают одинаковые 16 "
        "категориальных признаков с равными весами. Порядок уровней и различия внутри интервалов не учитываются.",
    )


def add_geometry_table(report, method):
    profiles = pd.read_csv(RESULTS_DIR / f"{method}_cluster_profiles.csv")
    report.add_table(
        f"Внутрикластерные расстояния, компактность и силуэт {method}",
        ["Кластер", "Внутри Dв", "До моды Q", "Радиус R", "Силуэт", "Отрицат., %"],
        [
            [
                int(row.cluster),
                format_decimal(row.mean_within_distance, 4),
                format_decimal(row.mean_distance_to_mode, 4),
                format_decimal(row.max_distance_to_mode, 4),
                format_decimal(row.silhouette_full, 4),
                format_decimal(row.negative_silhouette_percent_full),
            ]
            for row in profiles.itertuples()
        ],
        [1.8, 3, 3, 2.7, 3, 3],
    )


def add_clope_section(report):
    clope_metrics = pd.read_csv(RESULTS_DIR / "metrics.csv").set_index("method").loc["CLOPE"]
    parameter_search = pd.read_csv(RESULTS_DIR / "parameter_search.csv")
    profiles = pd.read_csv(RESULTS_DIR / "CLOPE_cluster_profiles.csv")
    count_comparison = pd.read_csv(RESULTS_DIR / "CLOPE_cluster_count_comparison.csv").set_index("variant")

    report.add_heading("3 Решение задачи кластеризации методом CLOPE. Оценка и интерпретация результатов")
    report.add_subheading("3.1 Представление данных и показатели качества")
    report.add_text(
        "Строка представлена транзакцией из 16 элементов вида «признак = категория». "
        "Общая мера различия для обоих методов – нормированный Хэмминг: "
        "d(x,z) = (1/16)·Σ I(xⱼ ≠ zⱼ). Это доля несовпавших категорий от 0 до 1. "
        "Для подготовленных номинальных данных с равными весами она совпадает с несходством Гауэра."
    )
    report.add_text(
        "Центр μc задаётся модой каждого признака; он может не совпадать с реальным наблюдением. "
        "Для CLOPE это описательный представитель группы, для k-modes – центр, использованный моделью. "
        "Полные модальные центры приведены в разделе 5. Расстояние между центрами также считается по Хэммингу."
    )
    report.add_text(
        "Внутрикластерное расстояние Dв(c) – среднее d(x,z) по всем различным парам объектов "
        "одного кластера. Межкластерное Dм(c,d) – среднее по всем парам, в которых один объект "
        "принадлежит c, другой – d. Компактность характеризуется средним расстоянием до моды "
        "Q(c) = (1/nc)·Σ d(x,μc); дополнительно указан максимальный радиус R(c) = max d(x,μc). "
        "Меньшие Dв, Q и R означают более тесную группу; большие Dм и расстояния между центрами – "
        "большую разделённость. Эти показатели оцениваются совместно."
    )
    report.add_text(
        "Силуэт объекта s = (b − a)/max(a,b), где a – среднее расстояние до остальных объектов "
        "своей группы, b – наименьшее среднее расстояние до другой группы. Чем ближе s к 1, "
        "тем лучше отделён объект; отрицательное значение указывает на возможное неудачное назначение. "
        "Итоговые расстояния, компактность и силуэты рассчитаны по всем 45 211 наблюдениям, включая малые группы."
    )
    report.add_text(
        "Для выбора r и k используется одна случайная подвыборка из 3000 строк; "
        "другая непересекающаяся подвыборка из 3000 строк служит дополнительным контролем. "
        "Seed отбора – 20261114. Обучение проводится на всём наборе. Ответ Y исключён из "
        "признаков и выбора параметров; разбиения оцениваются как описание имеющихся контактов."
    )

    report.add_subheading("3.2 Выбор параметра и проверка числа кластеров")
    report.add_text(
        "Критерий CLOPE: Profit(P,r) = (1/n)·Σ N(c)·S(c)/W(c)ʳ → max, где N – число "
        "транзакций, S – суммарное число элементов, W – число различных элементов кластера. "
        "Параметр r > 1 штрафует разнообразие внутри групп; число кластеров формируется алгоритмом."
    )
    report.add_text(
        "Проверены r = 1,01; 1,1; 1,2; 1,3; 1,4; 1,5; 1,6; 1,7; 1,8. Для каждого r "
        "использованы три порядка строк с seed 42, 43 и 44. При фиксированном r выбирается "
        "наибольший Profit; затем r выбирается по силуэту подвыборки настройки. Так учитывается "
        "зависимость CLOPE от начального порядка. Profit при разных r напрямую не сравнивается."
    )
    report.add_table(
        "Подбор r: лучший Profit из трёх порядков для каждого значения",
        ["r", "Групп", "Seed", "Силуэт выбора", "Секунды*", "Проходов"],
        [
            [format_decimal(row.parameter, 2), int(row.n_clusters), int(row.seed),
             format_decimal(row.silhouette_selection, 4), format_decimal(row.fit_seconds), int(row.iterations)]
            for row in parameter_search.loc[parameter_search.method == "CLOPE"].itertuples()
        ],
        [1.8, 1.9, 1.8, 4, 3.5, 3.5],
    )
    report.add_text(
        "* Суммарное время работы CLOPE для трёх порядков без расчёта метрик; число проходов "
        "относится к выбранному запуску. Число групп в эвристическом решении не обязано строго "
        "монотонно зависеть от r. При r = 1,1; 1,2; 1,3 получено одно и то же разбиение; "
        "из равных по силуэту вариантов выбран первый."
    )
    report.add_text(
        f"Основное решение: r = {format_decimal(clope_metrics.parameter, 1)}, seed = {int(clope_metrics.seed)}, "
        f"{int(clope_metrics.n_clusters)} кластера. Profit = {format_decimal(clope_metrics.objective)}, "
        f"силуэт выбора = {format_decimal(clope_metrics.silhouette_selection, 4)}, "
        f"контрольный = {format_decimal(clope_metrics.silhouette_validation, 4)}. "
        "Кластеры пронумерованы по убыванию размера."
    )
    report.add_table(
        "Два и три кластера при одинаковом r = 1,1; итоговые метрики по всему набору",
        ["Вариант", "Групп", "Profit", "Внутри Dв", "Между Dм", "До моды Q", "Силуэт"],
        [
            [name, int(row.n_clusters), format_decimal(row.objective),
             format_decimal(row.mean_within_distance, 4), format_decimal(row.mean_between_distance, 4),
             format_decimal(row.mean_distance_to_mode, 4), format_decimal(row.silhouette_full, 4)]
            for name, row in zip(["A", "B", "C"], count_comparison.itertuples())
        ],
        [1.5, 1.7, 3, 2.6, 2.6, 2.6, 2.5],
    )
    report.add_text(
        "A – естественное решение CLOPE с seed 43: 41 205 и 4006 записей. "
        "B – естественное решение с seed 42: 41 499, 3661 и 51 запись. "
        "C – диагностическое объединение третьей группы B с ближайшей по среднему "
        "межкластерному расстоянию второй группой: 41 499 и 3712 записей. "
        "Для C алгоритм повторно не запускался; это проверка необходимости малой группы, а не самостоятельный запуск CLOPE."
    )
    original = count_comparison.loc["three_clusters"]
    merged = count_comparison.loc["merged_smallest"]
    report.add_text(
        f"У варианта B силуэт {format_decimal(original.silhouette_full, 4)}, что выше, чем у A, "
        "но его Profit при том же r ниже. Третья группа содержит только 0,11% набора; "
        f"её собственный силуэт положителен: {format_decimal(original.smallest_cluster_silhouette, 4)}. "
        "Поэтому один малый размер не доказывает её ошибочность. Однако объединение в варианте C "
        f"повышает общий силуэт до {format_decimal(merged.silhouette_full, 4)} и немного увеличивает Profit "
        "при почти неизменном Dв. Для описательной сегментации выделять эти 51 запись в отдельный "
        "третий кластер не требуется. Варианты A и C имеют разный состав двух групп; "
        "CLOPE оптимизирует Profit, поэтому лучший Profit не обязан давать лучший силуэт."
    )

    report.add_subheading("3.3 Профили и интерпретация основного решения CLOPE")
    report.add_table(
        "Профили двух кластеров CLOPE по исходным данным",
        ["Кластер", "Число", "Доля, %", "Депозит, %", "Возраст*", "Баланс*"],
        [
            [int(row.cluster), format_integer(row.n), format_decimal(row.percent),
             format_decimal(row.yes_percent), format_decimal(row.age_median, 0), format_decimal(row.balance_median, 0)]
            for row in profiles.itertuples()
        ],
        [1.8, 3.5, 3, 3.2, 2.5, 2.5],
    )
    report.add_text(
        "* Медианы; баланс в евро. Кластер 1 объединяет 41 205 записей. "
        "У 84,11% не было предыдущих контактов, у 57,95% есть жилищный кредит. "
        "Наиболее частый месяц – май (31,84%); доля депозита – 9,72%. "
        "Это широкая группа контактов, которую нельзя считать узким однородным типом клиентов."
    )
    report.add_text(
        "Кластер 2 содержит 4006 записей. Доля retired составляет 23,69%, отсутствует "
        "жилищный кредит у 68,75%, исход прошлой кампании success встречается у 13,33%. "
        "Наиболее частый месяц – октябрь (18,42%), доля депозита – 32,00%. "
        "Такие модальные признаки описывают профиль группы, но не обязательно принадлежат большинству её объектов."
    )

    report.add_subheading("3.4 Расстояния, компактность и качество кластеров CLOPE")
    add_geometry_table(report, "CLOPE")
    report.add_text(
        f"Для всего решения среднее расстояние внутри групп Dв = {format_decimal(clope_metrics.mean_within_distance, 4)}, "
        f"между группами Dм = {format_decimal(clope_metrics.mean_between_distance, 4)}, "
        f"компактность Q = {format_decimal(clope_metrics.mean_distance_to_mode, 4)}. "
        "Расстояние между модальными центрами равно 0,2500: различаются 4 из 16 признаков. "
        "Dв и Q должны быть малыми; здесь их значения показывают существенную неоднородность."
    )
    report.add_text(
        f"Полный силуэт равен {format_decimal(clope_metrics.silhouette_full, 4)}, "
        f"отрицательны {format_percent(clope_metrics.negative_silhouette_percent_full)} индивидуальных значений. "
        "У второй группы средний силуэт отрицателен, а среднее внутрикластерное расстояние "
        "превышает межкластерное. Следовательно, высокую долю депозитов в её профиле нельзя "
        "считать доказательством хорошего геометрического отделения. Общая оценка преимущественно "
        "определяется большой первой группой."
    )
    feature_importance = pd.read_csv(RESULTS_DIR / "CLOPE_feature_importance.csv").head(6)
    report.add_table(
        "Связь признаков с основным разбиением CLOPE",
        ["Признак", "V Крамера"],
        [[row.feature, format_decimal(row.cramers_v, 4)] for row in feature_importance.itertuples()],
        [10, 6.5],
    )
    report.add_text(
        "V Крамера описывает связь признака с уже полученным номером кластера. "
        "Поскольку признаки участвовали в построении групп, это описательная оценка, "
        "а не независимый тест статистической значимости."
    )


def add_kmodes_section(report):
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv").set_index("method").loc["k-modes"]
    search = pd.read_csv(RESULTS_DIR / "parameter_search.csv")
    profiles = pd.read_csv(RESULTS_DIR / "k-modes_cluster_profiles.csv")

    report.add_heading("4 Кластеризация методом k-modes. Оценка и интерпретация результатов")
    report.add_subheading("4.1 Метод и выбор числа кластеров")
    report.add_text(
        "K-modes работает с категориями и минимизирует число несовпадений с модальными "
        "центрами: J = Σ по объектам Σ по признакам I(xⱼ ≠ μcⱼ) → min. Алгоритм чередует "
        "назначение ближайшей моде и пересчёт мод. Признаки и их веса совпадают с входом CLOPE. "
        "Использованы Huang, три начальных размещения и максимум 50 итераций, seed = 42; "
        "при фиксированном k выбирается наименьшая стоимость J."
    )
    report.add_table(
        "Подбор k по силуэту подвыборки настройки",
        ["k", "Силуэт", "Мин. размер", "Секунды*", "Итераций"],
        [
            [int(row.parameter), format_decimal(row.silhouette_selection, 4),
             format_integer(row.min_size), format_decimal(row.fit_seconds), int(row.iterations)]
            for row in search.loc[search.method == "k-modes"].itertuples()
        ],
        [2, 3.5, 4, 3.5, 3.5],
    )
    report.add_text(
        "* Время включает три начальных размещения; число итераций относится к выбранному из них. "
        f"Выбрано k = {int(metrics.parameter)}: силуэт настройки {format_decimal(metrics.silhouette_selection, 4)}, "
        f"контрольный {format_decimal(metrics.silhouette_validation, 4)}. Стоимость J = {format_integer(metrics.objective)} "
        "сравнивает назначения при одном представлении данных; напрямую сопоставлять её с Profit CLOPE нельзя."
    )
    report.add_figure("05_parameter_selection.png", "Подбор r и k по общему силуэту; подписи – число групп")

    report.add_subheading("4.2 Профили и интерпретация кластеров k-modes")
    report.add_table(
        "Профили кластеров k-modes по исходным данным",
        ["Кластер", "Число", "Доля, %", "Депозит, %", "Возраст*", "Баланс*"],
        [
            [int(row.cluster), format_integer(row.n), format_decimal(row.percent),
             format_decimal(row.yes_percent), format_decimal(row.age_median, 0), format_decimal(row.balance_median, 0)]
            for row in profiles.itertuples()
        ],
        [1.8, 3.5, 3, 3.2, 2.5, 2.5],
    )
    report.add_text(
        "* Медианы; баланс в евро. Кластер 1 содержит 32 098 записей (71,00%). "
        "Жилищный кредит есть у 70,31%, среднее образование – у 63,35%; наиболее частая "
        "профессия – blue-collar (28,95%). Медиана возраста – 40 лет, доля депозита – 8,29%."
    )
    report.add_text(
        "Кластер 2 включает 13 113 записей (29,00%). У 80,46% нет жилищного кредита, "
        "64,90% имеют высшее образование, 46,09% относятся к management; доля single – 47,42%. "
        "Медиана возраста – 37 лет, доля депозита – 20,04%. Группы отражают образовательные "
        "и кредитные профили, а не только оформление депозита."
    )

    report.add_subheading("4.3 Расстояния, компактность и качество кластеров k-modes")
    add_geometry_table(report, "k-modes")
    report.add_text(
        f"Среднее внутрикластерное расстояние Dв = {format_decimal(metrics.mean_within_distance, 4)}, "
        f"межкластерное Dм = {format_decimal(metrics.mean_between_distance, 4)}, "
        f"компактность Q = {format_decimal(metrics.mean_distance_to_mode, 4)}. "
        "Модальные центры отличаются по 7 из 16 признаков: расстояние 0,4375. "
        "У обеих групп средний силуэт положителен; вторая группа компактнее по Dв и Q."
    )
    report.add_text(
        f"Силуэт по всему набору равен {format_decimal(metrics.silhouette_full, 4)}, "
        f"отрицательны {format_percent(metrics.negative_silhouette_percent_full)} индивидуальных значений. "
        "Разница между Dм и Dв невелика, поэтому кластеры заметно перекрываются. "
        "Наличие разных модальных центров само по себе не означает хорошего разделения всех объектов."
    )
    report.add_figure("07_feature_importance.png", "Связь признаков с основными разбиениями двух методов")


def add_comparison_section(report):
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv").set_index("method")
    clope, kmodes = metrics.loc["CLOPE"], metrics.loc["k-modes"]
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    data_summary = json.loads((RESULTS_DIR / "data_summary.json").read_text())

    report.add_heading("5 Сравнение методов")
    report.add_subheading("5.1 Расстояния, компактность, центры и силуэт")
    comparisons = [
        ["Число кластеров", int(clope.n_clusters), int(kmodes.n_clusters)],
        ["Силуэт по всем 45 211 строкам ↑", format_decimal(clope.silhouette_full, 4), format_decimal(kmodes.silhouette_full, 4)],
        ["Среднее внутри групп Dв ↓", format_decimal(clope.mean_within_distance, 4), format_decimal(kmodes.mean_within_distance, 4)],
        ["Среднее между группами Dм ↑", format_decimal(clope.mean_between_distance, 4), format_decimal(kmodes.mean_between_distance, 4)],
        ["Компактность: до моды Q ↓", format_decimal(clope.mean_distance_to_mode, 4), format_decimal(kmodes.mean_distance_to_mode, 4)],
        ["Расстояние между центрами ↑", format_decimal(clope.min_center_distance, 4), format_decimal(kmodes.min_center_distance, 4)],
        ["Отрицательные силуэты, % ↓", format_decimal(clope.negative_silhouette_percent_full), format_decimal(kmodes.negative_silhouette_percent_full)],
        ["Максимальная группа, %", format_decimal(100 * clope.max_size / data_summary["n"]), format_decimal(100 * kmodes.max_size / data_summary["n"])],
        ["ARI с Y", format_decimal(clope.ARI_y, 4), format_decimal(kmodes.ARI_y, 4)],
    ]
    report.add_table(
        "Оценка основных решений на одинаковых признаках по нормированному Хэммингу",
        ["Показатель", "CLOPE", "k-modes"], comparisons, [9.5, 3.5, 3.5],
    )
    report.add_text(
        "Стрелки показывают предпочтительное направление изменения при совместной оценке. "
        "Общие Dв и Dм взвешены числом соответствующих пар объектов, а Q и силуэт – числом "
        "объектов. Поэтому средние значения могут скрывать качество малой группы. "
        "В обоих основных решениях по два кластера; приведённые Dм и расстояния между "
        "центрами относятся к единственной паре групп каждого метода."
    )

    center_tables = []
    center_columns = []
    for method in ("CLOPE", "k-modes"):
        modes = pd.read_csv(RESULTS_DIR / f"{method}_modes.csv", dtype={"mode": str})
        modes["center"] = method + " " + modes["cluster"].astype(str)
        center_columns.extend(modes["center"].unique())
        center_tables.append(modes)
    centers = pd.concat(center_tables).pivot(index="feature", columns="center", values="mode")
    feature_order = list(data_summary["prepared_levels"])
    category_labels = {"not_contacted": "нет контактов", "Q1": "≤103 с", "Q2": "104–180 с", "Q3": "181–319 с", "Q4": "≥320 с"}
    report.add_table(
        "Полные модальные центры: наиболее частые категории 16 признаков",
        ["Признак", *center_columns],
        [
            [feature, *[category_labels.get(centers.loc[feature, column], centers.loc[feature, column])
                        for column in center_columns]]
            for feature in feature_order
        ],
        [3.3, 3.3, 3.3, 3.3, 3.3],
    )
    report.add_text(
        "Первые модальные центры двух методов совпадают. В CLOPE второй центр отличается "
        "от первого профессией, жилищным кредитом, месяцем и длительностью контакта. "
        "В k-modes дополнительно различаются семейное положение, образование и число контактов "
        "в текущей кампании; это согласуется с большим расстоянием между центрами."
    )
    report.add_text(
        f"ARI между самими разбиениями CLOPE и k-modes равен {format_decimal(summary['cross_method_ARI'], 4)}. "
        "Он характеризует согласие назначений, а не компактность. В строке «ARI с Y» "
        "каждый метод отдельно сопоставляется с известным ответом о депозите; "
        "основная оценка структуры проводится по расстояниям и силуэту."
    )
    report.add_figure("06_cluster_sizes.png", "Размеры двух кластеров каждого метода")
    report.add_figure("08_silhouettes.png", "Индивидуальные силуэты по всем 45 211 наблюдениям")
    report.add_text(
        "У CLOPE выше общий силуэт и среднее межкластерное расстояние, однако вторая группа "
        "неоднородна и имеет отрицательный средний силуэт. У k-modes меньше расстояния внутри "
        "групп и до модальных центров, центры сильнее различаются, обе группы имеют положительные "
        "средние силуэты и более сопоставимые размеры. Таким образом, показатели оценивают разные "
        "стороны структуры и не дают универсального победителя."
    )

    report.add_subheading("6 Выводы")
    report.add_text(
        "Для основного решения CLOPE выбраны два кластера при r = 1,1. Отдельная проверка "
        "показывает, что выделение группы из 51 записи в третий кластер не требуется: её "
        "объединение с ближайшей группой повышает силуэт. При этом разные начальные порядки "
        "дают разные локальные решения, а Profit и силуэт не обязаны ранжировать их одинаково."
    )
    report.add_text(
        "По совокупности компактности, различия модальных центров, размеров и положительных "
        "средних силуэтов обеих групп k-modes удобнее для описательной сегментации клиентов. "
        "CLOPE имеет более высокий общий силуэт, но этот показатель преимущественно отражает "
        "большую группу и не подтверждает хорошее отделение второй. Обоим решениям свойственны "
        "перекрытие и зависимость от представления данных, поэтому выделенные группы следует "
        "считать описательными сегментами, а не доказанными естественными классами."
    )


def build_report():
    report = Report()
    report.add_goal("Изучить кластеризацию категориальных и смешанных данных.")
    add_dataset_section(report)
    add_descriptive_section(report)
    add_clope_section(report)
    add_kmodes_section(report)
    add_comparison_section(report)
    report.save()


if __name__ == "__main__":
    build_report()
