"""Создание Word-отчёта по сохранённым результатам кластеризации."""

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



def build_report():
    """Оформление отчёта по сохранённым таблицам, параметрам и рисункам."""
    reference = BASE.parent / "lab2" / "Отчёт_ЛР2_Классификация.docx"
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
        elif relation.reltype.endswith("/hyperlink"):
            doc.part.drop_rel(relation_id)

    def font(run, size=14, bold=False):
        run.font.name, run.font.size, run.font.bold = "Times New Roman", Pt(size), bold
        run.font.color.rgb = RGBColor(0, 0, 0)
        for name in ["ascii", "hAnsi", "eastAsia", "cs"]:
            run._element.get_or_add_rPr().rFonts.set(qn("w:" + name), "Times New Roman")

    for p in doc.paragraphs:
        if "ЛАБОРАТОРНАЯ РАБОТА №" in p.text:
            p.clear()
            font(p.add_run("ЛАБОРАТОРНАЯ РАБОТА №3"), bold=True)
        elif "Классификация" in p.text:
            p.clear()
            run = p.add_run("«Кластеризация категориальных и смешанных данных»")
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
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.keep_with_next = True
        pf.page_break_before = False
    section = doc.sections[-1]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(1.5)
    section.top_margin = section.bottom_margin = Cm(2)
    doc.core_properties.title = "Лабораторная работа №3. Кластеризация"
    doc.core_properties.subject = "Bank Marketing: CLOPE и k-modes"
    doc.core_properties.keywords = "кластеризация, Bank Marketing, CLOPE, k-modes"
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

    def blank_line(new_page=False):
        previous = doc.paragraphs[-1] if doc.paragraphs else None
        if (previous is not None and not previous.text
                and not previous._element.xpath('.//w:drawing | .//w:sectPr')):
            p = previous
        else:
            p = doc.add_paragraph()
        p.style = doc.styles["Normal"]
        font(p.add_run(""))
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p.paragraph_format.page_break_before = new_page
        p.paragraph_format.keep_with_next = True

    def heading(value, new_page=True):
        blank_line(new_page=new_page)
        p = doc.add_heading(value, level=1)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.page_break_before = False
        blank_line()

    def subheading(value):
        blank_line()
        p = doc.add_heading(value, level=2)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.page_break_before = False
        p.paragraph_format.space_after = Pt(0)
        blank_line()

    def number(value, digits=4):
        return f"{value:.{digits}f}".replace(".", ",")

    def scientific(value):
        return f"{value:.2e}".replace(".", ",")

    def table(caption, headers, rows, widths=None):
        nonlocal table_number
        table_number += 1
        rows = list(rows)
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


    # Content below is generated only from the saved experiment outputs.
    data = json.loads((RESULTS / "data_summary.json").read_text())
    config = json.loads((RESULTS / "summary.json").read_text())
    metrics = pd.read_csv(RESULTS / "metrics.csv").set_index("method")
    search = pd.read_csv(RESULTS / "parameter_search.csv")
    numeric = pd.read_csv(RESULTS / "numeric_statistics.csv", index_col="feature")
    sensitivity = pd.read_csv(RESULTS / "feature_sensitivity.csv")
    cp = pd.read_csv(RESULTS / "CLOPE_cluster_profiles.csv")
    kp = pd.read_csv(RESULTS / "k-modes_cluster_profiles.csv")
    clope, kmodes = metrics.loc["CLOPE"], metrics.loc["k-modes"]

    def integer(value):
        return f"{int(value):,}".replace(",", " ")

    def perc(value, digits=2):
        return number(float(value), digits) + "%"

    def stat(value, digits=2):
        return number(float(value), digits)

    blank_line()
    goal = doc.add_paragraph("Цель работы")
    goal.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    goal.paragraph_format.keep_with_next = True
    for run in goal.runs:
        font(run, bold=True)
    blank_line()
    text("Изучить кластеризацию категориальных и смешанных данных, применить методы "
         "CLOPE и k-modes к набору Bank Marketing, оценить качество, устойчивость и содержательный смысл полученных групп.")

    heading("1 Описание датасета")
    text("Использован набор Bank Marketing из UCI Machine Learning Repository: 45 211 записей "
         "о телефонных маркетинговых контактах португальского банка. Таблица содержит 16 входных "
         "признаков: девять категориальных, включая бинарные, и семь числовых. Ответ Y показывает, "
         "оформлен ли срочный депозит; он исключён из кластеризации и используется только для оценки результатов.")
    table("Признаки набора Bank Marketing", ["Группа", "Признаки", "Содержание"], [
        ["Демографические", "age, job, marital, education", "Возраст, профессия, семейное положение и образование"],
        ["Финансовые", "default, balance, housing, loan", "Просрочка, баланс и наличие кредитов"],
        ["Последний контакт", "contact, day, month, duration", "Способ, дата и длительность связи"],
        ["История кампаний", "campaign, pdays, previous, poutcome", "Число контактов, давность и исход прошлой кампании"],
        ["Внешний ответ", "Y", "Оформление депозита: yes / no"],
    ], [3.7, 6, 6.8])
    text("Каждая строка характеризует запись о контакте; одному клиенту могут соответствовать "
         "несколько контактов. Требуется выделить группы похожих записей и сравнить их характеристики.")

    heading("2 Дескриптивный анализ")
    subheading("2.1 Основные характеристики данных")
    text("Пустых ячеек и дубликатов исходных строк нет; используются все 45 211 наблюдений. "
         "Отдельная категория unknown встречается в job – 288 раз, education – 1857, contact – 13 020, "
         "poutcome – 36 959. Она сохранена без замены. Исход no представлен 39 922 записями "
         "(88,30%), yes – 5289 записями (11,70%); этот дисбаланс учитывается при оценке чистоты кластеров.")
    table("Описательные статистики числовых признаков", ["Признак", "Минимум", "Медиана", "Среднее", "Максимум"], [
        [name, stat(row["min"], 0), stat(row["50%"], 0), stat(row["mean"]), stat(row["max"], 0)]
        for name, row in numeric.iterrows()
    ], [3, 3.2, 3.5, 3.5, 3.3])
    text("Баланс и длительность разговора имеют выраженные правые хвосты: средние превышают "
         "медианы. Значение pdays = −1 означает отсутствие предыдущего контакта, поэтому выделяется "
         "в отдельную категорию. Наиболее связаны признаки истории контактов: V Крамера для pdays, "
         "previous и poutcome составляет около 0,58; их влияние дополнительно проверяется в разделе 5.")

    subheading("2.2 Подготовка признаков")
    table("Категоризация числовых признаков", ["Признак", "Категории"], [
        ["age", "18–30; 31–45; 46–60; 61 год и старше"],
        ["balance", "Отрицательный; 0; 1–1000; 1001–5000; более 5000 евро"],
        ["day", "1–10; 11–20; 21–31 день месяца"],
        ["duration", "≤103; 104–180; 181–319; ≥320 секунд"],
        ["campaign", "1; 2; 3–5; 6 и более контактов"],
        ["pdays", "Нет прошлого контакта; 0–90; 91–180; 181–365; более 365 дней"],
        ["previous", "0; 1; 2–3; 4 и более контактов"],
    ], [3.5, 13])
    text("Для duration использованы квартильные границы; остальные интервалы выбраны по смыслу "
         "признаков без обращения к Y. После преобразования 6939 строк повторяют ранее встречавшиеся "
         "профили; они сохранены для учёта исходных частот. Оба метода получают одинаковые 16 "
         "категориальных признаков с равными весами. Порядок уровней и различия внутри интервалов не учитываются.")

    heading("3 Решение задачи кластеризации методом CLOPE. Оценка и интерпретация результатов")
    subheading("3.1 Представление данных и оценка качества")
    text("Каждая строка преобразована в транзакцию из 16 элементов вида «признак = категория». "
         "CLOPE объединяет транзакции с перекрывающимися элементами, формируя число кластеров в ходе работы.")
    text("Для общей оценки двух методов используется нормированный Хэмминг: "
         "d(x,z) = (1/16)·Σ I(xⱼ ≠ zⱼ), то есть доля несовпавших категорий. Для подготовленных "
         "номинальных данных с равными весами он совпадает с несходством Гауэра.")
    text("Качество оценивается силуэтом s = (b − a)/max(a,b), где a – среднее расстояние "
         "до объектов своего кластера, b – наименьшее среднее расстояние до другого кластера. "
         "Чем ближе силуэт к 1, тем лучше отделён объект; отрицательные значения указывают "
         "на возможное неудачное назначение.")
    text("Кластеризация проводится на всех 45 211 строках. Параметры выбираются по силуэту "
         "на общей случайной подвыборке из 3000 объектов; контрольный силуэт рассчитывается "
         "на других 3000 объектах. Подвыборки не пересекаются; seed = 20261114. Ответ Y в выборе "
         "параметров не участвует. Такая проверка оценивает разбиение, а не предсказание новых клиентов.")

    subheading("3.2 Критерий и выбор параметра отталкивания")
    text("Для кластера c величины N(c), S(c) и W(c) обозначают число транзакций, суммарное число "
         "элементов и количество различных элементов. Средняя высота гистограммы H(c) = S(c)/W(c). "
         "Критерий: Profit(P,r) = (1/n)·Σ N(c)·S(c)/W(c)ʳ → max. Параметр r > 1 регулирует штраф "
         "за разнообразие внутри группы; число кластеров возникает в ходе работы.")
    text("Исследованы r = 1,1; 1,2; 1,3; 1,4; 1,5; 1,6; 1,7; 1,8. Для каждого значения строки "
         "переставлены с seed = 42, после чего выполняются начальное размещение и уточняющие проходы "
         "до прекращения перемещений. Параметр выбирается по максимальному силуэту на общей "
         "подвыборке выбора; внешняя метка Y в выборе не участвует.")
    table("Подбор r для CLOPE", ["r", "Кластеров", "Силуэт", "Мин. размер", "Секунды", "Проходов*"], [
        [stat(row.parameter, 1), int(row.n_clusters), stat(row.silhouette_selection, 4),
         integer(row.min_size), stat(row.fit_seconds, 2), int(row.iterations)]
        for row in search.loc[search.method == "CLOPE"].itertuples()
    ], [1.7, 2.5, 3, 3.5, 2.8, 3])
    text("* Число уточняющих проходов без начального размещения. Время относится к вызову алгоритма "
         "без расчёта метрик и подготовки данных. По мере изменения r число групп не обязано расти "
         "строго монотонно из-за последовательной эвристики и локальных решений. Значения Profit "
         "при разных r напрямую не сравниваются, поскольку меняется сам критерий.")
    text('В вариантах с очень малыми группами подвыборка может содержать не все кластеры. У выбранного '
         'варианта в обеих подвыборках представлены все три группы.')
    text(f"Выбрано r = {stat(clope.parameter, 1)}: силуэт выбора {stat(clope.silhouette_selection, 4)}, "
         f"контрольный силуэт {stat(clope.silhouette_validation, 4)}. Получено {int(clope.n_clusters)} непустых "
         f"кластера за {int(clope.iterations)} уточняющих прохода. Итоговый Profit равен "
         f"{stat(clope.objective, 4)}. Номера непустых групп упорядочены по убыванию размера.")

    subheading("3.3 Характеристики и интерпретация кластеров CLOPE")
    parameters = pd.read_csv(RESULTS / "CLOPE_cluster_parameters.csv")
    table("Параметры кластеров CLOPE", ["Кластер", "N", "Доля", "S", "W", "H"], [
        [int(row.cluster), integer(row.n), perc(100*row.n/data["n"]), integer(row.S), int(row.W), stat(row.H, 2)]
        for row in parameters.itertuples()
    ], [2, 3.2, 3, 3.2, 2, 3.1])
    table("Профили CLOPE по исходным данным", ["Кластер", "Депозит, %", "Возраст*", "Баланс*", "Длительность*", "Нет прошлых контактов, %"], [
        [int(row.cluster), stat(row.yes_percent), stat(row.age_median, 0), stat(row.balance_median, 0),
         stat(row.duration_median, 0), stat(row.previous_zero_percent)] for row in cp.itertuples()
    ], [2, 2.7, 2.4, 2.7, 3, 3.7])
    text("* Медианы; баланс в евро, длительность в секундах. Эти исходные числовые значения "
         "использованы для описания результатов, хотя алгоритм работал с интервальными категориями.")
    text("Кластер 1 – основная масса записей: 41 499 объектов, 91,79% набора. В нём у 87,58% "
         "не было предыдущих контактов; доля подписки на депозит – 10,54%. Это широкая остаточная "
         "группа, а не узкий однородный тип клиентов.")
    text("Кластер 2 содержит 3661 запись, 8,10%. У 74,41% был ровно один предыдущий контакт, "
         "у 49,36% исход прежней кампании – failure, у 87,00% способ связи – cellular. "
         "По сравнению со всем набором в группе заметно больше записей с известной историей контактов. "
         "Доля оформленного депозита составляет 24,36%.")
    text("Кластер 3 очень мал: 51 запись, 0,11%. Все контакты относятся к марту, 94,12% имеют "
         "один предыдущий контакт, 84,31% – без жилищного кредита. Депозит оформлен в 25 из 51 "
         "случаев, или 49,02%. Это узкая временная подгруппа; её размер и чувствительность к порядку "
         "строк не позволяют считать её надёжно установленным самостоятельным сегментом.")
    text("Общее для CLOPE различие формируется прежде всего историей контактов и месяцем. "
         "Профили являются описанием полученного разбиения, а не доказательством причинного влияния "
         "прошлых контактов на оформление депозита.")

    subheading("3.4 Качество и информативность признаков")
    text(f"На контрольной подвыборке силуэт равен {stat(clope.silhouette_validation, 4)}, "
         f"доля отрицательных индивидуальных силуэтов – {perc(clope.negative_silhouette_percent)}. "
         "Это указывает на ограниченную отделимость групп. Все три кластера представлены в подвыборке, "
         "но для третьего имеется только три объекта, поэтому его частная оценка особенно неустойчива.")
    text(f"Согласие с внешним ответом: ARI = {stat(clope.ARI_y, 4)}, AMI = {stat(clope.AMI_y, 4)}, "
         f"NMI = {stat(clope.NMI_y, 4)}. Чистота – {perc(100*clope.purity)}, равная доле исходного "
         "класса no. Во всех трёх группах no остаётся наиболее частым классом; чистота не показывает "
         "преимущества над единственным кластером по этому показателю.")
    imp = pd.read_csv(RESULTS / "CLOPE_feature_importance.csv").head(6)
    table("Связь признаков с кластерами CLOPE", ["Признак", "V Крамера"], [
        [row.feature, stat(row.cramers_v, 4)] for row in imp.itertuples()
    ], [10, 6.5])
    text("Наиболее связаны с разбиением previous, month, pdays и poutcome. Оценка V Крамера "
         "описывает уже полученное разбиение; поскольку признаки использовались при его построении, "
         "она не является независимым тестом статистической значимости.")

    heading("4 Кластеризация методом k-modes. Оценка и интерпретация результатов")
    subheading("4.1 Обоснование метода и параметры")
    text('Для сравнения выбран метод k-modes. Он работает с категориями и минимизирует сумму несовпадающих '
         'координат между объектами и модальными центрами. Центр задаётся наиболее частой категорией каждого '
         'признака. Данные и веса признаков совпадают с входом CLOPE.')
    text('Критерий J = Σ по объектам Σ по признакам I(xⱼ ≠ μсⱼ) → min, где μс – вектор мод соответствующей '
         'группы. Алгоритм чередует назначение ближайшей моде и пересчёт мод. Использованы инициализация '
         'Huang, n_init = 3, max_iter = 50, random_state = 42. Из трёх начальных размещений выбирается '
         'результат с наименьшей стоимостью.')
    table("Подбор k для k-modes", ["k", "Силуэт", "Мин. размер", "Секунды", "Итераций*"], [
        [int(row.parameter), stat(row.silhouette_selection, 4), integer(row.min_size), stat(row.fit_seconds, 2), int(row.iterations)]
        for row in search.loc[search.method == "k-modes"].itertuples()
    ], [2, 3.5, 4, 3.5, 3.5])
    text('* Итерации лучшей инициализации. Время включает три начальных размещения, поэтому его нельзя '
         'трактовать как чистое сравнение одной итерации с CLOPE. Выбор числа кластеров вновь основан только '
         'на силуэте подвыборки выбора.')
    text(f"Выбрано k = {int(kmodes.parameter)}. Силуэт выбора – {stat(kmodes.silhouette_selection, 4)}, "
         f"контрольный – {stat(kmodes.silhouette_validation, 4)}. Стоимость J = {integer(kmodes.objective)} "
         "означает суммарное число несовпадений с назначенными модами. Эта стоимость не сравнивается "
         "напрямую с Profit CLOPE.")
    figure("05_parameter_selection.png", "Выбор параметров по общему силуэту Хэмминга; подписи показывают число непустых групп")

    subheading("4.2 Профили и содержательная интерпретация k-modes")
    table("Профили кластеров k-modes", ["Кластер", "Число", "Доля", "Депозит, %", "Возраст*", "Баланс*"], [
        [int(row.cluster), integer(row.n), perc(row.percent), stat(row.yes_percent), stat(row.age_median, 0), stat(row.balance_median, 0)]
        for row in kp.itertuples()
    ], [2, 3.5, 3, 3, 2.5, 2.5])
    text("* Медианы. Кластер 1 содержит 32 098 записей, 71,00% набора. В нём 70,31% клиентов "
         "имеют жилищный кредит, 63,35% – среднее образование; наиболее частая профессия – "
         "blue-collar (28,95%). Медиана возраста – 40 лет, доля депозита – 8,29%.")
    text("Кластер 2 включает 13 113 записей, 29,00%. У 80,46% нет жилищного кредита, 64,90% "
         "имеют высшее образование, 46,09% относятся к management. Доля single – 47,42%. "
         "Медиана возраста – 37 лет; доля депозита – 20,04%. Это различие характеристик клиентов "
         "и периода контакта, а не разделение исключительно на подписавших и не подписавших депозит.")
    text("По сравнению с CLOPE группы k-modes более сопоставимы по размеру и удобнее для "
         "описания образовательных и кредитных профилей. При этом низкий силуэт означает существенное "
         "перекрытие: яркие средние профили не гарантируют однозначного назначения каждой записи.")

    subheading("4.3 Оценка качества и связи признаков с разбиением")
    text(f"Доля отрицательных индивидуальных силуэтов составляет {perc(kmodes.negative_silhouette_percent)}. "
         f"ARI с ответом Y равен {stat(kmodes.ARI_y, 4)}, AMI – {stat(kmodes.AMI_y, 4)}, "
         f"NMI – {stat(kmodes.NMI_y, 4)}. Чистота {perc(100*kmodes.purity)} снова совпадает "
         "с долей no во всём наборе. Следовательно, различие долей депозита между двумя сегментами "
         "не означает восстановление бинарной целевой метки.")
    imp = pd.read_csv(RESULTS / "k-modes_feature_importance.csv").head(6)
    table("Связь признаков с кластерами k-modes", ["Признак", "V Крамера"], [
        [row.feature, stat(row.cramers_v, 4)] for row in imp.itertuples()
    ], [10, 6.5])
    figure("07_feature_importance.png", "Описательная связь признаков с номером кластера в двух методах")
    text("В k-modes наиболее выражены образование, профессия, жилищный кредит и месяц контакта; "
         "в CLOPE – история контактов. Значимости имеют описательный характер. Не выполняются "
         "ANOVA или тесты средних произвольных категориальных кодов.")

    heading("5 Сравнение методов")
    subheading("5.1 Общие метрики и размеры групп")
    comparisons = [
        ["Число непустых кластеров", int(clope.n_clusters), int(kmodes.n_clusters)],
        ["Силуэт, выбор (3000)", stat(clope.silhouette_selection, 4), stat(kmodes.silhouette_selection, 4)],
        ["Силуэт, контроль (3000)", stat(clope.silhouette_validation, 4), stat(kmodes.silhouette_validation, 4)],
        ["Отрицательные силуэты, %", stat(clope.negative_silhouette_percent), stat(kmodes.negative_silhouette_percent)],
        ["Максимальный кластер, %", stat(100*clope.max_size/data["n"]), stat(100*kmodes.max_size/data["n"])],
        ["Минимальный кластер, строк", integer(clope.min_size), integer(kmodes.min_size)],
        ["ARI с Y", stat(clope.ARI_y, 4), stat(kmodes.ARI_y, 4)],
        ["AMI с Y", stat(clope.AMI_y, 4), stat(kmodes.AMI_y, 4)],
        ["NMI с Y", stat(clope.NMI_y, 4), stat(kmodes.NMI_y, 4)],
        ["Чистота, %", stat(100*clope.purity), stat(100*kmodes.purity)],
        ["Средний ARI между запусками", stat(clope.stability_ARI_mean, 4), stat(kmodes.stability_ARI_mean, 4)],
        ["Время выбранного запуска, с*", stat(clope.fit_seconds), stat(kmodes.fit_seconds)],
    ]
    table("Сравнение на одинаковых данных и по одной мере расстояния", ["Показатель", "CLOPE", "k-modes"], comparisons, [9.5, 3.5, 3.5])
    text("* CLOPE – один порядок строк, k-modes – лучший результат из трёх инициализаций. Время "
         "служит характеристикой конкретной конфигурации на данном компьютере, а не универсальным "
         "сравнением скорости алгоритмов.")
    text(f"В строке «ARI с Y» каждый метод сравнивается с известным ответом Y. "
         f"Прямое сравнение разбиений CLOPE и k-modes даёт один ARI = "
         f"{stat(config['cross_method_ARI'], 4)}; это отдельное сравнение двух методов.")
    figure("06_cluster_sizes.png", "Размеры кластеров, упорядоченных по убыванию численности")
    figure("08_silhouettes.png", "Индивидуальные силуэты на общей контрольной подвыборке")
    text("CLOPE имеет больший общий силуэт, но выделяет одну доминирующую группу и очень малую "
         "мартовскую подгруппу. У k-modes размеры более сбалансированы, однако геометрическая "
         "отделимость по Хэммингу ниже. Различия критериев объясняют, почему предпочтение по силуэту "
         "не означает преимущество по удобству сегментации.")

    subheading("5.2 Чувствительность к признакам и ограничения")
    text("В дополнительном опыте удалены pdays и previous, оставлен poutcome; параметры r = 1,1 "
         "и k = 2 и seed = 42 сохранены. Это уменьшает повторное представление истории контактов. "
         "Проверяется изменение меток по ARI на всех строках; силуэты разных пространств здесь "
         "напрямую не сравниваются.")
    table("Чувствительность после удаления двух исторических признаков", ["Метод", "Признаков", "Кластеров", "ARI с основным решением"], [
        [row.method, int(row.n_features), int(row.n_clusters), stat(row.ARI_with_full_feature_solution, 4)]
        for row in sensitivity.itertuples()
    ], [4, 3, 3, 6.5])
    text("Оба решения существенно меняются. Следовательно, выбор признаков и их весов имеет "
         "практическое значение; представление трёх связанных исторических полей не является нейтральным. "
         "Этот опыт проверяет чувствительность при фиксированных параметрах и не доказывает, "
         "что уменьшенная таблица даёт лучшее разбиение.")
    figure("10_deposit_rates.png", "Доли оформленного депозита в кластерах; пунктир – весь набор")
    text("Ограничения: числовая информация сведена к интервалам; порядок и цикличность уровней "
         "не учитываются; неизвестные значения сохранены как категории; календарные признаки "
         "и история кампаний могут отражать особенности сбора данных. Duration известна только "
         "после разговора, поэтому работа описывает уже состоявшиеся контакты и не является моделью "
         "выбора клиента до звонка. Известный Y не обязательно совпадает с естественной структурой "
         "признаков. Качество оценивается на общей категоризованной геометрии и на конечном диапазоне параметров.")

    subheading("5.3 Выводы")
    text("Оба метода применимы к подготовленным категориальным данным и позволяют описать "
         "разные аспекты Bank Marketing. CLOPE в выбранном запуске даёт больший силуэт и выделяет "
         "историю контактов, но формирует сильно неравные и чувствительные к порядку группы. "
         "k-modes даёт более понятные образовательные и кредитные профили с умеренно разными "
         "долями депозита, однако силуэт также невысок, а инициализация влияет на границы кластеров.")
    text("Устойчивого универсального победителя по совокупности критериев не установлено. "
         "Для описательной сегментации клиентов вариант k-modes удобнее по размерам и смыслу "
         "профилей; это прикладной выбор с учётом интерпретации, а не утверждение о более высокой "
         "внутренней метрике. Обоим решениям требуется проверка устойчивости при дальнейшем "
         "практическом использовании. Высокая чистота объясняется дисбалансом Y и сама по себе "
         "не подтверждает качественное восстановление ответа о депозите.")

    output = BASE / "Отчёт_ЛР3_Кластеризация.docx"
    doc.save(output)
    print(output)
    print("Tables:", table_number, "Figures:", figure_number)


if __name__ == "__main__":
    build_report()
