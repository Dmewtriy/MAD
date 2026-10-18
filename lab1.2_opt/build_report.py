"""Собрать дополнительный отчёт; главы 1 и 2 перенести без изменений.

Расчёты не выполняются: читаются CSV/JSON/PNG из results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from report_style import add_equation, add_figure, add_heading, add_paragraph, add_table, set_run_font

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
REFERENCE = BASE / "template/Исходный_отчёт.docx"
OUTPUT = BASE / "Отчёт_ЛР1.2_Дополнительное_задание.docx"
GOAL = ("Построить и сравнить две дополнительные регрессионные модели для порядковой оценки "
        "страхового риска: порядковую логистическую и порядковую пробит-регрессию. "
        "Оценить значимость коэффициентов и моделей, исследовать остатки и качество прогноза.")


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}".replace(".", ",")


def pformat(value):
    return f"{float(value):.3e}".replace(".", ",") if float(value) < .0001 else fmt(value, 5)


def read(name):
    return pd.read_csv(RESULTS / f"{name}.csv")


def blocks_for_report():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    blocks, table_number, figure_number = [], 4, 1

    def h(text, level=1):
        blocks.append({"kind": "heading", "text": text, "level": level})

    def p(text):
        blocks.append({"kind": "paragraph", "text": text})

    def eq(text):
        blocks.append({"kind": "equation", "text": text})

    def page():
        blocks.append({"kind": "page"})

    def table(title, headers, rows, widths):
        nonlocal table_number
        table_number += 1
        blocks.append({"kind": "table", "caption": f"Таблица {table_number}. {title}",
                       "headers": headers, "rows": rows, "widths": widths})

    def figure(filename, title):
        nonlocal figure_number
        figure_number += 1
        blocks.append({"kind": "figure", "file": filename, "number": figure_number, "title": title})

    h("3 Регрессионные модели")
    p("Для тех же 205 автомобилей построены порядковая логистическая и порядковая пробит-регрессия. "
      "Они учитывают порядок уровней symboling: −2 < −1 < 0 < 1 < 2 < 3. "
      "Уровень −3 в выборке отсутствует. Исходных переменных по-прежнему три: "
      "оценка риска, масса и тип привода. Главы 1 и 2 сохранены без изменений.")
    eq("w = curb-weight / 100;   F = I(drive-wheels = fwd);   R = I(drive-wheels = rwd)")
    p("Индикаторы F и R равны 1 для соответствующего привода и 0 в остальных случаях. "
      "Базовая категория — 4wd. Коэффициент массы относится к прибавке 100 фунтов. "
      "Параметры оцениваются максимальным правдоподобием; уровень значимости α = 0,05. "
      "Независимость записей и одинаковые коэффициенты при всех порогах приняты как предположения моделей [3–5].")
    p("Для диагностики используются квантильные остатки Данна–Смита [6], приведённые к стандартной "
      "нормальной шкале. Они подходят для категориального отклика. При верной модели их среднее должно "
      "быть близко к 0, а дисперсия — к 1. Случайная составляющая зафиксирована: seed = 12026. "
      "Гистограммы содержат 9 интервалов по формуле Стерджесса.")

    h("3.1 Порядковая логистическая регрессия", 2)
    p("Модель задаёт вероятность того, что оценка риска не превышает выбранный уровень cⱼ. "
      "Здесь θⱼ — границы категорий, η — линейный предиктор, Λ — логистическая функция [3–5].")
    eq("P(Y ≤ cⱼ | X) = Λ(θⱼ − η);   Λ(t) = 1 / (1 + exp(−t))")
    add_model("logit", "порядковой логистической регрессии", p, eq, table, figure, summary)

    page()
    h("3.2 Порядковая пробит-регрессия", 2)
    p("В пробит-модели используется функция стандартного нормального распределения Φ. "
      "Коэффициенты описывают изменение скрытой оценки риска в единицах стандартного отклонения "
      "скрытой ошибки. Нормальность самой symboling и массы не требуется [3–5].")
    eq("P(Y ≤ cⱼ | X) = Φ(θⱼ − η)")
    add_model("probit", "порядковой пробит-регрессии", p, eq, table, figure, summary)

    page()
    h("4 Сравнительный анализ и интерпретация результатов")
    logit, probit = (summary["models"][name] for name in ("logit", "probit"))
    table("Сравнение двух регрессионных моделей", ["Показатель", "Логистическая", "Пробит"],
          [["AIC", fmt(logit["aic"]), fmt(probit["aic"])],
           ["Псевдо-R² Мак-Фаддена", fmt(logit["mcfadden_pseudo_r_squared"]),
            fmt(probit["mcfadden_pseudo_r_squared"])],
           ["p-value модели", pformat(logit["model_p_value"]), pformat(probit["model_p_value"])]],
          [4200, 2577, 2577])
    p("Обе модели статистически значимы при α = 0,05. У логистической модели меньше AIC "
      "и немного выше псевдо-R², поэтому по качеству описания этой выборки ей можно отдать "
      "предпочтение. Различие между моделями невелико. Псевдо-R² характеризует улучшение "
      "правдоподобия и не является долей объяснённой дисперсии.")
    p("Обе модели показывают одинаковую основную связь: при фиксированном приводе увеличение массы "
      "связано со снижением вероятности высоких оценок риска. Различия между типами привода "
      "после учёта массы статистически не подтверждены; для группы 4wd вывод ограничен "
      "малым числом наблюдений — 9 автомобилей.")
    p("Низкий псевдо-R² и неодинаковый разброс остатков показывают, что масса и привод описывают "
      "риск лишь частично. Результаты характеризуют статистическую связь и не доказывают "
      "причинного влияния. Качество прогноза на новых данных в этой работе не оценивалось.")

    h("Список использованных источников")
    for text in [
        "1. Методические указания «Лабораторная работа 1.2. Регрессионный анализ»: ../lab1.2/ЛР1.2.pdf.",
        "2. Исходные данные 1985 Auto Imports и описание признаков: data/imports-85.data, data/imports-85.names.",
        "3. McCullagh P. Regression Models for Ordinal Data. Journal of the Royal Statistical Society. "
        "Series B. 1980. Vol. 42, No. 2. P. 109–142. https://www.stat.uchicago.edu/~pmcc/pubs/paper2.pdf",
        "4. statsmodels. OrderedModel: модель порядкового отклика. "
        "https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html",
        "5. statsmodels. Ordinal Regression: примеры logit и probit. "
        "https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html",
        "6. Dunn P. K., Smyth G. K. Randomized Quantile Residuals. Journal of Computational and Graphical "
        "Statistics. 1996. Vol. 5. P. 236–244. https://gksmyth.github.io/smyth/pubs/residual.html",
    ]:
        p(text)
    return blocks


def add_model(link, title, p, eq, table, figure, summary):
    coeff = read(f"{link}_coefficients")
    thresholds = read(f"{link}_thresholds")
    m = summary["models"][link]
    b = coeff.coefficient.to_numpy()
    sign = lambda value: ("+ " if value >= 0 else "− ") + fmt(abs(value), 6)
    eq(f"η = {fmt(b[0], 6)}w {sign(b[1])}F {sign(b[2])}R")
    table(f"Коэффициенты {title}", ["Параметр", "Коэффициент", "Ст. ошибка", "z", "p-value"],
          [[name, fmt(r.coefficient, 5), fmt(r.standard_error, 5), fmt(r.z, 3), pformat(r.p_value)]
           for name, r in zip(["w (100 фунтов)", "I(fwd)", "I(rwd)"], coeff.itertuples())],
          [2300, 2000, 1900, 1300, 1854])
    table(f"Пороги {title}", ["Уровень cⱼ", "−2", "−1", "0", "1", "2"],
          [["Порог θⱼ"] + [fmt(t, 5) for t in thresholds.threshold]], [1800, 1510, 1510, 1510, 1510, 1514])
    weight = coeff.iloc[0]
    if link == "logit":
        p(f"Коэффициент массы значим (p = {pformat(weight.p_value)}). При прибавке 100 фунтов "
          f"шансы оказаться выше любого выбранного порога умножаются на {fmt(weight.odds_ratio, 4)}, "
          f"то есть уменьшаются примерно на {fmt(100*(1-weight.odds_ratio), 2)}% при одинаковом приводе.")
    else:
        p(f"Коэффициент массы значим (p = {pformat(weight.p_value)}). При прибавке 100 фунтов "
          f"скрытая оценка риска снижается на {fmt(abs(weight.coefficient), 5)} стандартного отклонения "
          "скрытой ошибки при одинаковом приводе. Это соответствует снижению вероятности высоких уровней symboling.")
    p(f"Коэффициенты привода незначимы (p > 0,05). Модель в целом значима по критерию отношения "
      f"правдоподобий: χ²(3) = {fmt(m['likelihood_ratio'], 4)}, p = {pformat(m['model_p_value'])}. "
      "Проверяется улучшение по сравнению с моделью без факторов. Значимость отдельных коэффициентов "
      "оценена z-тестами Вальда.")
    figure(f"{link}_residuals.png", f"остатки и их гистограмма для {title}")
    p(f"Среднее квантильного остатка равно {fmt(m['residual_mean'], 4)}, дисперсия — "
      f"{fmt(m['residual_variance'], 4)}. Эти значения близки к 0 и 1. Однако отношение "
      f"максимальной групповой дисперсии к минимальной по четырём группам предиктора составляет "
      f"{fmt(m['group_variance_ratio'], 2)}, а максимальное абсолютное среднее в группе — "
      f"{fmt(m['max_group_abs_mean'], 3)}. Поэтому постоянство среднего и разброса остатков "
      "нельзя считать подтверждённым. Приблизительная колоколообразная форма гистограммы "
      "сама по себе не доказывает адекватность модели.")


def chapter_elements(doc):
    elements, inside = [], False
    for element in doc._element.body:
        if element.tag == qn("w:sectPr"):
            break
        text = ""
        # Paragraph.text исключает дублирование текста при обходе XML.
        if element.tag == qn("w:p"):
            text = Paragraph(element, doc._body).text
        if text.startswith("1 Описание исходных данных"):
            inside = True
        if text.startswith("3 Регрессионные модели"):
            break
        if inside:
            elements.append(element)
    return elements


def write_markdown(doc, blocks):
    text = ["# ЛР 1.2. Регрессионный анализ — дополнительное задание", "", "## Цель работы", "", GOAL, ""]
    for element in chapter_elements(doc):
        if element.tag == qn("w:p"):
            paragraph = Paragraph(element, doc._body)
            if paragraph.text:
                prefix = "## " if paragraph.text.startswith(("1 Описание", "2 Результаты")) else ""
                text += [prefix + paragraph.text, ""]
            if element.find(".//" + qn("w:drawing")) is not None:
                text += ["![Распределения выбранных переменных](results/selected_variables_histograms.png)", ""]
        elif element.tag == qn("w:tbl"):
            rows = [[cell.text.replace("\n", " ") for cell in row.cells] for row in Table(element, doc._body).rows]
            text += ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
            text += ["| " + " | ".join(row) + " |" for row in rows[1:]] + [""]
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            text += ["#" * (block["level"] + 1) + " " + block["text"], ""]
        elif kind in ("paragraph", "equation"):
            text += [block["text"], ""]
        elif kind == "table":
            text += [block["caption"], "", "| " + " | ".join(block["headers"]) + " |",
                     "| " + " | ".join(["---"] * len(block["headers"])) + " |"]
            text += ["| " + " | ".join(map(str, row)) + " |" for row in block["rows"]] + [""]
        elif kind == "figure":
            text += [f"![Рис. {block['number']} — {block['title']}](results/{block['file']})", ""]
    (BASE / "REPORT.md").write_text("\n".join(text), encoding="utf-8")


def build():
    doc = Document(REFERENCE)
    body = doc._element.body
    # Сохраняются исходные XML-узлы глав 1–2, включая таблицы и рисунок.
    original_chapters = [element.xml for element in chapter_elements(doc)]
    start = next(p._p for p in doc.paragraphs if p.text.startswith("3 Регрессионные модели"))
    removing = False
    for child in list(body):
        if child is start:
            removing = True
        if removing and child.tag != qn("w:sectPr"):
            body.remove(child)
    for paragraph in doc.paragraphs:
        if paragraph.text.strip() == "Изучение регрессионного анализа.":
            paragraph.clear()
            set_run_font(paragraph.add_run(GOAL), 14)
        elif "«Регрессионный анализ»" in paragraph.text:
            paragraph.clear()
            set_run_font(paragraph.add_run("«Регрессионный анализ»\n"), 14, bold=True)
            set_run_font(paragraph.add_run("Дополнительное задание\n"), 12, bold=True)
            set_run_font(paragraph.add_run("по дисциплине «Методы анализа данных»"), 12)
            # Компенсировать дополнительную строку на титульном листе.
            following = paragraph._p.getnext()
            if following is not None and following.tag == qn("w:p") and not Paragraph(following, doc._body).text:
                body.remove(following)
    assert [element.xml for element in chapter_elements(doc)] == original_chapters
    blocks = blocks_for_report()
    write_markdown(doc, blocks)
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            heading = add_heading(doc, block["text"], block["level"])
            heading.paragraph_format.keep_with_next = True
        elif kind == "paragraph":
            add_paragraph(doc, block["text"])
        elif kind == "equation":
            add_equation(doc, block["text"])
        elif kind == "page":
            doc.add_page_break()
        elif kind == "table":
            table = add_table(doc, block["caption"], block["headers"], block["rows"], block["widths"], font_size=10.5)
            # Короткие строки таблиц не должны разрываться между страницами.
            from docx.oxml import OxmlElement
            for row in table.rows:
                row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        elif kind == "figure":
            add_figure(doc, RESULTS / block["file"], block["title"], block["number"])
    doc.core_properties.title = "ЛР 1.2. Регрессионный анализ. Дополнительное задание"
    doc.core_properties.subject = "Порядковая логистическая и порядковая пробит-регрессии"
    doc.save(OUTPUT)
    # Проверка переноса выполняется и после повторного чтения DOCX.
    restored = Document(OUTPUT)
    assert [element.xml for element in chapter_elements(restored)] == original_chapters
    print(OUTPUT)
    print("Главы 1–2 совпадают с исходным отчётом на уровне XML.")


if __name__ == "__main__":
    build()
