"""Сборка DOCX из актуального оформления ЛР 1.3 и рассчитанных результатов.

Стили, геометрия таблиц и образцы абзацев читаются из копии текущего отчёта,
а не из более раннего генератора или отчёта ЛР 1.2.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

from report_content import blocks_for_report

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "template/Оформление_ЛР1.3.docx"
RESULTS = BASE / "results"
OUTPUT = BASE / "Отчёт_ЛР1.3_Дополнительное_задание.docx"
TABLE_INDICES = {"quality": 1, "descriptive": 2, "histograms": 3, "normality": 4, "test": 9}


def insert_paragraph(doc, sample, text=""):
    xml = deepcopy(sample._p)
    # Закладки исходных результатов и их идентификаторы не переносятся.
    for node in list(xml):
        if node.tag != qn("w:pPr"):
            xml.remove(node)
    doc._element.body.insert_element_before(xml, "w:sectPr")
    paragraph = Paragraph(xml, doc._body)
    if text:
        paragraph.add_run(text)
    return paragraph


def set_cell_text(cell, text, header):
    paragraph = cell.paragraphs[0]
    run_properties = deepcopy(paragraph.runs[0]._r.rPr) if paragraph.runs and paragraph.runs[0]._r.rPr is not None else None
    for other in list(cell.paragraphs[1:]):
        cell._tc.remove(other._p)
    paragraph.clear()
    run = paragraph.add_run(str(text))
    if run_properties is not None:
        run._r.insert(0, run_properties)
    run.font.name, run.font.size, run.bold = "Times New Roman", Pt(12), header
    fonts = run._r.get_or_add_rPr().rFonts
    for name in ["ascii", "hAnsi", "eastAsia", "cs"]:
        fonts.set(qn(f"w:{name}"), "Times New Roman")


def add_table(doc, block, template, caption_sample):
    insert_paragraph(doc, caption_sample, block["caption"])
    xml = deepcopy(template)
    doc._element.body.insert_element_before(xml, "w:sectPr")
    table = Table(xml, doc._body)
    assert len(table.columns) == len(block["headers"])
    row_sample = deepcopy(table.rows[1]._tr)
    for row in list(table.rows)[1:]:
        table._tbl.remove(row._tr)
    for cell, text in zip(table.rows[0].cells, block["headers"]):
        set_cell_text(cell, text, True)
    for values in block["rows"]:
        assert len(values) == len(block["headers"])
        table._tbl.append(deepcopy(row_sample))
        for cell, text in zip(table.rows[-1].cells, values):
            set_cell_text(cell, text, False)
    for row in table.rows:
        properties = row._tr.get_or_add_trPr()
        if properties.find(qn("w:cantSplit")) is None:
            properties.append(OxmlElement("w:cantSplit"))
    return table


def prepare_document():
    doc = Document(REFERENCE)
    paragraphs = doc.paragraphs
    find = lambda text: next(p for p in paragraphs if p.text == text)
    samples = {
        "heading1": find("1 Описание исходных данных"),
        "heading2": find("2.1 Базовые статистики цены"),
        "body": next(p for p in paragraphs if p.text.startswith("Использован набор 1985")),
        "equation": next(p for p in paragraphs if p.text.startswith("k = ceil")),
        "table_caption": find("Таблица 1. Проверка исходных данных"),
        "figure_caption": next(p for p in paragraphs if p.text.startswith("Рис. 1 -")),
    }
    # python-docx может создавать новые обёртки; положение определяем по XML.
    heading_index = next(i for i, p in enumerate(paragraphs) if p._p is samples["heading1"]._p)
    samples["heading_blank"] = paragraphs[heading_index + 1]
    figure_index = next(i for i, p in enumerate(paragraphs) if p._p is samples["figure_caption"]._p)
    samples["figure_blank"] = paragraphs[figure_index + 1]
    samples["image"] = paragraphs[figure_index - 1]
    samples["table_blank"] = Paragraph(doc.tables[1]._tbl.getnext(), doc._body)
    tables = {key: deepcopy(doc.tables[index]._tbl) for key, index in TABLE_INDICES.items()}
    # Абзацы сохраняем в отдельном дереве до удаления старого текста.
    samples = {key: Paragraph(deepcopy(p._p), doc._body) for key, p in samples.items()}
    body = doc._element.body
    split = next(child for child in body if child.tag == qn("w:p")
                 and child.find(".//" + qn("w:sectPr")) is not None)
    after_title = False
    for child in list(body):
        if after_title and child.tag != qn("w:sectPr"):
            body.remove(child)
        if child is split:
            after_title = True
    used = set(doc._element.xpath("//@r:embed"))
    for relation_id, relation in list(doc.part.rels.items()):
        if relation.reltype.endswith("/image") and relation_id not in used:
            doc.part.drop_rel(relation_id)
    title = next(p for p in doc.paragraphs if "«Дисперсионный анализ»" in p.text)
    # Дополнение помещается на титульный лист без изменения данных исполнителей.
    title.clear()
    for text, size, bold in [("«Дисперсионный анализ»\n", 14, True),
                             ("Дополнительное задание\n", 12, True),
                             ("по дисциплине «Методы анализа данных»", 12, False)]:
        run = title.add_run(text)
        run.font.name, run.font.size, run.bold = "Times New Roman", Pt(size), bold
    following = title._p.getnext()
    if following is not None and following.tag == qn("w:p") and not Paragraph(following, doc._body).text:
        body.remove(following)
    doc.core_properties.title = "ЛР 1.3. Дисперсионный анализ. Дополнительное задание"
    doc.core_properties.subject = "ANOVA Уэлча и Брауна–Форсайта"
    return doc, samples, tables


def write_word(blocks):
    doc, samples, tables = prepare_document()
    previous = None
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            if block["level"] == 2 and previous not in ["heading", "figure"]:
                insert_paragraph(doc, samples["figure_blank"])
            p = insert_paragraph(doc, samples[f"heading{block['level']}"], block["text"])
            if block["level"] == 1 and previous is not None:
                p.paragraph_format.page_break_before = True
            insert_paragraph(doc, samples["heading_blank"])
        elif kind == "paragraph":
            insert_paragraph(doc, samples["body"], block["text"])
        elif kind == "equation":
            p = insert_paragraph(doc, samples["equation"], block["text"])
            p.paragraph_format.keep_together = True
            for run in p.runs:
                run.font.name, run.font.size, run.italic = "Times New Roman", Pt(12), True
        elif kind == "table":
            add_table(doc, block, tables[block["key"]], samples["table_caption"])
            insert_paragraph(doc, samples["table_blank"])
        elif kind == "figure":
            p = insert_paragraph(doc, samples["image"])
            picture = p.add_run().add_picture(str(RESULTS / block["file"]), width=Cm(16.3))
            picture._inline.docPr.set("descr", block["caption"])
            insert_paragraph(doc, samples["figure_caption"], block["caption"])
            insert_paragraph(doc, samples["figure_blank"])
        previous = kind
    doc.save(OUTPUT)
    return OUTPUT


def write_markdown(blocks):
    lines = ["# ЛР 1.3. Дисперсионный анализ – дополнительное задание", ""]
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            lines += ["#" * (block["level"] + 1) + " " + block["text"], ""]
        elif kind in ["paragraph", "equation"]:
            lines += [block["text"], ""]
        elif kind == "table":
            lines += [block["caption"], "", "| " + " | ".join(block["headers"]) + " |",
                      "| " + " | ".join(["---"] * len(block["headers"])) + " |"]
            lines += ["| " + " | ".join(map(str, row)) + " |" for row in block["rows"]] + [""]
        elif kind == "figure":
            lines += [f"![{block['caption']}](results/{block['file']})", ""]
    path = BASE / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    blocks = blocks_for_report(RESULTS)
    print(write_markdown(blocks))
    print(write_word(blocks))
