"""Собрать согласованные отчёты Markdown и Word из результатов ЛР 1.3.

Титульный лист и формат таблиц сохраняются из шаблона текущего отчёта.
Статистические расчёты выполняет anova_analysis.py.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table

from report_content import blocks_for_report

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "report_template.docx"
OUTPUT_NAME = "Отчёт_ЛР1.3_Дисперсионный_анализ.docx"
FONT = "Times New Roman"
TEMPLATE_TABLE_KEYS = ["quality", "descriptive", "frequencies", "histograms", "normality",
                       "one_way", "ranks", "kruskal", "cells", "two_way"]


def set_font(run, size=14, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    fonts = run._element.get_or_add_rPr().rFonts
    for name in ["ascii", "hAnsi", "eastAsia", "cs"]:
        fonts.set(qn(f"w:{name}"), FONT)


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = FONT, Pt(14)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = Cm(1.25)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = pf.space_after = Pt(0)
    for name in ["Heading 1", "Heading 2"]:
        style = doc.styles[name]
        style.font.name, style.font.size, style.font.bold = FONT, Pt(14), True
        style.font.color.rgb = RGBColor(0, 0, 0)
        pf = style.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = pf.left_indent = Cm(0)
        # Одна пустая строка перед заголовком. Word подавляет этот интервал
        # вверху страницы; после заголовка добавляется пустой абзац.
        pf.space_before, pf.space_after = Pt(21), Pt(0)
        pf.line_spacing, pf.keep_with_next = 1.0, True
    settings = doc.settings._element
    compat = settings.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        settings.append(compat)
    suppress = compat.find(qn("w:suppressSpBfAfterPgBrk"))
    if suppress is None:
        suppress = OxmlElement("w:suppressSpBfAfterPgBrk")
        compat.append(suppress)
    suppress.set(qn("w:val"), "true")


def prepare_title(reference):
    doc = Document(reference)
    if len(doc.tables) != len(TEMPLATE_TABLE_KEYS) + 1:
        raise RuntimeError("В шаблоне должны находиться титульная таблица и 10 образцов таблиц")
    templates = {key: deepcopy(table._tbl)
                 for key, table in zip(TEMPLATE_TABLE_KEYS, doc.tables[1:])}
    body = doc._element.body
    split = next((child for child in body if child.tag == qn("w:p")
                  and child.find(".//" + qn("w:sectPr")) is not None), None)
    if split is None:
        raise RuntimeError("В отчёте-образце не найден разрыв после титульного листа")
    after = False
    for child in list(body):
        if after and child.tag != qn("w:sectPr"):
            body.remove(child)
        if child is split:
            after = True
    # После удаления основного текста убираем связи с рисунками прошлой работы.
    # Иначе Word сохраняет их внутри архива, хотя в новом отчёте они не видны.
    used_images = set(doc._element.xpath("//@r:embed"))
    for relation_id, relation in list(doc.part.rels.items()):
        if relation.reltype.endswith("/image") and relation_id not in used_images:
            doc.part.drop_rel(relation_id)
    for paragraph in doc.paragraphs:
        if ("ЛАБОРАТОРНАЯ РАБОТА №" in paragraph.text
                and paragraph.text.strip() != "ЛАБОРАТОРНАЯ РАБОТА №1.3"):
            paragraph.clear()
            set_font(paragraph.add_run("ЛАБОРАТОРНАЯ РАБОТА №1.3"), bold=True)
        elif "Регрессионный анализ" in paragraph.text:
            paragraph.clear()
            first = paragraph.add_run("«Дисперсионный анализ»")
            set_font(first, bold=True)
            first.add_break()
            set_font(paragraph.add_run("по дисциплине «Методы анализа данных»"), size=12)
    configure_styles(doc)
    section = doc.sections[-1]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(1.5)
    section.top_margin = section.bottom_margin = Cm(2)
    doc.core_properties.title = "Лабораторная работа 1.3. Дисперсионный анализ"
    doc.core_properties.subject = "Цена автомобилей Automobile по приводу и наддуву"
    return doc, templates


def paragraph(doc, text, *, caption=False, centered=False, keep_next=False):
    p = doc.add_paragraph()
    set_font(p.add_run(text))
    if caption or centered:
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(4)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if centered else WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.keep_with_next = keep_next
    return p


def blank_line(doc, *, keep_next=False):
    p = doc.add_paragraph()
    set_font(p.add_run(""))
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = pf.space_after = Pt(0)
    pf.line_spacing = Pt(21)
    pf.keep_with_next = keep_next
    return p


def add_field_run(p, field_type=None, instruction=None, text=None):
    run = p.add_run()
    set_font(run, size=12)
    if field_type:
        node = OxmlElement("w:fldChar")
        node.set(qn("w:fldCharType"), field_type)
        if field_type == "begin":
            node.set(qn("w:dirty"), "true")
    elif instruction:
        node = OxmlElement("w:instrText")
        node.set(qn("xml:space"), "preserve")
        node.text = instruction
    else:
        run.text = text or ""
        return run
    run._r.append(node)
    return run


def add_toc(doc, blocks):
    title = doc.add_paragraph("Оглавление", style="TOC Heading")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = title.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = pf.space_after = Pt(0)
    pf.keep_with_next = True
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "9")
    title._p.get_or_add_pPr().append(outline)
    for run in title.runs:
        set_font(run, bold=True)
    blank_line(doc, keep_next=True)
    bookmark_ids = [int(value) for value in doc._element.xpath("//w:bookmarkStart/@w:id")]
    next_id = max(bookmark_ids, default=0) + 1
    headings = [block for block in blocks if block["kind"] == "heading"]
    bookmarks = []
    for i, block in enumerate(headings):
        name = f"_Lab13Heading{i + 1:03d}"
        bookmarks.append((name, next_id + i))
        p = doc.add_paragraph(style=f"toc {block['level']}")
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.left_indent = Cm(0 if block["level"] == 1 else 0.6)
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.keep_together = True
        pf.tab_stops.add_tab_stop(Cm(16.3), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if i == 0:
            add_field_run(p, field_type="begin")
            add_field_run(p, instruction=' TOC \\o "1-2" \\h \\z \\u ')
            add_field_run(p, field_type="separate")
        # Сохраняем читаемые пункты оглавления ещё до обновления полей Word.
        link = OxmlElement("w:hyperlink")
        link.set(qn("w:anchor"), name)
        link.set(qn("w:history"), "1")
        run = add_field_run(p, text=block["text"])
        link.append(run._r)
        p._p.append(link)
        add_field_run(p, text="\t")
        add_field_run(p, field_type="begin")
        add_field_run(p, instruction=f" PAGEREF {name} \\h ")
        add_field_run(p, field_type="separate")
        add_field_run(p, text=" ")
        add_field_run(p, field_type="end")
        if i == len(headings) - 1:
            add_field_run(p, field_type="end")
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")
    doc.add_page_break()
    return iter(bookmarks)


def set_cell_text(cell, text, *, bold):
    p = cell.paragraphs[0]
    for other in list(cell.paragraphs[1:]):
        cell._tc.remove(other._p)
    p.clear()
    set_font(p.add_run(str(text)), size=12, bold=bold)


def add_word_table(doc, block, template):
    paragraph(doc, block["caption"], caption=True, keep_next=True)
    xml = deepcopy(template)
    doc._element.body.insert_element_before(xml, "w:sectPr")
    table = Table(xml, doc._body)
    row_template = deepcopy(table.rows[1]._tr)
    for row in list(table.rows)[1:]:
        table._tbl.remove(row._tr)
    for cell, header in zip(table.rows[0].cells, block["headers"]):
        set_cell_text(cell, header, bold=True)
    properties = table.rows[0]._tr.get_or_add_trPr()
    if properties.find(qn("w:tblHeader")) is None:
        properties.append(OxmlElement("w:tblHeader"))
    for values in block["rows"]:
        table._tbl.append(deepcopy(row_template))
        cells = table.rows[-1].cells
        for cell, value in zip(cells, values):
            set_cell_text(cell, value, bold=False)
    for row_index, row in enumerate(table.rows):
        row_properties = row._tr.get_or_add_trPr()
        if row_properties.find(qn("w:cantSplit")) is None:
            row_properties.append(OxmlElement("w:cantSplit"))
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    set_font(run, size=12, bold=row_index == 0)
    return table


def write_word(blocks, results, output, reference):
    doc, templates = prepare_title(reference)
    bookmarks = add_toc(doc, blocks)
    previous_kind = None
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            p = doc.add_paragraph(block["text"], style=f"Heading {block['level']}")
            if previous_kind in [None, "heading", "figure"] or p.paragraph_format.page_break_before:
                p.paragraph_format.space_before = Pt(0)
            for run in p.runs:
                set_font(run, bold=True)
            name, bookmark_id = next(bookmarks)
            start, end = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
            start.set(qn("w:id"), str(bookmark_id))
            start.set(qn("w:name"), name)
            end.set(qn("w:id"), str(bookmark_id))
            p._p.insert(1, start)
            p._p.append(end)
            blank_line(doc, keep_next=True)
        elif kind == "paragraph":
            paragraph(doc, block["text"])
        elif kind == "equation":
            p = paragraph(doc, block["text"], centered=True)
            p.paragraph_format.keep_together = True
            for run in p.runs:
                set_font(run, size=12, italic=True)
        elif kind == "table":
            add_word_table(doc, block, templates[block["key"]])
        elif kind == "figure":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.line_spacing = 1.0
            picture = p.add_run().add_picture(str(results / block["file"]), width=Cm(16.3))
            picture._inline.docPr.set("descr", block["caption"])
            paragraph(doc, block["caption"], caption=True, centered=True, keep_next=True)
            blank_line(doc)
        previous_kind = kind
    path = output / OUTPUT_NAME
    doc.save(path)
    return path


def write_markdown(blocks, results, output):
    lines = ["# Лабораторная работа 1.3. Дисперсионный анализ", "", "## Оглавление", ""]
    headings = [block for block in blocks if block["kind"] == "heading"]
    for i, block in enumerate(headings, 1):
        lines.append("    " * (block["level"] - 1) + f"- [{block['text']}](#section-{i:02d})")
    lines.append("")
    heading_number = 0
    for b in blocks:
        if b["kind"] == "heading":
            heading_number += 1
            lines += [f'<a id="section-{heading_number:02d}"></a>', "",
                      "#" * (b["level"] + 1) + " " + b["text"], ""]
        elif b["kind"] == "paragraph":
            lines += [b["text"], ""]
        elif b["kind"] == "equation":
            lines += [f"`{b['text']}`", ""]
        elif b["kind"] == "table":
            lines += [b["caption"], "", "| " + " | ".join(b["headers"]) + " |",
                      "| " + " | ".join(["---"] * len(b["headers"])) + " |"]
            lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in b["rows"]]
            lines.append("")
        elif b["kind"] == "figure":
            # При обычном запуске results находится рядом с Markdown-отчётом.
            import os
            relative = Path(os.path.relpath(results / b["file"], output)).as_posix()
            lines += [f"![{b['caption']}]({relative})", ""]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results")
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = blocks_for_report(args.results)
    print(write_markdown(blocks, args.results, args.output))
    print(write_word(blocks, args.results, args.output, args.reference))


if __name__ == "__main__":
    main()
