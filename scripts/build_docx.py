"""把 build/report.md 直接构建为排好版的 .docx（不依赖 HTML 转换）。

支持的报告语法（自定义精简 Markdown）：
    # 标题            -> 文档标题
    > 副标题          -> 居中副标题
    @段落             -> 无首行缩进的段落（用于带粗体小标题的正文行）
    ## 小节           -> 二级标题
    ### 小节          -> 三级标题
    $$公式$$          -> 居中公式行
    %% 表 N 说明      -> 表题
    | a | b |         -> 表格（第二行为分隔行）
    [[FIG:name|caption|width]] -> 插图 + 图注
    - 项目            -> 无序列表
    [1] 文献          -> 参考文献条目
正文中 **粗体**、`代码`、$斜体$ 会被解析。
"""
import os
import re
import sys
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
FIG = os.path.join(ROOT, "figures")

CN = "宋体"
EN = "Times New Roman"
HEI = "黑体"

BODY = 10.5
LINE = 1.16
INDENT = 0.74


def style_run(run, size=BODY, bold=False, italic=False, cn=CN, en=EN):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def add_rich(p, text, size=BODY, base_bold=False):
    """解析 **粗体**、`代码`、$斜体$。"""
    tokens = re.split(r"(\*\*.+?\*\*|`.+?`|\$.+?\$)", text)
    for tk in tokens:
        if not tk:
            continue
        if tk.startswith("**") and tk.endswith("**"):
            style_run(p.add_run(tk[2:-2]), size, bold=True)
        elif tk.startswith("`") and tk.endswith("`"):
            r = p.add_run(tk[1:-1])
            style_run(r, size - 0.5, bold=base_bold, en="Consolas")
        elif tk.startswith("$") and tk.endswith("$") and len(tk) > 2:
            style_run(p.add_run(tk[1:-1]), size, bold=base_bold, italic=True, en="Cambria Math")
        else:
            style_run(p.add_run(tk), size, bold=base_bold)


def fmt(p, size=BODY, first=INDENT, before=0, after=3, line=LINE,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, keep_next=False):
    pf = p.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Cm(first)
    pf.left_indent = Cm(0)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.keep_with_next = keep_next


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _disp_width(txt):
    w = 0
    for ch in re.sub(r"\*\*|`", "", txt):
        w += 2 if ord(ch) > 127 else 1
    return w


def add_table(doc, rows):
    header, body = rows[0], rows[2:] if len(rows) > 2 else []
    t = doc.add_table(rows=1 + len(body), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        cell = t.cell(0, j)
        cell.text = ""
        p = cell.paragraphs[0]
        fmt(p, size=8.5, first=0, before=1, after=1, line=1.0,
            align=WD_ALIGN_PARAGRAPH.CENTER)
        add_rich(p, h, size=8.5, base_bold=True)
        for r in p.runs:
            style_run(r, 8.5, bold=True, cn=HEI, en=HEI)
        shade(cell, "E8EDF4")
    for i, row in enumerate(body, start=1):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            fmt(p, size=8.5, first=0, before=1, after=1, line=1.0,
                align=WD_ALIGN_PARAGRAPH.CENTER)
            add_rich(p, val, size=8.5)
    # 列宽按内容长度分配，总宽 15.6 cm
    all_rows = [header] + body
    weights = [max(_disp_width(r[j]) for r in all_rows) for j in range(len(header))]
    total_w = 15.6
    floor = 1.9
    wsum = sum(weights)
    widths = [max(floor, total_w * w / wsum) for w in weights]
    scale = total_w / sum(widths)
    widths = [w * scale for w in widths]
    for j, w in enumerate(widths):
        for r in t.rows:
            r.cells[j].width = Cm(w)

    # 防止表格跨页：除最后一行外都设置 keep_with_next
    for row in t.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    return t


def build(md_path, out_path):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.9)
    sec.top_margin = sec.bottom_margin = Cm(1.9)

    st = doc.styles["Normal"]
    st.font.name = EN
    st.font.size = Pt(BODY)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    st.paragraph_format.line_spacing = LINE
    st.paragraph_format.space_after = Pt(3)

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        m = re.match(r"\[\[FIG:(.+?)\|(.+?)(?:\|(\d+)%?)?\]\]", s)
        if m:
            name, cap, w = m.group(1), m.group(2), m.group(3)
            width = Cm(15.0) if not w else Cm(15.0 * int(w) / 100)
            p = doc.add_paragraph()
            fmt(p, first=0, before=5, after=0, line=1.0,
                align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
            p.add_run().add_picture(os.path.join(FIG, name + ".png"), width=width)
            pc = doc.add_paragraph()
            fmt(pc, size=8.5, first=0, before=2, after=7, line=1.05,
                align=WD_ALIGN_PARAGRAPH.CENTER)
            add_rich(pc, cap, size=8.5)
            for r in pc.runs:
                style_run(r, 8.5, cn=HEI, en=HEI)
            i += 1
            continue

        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            p = doc.add_paragraph()
            fmt(p, size=10, first=0, before=1, after=3, line=1.15,
                align=WD_ALIGN_PARAGRAPH.CENTER)
            add_rich(p, s[2:-2], size=10)
            i += 1
            continue

        if s.startswith("%% "):
            p = doc.add_paragraph()
            fmt(p, size=8.5, first=0, before=7, after=1, line=1.05,
                align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
            add_rich(p, s[3:], size=8.5, base_bold=True)
            for r in p.runs:
                style_run(r, 8.5, bold=True, cn=HEI, en=HEI)
            i += 1
            continue

        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) >= 2:
                add_table(doc, rows)
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        if s.startswith("# "):
            p = doc.add_paragraph()
            fmt(p, size=15, first=0, before=0, after=4, line=1.15,
                align=WD_ALIGN_PARAGRAPH.CENTER)
            add_rich(p, s[2:], size=15, base_bold=True)
            for r in p.runs:
                style_run(r, 15, bold=True, cn=HEI, en=HEI)
            i += 1
            continue
        if s.startswith("### "):
            p = doc.add_paragraph()
            fmt(p, size=10.5, first=0, before=5, after=1, line=1.15, keep_next=True)
            add_rich(p, s[4:], size=10.5, base_bold=True)
            for r in p.runs:
                style_run(r, 10.5, bold=True, cn=HEI, en=HEI)
            i += 1
            continue
        if s.startswith("## "):
            p = doc.add_paragraph()
            fmt(p, size=11.5, first=0, before=8, after=2, line=1.15, keep_next=True)
            add_rich(p, s[3:], size=11.5, base_bold=True)
            for r in p.runs:
                style_run(r, 11.5, bold=True, cn=HEI, en=HEI)
            i += 1
            continue
        if s.startswith("> "):
            p = doc.add_paragraph()
            fmt(p, size=10, first=0, before=1, after=1, line=1.1,
                align=WD_ALIGN_PARAGRAPH.CENTER)
            add_rich(p, s[2:], size=10)
            i += 1
            continue
        if s.startswith("- "):
            p = doc.add_paragraph()
            fmt(p, size=BODY, first=0, before=0, after=2, line=LINE)
            p.paragraph_format.left_indent = Cm(0.95)
            p.paragraph_format.first_line_indent = Cm(-0.35)
            add_rich(p, "· " + s[2:], size=BODY)
            i += 1
            continue
        if re.match(r"^\[\d+\]", s):
            p = doc.add_paragraph()
            fmt(p, size=8.5, first=0, before=0, after=1.5, line=1.1)
            p.paragraph_format.left_indent = Cm(0.65)
            p.paragraph_format.first_line_indent = Cm(-0.65)
            add_rich(p, s, size=8.5)
            i += 1
            continue

        if s.startswith("@"):
            p = doc.add_paragraph()
            fmt(p, size=BODY, first=INDENT, before=0, after=3)
            add_rich(p, s[1:], size=BODY)
            i += 1
            continue

        p = doc.add_paragraph()
        fmt(p, size=BODY, first=INDENT, before=0, after=3)
        add_rich(p, s, size=BODY)
        i += 1

    doc.save(out_path)
    print("built:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, "report.docx")
    build(os.path.join(BUILD, "report.md"), out)
