from pathlib import Path
import re
import html

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from svglib.svglib import svg2rlg

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "Project_Report.md"
OUTPUT = ROOT / "Project_Report.pdf"
PLOT_DIR = ROOT / "outputs" / "visualizations"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle2", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=23, leading=29, textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=16))
styles.add(ParagraphStyle(name="Center2", parent=styles["Normal"], fontSize=10, leading=15,
    textColor=colors.HexColor("#526477"), alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=16, leading=20,
    textColor=colors.HexColor("#17324D"), spaceBefore=5, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11.5, leading=15,
    textColor=colors.HexColor("#22577A"), spaceBefore=6, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5, leading=12.5, spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.3, leading=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Captionx", parent=styles["Normal"], fontSize=7.5, leading=10,
    textColor=colors.HexColor("#667085"), alignment=TA_CENTER, spaceAfter=7))

def fmt_inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\x60(.+?)\x60", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\[(.+?)\]\((https?://.+?)\)", r'<link href="\2" color="#2673A6">\1</link>', text)
    return text

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#65748B"))
    canvas.drawString(20*mm, 12*mm, "Student Performance Analysis and Prediction System")
    canvas.drawRightString(A4[0]-20*mm, 12*mm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#D5DDE5"))
    canvas.line(20*mm, 16*mm, A4[0]-20*mm, 16*mm)
    canvas.restoreState()

def scale_svg(path, max_w=160*mm, max_h=91*mm):
    drawing = svg2rlg(str(path))
    if drawing is None or not drawing.width or not drawing.height:
        return None
    factor = min(max_w / drawing.width, max_h / drawing.height)
    drawing.width *= factor
    drawing.height *= factor
    drawing.scale(factor, factor)
    return drawing

lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
title = next((line.lstrip("# ").strip() for line in lines if line.startswith("# ")),
             "Student Performance Analysis and Prediction System")
story = [
    Spacer(1, 30*mm),
    Paragraph("MINOR PROJECT REPORT", styles["Center2"]),
    Spacer(1, 10*mm),
    Paragraph(fmt_inline(title), styles["CoverTitle2"]),
    Paragraph("Python | Exploratory Data Analysis | Linear Regression", styles["Center2"]),
    Spacer(1, 16*mm),
    Paragraph("<b>Submitted by:</b> Shivam A Pandya", styles["Center2"]),
    Paragraph("<b>Program:</b> B.Tech - AI Engineering", styles["Center2"]),
    Paragraph("<b>Institution:</b> SRM Ramapuram", styles["Center2"]),
    Spacer(1, 18*mm),
    Paragraph("AI/ML with Python Training Program", styles["Center2"]),
    Paragraph("Dataset: UCI Student Performance (Portuguese course) | DOI: 10.24432/C5TG7T | CC BY 4.0", styles["Smallx"]),
    PageBreak(),
]

i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line or line.startswith("# ") or line.startswith("**Minor Project") or line.startswith("**Student:**") or line.startswith("**Program:**"):
        i += 1
        continue
    if line.startswith("## "):
        story.append(Paragraph(fmt_inline(line[3:].strip()), styles["H1x"]))
        i += 1
        continue
    if line.startswith("|"):
        table_lines = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c or "-") for c in cells):
                table_lines.append(cells)
            i += 1
        if table_lines:
            width = max(len(row) for row in table_lines)
            table_data = [[Paragraph(fmt_inline(c), styles["Smallx"]) for c in row + [""]*(width-len(row))] for row in table_lines]
            col_width = (A4[0] - 40*mm) / width
            tbl = Table(table_data, colWidths=[col_width]*width, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C9D3DD")),
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF1F6")),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ]))
            story += [tbl, Spacer(1, 5)]
        continue
    if line.startswith("- "):
        while i < len(lines) and lines[i].strip().startswith("- "):
            item = lines[i].strip()[2:].strip()
            chart = re.search(r"\x60([^ \x60]+\.svg)\x60", item)
            if chart:
                svg_path = PLOT_DIR / chart.group(1)
                if svg_path.exists():
                    drawing = scale_svg(svg_path)
                    if drawing:
                        story += [Spacer(1, 4), drawing, Paragraph(fmt_inline(chart.group(1)), styles["Captionx"])]
            else:
                story.append(Paragraph("&#8226; " + fmt_inline(item), styles["Bodyx"]))
            i += 1
        continue
    paragraph_lines = [line]
    i += 1
    while i < len(lines):
        nxt = lines[i].strip()
        if not nxt or nxt.startswith("#") or nxt.startswith("|") or nxt.startswith("- "):
            break
        paragraph_lines.append(nxt)
        i += 1
    story.append(Paragraph(fmt_inline(" ".join(paragraph_lines)), styles["Bodyx"]))

doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
    topMargin=18*mm, bottomMargin=22*mm, title=title, author="Shivam A Pandya")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Created {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
