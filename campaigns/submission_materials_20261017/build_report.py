"""Build the three-page jury report from Markdown and saved monthly metrics.

Optional authoring dependencies: requirements-report.txt. No model execution.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'delivery/report/report.md'
OUTPUT = ROOT / 'delivery/report/report.pdf'
MONTHLY = ROOT / 'results/minimize_expanding_20261017/monthly_scores.csv'
REGISTRY = ROOT / 'best_model.json'
RECEIPT = ROOT / 'results/submission_organization_20261017/report_build.json'
NAVY = colors.HexColor('#13334B')
TEAL = colors.HexColor('#008978')
GREY = colors.HexColor('#536572')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inline(value):
    value = escape(value)
    value = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', value)
    # Plain compact path labels keep references readable in the PDF.
    return re.sub(r'`([^`]+)`', r'<font name="Courier" size="8">\1</font>', value)


def monthly_chart():
    with MONTHLY.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['recipe'] == 'fixed_blend_ratio']
    assert [row['month'] for row in rows] == [f'2026-{m:02}' for m in range(2, 10)]
    registry = json.loads(REGISTRY.read_text())
    ratio = sum(float(row['ratio']) for row in rows) / 8
    assert abs(ratio - registry['mean_monthly_ratio']) < 1e-12
    drawing = Drawing(504, 184)
    left, bottom, width, height = 42, 35, 440, 124
    low, high = 0.5, 1.05
    y = lambda value: bottom + (value - low) / (high - low) * height
    for value in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        drawing.add(Line(left, y(value), left + width, y(value),
                         strokeColor=colors.HexColor('#DEE6EA'), strokeWidth=0.6))
        drawing.add(String(left - 9, y(value) - 3, f'{value:.1f}',
                           fontName='Helvetica', fontSize=8, textAnchor='end', fillColor=GREY))
    drawing.add(Line(left, y(1), left + width, y(1), strokeColor=NAVY,
                     strokeWidth=1, strokeDashArray=[4, 3]))
    drawing.add(String(left, 174, 'Monthly MAE / mean-corrected-interpolation MAE',
                       fontName='Helvetica-Bold', fontSize=9, fillColor=NAVY))
    months = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    for i, (row, month) in enumerate(zip(rows, months)):
        value = float(row['ratio'])
        x = left + width * (i + 0.5) / 8
        drawing.add(Rect(x - 16, y(low), 32, y(value) - y(low), fillColor=TEAL, strokeColor=None))
        drawing.add(String(x, y(value) + 6, f'{value:.3f}', fontSize=8,
                           fontName='Helvetica', textAnchor='middle', fillColor=NAVY))
        drawing.add(String(x, bottom - 15, month, fontSize=8,
                           fontName='Helvetica', textAnchor='middle', fillColor=GREY))
    drawing.add(String(left, 5, 'Developmental folds. Baseline = 1; lower is better. Bar axis starts at 0.5.',
                       fontName='Helvetica', fontSize=8, fillColor=GREY))
    return drawing


def footer(canvas, document):
    canvas.setTitle('Quarter hour electricity prices with conserved hourly means - report draft')
    canvas.setAuthor('Neural Wave AET team - member list pending')
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(2)
    canvas.line(42, A4[1] - 29, A4[0] - 42, A4[1] - 29)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    canvas.drawString(42, 25, 'DRAFT | AET / Neural Wave 2026 | 17 October 2026')
    canvas.drawRightString(A4[0] - 42, 25, f'{document.page} / 3')


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=21,
                              leading=24, textColor=NAVY, spaceAfter=11))
    styles.add(ParagraphStyle('ReportHeading', fontName='Helvetica-Bold', fontSize=12,
                              leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=6,
                              keepWithNext=True))
    styles.add(ParagraphStyle('ReportBody', fontName='Helvetica', fontSize=10,
                              leading=14, textColor=colors.HexColor('#243B48'), spaceAfter=8))
    styles.add(ParagraphStyle('ReportCell', parent=styles['ReportBody'], fontSize=8.5,
                              leading=11, spaceAfter=0))
    styles.add(ParagraphStyle('ReportHeader', parent=styles['ReportCell'],
                              fontName='Helvetica-Bold', textColor=colors.white))
    story = []
    blocks = SOURCE.read_text().strip().split('\n\n')
    for block in blocks:
        if block == '<!-- pagebreak -->':
            story.append(PageBreak())
        elif block == '<!-- monthly_chart -->':
            story.extend([monthly_chart(), Spacer(1, 7)])
        elif block.startswith('# '):
            story.append(Paragraph(inline(block[2:]), styles['ReportTitle']))
        elif block.startswith('## '):
            story.append(Paragraph(inline(block[3:]), styles['ReportHeading']))
        elif block.startswith('|'):
            lines = block.splitlines()
            rows = [[cell.strip() for cell in line.strip('|').split('|')]
                    for i, line in enumerate(lines) if i != 1]
            cells = [[Paragraph(inline(cell), styles['ReportHeader' if i == 0 else 'ReportCell'])
                      for cell in row] for i, row in enumerate(rows)]
            table = Table(cells, colWidths=[237, 130, 144], hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#EEF5F5'), colors.white]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor('#CEDBE0')),
            ]))
            story.extend([KeepTogether(table), Spacer(1, 10)])
        else:
            story.append(Paragraph(inline(block.replace('\n', ' ')), styles['ReportBody']))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=42, leftMargin=42,
                                 topMargin=42, bottomMargin=43)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(OUTPUT).pages)
    assert pages == 3, f'Expected exactly three pages, generated {pages}'
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps({
        'status': 'draft', 'pages': pages, 'visual_review': 'pending',
        'output': str(OUTPUT.relative_to(ROOT)), 'output_sha256': digest(OUTPUT),
        'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': digest(SOURCE),
        'monthly_metrics': str(MONTHLY.relative_to(ROOT)), 'monthly_sha256': digest(MONTHLY),
        'registry_sha256': digest(REGISTRY), 'model_execution': False,
    }, indent=2) + '\n')
    print(f'Created {OUTPUT.relative_to(ROOT)} ({pages} pages; visual review pending)')


if __name__ == '__main__':
    main()
