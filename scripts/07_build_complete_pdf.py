"""Build the 'Complete Project PDF' required for submission: table of contents, the final report,
then every code and documentation file as real text (no screenshots, no outputs, no data files)."""
import io, json, textwrap, sys
from pathlib import Path
import matplotlib
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "Project Submission"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "Bounds Complete Project PDF.pdf"
REPORT = ROOT / "docs" / "Final_Project_Overview.pdf"

fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Mono", str(fdir / "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("Sans", str(fdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("SansB", str(fdir / "DejaVuSans-Bold.ttf")))

GROUPS = [
    ("Documentation", ["README.md", "requirements.txt", "app/requirements.txt", "docs/AI_USAGE_LOG.md"]),
    ("Source code (src/)", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").glob("*.py"))),
    ("Pipeline scripts (scripts/)", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.py"))),
    ("Streamlit application (app/)", ["app/Home.py", "app/content.py", "app/utils.py"] +
        sorted(str(p.relative_to(ROOT)) for p in (ROOT / "app" / "pages").glob("*.py"))),
    ("Tests (tests/)", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*.py"))),
    ("EDA notebook (code and text cells only, no outputs)", ["notebooks/01_eda.ipynb"]),
]

def file_text(rel):
    p = ROOT / rel
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text(encoding="utf-8")); parts = []
        for i, c in enumerate(nb["cells"], 1):
            src = "".join(c["source"])
            tag = "# [code cell %d]" % i if c["cell_type"] == "code" else "# [markdown cell %d]" % i
            if c["cell_type"] == "markdown": src = "\n".join("# " + l for l in src.splitlines())
            parts.append(tag + "\n" + src + "\n")
        return "\n".join(parts)
    return p.read_text(encoding="utf-8", errors="replace")

FS, LEAD, W = 7.4, 9.0, 112
def code_pdf(rel):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=letter)
    pw, ph = letter; lm, tm, bm = 0.7 * inch, 0.75 * inch, 0.7 * inch
    y = ph - tm
    def header():
        nonlocal y
        c.setFont("SansB", 10); c.setFillColor(colors.HexColor("#1f5f3f")); c.drawString(lm, ph - 0.5 * inch, rel)
        c.setFillColor(colors.black); c.setStrokeColor(colors.grey); c.line(lm, ph - 0.56 * inch, pw - lm, ph - 0.56 * inch)
        y = ph - tm
    header(); c.setFont("Mono", FS)
    for n, raw in enumerate(file_text(rel).expandtabs(4).splitlines(), 1):
        chunks = textwrap.wrap(raw, W - 6, replace_whitespace=False, drop_whitespace=False,
                               subsequent_indent="    ") or [""]
        for k, ch in enumerate(chunks):
            if y < bm:
                c.showPage(); header(); c.setFont("Mono", FS)
            c.setFillColor(colors.grey); c.drawString(lm, y, f"{n:>4}" if k == 0 else "    ")
            c.setFillColor(colors.black); c.drawString(lm + 0.35 * inch, y, ch); y -= LEAD
    c.save(); buf.seek(0); return PdfReader(buf)

def toc_pdf(entries, total_pages):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=letter); pw, ph = letter
    c.setFont("SansB", 17); c.drawString(0.8 * inch, ph - 0.9 * inch, "Complete Project PDF: Table of Contents")
    c.setFont("Sans", 9.5)
    c.setFont("Sans", 8.5); c.drawString(0.8 * inch, ph - 1.15 * inch, "Julie Bounds | DTSC 691 | Building Energy Benchmarking Anomaly Detection (NYC LL84) | Mentor: Jasmeet Bhatia")
    y = ph - 1.55 * inch
    for kind, label, page in entries:
        if kind == "group":
            y -= 5; c.setFont("SansB", 10)
        else:
            c.setFont("Sans", 9)
        x = 0.8 * inch if kind == "group" else 1.1 * inch
        c.drawString(x, y, label)
        if page:
            c.drawRightString(pw - 0.8 * inch, y, str(page))
            if kind != "group":
                tw = pdfmetrics.stringWidth(label, "Sans", 9) + x + 6; c.setFillColor(colors.lightgrey)
                dots = int((pw - 0.8 * inch - 24 - tw) / pdfmetrics.stringWidth(".", "Sans", 9))
                c.setFont("Sans", 9); c.drawString(tw, y, "." * max(dots, 0)); c.setFillColor(colors.black)
        y -= 12.6
    c.setFont("Sans", 8); c.setFillColor(colors.grey)
    c.drawString(0.8 * inch, 0.6 * inch, "Code is included as selectable text. Outputs and data files are intentionally excluded (see All Project Files on Google Drive).")
    c.save(); buf.seek(0); return PdfReader(buf)

report = PdfReader(str(REPORT)); n_report = len(report.pages)
code = []   # (group, rel, reader)
for g, files in GROUPS:
    for rel in files:
        code.append((g, rel, code_pdf(rel)))

page = 2; entries = [("group", "Project Overview (final report)", None), ("item", "Final Project Overview: problem, data, methods, results, audit, app, limitations", page)]
page += n_report; starts = []
last = None
for g, rel, r in code:
    if g != last: entries.append(("group", "Appendix: " + g, None)); last = g
    entries.append(("item", rel, page)); starts.append(page); page += len(r.pages)
total = page - 1

w = PdfWriter()
w.append(toc_pdf(entries, total)); assert len(w.pages) == 1, "TOC must fit on one page"
w.append(report)
for g, rel, r in code: w.append(r)
w.add_outline_item("Table of contents", 0); w.add_outline_item("Project Overview", 1)
last = None; parent = None
for (g, rel, r), s in zip(code, starts):
    if g != last: parent = w.add_outline_item("Appendix: " + g, s - 1); last = g
    w.add_outline_item(rel, s - 1, parent=parent)
w.add_metadata({"/Title": "Bounds Complete Project PDF", "/Author": "Julie Bounds"})
with open(OUT, "wb") as f: w.write(f)
print("wrote", OUT, "pages:", total)
