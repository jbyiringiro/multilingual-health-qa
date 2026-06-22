"""Convert report/REPORT.md -> a self-contained, print-ready HTML.

Images are inlined as base64 so the single HTML file is fully portable. Open it
in a browser and use Print -> Save as PDF (A4) to produce the submission PDF.

Run:  python scripts/build_report_html.py
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

REPORT = Path("report/REPORT.md")
OUT = Path("report/REPORT.html")

CSS = """
@page { size: A4 portrait; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13pt; line-height: 1.5;
       color: #000; max-width: 820px; margin: 0 auto; padding: 24px; }
h1, h2, h3, h4, p, li, td, th, blockquote, a, code { color: #000; }
h1 { font-size: 22pt; border-bottom: 2px solid #000; padding-bottom: 6px; }
h2 { font-size: 16pt; margin-top: 22px; border-bottom: 1px solid #999; }
h3 { font-size: 14pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12pt; }
th, td { border: 1px solid #888; padding: 5px 8px; text-align: left; }
th { background: #f0f0f0; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 11.5pt; }
img { max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ccc; }
blockquote { border-left: 4px solid #888; background: #f7f7f7; padding: 8px 14px; margin: 12px 0; }
a { color: #000; text-decoration: underline; }
"""


def inline_images(html: str) -> str:
    def repl(m):
        alt, src = m.group(1), m.group(2)
        p = (REPORT.parent / src).resolve()
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        ext = p.suffix.lstrip(".") or "png"
        return f'<img alt="{alt}" src="data:image/{ext};base64,{b64}"/>'
    return re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', repl, html)


def main():
    text = REPORT.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = inline_images(body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Final Project Report</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB, images inlined).")
    print("Open it in a browser -> Print -> Save as PDF (A4) -> StudentName_FinalProject.pdf")


if __name__ == "__main__":
    main()
