"""Shared labels, palette, and font setup for report renderers."""
from __future__ import annotations

import base64
import os
from html import escape

import matplotlib

COLOR_NAVY = "#1C3155"
COLOR_NAVY_LIGHT = "#314B70"
COLOR_TEAL = "#008F87"
COLOR_ORANGE = "#E87524"
COLOR_GRAY_TEXT = "#29384D"
COLOR_GRAY_MUTED = "#74849B"


def friendly_label(value: str) -> str:
    text = str(value).replace("_", " ").strip()
    replacements = {
        "sample size": "Cỡ mẫu",
        "count": "Số lượng",
        "period": "Kỳ",
        "mean ": "Trung bình ",
        "sum ": "Tổng ",
    }
    lowered = text.casefold()
    for source, target in replacements.items():
        if lowered == source or lowered.startswith(source):
            text = target + text[len(source):]
            break
    return text[:1].upper() + text[1:]


def display_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def evidence_table_html(table) -> str:
    headers = "".join(f"<th>{escape(friendly_label(column))}</th>" for column in table.columns)
    rows = []
    for row in table.rows:
        cells = "".join(
            f"<td>{escape(display_value(row.get(column)))}</td>" for column in table.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        '\n<div class="evidence-table-block">\n'
        f'<p class="table-title">{escape(table.title)}</p>\n'
        f'<table class="evidence-table"><thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>\n</div>\n'
    )


_MPL_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT_REGULAR_PATH = os.path.join(_MPL_FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD_PATH = os.path.join(_MPL_FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_ITALIC_PATH = os.path.join(_MPL_FONT_DIR, "DejaVuSans-Oblique.ttf")


def _file_to_base64(path: str) -> str | None:
    try:
        with open(path, "rb") as source:
            return base64.b64encode(source.read()).decode("ascii")
    except OSError:
        return None


def build_font_face_css() -> str:
    """Embed fonts so Vietnamese text renders consistently on every host."""
    regular_b64 = _file_to_base64(FONT_REGULAR_PATH)
    bold_b64 = _file_to_base64(FONT_BOLD_PATH)
    italic_b64 = _file_to_base64(FONT_ITALIC_PATH)
    if not regular_b64:
        return ""
    faces = [f"""
        @font-face {{
            font-family: 'ReportSans';
            src: url(data:font/ttf;base64,{regular_b64}) format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
    """]
    if bold_b64:
        faces.append(f"""
        @font-face {{
            font-family: 'ReportSans';
            src: url(data:font/ttf;base64,{bold_b64}) format('truetype');
            font-weight: bold;
            font-style: normal;
        }}
        """)
    if italic_b64:
        faces.append(f"""
        @font-face {{
            font-family: 'ReportSans';
            src: url(data:font/ttf;base64,{italic_b64}) format('truetype');
            font-weight: normal;
            font-style: italic;
        }}
        """)
    return "\n".join(faces)


def register_reportlab_fonts() -> tuple[str, str]:
    """Register bundled Vietnamese-capable fonts and return regular/bold names."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_name = "Helvetica"
    bold_name = "Helvetica-Bold"
    if not os.path.exists(FONT_REGULAR_PATH):
        return regular_name, bold_name
    pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_REGULAR_PATH))
    regular_name = "DejaVuSans"
    bold_name = regular_name
    if os.path.exists(FONT_BOLD_PATH):
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", FONT_BOLD_PATH))
        bold_name = "DejaVuSans-Bold"
    pdfmetrics.registerFontFamily(
        "DejaVuSans",
        normal=regular_name,
        bold=bold_name,
        italic=regular_name,
        boldItalic=bold_name,
    )
    return regular_name, bold_name
