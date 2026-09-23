import logging
import os
import re
import base64
from html import escape, unescape
from datetime import datetime
import markdown
from graph.state import GraphState
from schemas.messages import ReportFormat
from agents.report_formatting import (
    COLOR_GRAY_MUTED,
    COLOR_GRAY_TEXT,
    COLOR_NAVY,
    COLOR_NAVY_LIGHT,
    COLOR_ORANGE,
    COLOR_TEAL,
    build_font_face_css,
    evidence_table_html,
    register_reportlab_fonts,
)

logger = logging.getLogger(__name__)





_evidence_table_html = evidence_table_html


def _tables_without_visual_coverage(tables, theme_question_ids, matching_visuals):
    """Keep a table only when no chart already presents the same evidence."""
    visual_question_ids = {
        question_id
        for visual in matching_visuals
        for question_id in visual.evidence_question_ids
    }
    return [
        table for table in tables
        if theme_question_ids.intersection(table.evidence_question_ids)
        and not set(table.evidence_question_ids).issubset(visual_question_ids)
    ]














def _create_pdf_report(markdown_text: str, output_path: str, visuals_by_figure=None) -> None:
    """Create the report PDF with the configured ReportLab renderer."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_name, font_bold = register_reportlab_fonts()

    navy = colors.HexColor(COLOR_NAVY)
    navy_light = colors.HexColor(COLOR_NAVY_LIGHT)
    teal = colors.HexColor(COLOR_TEAL)
    orange = colors.HexColor(COLOR_ORANGE)
    gray_muted = colors.HexColor(COLOR_GRAY_MUTED)
    gray_text = colors.HexColor(COLOR_GRAY_TEXT)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_LEFT,
        textColor=navy, fontName=font_bold, fontSize=21, leading=25, spaceAfter=8,
    )
    date_style = ParagraphStyle(
        "ReportDate", parent=styles["BodyText"], alignment=TA_LEFT,
        textColor=gray_muted, fontName=font_name, fontSize=9, spaceAfter=10,
    )
    heading2 = ParagraphStyle(
        "SectionHeading2", parent=styles["Heading2"],
        textColor=navy, fontName=font_bold, fontSize=14, leading=18,
        spaceBefore=8, spaceAfter=4,
    )
    heading3 = ParagraphStyle(
        "SectionHeading3", parent=styles["Heading3"],
        textColor=navy, fontName=font_bold, fontSize=11.5, leading=14,
        spaceBefore=6, spaceAfter=3, keepWithNext=True,
    )
    bullet_style = ParagraphStyle(
        "BulletItem", parent=styles["BodyText"],
        fontName=font_name, fontSize=9.5, leading=14, leftIndent=18,
        bulletIndent=4, spaceAfter=5, textColor=gray_text,
        borderColor=orange, borderWidth=0, borderPadding=(5, 8, 5, 10), backColor=colors.HexColor("#F3F6F8"),
    )
    caption_style = ParagraphStyle(
        "FigureCaption", parent=styles["BodyText"],
        fontName=font_name, alignment=TA_CENTER, fontSize=8.5, textColor=gray_muted, spaceAfter=12,
    )
    body = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName=font_name,
        fontSize=10.5, leading=16, spaceBefore=0, spaceAfter=5,
        textColor=gray_text,
    )

    story = []

    def draw_page_chrome(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(navy)
        canvas.rect(0, height - 7, width, 7, stroke=0, fill=1)
        canvas.setFillColor(teal)
        canvas.rect(0, height - 10, width, 3, stroke=0, fill=1)
        if document.page > 1:
            canvas.setFont(font_bold, 7.5)
            canvas.setFillColor(gray_muted)
            canvas.drawString(42, height - 27, "BÁO CÁO PHÂN TÍCH DỮ LIỆU")
            canvas.drawRightString(width - 42, height - 27, "REPORT AGENT")
            canvas.setStrokeColor(teal)
            canvas.setLineWidth(0.8)
            canvas.line(42, height - 31, width - 42, height - 31)
        canvas.setStrokeColor(colors.HexColor("#DCE4EA"))
        canvas.line(40, 30, width - 40, 30)
        canvas.setFont(font_name, 7.5)
        canvas.setFillColor(gray_muted)
        canvas.drawString(40, 17, "REPORT AGENT - DATA ANALYTICS")
        canvas.drawRightString(width - 40, 17, f"Trang {document.page}")
        canvas.restoreState()

    def rich_text(value: str) -> str:
        safe = escape(value)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        return safe

    def plain_html(value: str) -> str:
        return unescape(re.sub(r"<[^>]+>", "", value)).strip()

    is_title_block = True
    summary_pending = False
    lines = markdown_text.splitlines()
    line_index = 0
    while line_index < len(lines):
        raw_line = lines[line_index]
        line = raw_line.strip()
        if not line:

            if not story or not isinstance(story[-1], Spacer):
                story.append(Spacer(1, 2))
            line_index += 1
            continue
        if line == "---":
            if is_title_block:
                story.append(HRFlowable(width="100%", thickness=1.5, color=teal, spaceBefore=2, spaceAfter=10))
                is_title_block = False
            else:
                story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#dddddd"),
                                         spaceBefore=6, spaceAfter=10))
            line_index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(rich_text(line[2:]), title))
        elif line.startswith("**Date Generated:**") or line.startswith("**Ngày tạo:**"):
            story.append(Paragraph(rich_text(line), date_style))
        elif line.startswith("## "):
            story.append(Paragraph(rich_text(line[3:]), heading2))
            story.append(HRFlowable(width="100%", thickness=1.5, color=teal, spaceBefore=0, spaceAfter=8, hAlign="LEFT"))
            summary_pending = line[3:].lstrip().startswith("1.")
        elif line.startswith("### "):
            story.append(Paragraph(rich_text(line[4:]), heading3))
        elif line.startswith("- "):
            story.append(Paragraph(f'<font color="{COLOR_ORANGE}">&#9679;</font>&nbsp;&nbsp;{rich_text(line[2:])}', bullet_style))
        elif line.startswith("**Figure") or line.startswith("**Hình"):
            story.append(Paragraph(f"<i>{rich_text(line.replace('**', ''))}</i>", caption_style))
        elif line.startswith("<") :


            if line.startswith('<p class="report-subtitle">'):
                story.append(Paragraph(f"<b>{rich_text(plain_html(line))}</b>", body))
            elif line.startswith('<div class="kpi-grid">'):
                block_lines = [line]
                line_index += 1
                while line_index < len(lines):
                    block_line = lines[line_index].strip()
                    block_lines.append(block_line)
                    if block_line == "</div>":
                        break
                    line_index += 1
                block_text = " ".join(block_lines)
                cards = []
                for card in re.findall(r'<div class="kpi-card">(.*?)</div>', block_text, re.DOTALL):
                    name = plain_html(re.search(r'class="kpi-name">(.*?)</span>', card, re.DOTALL).group(1))
                    value = plain_html(re.search(r'class="kpi-value">(.*?)</strong>', card, re.DOTALL).group(1))
                    scope_match = re.search(r'class="kpi-scope">(.*?)</span>', card, re.DOTALL)
                    scope = plain_html(scope_match.group(1)) if scope_match else ""
                    cards.append(Paragraph(f'<font color="{COLOR_GRAY_MUTED}" size="7">{escape(name.upper())}</font><br/><font color="{COLOR_NAVY}" size="14"><b>{escape(value)}</b></font><br/><font color="{COLOR_GRAY_MUTED}" size="7">{escape(scope)}</font>', body))
                if cards:
                    kpi_table = Table([cards], colWidths=[6.35 * inch / len(cards)] * len(cards))
                    kpi_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6F8")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE4EA")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]))
                    story.append(kpi_table)
            elif line.startswith('<div class="evidence-table-block">'):
                block_lines = [line]
                line_index += 1
                while line_index < len(lines):
                    block_line = lines[line_index].strip()
                    block_lines.append(block_line)
                    if block_line == "</div>":
                        break
                    line_index += 1
                block_text = " ".join(block_lines)
                table_title_match = re.search(r'class="table-title">(.*?)</p>', block_text, re.DOTALL)
                if table_title_match:
                    story.append(Paragraph(f"<b>{rich_text(plain_html(table_title_match.group(1)))}</b>", heading3))
                row_html = re.findall(r"<tr>(.*?)</tr>", block_text, re.DOTALL)
                table_rows = []
                for row_index, row in enumerate(row_html):
                    cells = [
                        plain_html(cell)
                        for cell in re.findall(r"<t[hd](?:\s[^>]*)?>(.*?)</t[hd]>", row, re.DOTALL)
                    ]
                    if cells:
                        table_rows.append([
                            Paragraph(
                                f'<font color="#FFFFFF"><b>{rich_text(cell)}</b></font>'
                                if row_index == 0 else rich_text(cell),
                                body,
                            )
                            for cell in cells
                        ])
                if table_rows:
                    evidence_table = Table(table_rows, repeatRows=1, hAlign="LEFT")
                    evidence_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), navy),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE4EA")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
                    ]))
                    story.append(evidence_table)
            elif line.startswith('<div class="figure">'):
                figure_match = None
                block_lines = [line]
                line_index += 1
                while line_index < len(lines):
                    block_line = lines[line_index].strip()
                    block_lines.append(block_line)
                    if block_line == "</div>":
                        break
                    line_index += 1
                block_text = " ".join(block_lines)
                figure_match = re.search(r"<(?:strong|b)>\s*(?:Figure|Hình)\s+(\d+):", block_text, re.IGNORECASE)
                if figure_match and visuals_by_figure:
                    visual = visuals_by_figure.get(figure_match.group(1))
                    if visual and os.path.exists(visual.file_path):
                        story.extend([
                            Spacer(1, 6),
                            Image(visual.file_path, width=5.7 * inch, height=3.2 * inch),
                        ])
                        if getattr(visual, "description", None):
                            story.append(Paragraph(
                                f"<i>{rich_text(visual.description)}</i>", caption_style
                            ))
            line_index += 1
            continue
        elif not line.startswith("!"):
            paragraph = Paragraph(rich_text(line), body)
            if summary_pending:
                summary_box = Table([[paragraph]], colWidths=[6.35 * inch])
                summary_box.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6F8")),
                    ("LINEBEFORE", (0, 0), (0, -1), 4, teal),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(summary_box)
                summary_pending = False
            else:
                story.append(paragraph)
        line_index += 1
    document = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=42, leftMargin=42, topMargin=44, bottomMargin=42,
    )
    document.build(story, onFirstPage=draw_page_chrome, onLaterPages=draw_page_chrome)

def export_report(state: GraphState) -> GraphState:
    """
    Finalizes the report by assembling all drafted sections and generated visuals
    into a complete document (e.g., Markdown), saves it, and also generates a PDF.
    """
    request_id = state['request_id']
    report_sections_draft = state.get('report_sections_draft', None)
    report_plan = state.get("report_plan")
    analysis_insights = state.get("analysis_insights") or []
    generated_visuals = state.get('generated_visuals', [])
    dataset_name = state.get('dataset_name', 'Unnamed Dataset')
    report_language = state.get('report_language', 'Tiếng Việt')
    report_output_dir = state.get('report_output_dir', os.path.join("local_app_data", "reports"))
    chart_output_dir = state.get('chart_output_dir', os.path.join("local_app_data", "charts"))

    logger.info(f"ReportFinalizationNode processing request: {request_id}")
    logger.info("Report finalization started with status: %s", state["status"])

    os.makedirs(report_output_dir, exist_ok=True)
    os.makedirs(chart_output_dir, exist_ok=True)

    if not report_sections_draft:
        logger.error(f"Report sections draft is missing for request {request_id}.")
        state['status'] = "error"
        state['error_message'] = "Cannot finalize report: Report sections draft is missing."
        return state

    final_report_content_md_sections = []
    pdf_file_path = None
    try:

        dataset_name =  report_sections_draft.dataset_title
        if report_language == "English":
            final_report_content_md_sections.append(f"# {('Data Analysis Report: ' + dataset_name).upper()}\n\n")
            if report_sections_draft.report_subtitle.strip():
                final_report_content_md_sections.append(
                    f'<p class="report-subtitle">{escape(report_sections_draft.report_subtitle)}</p>'
                )
            final_report_content_md_sections.append("---")
            final_report_content_md_sections.append(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            final_report_content_md_sections.append(f"# {dataset_name.upper()}\n\n")
            if report_sections_draft.report_subtitle.strip():
                final_report_content_md_sections.append(
                    f'<p class="report-subtitle">{escape(report_sections_draft.report_subtitle)}</p>'
                )
            final_report_content_md_sections.append("---")
            final_report_content_md_sections.append(f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        final_report_content_md_sections.append("---")


        final_report_content_md_sections.append(
            "\n## 1. Report Overview and Analytical Scope\n"
            if report_language == "English"
            else "\n## 1. Tổng quan và phạm vi phân tích\n"
        )
        final_report_content_md_sections.append(report_sections_draft.introduction_text)

        section_number = 2

        if report_sections_draft.analysis_narratives or generated_visuals:
            analysis_section_number = section_number
            section_number += 1
            final_report_content_md_sections.append(
                f"\n## {analysis_section_number}. Detailed Analysis and Key Findings\n"
                if report_language == "English"
                else f"\n## {analysis_section_number}. Các phát hiện trọng yếu\n"
            )

        multi_partition = False
        visuals_by_id = {visual.visual_id: visual for visual in generated_visuals} if generated_visuals else {}
        embedded_visual_ids = set()
        all_visual_question_ids = {
            question_id
            for visual in (generated_visuals or [])
            for question_id in visual.evidence_question_ids
        }
        # Tables covered by a chart are intentionally considered handled so they
        # are not reintroduced later in the unmatched-table appendix.
        embedded_table_ids = {
            id(table)
            for table in (report_plan.evidence_tables if report_plan else [])
            if set(table.evidence_question_ids).issubset(all_visual_question_ids)
        }
        current_partition = None
        partition_index = 0
        finding_index = 0
        insight_by_id = {insight.insight_id: insight for insight in analysis_insights}
        for i, narrative_original_md in enumerate(report_sections_draft.analysis_narratives):
            theme = (report_plan.themes[i] if report_plan and i < len(report_plan.themes) else None)
            theme_question_ids = set()
            for insight_id in (theme.insight_ids if theme else []):
                insight = insight_by_id.get(insight_id)
                if insight:
                    theme_question_ids.update(insight.evidence_question_ids)
            matching_tables = _tables_without_visual_coverage(
                report_plan.evidence_tables if report_plan else [],
                theme_question_ids,
                generated_visuals or [],
            )
            current_narrative_text = narrative_original_md
            embedded_visuals_html_for_this_narrative = []
            figure_id_map = report_sections_draft.figure_id_map or {}
            figure_placeholders_in_narrative = re.findall(r'\[FIGURE (\d+)\]', narrative_original_md)
            unique_figure_numbers = sorted(list(set(figure_placeholders_in_narrative)), key=int)

            for fig_num_str in unique_figure_numbers:
                generic_figure_placeholder = f"[FIGURE {fig_num_str}]"
                figure_label = "Figure" if report_language == "English" else "Hình"
                if generic_figure_placeholder in figure_id_map:
                    visual_id_from_map = figure_id_map[generic_figure_placeholder]
                    visual_obj = visuals_by_id.get(visual_id_from_map)

                    if visual_obj and os.path.exists(visual_obj.file_path):
                        current_narrative_text = current_narrative_text.replace(
                            generic_figure_placeholder,
                            f"{figure_label} {fig_num_str}"
                        )
                        try:
                            encoded = base64.b64encode(open(visual_obj.file_path, "rb").read()).decode("ascii")
                            abs_chart_path_url = f"data:image/png;base64,{encoded}"
                        except OSError:
                            abs_chart_path_url = f"file:///{os.path.abspath(visual_obj.file_path).replace(os.sep, '/')}"
                        visible_description = re.sub(
                            r"^\[[^\]]+\]\s*", "", visual_obj.description or ""
                        )
                        safe_desc = escape(visible_description)










                        embedded_visuals_html_for_this_narrative.append(
                            "\n\n"
                            f'<div class="figure">\n'
                            f'<img src="{abs_chart_path_url}" alt="{safe_desc}">\n'
                            f'<p class="figcaption"><strong>{figure_label} {fig_num_str}:</strong> {safe_desc}</p>\n'
                            "</div>\n\n"
                        )
                        embedded_visual_ids.add(visual_obj.visual_id)
                    else:
                        logger.warning(
                            f"Visual ID '{visual_id_from_map}' mapped to '{generic_figure_placeholder}' not found or file missing at '{visual_obj.file_path if visual_obj else 'N/A'}' for request {request_id}.")
                        current_narrative_text = current_narrative_text.replace(
                            generic_figure_placeholder,
                            f"*(Visual for {generic_figure_placeholder} missing)*"
                        )
                else:
                    logger.warning(
                        f"'{generic_figure_placeholder}' found in narrative but no corresponding 'visual_id' in 'figure_id_map' for request {request_id}.")
                    current_narrative_text = current_narrative_text.replace(
                        generic_figure_placeholder,
                        f"*(Visual for {generic_figure_placeholder} not mapped)*"
                    )
            if current_narrative_text:
                parts = current_narrative_text.split(":-", 1)
                title_from_narrative = parts[0].strip()
                body_from_narrative = parts[1].strip() if len(parts) > 1 else ''
            else:
                title_from_narrative = 'Finding'
                body_from_narrative = ''

            if multi_partition:
                if "|" in title_from_narrative:
                    partition_name, finding_title = [part.strip() for part in title_from_narrative.split("|", 1)]
                else:
                    partition_name, finding_title = "Phần dữ liệu", title_from_narrative

                finding_title = re.sub(re.escape(partition_name), "", finding_title,
                                       flags=re.IGNORECASE).strip(" :-|")
                if not finding_title:
                    finding_title = ("Indicator relationships and key findings"
                                     if report_language == "English"
                                     else "Mối liên hệ giữa các chỉ báo và phát hiện chính")
                if partition_name != current_partition:
                    partition_index += 1
                    finding_index = 0
                    current_partition = partition_name
                    partition_heading = (
                        "Thematic analysis of indicators and relationships"
                        if report_language == "English"
                        else "Phân tích chuyên đề về các chỉ báo và mối liên hệ"
                    )
                    final_report_content_md_sections.append(
                        f"\n## 2.{partition_index}. {partition_heading}\n"
                    )
                finding_index += 1
                final_report_content_md_sections.append(
                    f"\n### 2.{partition_index}.{finding_index}. {finding_title}\n"
                )
            else:
                final_report_content_md_sections.append(
                    f"\n### {analysis_section_number}.{i + 1}. {title_from_narrative}\n"
                )
            for table in matching_tables:
                final_report_content_md_sections.append(_evidence_table_html(table))
                embedded_table_ids.add(id(table))
            final_report_content_md_sections.append(body_from_narrative)
            final_report_content_md_sections.extend(embedded_visuals_html_for_this_narrative)
            final_report_content_md_sections.append("\n")

        remaining_visuals = [
            visual for visual in (generated_visuals or [])
            if visual.visual_id not in embedded_visual_ids and os.path.exists(visual.file_path)
        ]
        if remaining_visuals:
            raise ValueError(
                "Report draft chưa ánh xạ các biểu đồ vào narrative: "
                f"{[visual.visual_id for visual in remaining_visuals]}"
            )

        remaining_tables = [
            table for table in (report_plan.evidence_tables if report_plan else [])
            if id(table) not in embedded_table_ids
        ]
        if remaining_tables:
            final_report_content_md_sections.append(
                "\n### Supplementary Evidence Tables\n"
                if report_language == "English"
                else "\n### Bảng dữ liệu kiểm chứng bổ sung\n"
            )
            for table in remaining_tables:
                final_report_content_md_sections.append(_evidence_table_html(table))

        if report_sections_draft.notable_issues:
            final_report_content_md_sections.append(
                f"\n## {section_number}. Notable Issues and Anomalies\n"
                if report_language == "English"
                else f"\n## {section_number}. Các vấn đề và bất thường đáng chú ý\n"
            )
            for issue in report_sections_draft.notable_issues:
                final_report_content_md_sections.append(f"- {issue}\n")
            section_number += 1

        final_report_content_md_sections.append(
            f"\n## {section_number}. Overall Conclusion and Recommended Next Steps\n"
            if report_language == "English"
            else f"\n## {section_number}. Kết luận và khuyến nghị\n"
        )
        for takeaway in report_sections_draft.key_takeaways_bullet_points:
            final_report_content_md_sections.append(f"- {takeaway}\n")
        final_report_content_md_sections.append(report_sections_draft.conclusion_text)

        full_report_string_md = "\n".join(final_report_content_md_sections)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename_base = f"report_{timestamp_str}"

        try:
            font_face_css = build_font_face_css()
            font_family_css = "'ReportSans', 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" if font_face_css else "'Noto Sans', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif"

            html_content = f"""
            <html>
            <head>
                <title>Data Analysis Report - {dataset_name}</title>
                <style>
                    {font_face_css}

                    @page {{
                        size: A4;
                        margin: 20mm 18mm 18mm 18mm;
                        border-top: 2.5mm solid {COLOR_NAVY};
                        @bottom-left {{
                            content: "REPORT AGENT - DATA ANALYTICS";
                            color: {COLOR_GRAY_MUTED};
                            font-size: 7.5pt;
                            border-top: 0.5pt solid #dce4ea;
                            padding-top: 5px;
                        }}
                        @bottom-right {{
                            content: "Trang " counter(page);
                            color: {COLOR_GRAY_MUTED};
                            font-size: 7.5pt;
                            border-top: 0.5pt solid #dce4ea;
                            padding-top: 5px;
                        }}
                    }}

                    body {{
                        font-family: {font_family_css};
                        line-height: 1.52;
                        color: {COLOR_GRAY_TEXT};
                        font-size: 10pt;
                        letter-spacing: 0.01em;
                    }}

                    /* ---- Title block ---- */
                    h1 {{
                        color: {COLOR_NAVY};
                        text-align: left;
                        text-transform: uppercase;
                        font-size: 22pt;
                        font-weight: 800;
                        line-height: 1.18;
                        letter-spacing: 0.015em;
                        margin: 5mm 0 8px 0;
                    }}
                    h1 + hr {{
                        border: 0;
                        height: 2px;
                        background: {COLOR_TEAL};
                        width: 100%;
                        margin: 0 0 8px 0;
                    }}
                    h1 + hr + p {{
                        text-align: left;
                        color: {COLOR_GRAY_MUTED};
                        font-size: 8.5pt;
                        margin-bottom: 0;
                    }}
                    h1 + hr + p + hr {{
                        border: 0;
                        height: 1px;
                        background: #dce4ea;
                        margin: 10px 0 8px 0;
                    }}
                    p.report-subtitle {{
                        color: {COLOR_GRAY_MUTED};
                        font-size: 10.5pt;
                        font-weight: 600;
                        margin: -2px 0 8px 0;
                        text-align: left;
                    }}

                    /* ---- KPI cards ---- */
                    div.kpi-grid {{
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 8px;
                        margin: 8px 0 14px 0;
                    }}
                    div.kpi-card {{
                        background: #F3F6F8;
                        border-top: 3px solid {COLOR_TEAL};
                        padding: 8px 9px;
                        min-height: 52px;
                        break-inside: avoid;
                    }}
                    span.kpi-name, span.kpi-value, span.kpi-scope {{ display: block; }}
                    span.kpi-name {{
                        color: {COLOR_GRAY_MUTED};
                        font-size: 7.7pt;
                        font-weight: 700;
                        text-transform: uppercase;
                    }}
                    strong.kpi-value {{
                        color: {COLOR_NAVY};
                        font-size: 16pt;
                        line-height: 1.25;
                        margin: 3px 0;
                    }}
                    span.kpi-scope {{ color: {COLOR_GRAY_MUTED}; font-size: 7.3pt; }}

                    /* ---- Section headings ---- */
                h2 {{
                        color: {COLOR_NAVY};
                        font-size: 14.5pt;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.02em;
                        margin-top: 14px;
                        margin-bottom: 7px;
                        padding-bottom: 3px;
                        border-bottom: 2px solid {COLOR_TEAL};
                    }}
                    h3 {{
                        color: {COLOR_NAVY};
                        font-size: 11.5pt;
                        font-weight: 700;
                        margin-top: 10px;
                        margin-bottom: 4px;
                        break-after: avoid;
                    }}

                    p {{ margin-top: 0.12em; margin-bottom: 0.62em; text-align: justify; }}
                    h2:first-of-type + p {{
                        background: #F3F6F8;
                        border-left: 5px solid {COLOR_TEAL};
                        border-radius: 3px;
                        padding: 10px 12px;
                    }}
                    ul, ol {{ margin-top: 0.25em; margin-bottom: 0.75em; }}
                    strong {{ font-weight: bold; }}
                    em {{ font-style: italic; }}
                    code {{
                        font-family: 'Courier New', Courier, monospace;
                        background-color: #eee;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                    pre {{
                        background-color: #f4f4f4;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                        font-family: 'Courier New', Courier, monospace;
                        font-size: 0.9em;
                    }}

                    /* ---- Key takeaways as accent callout cards ---- */
                    ul {{ list-style: none; padding-left: 0; }}
                    ul li {{
                        position: relative;
                        background: #F3F6F8;
                        border-left: 4px solid {COLOR_ORANGE};
                        border-radius: 3px;
                        padding: 6px 12px 6px 28px;
                        margin-bottom: 5px;
                        line-height: 1.5;
                    }}
                    ul li::before {{
                        content: "\\2713";
                        position: absolute;
                        left: 5px;
                        color: {COLOR_ORANGE};
                        font-weight: bold;
                    }}
                    ol {{ padding-left: 22px; }}
                    ol li {{ margin-bottom: 6px; }}

                    /* ---- Figures / charts ---- */
                    div.figure {{
                        margin: 14px 0 16px 0;
                        text-align: center;
                        break-inside: avoid;
                    }}
                    div.figure img {{
                        max-width: 96%;
                        max-height: 78mm;
                        height: auto;
                        width: auto;
                        border: 0;
                        border-radius: 0;
                        box-shadow: none;
                    }}
                    p.figcaption {{
                        margin-top: 8px;
                        font-size: 9.5pt;
                        font-style: normal;
                        color: {COLOR_GRAY_MUTED};
                    }}
                    img {{ max-width: 90%; height: auto; display: block; margin: 15px auto; }}

                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 10px 0 14px 0;
                        font-size: 8.7pt;
                    }}
                    th {{
                        background: {COLOR_NAVY};
                        color: white;
                        font-weight: bold;
                        padding: 7px 6px;
                        border: 0.5pt solid #aebbc8;
                    }}
                    td {{ padding: 6px; border: 0.5pt solid #dce4ea; }}
                    tbody tr:nth-child(even) {{ background: #F3F6F8; }}
                    div.evidence-table-block {{
                        margin: 8px 0 12px 0;
                        break-inside: avoid;
                    }}
                    p.table-title {{
                        color: {COLOR_NAVY};
                        font-weight: 700;
                        font-size: 9.3pt;
                        margin: 0 0 4px 0;
                        text-align: left;
                    }}
                    table.evidence-table td:not(:first-child) {{ text-align: right; }}

                    hr {{ border: 0; height: 1px; background: #eee; margin: 2em 0; }}
                </style>
            </head>
            <body>
                {markdown.markdown(full_report_string_md, extensions=['fenced_code', 'tables', 'nl2br', 'md_in_html'])}
            </body>
            </html>
            """

            report_pdf_file_path = os.path.join(report_output_dir, f"{report_filename_base}.pdf")
            visuals_by_figure = {
                match.group(0): visual
                for placeholder, visual_id in (report_sections_draft.figure_id_map or {}).items()
                for match in [re.search(r"\d+", placeholder)]
                if match and (visual := visuals_by_id.get(visual_id))
            }
            _create_pdf_report(
                full_report_string_md,
                report_pdf_file_path,
                visuals_by_figure,
            )
            pdf_file_path = report_pdf_file_path
            logger.info("Final PDF report saved to: %s", pdf_file_path)

        except Exception as e:
            logger.error(f"Error generating PDF report for request {request_id}: {e}", exc_info=True)
            state['status'] = "error"
            state['error_message'] = f"Error generating PDF: {e}"
            return state


        state['final_report'] = ReportFormat(
            content=full_report_string_md,
            format_type="pdf",
            pdf_file_path=pdf_file_path
        )
        state['status'] = "report_finalized"
        logger.info(f"ReportFinalizationNode completed for request: {request_id}. Report finalized and saved.")
        return state

    except Exception as e:
        logger.error(f"An unexpected error occurred in ReportFinalizationNode for request {request_id}: {e}",
                     exc_info=True)
        state['status'] = "error"
        state['error_message'] = f"An unexpected error occurred during report finalization: {e}"
        return state
