"""Build a compact report plan deterministically from validated evidence."""
from __future__ import annotations

from graph.state import GraphState
from schemas.messages import EvidenceTable, ReportPlan, ReportTheme


THEMES = {
    "trend_over_time": ("Xu hướng theo thời gian", "Đánh giá biến động giữa các kỳ đủ điều kiện."),
    "correlation": ("Mối liên hệ giữa các chỉ báo", "Trình bày mối liên hệ quan sát được, không suy luận nhân quả."),
    "distribution_summary": ("Phân phối và mức điển hình", "Mô tả trung tâm, độ phân tán và phạm vi của chỉ báo."),
    "aggregate_summary": ("Tổng quan các chỉ số chính", "Tóm tắt quy mô và mức điển hình của các metric trọng yếu."),
    "missing_rate": ("Chất lượng dữ liệu", "Đánh giá mức độ thiếu dữ liệu và ảnh hưởng đến kết luận."),
    "top_n": ("Các nhóm nổi bật", "Xác định nhóm đứng đầu trên metric đã kiểm chứng."),
    "mean_by_group": ("So sánh giữa các nhóm", "So sánh metric giữa các nhóm có đủ mẫu."),
    "count_by_category": ("Cơ cấu theo nhóm", "Mô tả số lượng và tỷ trọng giữa các nhóm."),
}


def _table_title(item) -> str:
    """Describe evidence without exposing the internal analytical question."""
    columns = " và ".join(item.columns)
    labels = {
        "trend_over_time": f"Xu hướng {columns} theo thời gian",
        "mean_by_group": f"Giá trị trung bình của {columns}",
        "count_by_category": f"Cơ cấu {columns}",
        "top_n": f"Các nhóm dẫn đầu theo {columns}",
    }
    return labels.get(item.operation, f"Chi tiết {columns}")


def build_report_plan(state: GraphState) -> GraphState:
    insights = [item for item in state.get("analysis_insights") or [] if item.evidence_valid]
    results = {item.question_id: item for item in state.get("computed_question_results") or []}
    allowed_ids = {qid for insight in insights for qid in insight.evidence_question_ids}
    selected = [item for qid, item in results.items() if qid in allowed_ids]

    theme_groups: dict[str, list[str]] = {}
    for insight in insights:
        operations = [results[qid].operation for qid in insight.evidence_question_ids if qid in results]
        key = operations[0] if operations else "count_by_category"
        theme_groups.setdefault(key, []).append(insight.insight_id)
    themes = [
        ReportTheme(title=THEMES[key][0], purpose=THEMES[key][1], insight_ids=ids)
        for key, ids in theme_groups.items() if ids
    ]

    tables = []
    for item in selected:
        if not isinstance(item.result, list) or len(item.result) < 2:
            continue
        rows = [row for row in item.result[:10] if isinstance(row, dict)]
        if not rows:
            continue
        columns = list(dict.fromkeys(key for row in rows for key in row))
        tables.append(EvidenceTable(title=_table_title(item), columns=columns,
                                    rows=rows, evidence_question_ids=[item.question_id]))
        if len(tables) >= 4:
            break

    state["report_plan"] = ReportPlan(kpis=[], themes=themes, evidence_tables=tables)
    state["status"] = "report_planned"
    return state
