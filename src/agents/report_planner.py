"""Build a compact report plan deterministically from validated evidence."""
from __future__ import annotations

import logging

from graph.state import GraphState
from schemas.messages import EvidenceTable, ReportPlan, ReportTheme

logger = logging.getLogger(__name__)


def _theme_for_result(item) -> tuple[str, str]:
    title = (item.question or "").strip().rstrip("?").strip()
    purpose = str(item.parameters.get("expected_result") or "").strip()
    if not title or not purpose:
        raise ValueError(
            f"Kết quả {item.question_id} thiếu question/expected_result để lập report theme."
        )
    return title, purpose


def _table_title(item) -> str:
    """Describe evidence without exposing the internal analytical question."""
    plan = item.parameters.get("analysis_plan") or {}
    groups = plan.get("group_by") or []
    metrics = plan.get("metrics") or []
    transforms = [entry.get("type") for entry in plan.get("transforms") or []]
    if groups and metrics:
        if "pct_change" in transforms:
            return f"Tăng trưởng {metrics[0]} theo {groups[0]}"
        if "share_of_total" in transforms:
            return f"Tỷ trọng {metrics[0]} theo {groups[0]}"
        return f"{metrics[0]} theo {groups[0]}"
    if groups:
        return f"Số lượng theo {groups[0]}"
    columns = " và ".join(item.columns)
    labels = {
        "trend_over_time": f"Xu hướng {columns} theo thời gian",
        "mean_by_group": f"Giá trị trung bình của {columns}",
        "count_by_category": f"Cơ cấu {columns}",
        "top_n": f"Các nhóm dẫn đầu theo {columns}",
        "sum_by_group": f"Tổng {columns} theo nhóm",
        "share_of_total": f"Tỷ trọng {columns} trong tổng thể",
        "share_by_group": f"Tỷ trọng {columns} theo nhóm",
        "period_over_period_growth": f"Tăng trưởng {columns} theo kỳ",
        "period_growth": f"Tăng trưởng {columns} theo kỳ",
        "ratio_by_group": f"Tỷ lệ {columns} theo nhóm",
        "difference_between_groups": f"Chênh lệch {columns} giữa các nhóm",
        "outlier_analysis": f"Thống kê ngoại lệ của {columns}",
    }
    return labels.get(item.operation, f"Chi tiết {columns}")


def build_report_plan(state: GraphState) -> GraphState:
    insights = [item for item in state.get("analysis_insights") or [] if item.evidence_valid]
    results = {item.question_id: item for item in state.get("computed_question_results") or []}
    allowed_ids = {qid for insight in insights for qid in insight.evidence_question_ids}
    unknown_insight_ids = allowed_ids - set(results)
    if unknown_insight_ids:
        raise ValueError(
            "Validated insight tham chiếu computed result không tồn tại: "
            f"{sorted(unknown_insight_ids)}."
        )
    missing_insight_ids = set(results) - allowed_ids
    if missing_insight_ids:
        logger.warning(
            "Report plan bỏ qua %s computed result không có evidence insight hợp lệ: %s",
            len(missing_insight_ids), sorted(missing_insight_ids),
        )
    selected = [item for qid, item in results.items() if qid in allowed_ids]

    theme_groups: dict[tuple[str, str], list[str]] = {}
    for insight in insights:
        linked_results = [results[qid] for qid in insight.evidence_question_ids if qid in results]
        if len(linked_results) != 1:
            raise ValueError(
                f"Insight {insight.insight_id} phải liên kết đúng một computed result, "
                f"nhận {len(linked_results)}."
            )
        key = _theme_for_result(linked_results[0])
        theme_groups.setdefault(key, []).append(insight.insight_id)
    themes = []
    for (title, purpose), ids in theme_groups.items():
        if not ids:
            continue
        themes.append(ReportTheme(title=title, purpose=purpose, insight_ids=ids))

    tables = []
    for item in selected:
        if not isinstance(item.result, list) or len(item.result) < 2:
            continue
        rows = [row for row in item.result if isinstance(row, dict)]
        if not rows:
            continue
        columns = list(dict.fromkeys(key for row in rows for key in row))
        tables.append(EvidenceTable(title=_table_title(item), columns=columns,
                                    rows=rows, evidence_question_ids=[item.question_id]))

    state["report_plan"] = ReportPlan(kpis=[], themes=themes, evidence_tables=tables)
    state["status"] = "report_planned"
    return state
