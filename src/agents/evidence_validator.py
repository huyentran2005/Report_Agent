"""Deterministic validation and deduplication of LLM-written insights."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from graph.state import GraphState

logger = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?")
VAGUE_TERMS = ("phần lớn", "khá cao", "đáng kể", "significant", "majority", "substantial")
METRIC_TERMS = ("tỷ lệ", "số lượng", "trung bình", "tổng", "chênh lệch", "rate", "count", "average", "total")


def _numbers(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [number for nested in value.values() for number in _numbers(nested)]
    if isinstance(value, (list, tuple)):
        return [number for nested in value for number in _numbers(nested)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        return [float(token.replace(",", "")) for token in NUMBER_PATTERN.findall(value)]
    return []


def _grounded(displayed: float, source: float) -> bool:
    return abs(displayed - source) <= max(0.005, abs(source) * 0.0005)


def _sample_size(item) -> int | None:
    result = item.result
    if isinstance(result, dict):
        for key in ("row_count", "complete_pairs", "count", "sample_size"):
            value = result.get(key)
            if isinstance(value, (int, float)):
                return int(value)
    if isinstance(result, list) and (
        item.operation == "count_by_category"
        or (item.operation == "trend_over_time" and item.parameters.get("aggregation") == "count")
    ):
        counts = [
            record.get("count") for record in result if isinstance(record, dict)
            and isinstance(record.get("count"), (int, float))
        ]
        return int(sum(counts)) if counts else None
    if isinstance(result, list):
        sizes = [
            record.get("sample_size") for record in result if isinstance(record, dict)
            and isinstance(record.get("sample_size"), (int, float))
        ]
        return int(sum(sizes)) if sizes else None
    return None


def validate_evidence(state: GraphState) -> GraphState:
    results = {item.question_id: item for item in state.get("computed_question_results") or []}
    signatures: dict[tuple, set[str]] = {}
    signature_ids: dict[tuple, set[str]] = {}
    for item in results.values():
        parameters = {key: value for key, value in item.parameters.items() if key != "target_column"}
        signature = (
            item.source_partition, item.operation, tuple(item.columns),
            json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str),
        )
        signatures.setdefault(signature, set()).add(
            json.dumps(item.result, ensure_ascii=False, sort_keys=True, default=str)
        )
        signature_ids.setdefault(signature, set()).add(item.question_id)
    conflicting_ids = {
        question_id for signature, values in signatures.items() if len(values) > 1
        for question_id in signature_ids[signature]
    }
    candidates = state.get("analysis_insights") or []
    valid = []
    rejected = []
    for insight in candidates:
        linked = [results[qid] for qid in insight.evidence_question_ids if qid in results]
        if not linked:
            rejected.append((insight.insight_id, "không liên kết kết quả pandas"))
            continue
        if any(item.question_id in conflicting_ids for item in linked):
            rejected.append((insight.insight_id, "metric cùng phạm vi có kết quả mâu thuẫn"))
            continue
        allowed = [
            number for item in linked
            for number in (*_numbers(item.result), *_numbers(item.parameters))
        ]
        prose = f"{insight.title} {insight.finding} {insight.narrative}"
        displayed = _numbers(prose)
        reasons = []
        if not allowed or not displayed:
            reasons.append("phát hiện không hiển thị số liệu cụ thể")
        unsupported = [number for number in displayed if not any(_grounded(number, source) for source in allowed)]
        if unsupported:
            reasons.append(f"số không có trong evidence: {unsupported}")
        lower = prose.casefold()
        if any(term in lower for term in VAGUE_TERMS):
            reasons.append("dùng nhận xét định tính mơ hồ")
        if any(term in lower for term in METRIC_TERMS) and not displayed:
            reasons.append("metric định lượng thiếu value")
        if reasons:
            rejected.append((insight.insight_id, "; ".join(reasons)))
            continue

        insight.question = " | ".join(item.question for item in linked)
        insight.finding = insight.finding or insight.title
        insight.metrics = {item.question_id: item.result for item in linked}
        insight.source_sheet = insight.source_partition
        insight.source_columns = list(dict.fromkeys(column for item in linked for column in item.columns))
        insight.filters = {
            item.question_id: (item.parameters.get("analysis_plan") or {}).get("filters", [])
            for item in linked if (item.parameters.get("analysis_plan") or {}).get("filters")
        }
        sizes = [size for item in linked if (size := _sample_size(item)) is not None]
        insight.sample_size = max(sizes) if sizes else None
        denominators = [
            int(item.result["row_count"]) for item in linked
            if isinstance(item.result, dict) and isinstance(item.result.get("row_count"), (int, float))
        ]
        insight.denominator = max(denominators) if denominators else insight.sample_size
        profile = (state.get("sheet_profiles") or {}).get(insight.source_partition or "", {})
        details = profile.get("column_details") or {}
        insight.missing_values = sum(
            int((details.get(column) or {}).get("missing_values_count", 0) or 0)
            for column in insight.source_columns
        )


        insight.limitations = list(dict.fromkeys(insight.limitations))
        insight.evidence_valid = True

        valid.append(insight)

    if not valid:
        state["status"] = "error"
        state["error_message"] = (
            "Không có insight nào vượt qua Evidence Validation. "
            f"Chi tiết: {rejected}"
        )
        return state
    if rejected:
        state["status"] = "error"
        state["error_message"] = (
            f"Evidence validation không bao phủ đủ {len(candidates)} câu hỏi; "
            f"đã loại {len(rejected)} insight: {rejected}"
        )
        return state
    state["analysis_insights"] = valid
    state["validated_insights"] = valid
    state["status"] = "evidence_validated"
    logger.info("Evidence validation kept %s/%s insights; rejected=%s", len(valid), len(candidates), rejected)
    return state
