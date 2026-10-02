"""Deterministic validation and deduplication of LLM-written insights."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from graph.state import GraphState
from llm import get_llm, text_from_response
from schemas.messages import AnalysisInsight

logger = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?")
DATE_LITERAL_PATTERN = re.compile(
    r"(?:\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?)?\b"
    r"|\b(?:tháng|thang|month)\s+\d{1,2}(?:\s+(?:năm|nam|year)\s+\d{4})?\b"
    r"|\b(?:năm|nam|year)\s+\d{4}\b)",
    re.IGNORECASE,
)


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


def _prose_numbers(value: str) -> list[float]:
    """Extract claimed metrics while ignoring date/month literals."""
    masked = DATE_LITERAL_PATTERN.sub(" ", value)
    return _numbers(masked)


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


def _deterministic_insight(item, insight_id: str, reason: str) -> AnalysisInsight:
    """Preserve one-to-one coverage using only audited Pandas evidence."""
    evidence = json.dumps(item.result, ensure_ascii=False, default=str)
    return AnalysisInsight(
        insight_id=insight_id,
        title=item.question,
        narrative=f"Kết quả Pandas đã kiểm chứng cho yêu cầu phân tích: {evidence}",
        finding=f"Kết quả đã kiểm chứng: {evidence}",
        source_partition=item.source_partition,
        question=item.question,
        metrics={item.question_id: item.result},
        source_sheet=item.source_partition,
        source_columns=list(item.columns),
        evidence_question_ids=[item.question_id],
        limitations=[f"Nội dung LLM ban đầu đã được thay bằng evidence trực tiếp: {reason}"],
        evidence_valid=True,
    )


def _clean_json_response(value: str) -> str:
    text = value.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _rewrite_insight(llm, insight: AnalysisInsight, item, reason: str) -> AnalysisInsight:
    """Ask the LLM to rewrite invalid prose using only one audited result."""
    prompt = f"""
Bạn là Evidence Rewrite Agent. Insight dưới đây không vượt qua kiểm tra vì: {reason}

Hãy viết lại title, finding và narrative bằng cách chỉ sử dụng số, nhãn, kỳ và kết luận có trực tiếp
trong computed result. Không thêm phép tính, ước lượng, so sánh, nguyên nhân hoặc số mới. Giữ nguyên
insight_id, source_partition và evidence_question_ids. Mỗi
evidence_question_ids phải chứa duy nhất question_id đã cung cấp. Trả về đúng một JSON theo schema
AnalysisInsight, không Markdown và không giải thích.

Insight ban đầu:
{json.dumps(insight.model_dump(mode="python"), ensure_ascii=False, default=str)}

Question ID: {item.question_id}
Câu hỏi: {item.question}
Computed result đã kiểm chứng:
{json.dumps(item.result, ensure_ascii=False, default=str)}
"""
    response = text_from_response(llm.invoke(prompt, config={"request_options": {"timeout": 60}}))
    rewritten = AnalysisInsight.model_validate_json(_clean_json_response(response))
    rewritten.insight_id = insight.insight_id
    rewritten.source_partition = item.source_partition
    rewritten.evidence_question_ids = [item.question_id]
    rewritten.evidence_valid = False
    return rewritten


def _prose_issues(insight: AnalysisInsight, allowed: list[float]) -> list[str]:
    prose = f"{insight.title} {insight.finding} {insight.narrative}"
    displayed = _prose_numbers(prose)
    reasons = []
    unsupported = [number for number in displayed
                   if not any(_grounded(number, source) for source in allowed)]
    if unsupported:
        reasons.append(f"số không có trong evidence: {unsupported}")
    return reasons


def validate_evidence(state: GraphState) -> GraphState:
    llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
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
        reasons = _prose_issues(insight, allowed)
        if reasons:
            reason = "; ".join(reasons)
            rewritten = None
            rewrite_error = ""
            for attempt in range(2):
                try:
                    candidate = _rewrite_insight(llm, insight, linked[0], reason)
                    candidate_issues = _prose_issues(candidate, allowed)
                    if candidate_issues:
                        reason = "; ".join(candidate_issues)
                        continue
                    rewritten = candidate
                    logger.info(
                        "Evidence Rewrite Agent repaired %s on attempt %s.",
                        insight.insight_id, attempt + 1,
                    )
                    break
                except Exception as exc:
                    rewrite_error = str(exc)
                    logger.warning(
                        "Evidence rewrite failed for %s on attempt %s: %s",
                        insight.insight_id, attempt + 1, exc,
                    )
            if rewritten is not None:
                insight = rewritten
            else:
                final_reason = reason + (f"; rewrite error: {rewrite_error}" if rewrite_error else "")
                rejected.append((insight.insight_id, final_reason))
                insight = _deterministic_insight(linked[0], insight.insight_id, final_reason)

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

    covered_ids = {
        question_id for insight in valid for question_id in insight.evidence_question_ids
    }
    for question_id, item in results.items():
        if question_id in covered_ids:
            continue
        reason = "LLM không tạo insight hợp lệ liên kết với computed result"
        fallback = _deterministic_insight(
            item, f"insight_fallback_{question_id}", reason
        )
        valid.append(fallback)
        rejected.append((fallback.insight_id, reason))

    if not valid:
        state["status"] = "error"
        state["error_message"] = (
            "Không có insight nào vượt qua Evidence Validation. "
            f"Chi tiết: {rejected}"
        )
        return state
    if rejected:
        logger.warning(
            "Evidence validation repaired %s insight issue(s) without dropping computed results: %s",
            len(rejected), rejected,
        )
    state["analysis_insights"] = valid
    state["validated_insights"] = valid
    state["status"] = "evidence_validated"
    logger.info("Evidence validation retained %s/%s computed results; repaired=%s",
                len(valid), len(results), rejected)
    return state
