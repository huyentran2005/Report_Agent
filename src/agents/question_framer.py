"""Frame report questions and answer them with audited pandas operations."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError, model_validator

from data_io import (effective_instructions, infer_column_semantic, is_date_like_series, read_dataset_partitions,
                     split_partitions_by_source, to_datetime_series)
from graph.state import GraphState
from llm import get_llm, text_from_response
from schemas.messages import ComputedQuestionResult, DataProfile, FramedQuestion
from privacy import is_person_name_column, safe_column_details

logger = logging.getLogger(__name__)


class FramedQuestionsOutput(BaseModel):
    questions: List[FramedQuestion] = Field(min_length=20, max_length=20)
    selected_question_ids: List[str] = Field(
        min_length=5, max_length=10,
        description="ID của 5-10 câu hỏi quan trọng nhất được chọn từ đúng 20 ứng viên.",
    )

    @model_validator(mode="after")
    def validate_selected_questions(self):
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Mỗi câu hỏi ứng viên phải có question_id duy nhất.")
        if len(self.selected_question_ids) != len(set(self.selected_question_ids)):
            raise ValueError("selected_question_ids không được chứa ID trùng lặp.")
        unknown = set(self.selected_question_ids) - set(question_ids)
        if unknown:
            raise ValueError(f"selected_question_ids chứa ID không tồn tại: {sorted(unknown)}")
        return self


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _records(series: pd.Series, key_name: str, value_name: str) -> List[Dict[str, Any]]:
    return [{key_name: _json_value(key), value_name: _json_value(value)} for key, value in series.items()]


def _require_columns(df: pd.DataFrame, question: FramedQuestion, count: int | tuple[int, ...]) -> None:
    allowed_counts = (count,) if isinstance(count, int) else count
    if len(question.columns) not in allowed_counts:
        raise ValueError(f"{question.operation} requires {allowed_counts} column(s)")
    missing = [column for column in question.columns if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns: {missing}")
    if any(str(column).startswith("_") for column in question.columns):
        raise ValueError("Metadata columns cannot be analyzed")
    sensitive = [column for column in question.columns if is_person_name_column(column, df[column])]
    if sensitive:
        raise ValueError(f"Personal-name columns cannot be analyzed or grouped: {sensitive}")
    unknown = [column for column in question.columns
               if infer_column_semantic(df[column], column) == "unknown"]
    if unknown:
        raise ValueError(f"UNKNOWN columns cannot be analyzed: {unknown}")


def _require_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    invalid = [column for column in columns
               if infer_column_semantic(df[column], column) != "numeric_measure"]
    if invalid:
        raise TypeError(f"Numeric columns required: {invalid}")


def _mean_by_group(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, 2)
    group, value = q.columns
    _require_numeric(df, [value])
    if infer_column_semantic(df[group], group) not in {"categorical", "boolean/status"}:
        raise TypeError(f"Categorical group column required: {group!r}")
    work = df[[group, value]].dropna()
    if len(work) < 2 or work[group].nunique() < 2:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    metric_name = f"mean_{value}"
    grouped = work.groupby(group)[value].agg(["mean", "count"]).sort_values("mean", ascending=False)
    result = [
        {group: _json_value(key), metric_name: _json_value(row["mean"]),
         "sample_size": int(row["count"])}
        for key, row in grouped.iterrows()
    ]
    return result, f"df.groupby({group!r})[{value!r}].agg(['mean', 'count'])"


def _count_by_category(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, 1)
    column = q.columns[0]
    if infer_column_semantic(df[column], column) not in {"categorical", "boolean/status", "datetime"}:
        raise TypeError(f"Categorical/status column required: {column!r}")
    if df[column].dropna().nunique() < 2:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    result = df[column].value_counts(dropna=False)
    return _records(result, column, "count"), f"df[{column!r}].value_counts(dropna=False)"


def _trend_over_time(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, (1, 2))
    date_column = q.columns[0]
    dates = to_datetime_series(df[date_column])
    if dates.notna().mean() < 0.7:
        raise TypeError(f"Column {date_column!r} is not sufficiently date-like")
    work = df.assign(__period=dates.dt.to_period("M").astype("string")).dropna(subset=["__period"])
    if work["__period"].nunique() < 2:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    if len(q.columns) == 1 or q.aggregation == "count":
        result = work.groupby("__period").size()
        value_name = "count"
    else:
        value = q.columns[1]
        _require_numeric(df, [value])
        value_name = f"{q.aggregation}_{value}"
        grouped = work.groupby("__period")[value].agg([q.aggregation, "count"]).sort_index()
        records = [
            {"period": _json_value(key), value_name: _json_value(row[q.aggregation]),
             "sample_size": int(row["count"])}
            for key, row in grouped.iterrows()
        ]
        return records, f"monthly {q.aggregation} and count grouped by parsed {date_column!r}"
    result = result.sort_index()
    return _records(result, "period", value_name), f"monthly {q.aggregation} grouped by parsed {date_column!r}"


def _top_n(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, (1, 2))
    group = q.columns[0]
    if infer_column_semantic(df[group], group) not in {"categorical", "boolean/status", "datetime"}:
        raise TypeError(f"Categorical/status/time group required: {group!r}")
    if df[group].dropna().nunique() < 2:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    if len(q.columns) == 1 or q.aggregation == "count":
        result = df[group].value_counts(dropna=False).nlargest(q.n)
        value_name = "count"
    else:
        value = q.columns[1]
        _require_numeric(df, [value])
        value_name = f"{q.aggregation}_{value}"
        grouped = df[[group, value]].dropna().groupby(group)[value].agg([q.aggregation, "count"])
        grouped = grouped.nlargest(q.n, q.aggregation)
        records = [
            {group: _json_value(key), value_name: _json_value(row[q.aggregation]),
             "sample_size": int(row["count"])}
            for key, row in grouped.iterrows()
        ]
        return records, f"top {q.n} {group!r} by {q.aggregation} with group sample size"
    return _records(result, group, value_name), f"top {q.n} {group!r} by {q.aggregation}"


def _missing_rate(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, 1)
    column = q.columns[0]
    rate = float(df[column].isna().mean() * 100)
    return {"missing_count": int(df[column].isna().sum()), "row_count": int(len(df)), "missing_rate_percent": rate}, f"df[{column!r}].isna().mean() * 100"


def _correlation(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, 2)
    _require_numeric(df, q.columns)
    left, right = q.columns
    pairs = df[[left, right]].dropna()
    if len(pairs) < 3 or pairs[left].nunique() < 2 or pairs[right].nunique() < 2:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    value = pairs[left].corr(pairs[right])
    if pd.isna(value):
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    return {"pearson_correlation": _json_value(value), "complete_pairs": int(len(pairs))}, f"df[{left!r}].corr(df[{right!r}])"


def _distribution_summary(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, 1)
    column = q.columns[0]
    _require_numeric(df, [column])
    values = df[column].dropna()
    if len(values) < 2:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    result = {"count": int(values.count()), "mean": _json_value(values.mean()), "std": _json_value(values.std()), "min": _json_value(values.min()), "q25": _json_value(values.quantile(.25)), "median": _json_value(values.median()), "q75": _json_value(values.quantile(.75)), "max": _json_value(values.max())}
    return result, f"df[{column!r}].dropna().describe()"


def _aggregate_summary(df: pd.DataFrame, q: FramedQuestion) -> tuple[Any, str]:
    _require_columns(df, q, 1)
    column = q.columns[0]
    _require_numeric(df, [column])
    values = df[column].dropna()
    if values.empty:
        raise ValueError("Không đủ dữ liệu để đánh giá.")
    return {
        "count": int(values.count()), "sum": _json_value(values.sum()),
        "mean": _json_value(values.mean()), "min": _json_value(values.min()),
        "max": _json_value(values.max()),
    }, f"df[{column!r}].dropna().agg(['count', 'sum', 'mean', 'min', 'max'])"


OPERATIONS: Dict[str, Callable[[pd.DataFrame, FramedQuestion], tuple[Any, str]]] = {
    "mean_by_group": _mean_by_group,
    "count_by_category": _count_by_category,
    "trend_over_time": _trend_over_time,
    "top_n": _top_n,
    "missing_rate": _missing_rate,
    "correlation": _correlation,
    "distribution_summary": _distribution_summary,
    "aggregate_summary": _aggregate_summary,
}


def compact_computed_results(results, max_items: int = 20) -> List[Dict[str, Any]]:
    """Create a bounded LLM context while preserving full results in graph state."""
    compacted: List[Dict[str, Any]] = []
    for item in results:
        if any(is_person_name_column(column) for column in item.columns):
            logger.warning("Omitting audited result based on a personal-name column from LLM context: %s", item.columns)
            continue
        payload = item.model_dump(mode="json")
        value = payload.get("result")
        if isinstance(value, list) and len(value) > max_items:
            if item.operation == "trend_over_time":
                payload["result"] = value[-max_items:]
                payload["context_note"] = "Chỉ cung cấp các kỳ gần nhất; kết quả đầy đủ vẫn được lưu trong state."
            else:
                payload["result"] = value[:max_items]
                payload["context_note"] = "Chỉ cung cấp các mục đứng đầu; kết quả đầy đủ vẫn được lưu trong state."
        compacted.append(payload)
    return compacted


def execute_question(df: pd.DataFrame, question: FramedQuestion) -> ComputedQuestionResult:
    result, calculation = OPERATIONS[question.operation](df, question)
    parameters: Dict[str, Any] = {"aggregation": question.aggregation, "n": question.n}
    if question.operation == "distribution_summary":

        parameters["percentiles"] = [25, 50, 75]
    return ComputedQuestionResult(question_id=question.question_id, question=question.question,
                                  operation=question.operation, columns=question.columns,
                                  parameters=parameters,
                                  result=result, calculation=calculation,
                                  source_partition=question.source_partition)


def _fallback_questions(profile: DataProfile) -> List[FramedQuestion]:
    details = {
        name: info for name, info in profile.column_details.items()
        if not str(name).startswith("_")
        and not info.get("is_sensitive_person_name")
        and not is_person_name_column(name)
    }
    numeric = [name for name, info in details.items() if any(token in str(info.get("type", "")).lower() for token in ("int", "float", "decimal"))]
    categorical = [name for name in details if name not in numeric]
    date_like = [name for name in categorical if any(token in name.lower() for token in ("date", "time", "day", "month", "year", "ngày", "tháng", "năm"))]
    specs: List[tuple[str, List[str], str, str]] = []
    if categorical:
        specs.append(("count_by_category", [categorical[0]], f"Các nhóm {categorical[0]} có bao nhiêu bản ghi?", "count"))
        specs.append(("top_n", [categorical[0]], f"5 nhóm {categorical[0]} phổ biến nhất là gì?", "count"))
    if categorical and numeric:
        specs.append(("mean_by_group", [categorical[0], numeric[0]], f"Trung bình {numeric[0]} theo {categorical[0]} là bao nhiêu?", "mean"))
    if date_like and numeric:
        specs.append(("trend_over_time", [date_like[0], numeric[0]], f"{numeric[0]} thay đổi theo thời gian như thế nào?", "sum"))
    for column in numeric[:2]:
        specs.append(("aggregate_summary", [column], f"Các KPI tổng hợp của {column} là gì?", "sum"))
        specs.append(("distribution_summary", [column], f"Phân phối của {column} có đặc điểm gì?", "mean"))
    if len(numeric) >= 2:
        specs.append(("correlation", numeric[:2], f"{numeric[0]} và {numeric[1]} tương quan ra sao?", "mean"))
    for column in list(details)[:4]:
        if len(specs) >= 4:
            break
        specs.append(("missing_rate", [column], f"Tỷ lệ thiếu của {column} là bao nhiêu?", "mean"))
    return [FramedQuestion(question_id=f"question_{i}", operation=op, columns=columns,
                           question=text, aggregation=aggregation) for i, (op, columns, text, aggregation) in enumerate(specs[:8], 1)]


def _fallback_questions_for_column(df: pd.DataFrame, target: str) -> List[FramedQuestion]:
    """Build 4-8 questions per column, including cross-column relationships when valid."""
    numeric = [str(column) for column in df.columns
               if not str(column).startswith("_")
               and not is_person_name_column(column, df[column])
               and infer_column_semantic(df[column], str(column)) == "numeric_measure"]
    other_numeric = [column for column in numeric if column != target]
    categorical = [str(column) for column in df.columns
                   if not str(column).startswith("_")
                   and infer_column_semantic(df[column], str(column)) in {"categorical", "boolean/status"}
                   and not is_person_name_column(column, df[column])]
    other_categorical = [column for column in categorical if column != target]
    target_is_numeric = target in numeric
    values = df[target]
    target_is_date = is_date_like_series(values, target)
    date_column = next((str(column) for column in df.columns
                        if not str(column).startswith("_") and column != target
                        and is_date_like_series(df[column], str(column))), None)
    specs = [("missing_rate", [target], f"Tỷ lệ thiếu dữ liệu của {target} là bao nhiêu?", "mean")]
    if target_is_numeric:
        specs.append(("aggregate_summary", [target], f"Các KPI tổng hợp của {target} là gì?", "sum"))
        specs.append(("distribution_summary", [target], f"Phân phối thống kê của {target} có đặc điểm gì?", "mean"))
        specs.append(("top_n", [target], f"Những giá trị {target} xuất hiện nhiều nhất là gì?", "count"))
        for related in other_numeric[:2]:
            specs.append(("correlation", [target, related], f"{target} có mối liên hệ tuyến tính với {related} như thế nào?", "mean"))
        if other_categorical:
            specs.append(("mean_by_group", [other_categorical[0], target], f"Trung bình {target} thay đổi giữa các nhóm {other_categorical[0]} như thế nào?", "mean"))
            specs.append(("top_n", [other_categorical[0], target], f"Những nhóm {other_categorical[0]} nào có tổng {target} cao nhất?", "sum"))
        if date_column:
            specs.append(("trend_over_time", [date_column, target], f"{target} thay đổi theo thời gian như thế nào?", "mean"))
            specs.append(("trend_over_time", [date_column, target], f"Tổng {target} theo từng kỳ thời gian là bao nhiêu?", "sum"))
    elif target_is_date:
        specs.append(("trend_over_time", [target], f"Số bản ghi theo {target} biến động như thế nào?", "count"))
        specs.append(("count_by_category", [target], f"Số bản ghi tại từng mốc {target} là bao nhiêu?", "count"))
        if numeric:
            specs.append(("trend_over_time", [target, numeric[0]], f"{numeric[0]} thay đổi theo {target} như thế nào?", "mean"))
            specs.append(("trend_over_time", [target, numeric[0]], f"Tổng {numeric[0]} tại từng mốc {target} là bao nhiêu?", "sum"))
        specs.append(("top_n", [target], f"Những mốc {target} có nhiều bản ghi nhất là gì?", "count"))
    else:
        specs.append(("count_by_category", [target], f"Số bản ghi trong từng nhóm {target} là bao nhiêu?", "count"))
        specs.append(("top_n", [target], f"Các nhóm {target} xuất hiện nhiều nhất là gì?", "count"))
        if numeric:
            specs.append(("mean_by_group", [target, numeric[0]], f"Trung bình {numeric[0]} theo {target} là bao nhiêu?", "mean"))
            specs.append(("top_n", [target, numeric[0]], f"Những nhóm {target} có tổng {numeric[0]} cao nhất là gì?", "sum"))
        elif other_categorical:
            specs.append(("top_n", [target], f"10 nhóm {target} phổ biến nhất là gì?", "count"))
    return [FramedQuestion(question_id=f"candidate_{index}", operation=operation,
                           columns=columns, question=text, aggregation=aggregation)
            for index, (operation, columns, text, aggregation) in enumerate(specs[:8], 1)]


def _validated_questions(questions: List[FramedQuestion], df: pd.DataFrame,
                         target_column: str | None = None) -> List[FramedQuestion]:
    valid: List[FramedQuestion] = []
    seen = set()
    for question in questions[:8]:
        try:
            if target_column is not None and target_column not in question.columns:
                raise ValueError(f"Question must include target column {target_column!r}")
            _validate_question_wording(question)
            OPERATIONS[question.operation](df, question)
            signature = (question.operation, tuple(question.columns), question.aggregation, question.n)
            if signature in seen:
                continue
            seen.add(signature)
            valid.append(question)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Rejected framed question %s: %s", question.question_id, exc)
    if target_column is None:
        return valid if len(valid) >= 4 else []
    other_columns_exist = any(not str(column).startswith("_") and str(column) != target_column
                              for column in df.columns)
    has_relationship = any(len(question.columns) >= 2 for question in valid)
    minimum = 4 if len(df.columns) > 1 else 2
    if len(valid) < minimum or (other_columns_exist and not has_relationship):
        return []
    return valid


def _validated_workbook_questions(questions: List[FramedQuestion],
                                  partitions: Dict[str, pd.DataFrame], minimum: int = 15) -> List[FramedQuestion]:
    valid: List[FramedQuestion] = []
    seen = set()
    only_partition = next(iter(partitions)) if len(partitions) == 1 else None
    for question in questions[:20]:
        partition = question.source_partition or only_partition
        if partition not in partitions:
            logger.warning("Rejected question %s: invalid source_partition %r",
                           question.question_id, partition)
            continue
        question.source_partition = partition
        try:
            _validate_question_wording(question)
            OPERATIONS[question.operation](partitions[partition], question)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Rejected workbook question %s: %s", question.question_id, exc)
            continue
        signature_columns = (tuple(sorted(question.columns)) if question.operation == "correlation"
                             else tuple(question.columns))
        signature = (partition, question.operation, signature_columns,
                     question.aggregation, question.n)
        if signature in seen:
            continue
        seen.add(signature)
        valid.append(question)
    return valid if len(valid) >= minimum else []


def _validate_question_wording(question: FramedQuestion) -> None:
    """Reject labels that claim a metric or time grain the operation does not calculate."""
    wording = question.question.casefold()
    is_count = (
        question.operation == "count_by_category"
        or (question.operation == "top_n" and question.aggregation == "count")
    )
    if is_count and any(token in wording for token in ("tỷ lệ", "ti le", "phần trăm", "percent", "%")):
        raise ValueError("Phép đếm không được mô tả là tỷ lệ/phần trăm.")
    if question.operation == "trend_over_time" and any(
        token in wording for token in ("theo từng ngày", "theo ngày", "mỗi ngày", "hàng ngày", "daily")
    ):
        raise ValueError("trend_over_time được tổng hợp theo tháng, không phải theo ngày.")


def _fallback_workbook_questions(partitions: Dict[str, pd.DataFrame]) -> List[FramedQuestion]:
    """Rank deterministic candidates across the whole file, with no per-sheet quota."""
    weighted = []
    priority = {
        "correlation": 6, "trend_over_time": 6, "mean_by_group": 5,
        "top_n": 4, "aggregate_summary": 4, "distribution_summary": 3, "count_by_category": 2,
        "missing_rate": 1,
    }
    seen = set()
    for partition, df in partitions.items():
        for column in df.columns:
            target = str(column)
            if target.startswith("_") or is_person_name_column(column, df[column]):
                continue
            if infer_column_semantic(df[column], target) in {"identifier", "text", "unknown"}:
                continue
            for question in _fallback_questions_for_column(df, target):
                question.source_partition = partition
                signature_columns = (tuple(sorted(question.columns)) if question.operation == "correlation"
                                     else tuple(question.columns))
                signature = (partition, question.operation, signature_columns,
                             question.aggregation, question.n)
                if signature in seen:
                    continue
                seen.add(signature)
                relationship_bonus = 2 if len(question.columns) >= 2 else 0
                weighted.append((priority[question.operation] + relationship_bonus, question))
    weighted.sort(key=lambda item: item[0], reverse=True)
    return [question for _, question in weighted[:20]]


def frame_questions(state: GraphState) -> GraphState:
    profile = state.get("dataframe_profile")
    if not profile:
        state["status"] = "error"
        state["error_message"] = "Cannot frame questions without a dataframe profile."
        return state
    try:
        partitions = split_partitions_by_source(read_dataset_partitions(
            state["file_path"], state.get("workbook_sheets"), state.get("instructions", "")
        ))
    except Exception as exc:
        state["status"] = "error"
        state["error_message"] = f"Cannot execute framed questions: {exc}"
        return state
    parser = JsonOutputParser(pydantic_object=FramedQuestionsOutput)
    prompt = PromptTemplate(template="""
Bạn là chuyên gia phân tích số liệu đồng thời là người viết báo cáo chuyên nghiệp. Hãy xem toàn bộ
hồ sơ của file, nhận diện lĩnh vực từ bằng chứng trong dữ liệu và tạo đúng 20 câu hỏi ứng viên cho
một báo cáo duy nhất. Sau đó điền `selected_question_ids` bằng ID duy nhất của 5-10 câu hỏi quan
trọng nhất trong 20 ứng viên.

Đánh giá và lựa chọn câu hỏi theo thứ tự ưu tiên:
1. Trả lời trực tiếp mục tiêu và yêu cầu của người dùng.
2. Phù hợp với lĩnh vực được chứng minh bởi tên cột và instruction, không tự gán bối cảnh.
3. Làm rõ kết quả trọng yếu, xu hướng, bất thường hoặc mối liên hệ có giá trị hành động.
4. Có thể được trả lời chính xác bằng một operation pandas trong danh sách đóng.
5. Bổ sung cho nhau và tạo thành mạch báo cáo; không chọn hai câu chỉ khác cách diễn đạt.
6. Ưu tiên câu quan hệ/xu hướng có ý nghĩa hơn thống kê mô tả đơn giản. Chỉ ưu tiên missing_rate
   khi mức độ thiếu thực sự quan trọng đối với độ tin cậy của báo cáo.

Không chia quota theo sheet hoặc theo cột. 20 câu ứng viên phải đủ rộng để cân nhắc, nhưng 5-10 câu
được chọn phải là một bộ phân tích cô đọng, phù hợp nhất cho báo cáo cuối cùng.

Chỉ được chọn operation trong danh sách sau:
mean_by_group, count_by_category, trend_over_time, top_n, missing_rate, correlation,
distribution_summary, aggregate_summary.
Chỉ dùng tên cột chính xác có trong hồ sơ. Không viết mã pandas/Python và không bịa cột.
Mỗi câu hỏi phải đặt `source_partition` bằng đúng tên phần dữ liệu chứa các cột được chọn.
Không kết hợp cột thuộc hai source_partition khác nhau trong cùng một phép tính.
Chỉ dùng cột `semantic_type=numeric_measure` cho SUM/MEAN/correlation. Không dùng identifier,
số điện thoại, mã giao dịch, mã khách hàng, text hoặc unknown như một measurement, kể cả khi
pandas lưu chúng dưới kiểu số. Không chọn phép tính nếu valid_count, unique_count, variance,
period_count hoặc số cặp hợp lệ không đủ.
Thứ tự cột: mean_by_group=[nhóm,giá_trị]; count_by_category=[phân_loại];
trend_over_time=[thời_gian,giá_trị] (hoặc [thời_gian] khi đếm);
top_n=[phân_loại] hoặc [phân_loại,giá_trị]; missing_rate=[cột];
correlation=[cột_số_x,cột_số_y]; distribution_summary=[cột_số]; aggregate_summary=[cột_số].
Với trend_over_time/top_n, aggregation chỉ được là sum, mean hoặc count.
`count_by_category` và `top_n` với aggregation=count chỉ tạo SỐ LƯỢNG; câu hỏi không được gọi
kết quả đó là "tỷ lệ", "phần trăm" hoặc dùng ký hiệu `%`.
Câu hỏi thời gian phải dùng đúng cột thời gian và đúng độ phân giải mà operation hỗ trợ;
`trend_over_time` hiện tổng hợp theo THÁNG, vì vậy không được viết "theo ngày", "từng ngày"
hoặc khẳng định một độ phân giải khác. Không tự đổi ngày, tháng, năm, dịch kỳ hoặc tạo mốc
thời gian không có trong dữ liệu.
Câu hỏi phải dùng thuật ngữ chuyên nghiệp phù hợp với lĩnh vực và bám sát mục tiêu người dùng.
Tuyệt đối không chọn cột có `is_sensitive_person_name=true`; không đếm, xếp hạng,
phân nhóm hoặc đưa tên người vào câu hỏi và kết quả.

Hồ sơ toàn bộ file theo source_partition: {profiles}
Yêu cầu người dùng: {instructions}
{format_instructions}
Chỉ trả về JSON hợp lệ.
""", input_variables=["profiles", "instructions"], partial_variables={"format_instructions": parser.get_format_instructions()})
    combined_instructions = effective_instructions(
        state.get("instructions", ""), state.get("workbook_instruction_context")
    )
    profiles = state.get("sheet_profiles") or {}
    all_questions: List[FramedQuestion] = []
    all_results: List[ComputedQuestionResult] = []
    try:
        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
    except Exception as exc:
        state["status"] = "error"
        state["error_message"] = f"Không thể khởi tạo LLM để tạo câu hỏi: {exc}"
        return state
    profile_context = {}
    for partition_name, df in partitions.items():
        raw_profile = profiles.get(partition_name)
        partition_profile = DataProfile(**raw_profile) if raw_profile else DataProfile(
            num_rows=len(df), num_columns=len(df.columns), column_details={}, key_observations=""
        )
        profile_context[partition_name] = {
            "num_rows": partition_profile.num_rows,
            "column_details": safe_column_details(partition_profile.column_details),
        }
    try:
        response = text_from_response(llm.invoke(prompt.invoke({
            "profiles": json.dumps(profile_context, ensure_ascii=False, indent=2),
            "instructions": combined_instructions,
        }), config={"request_options": {"timeout": 60}}))
        parsed = FramedQuestionsOutput.model_validate_json(
            response.strip().removeprefix("```json").removesuffix("```").strip()
        )
        valid_candidates = _validated_workbook_questions(
            parsed.questions, partitions, minimum=0
        )
        selected_ids = set(parsed.selected_question_ids)
        selected = [question for question in valid_candidates
                    if question.question_id in selected_ids]
        if len(selected) < 5:
            selected_signatures = {question.question_id for question in selected}
            selected.extend(question for question in valid_candidates
                            if question.question_id not in selected_signatures)
        all_questions = selected[:10] if len(selected) >= 5 else []
    except Exception as exc:
        logger.warning("Tạo bộ câu hỏi cấp file thất bại; dùng fallback: %s", exc)
    if not all_questions:
        fallback_questions = _fallback_workbook_questions(partitions)
        fallback_questions = fallback_questions[:10]
        fallback_minimum = min(5, len(fallback_questions))
        all_questions = _validated_workbook_questions(
            fallback_questions, partitions, minimum=fallback_minimum
        ) if fallback_minimum else []




    covered_partitions = {question.source_partition for question in all_questions}
    operation_priority = {
        "trend_over_time": 0, "count_by_category": 1, "aggregate_summary": 2,
        "distribution_summary": 3,
        "mean_by_group": 3, "top_n": 4, "missing_rate": 5, "correlation": 6,
    }
    for partition, frame in partitions.items():
        if partition in covered_partitions:
            continue
        candidates = []
        for column in frame.columns:
            target = str(column)
            if target.startswith("_") or is_person_name_column(column, frame[column]):
                continue
            candidates.extend(_fallback_questions_for_column(frame, target))
        candidates.sort(key=lambda item: operation_priority.get(item.operation, 99))
        for candidate in candidates:
            candidate.source_partition = partition
            if _validated_workbook_questions([candidate], partitions, minimum=1):
                all_questions.append(candidate)
                covered_partitions.add(partition)
                break
    if not all_questions:
        state["status"] = "error"
        state["error_message"] = "Không đủ dữ liệu để đánh giá."
        return state
    for question in all_questions:
        question.question_id = f"question_{len(all_results) + 1}"
        computed = execute_question(partitions[question.source_partition], question)
        computed.parameters["target_column"] = question.columns[0]
        all_results.append(computed)
    if not all_results:
        state["status"] = "error"
        state["error_message"] = "Không đủ dữ liệu để đánh giá."
        return state
    state["framed_questions"] = all_questions
    state["computed_question_results"] = all_results
    state["status"] = "questions_computed"
    return state
