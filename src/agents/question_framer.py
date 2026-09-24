"""Let the LLM frame questions and compile them into validated analysis plans."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

from agents.analysis_plan_executor import execute_analysis_plan
from agents.pandas_sandbox import execute_pandas
from data_io import (effective_instructions, infer_column_semantic, read_dataset_partitions,
                     split_partitions_by_source, to_datetime_series)
from graph.state import GraphState
from llm import get_llm, text_from_response
from privacy import is_person_name_column, safe_column_details
from schemas.messages import (AnalysisCoverageMap, ComputedQuestionResult, DataProfile,
                              ColumnAnalysisRole, FramedQuestion, QuestionDataScope,
                              StructuredAnalysisPlan)

logger = logging.getLogger(__name__)
MAX_PLAN_REVISIONS = 2


class FramedQuestionsOutput(BaseModel):
    questions: List[FramedQuestion] = Field(min_length=1)
    selected_question_ids: List[str] = Field(
        min_length=1,
        description="Toàn bộ question_id trong questions; không có quota hoặc giới hạn số lượng cố định.",
    )

    @model_validator(mode="after")
    def validate_selected_questions(self):
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Mỗi câu hỏi phải có question_id duy nhất.")
        if len(self.selected_question_ids) != len(set(self.selected_question_ids)):
            raise ValueError("selected_question_ids không được chứa ID trùng lặp.")
        unknown = set(self.selected_question_ids) - set(question_ids)
        if unknown:
            raise ValueError(f"selected_question_ids chứa ID không tồn tại: {sorted(unknown)}")
        omitted = set(question_ids) - set(self.selected_question_ids)
        if omitted:
            raise ValueError(
                "Mọi câu hỏi đã tạo phải được chọn để thực thi; "
                f"selected_question_ids còn thiếu: {sorted(omitted)}"
            )
        return self


class RepairedPlanOutput(BaseModel):
    plan: StructuredAnalysisPlan
    question: str | None = None
    scope: QuestionDataScope | None = None
    explanation: str = ""


class CoverageSupplementOutput(BaseModel):
    questions: List[FramedQuestion] = Field(min_length=1)


class PandasPlannerAction(BaseModel):
    action: Literal["inspect_schema", "sample_rows", "execute_pandas", "finish"]
    code: str | None = None
    reasoning: str = ""


MAX_PANDAS_ACTIONS = 8


def _clean_json_response(response: str) -> str:
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _question_columns(question: FramedQuestion) -> list[str]:
    plan = question.plan
    columns = [*plan.group_by, *plan.metrics]
    columns.extend(item.column for item in plan.filters if item.column)
    for transform in plan.transforms:
        columns.extend(value for value in (transform.column, transform.numerator, transform.denominator) if value)
    return list(dict.fromkeys(columns))


def _sync_question_plan(question: FramedQuestion) -> None:
    question.source_partition = question.plan.source_partition
    question.columns = _question_columns(question)
    question.operation = question.analysis_type or "structured_plan"


def _filter_columns(question: FramedQuestion) -> set[str]:
    return {item.column for item in question.plan.filters}


def _instruction_authorizes_filter(filter_spec, instructions: str) -> bool:
    """Require concrete evidence in the user instruction for a requested subset."""
    text = (instructions or "").casefold()
    values = filter_spec.value if isinstance(filter_spec.value, list) else [filter_spec.value]
    tokens = [str(value).casefold() for value in values if value is not None]
    years = set()
    for value in values:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            year = parsed.year
            if filter_spec.operator == "lt" and parsed.month == 1 and parsed.day == 1:
                year -= 1
            years.add(str(year))
        else:
            years.update(re.findall(r"\b(?:19|20)\d{2}\b", str(value)))
    return (
        str(filter_spec.column).casefold() in text
        or any(token and token in text for token in tokens)
        or bool(years and years.issubset(set(re.findall(r"\b(?:19|20)\d{2}\b", text))))
    )


def _validate_question_scope(df, question: FramedQuestion, instructions: str) -> None:
    """Validate the declared scope against the executable plan and real partition."""
    scope = question.scope
    expected_partition = question.plan.source_partition
    if scope.source_partitions != [expected_partition]:
        raise ValueError(
            f"scope.source_partitions phải bằng [{expected_partition!r}], nhận {scope.source_partitions}."
        )
    unknown = (set(scope.time_columns) | set(scope.dimension_columns)) - set(df.columns)
    if unknown:
        raise ValueError(f"Scope dùng cột không tồn tại: {sorted(unknown)}")

    time_columns = {
        str(column) for column in df.columns
        if infer_column_semantic(df[column], str(column)) == "datetime"
    }
    declared = set(scope.time_columns) | set(scope.dimension_columns)
    filter_columns = _filter_columns(question)
    undeclared = filter_columns - declared
    if undeclared:
        raise ValueError(f"Filter chưa được khai báo trong scope: {sorted(undeclared)}")
    wrong_time = set(scope.time_columns) - time_columns
    if wrong_time:
        raise ValueError(f"scope.time_columns chứa cột không phải thời gian: {sorted(wrong_time)}")

    if scope.coverage_mode == "full_dataset" and question.plan.filters:
        raise ValueError("scope=full_dataset không được chứa filter làm hẹp dữ liệu.")
    if scope.coverage_mode != "full_dataset" and not question.plan.filters:
        raise ValueError(f"scope={scope.coverage_mode} phải có filter tương ứng trong plan.")
    if scope.coverage_mode == "user_requested_subset":
        unauthorized = [item.column for item in question.plan.filters
                        if not _instruction_authorizes_filter(item, instructions)]
        if unauthorized:
            raise ValueError(
                f"Không tìm thấy yêu cầu người dùng cho subset trên cột: {sorted(set(unauthorized))}"
            )
    if scope.coverage_mode == "analytical_subset":
        if not question.depends_on_question_ids:
            raise ValueError("analytical_subset chỉ hợp lệ cho câu hỏi drill-down có dependency.")
        if len(scope.rationale.strip()) < 12:
            raise ValueError("analytical_subset phải có rationale phân tích rõ ràng.")


def _order_questions(questions: List[FramedQuestion]) -> List[FramedQuestion]:
    """Stable topological order; reject unknown dependencies and dependency cycles."""
    by_id = {question.question_id: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("question_id phải duy nhất trước khi sắp xếp phụ thuộc.")
    unknown = {
        dependency for question in questions for dependency in question.depends_on_question_ids
        if dependency not in by_id
    }
    if unknown:
        raise ValueError(f"Câu hỏi phụ thuộc vào ID không tồn tại: {sorted(unknown)}")
    ordered: List[FramedQuestion] = []
    pending = list(questions)
    completed: set[str] = set()
    while pending:
        ready = [question for question in pending
                 if set(question.depends_on_question_ids).issubset(completed)]
        if not ready:
            cycle = [question.question_id for question in pending]
            raise ValueError(f"Phát hiện vòng lặp phụ thuộc giữa các câu hỏi: {cycle}")
        for question in ready:
            ordered.append(question)
            completed.add(question.question_id)
            pending.remove(question)
    return ordered


def _remove_spurious_dependencies(questions: List[FramedQuestion]) -> None:
    """Keep dependencies only for drill-down questions within a shared analytical theme."""
    by_id = {question.question_id: question for question in questions}
    for question in questions:
        retained = []
        for dependency_id in question.depends_on_question_ids:
            dependency = by_id.get(dependency_id)
            if dependency and set(question.theme_ids) & set(dependency.theme_ids):
                retained.append(dependency_id)
            else:
                logger.warning(
                    "Bỏ dependency không có quan hệ theme: %s -> %s",
                    question.question_id, dependency_id,
                )
        question.depends_on_question_ids = retained


def _print_question_outcomes(answered, unanswered) -> None:
    print("\n=== CÂU HỎI ĐÃ ĐƯỢC TRẢ LỜI ===")
    if not answered:
        print("(không có)")
    for question in answered:
        print(
            f"[LLM] [ĐÃ TRẢ LỜI] {question.question_id}: {question.question} "
            f"| themes={question.theme_ids} "
            f"| depends_on={question.depends_on_question_ids or 'none'} "
            f"| plan={question.operation} | columns={question.columns} | source={question.source_partition}"
        )
    print("\n=== CÂU HỎI KHÔNG ĐƯỢC TRẢ LỜI ===")
    if not unanswered:
        print("(không có)")
    for question, reason in unanswered:
        print(
            f"[LLM] [KHÔNG TRẢ LỜI] {question.question_id}: {question.question} "
            f"| plan={question.operation} | lý do={reason}"
        )
    print("=== KẾT THÚC DANH SÁCH CÂU HỎI ===\n")


def compact_computed_results(results, max_items: int = 20) -> List[Dict[str, Any]]:
    """Bound LLM context while preserving complete computed results in graph state."""
    compacted: List[Dict[str, Any]] = []
    for item in results:
        if any(is_person_name_column(column) for column in item.columns):
            logger.warning("Omitting result based on a personal-name column: %s", item.columns)
            continue
        payload = item.model_dump(mode="json")
        value = payload.get("result")
        if isinstance(value, list) and len(value) > max_items:
            payload["result"] = value[:max_items]
            payload["context_note"] = "Kết quả rút gọn cho LLM; bản đầy đủ vẫn được lưu trong state."
        compacted.append(payload)
    return compacted


def _repair_plan(llm, question: FramedQuestion, error: Exception,
                 profile_context: dict[str, Any]) -> RepairedPlanOutput:
    parser = JsonOutputParser(pydantic_object=RepairedPlanOutput)
    prompt = PromptTemplate(
        template="""
Bạn là Query Planner. Analysis plan JSON dưới đây không chạy được bằng Generic Pandas Executor.
Quan sát lỗi và sửa cả `question` lẫn `plan` để chúng mô tả đúng cùng một phạm vi. Chỉ dùng cột trong hồ sơ, không sinh Python,
không đổi source_partition tùy tiện và không viết nội dung ngoài JSON.
Không tự đổi ý nghĩa metric theo suy đoán. Nếu hồ sơ của partition không có cột mang vai trò thời gian,
phải loại bỏ mọi tham chiếu thời gian và không được tạo cột giả.
Với filter thời gian, dùng mốc ISO chính xác. Một năm phải dùng khoảng nửa mở
gte YYYY-01-01 và lt (YYYY+1)-01-01 để không loại sai bản ghi có thành phần giờ.

Câu hỏi: {question}
Scope hiện tại: {scope}
Plan lỗi: {plan}
Lỗi executor/validator: {error}
Hồ sơ dữ liệu: {profile}
{format_instructions}
""",
        input_variables=["question", "scope", "plan", "error", "profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    response = text_from_response(llm.invoke(prompt.invoke({
        "question": question.question,
        "scope": question.scope.model_dump_json(),
        "plan": question.plan.model_dump_json(exclude_none=True),
        "error": str(error),
        "profile": json.dumps(profile_context, ensure_ascii=False),
    }), config={"request_options": {"timeout": 60}}))
    return RepairedPlanOutput.model_validate_json(_clean_json_response(response))


def _refine_dependent_plan(llm, question: FramedQuestion,
                           completed_results: dict[str, ComputedQuestionResult],
                           profile_context: dict[str, Any]) -> None:
    """Re-plan a dependent question only after its prerequisite results are available."""
    if not question.depends_on_question_ids:
        return
    parser = JsonOutputParser(pydantic_object=RepairedPlanOutput)
    evidence = {
        question_id: completed_results[question_id].model_dump(mode="json")
        for question_id in question.depends_on_question_ids
    }
    prompt = PromptTemplate(
        template="""
Bạn là Query Planner theo chuỗi. Câu hỏi hiện tại phụ thuộc vào các kết quả tiền đề bên dưới.
Hãy quan sát chính các kết quả đó và tinh chỉnh Structured Analysis Plan để đào sâu phát hiện liên
quan. Giữ nguyên mục tiêu câu hỏi, chỉ dùng cột trong hồ sơ, không sinh Python và không lặp nguyên
phép tính tiền đề. Nếu plan hiện tại đã đúng thì trả lại nguyên plan.

Câu hỏi phụ thuộc: {question}
Plan hiện tại: {plan}
Kết quả tiền đề: {evidence}
Hồ sơ dữ liệu: {profile}
{format_instructions}
Chỉ trả về JSON hợp lệ.
""",
        input_variables=["question", "plan", "evidence", "profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    response = text_from_response(llm.invoke(prompt.invoke({
        "question": question.question,
        "plan": question.plan.model_dump_json(exclude_none=True),
        "evidence": json.dumps(evidence, ensure_ascii=False, default=str),
        "profile": json.dumps(profile_context, ensure_ascii=False),
    }), config={"request_options": {"timeout": 60}}))
    repaired = RepairedPlanOutput.model_validate_json(_clean_json_response(response))
    question.plan = repaired.plan
    if repaired.scope:
        question.scope = repaired.scope
    if repaired.question:
        question.question = repaired.question


def _validate_time_scope(df, question: FramedQuestion, instructions: str) -> None:
    """Reject LLM-invented time windows that exclude available periods."""
    date_filters = []
    for filter_spec in question.plan.filters:
        if filter_spec.column in df.columns and infer_column_semantic(
            df[filter_spec.column], filter_spec.column
        ) == "datetime":
            date_filters.append(filter_spec)
    if not date_filters:
        return
    instruction_text = (instructions or "").casefold()
    authorized_years = set(re.findall(r"\b(?:19|20)\d{2}\b", instruction_text))
    relative_scope_terms = (
        "gần nhất", "mới nhất", "latest", "most recent", "ytd", "year to date",
        "từ ngày", "đến ngày", "giai đoạn", "khoảng thời gian",
    )
    filter_years = set()
    for item in date_filters:
        values = item.value if isinstance(item.value, list) else [item.value]
        for value in values:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                year = parsed.year - 1 if item.operator == "lt" and parsed.month == 1 and parsed.day == 1 else parsed.year
                filter_years.add(str(year))
            else:
                filter_years.update(re.findall(r"\b(?:19|20)\d{2}\b", str(value)))
    if filter_years and filter_years.issubset(authorized_years):
        return
    if any(term in instruction_text for term in relative_scope_terms):
        return
    date_column = date_filters[0].column
    dates = to_datetime_series(df[date_column]).dropna()
    available = (
        f"{dates.min().isoformat()} đến {dates.max().isoformat()}"
        if not dates.empty else "không xác định"
    )
    raise ValueError(
        f"Không được tự giới hạn thời gian bằng filter {[(item.operator, item.value) for item in date_filters]}; "
        f"người dùng không yêu cầu phạm vi này và dữ liệu {date_column!r} có phạm vi {available}. "
        "Hãy bỏ filter thời gian và sửa câu hỏi để phân tích toàn bộ phạm vi dữ liệu."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_invalid_mean_transform(cls, value):
        """LLM đôi khi đặt ``mean`` vào transforms thay vì aggregation.

        ``mean`` không phải phép biến đổi hậu xử lý; nó là một aggregation hợp
        lệ của StructuredAnalysisPlan. Loại bỏ transform sai này để plan được
        validate và executor dùng aggregation đã khai báo.
        """
        if not isinstance(value, dict):
            return value
        questions = value.get("questions") or []
        for question in questions:
            plan = question.get("plan") if isinstance(question, dict) else None
            if not isinstance(plan, dict):
                continue
            transforms = plan.get("transforms") or []
            plan["transforms"] = [
                transform for transform in transforms
                if not (isinstance(transform, dict) and transform.get("type") == "mean")
            ]
            if any(isinstance(transform, dict) and transform.get("type") == "mean"
                   for transform in transforms):
                plan.setdefault("aggregation", "mean")
        return value


def _execute_question_with_actions(llm, df: pd.DataFrame,
                                   question: FramedQuestion) -> ComputedQuestionResult:
    """Let the planner inspect, execute Pandas, observe, and finish iteratively."""
    parser = JsonOutputParser(pydantic_object=PandasPlannerAction)
    history: list[dict[str, Any]] = []
    latest_result: Any = None
    latest_code: str | None = None
    schema = [
        {"name": str(column), "dtype": str(df[column].dtype),
         "semantic_type": infer_column_semantic(df[column], str(column))}
        for column in df.columns if not str(column).startswith("_")
    ]
    prompt = PromptTemplate(
        template="""
Bạn là Pandas Planner Agent. Hãy giải quyết yêu cầu phân tích bằng vòng lặp action-observation.
Mỗi lượt chỉ trả về một action JSON. Có bốn action:
- inspect_schema: xem schema đã kiểm chứng;
- sample_rows: xem tối đa 5 dòng mẫu;
- execute_pandas: chạy code Pandas đọc dữ liệu; code bắt buộc dùng dataframe `df`, không import,
  không đọc/ghi file, không gọi mạng và phải gán kết quả cuối vào biến `result`;
- finish: chỉ dùng sau khi đã có kết quả đủ trả lời câu hỏi.

Không dùng tên cột ngoài schema. Không tự gán ý nghĩa nghiệp vụ hoặc phạm vi không có trong yêu cầu.
Quan sát lỗi của action trước và sửa code ở lượt sau. Không trả Markdown hoặc nội dung ngoài JSON.

Câu hỏi: {question}
Schema: {schema}
Kế hoạch định hướng ban đầu: {plan}
Lịch sử action-observation: {history}
Đã có kết quả thực thi: {has_result}
{format_instructions}
""",
        input_variables=["question", "schema", "plan", "history", "has_result"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    for _ in range(MAX_PANDAS_ACTIONS):
        raw = text_from_response(llm.invoke(prompt.invoke({
            "question": question.question,
            "schema": json.dumps(schema, ensure_ascii=False),
            "plan": question.plan.model_dump_json(exclude_none=True),
            "history": json.dumps(history, ensure_ascii=False, default=str),
            "has_result": latest_result is not None,
        }), config={"request_options": {"timeout": 60}}))
        action = PandasPlannerAction.model_validate_json(_clean_json_response(raw))
        if action.action == "inspect_schema":
            observation = {"schema": schema, "row_count": len(df)}
        elif action.action == "sample_rows":
            observation = {"rows": df.head(5).astype(object).where(pd.notna(df.head(5)), None).to_dict("records")}
        elif action.action == "execute_pandas":
            if not action.code:
                observation = {"error": "execute_pandas yêu cầu trường code."}
            else:
                try:
                    latest_result = execute_pandas(action.code, df)
                    latest_code = action.code
                    observation = {"result": latest_result}
                except Exception as exc:
                    observation = {"error": str(exc)}
        else:
            if latest_result is None or not latest_code:
                observation = {"error": "Chưa có kết quả execute_pandas để hoàn tất."}
            else:
                return ComputedQuestionResult(
                    question_id=question.question_id, question=question.question,
                    operation=question.operation, columns=question.columns,
                    parameters={
                        "analysis_type": question.analysis_type,
                        "expected_result": question.expected_result,
                        "visualization": question.visualization,
                        "theme_ids": question.theme_ids,
                        "data_scope": question.scope.model_dump(mode="json"),
                        "analysis_plan": question.plan.model_dump(mode="json", exclude_none=True),
                        "pandas_action": latest_code,
                    },
                    result=latest_result, calculation=latest_code,
                    source_partition=question.source_partition,
                )
        history.append({"action": action.model_dump(mode="json"), "observation": observation})
    if latest_result is not None and latest_code:
        logger.warning(
            "Pandas Planner reached %s actions without finish; auto-finalizing the latest result.",
            MAX_PANDAS_ACTIONS,
        )
        return ComputedQuestionResult(
            question_id=question.question_id, question=question.question,
            operation=question.operation, columns=question.columns,
            parameters={
                "analysis_type": question.analysis_type,
                "expected_result": question.expected_result,
                "visualization": question.visualization,
                "theme_ids": question.theme_ids,
                "data_scope": question.scope.model_dump(mode="json"),
                "analysis_plan": question.plan.model_dump(mode="json", exclude_none=True),
                "pandas_action": latest_code,
            },
            result=latest_result, calculation=latest_code,
            source_partition=question.source_partition,
        )
    try:
        fallback_result, fallback_code, _ = execute_analysis_plan(df, question.plan)
        return ComputedQuestionResult(
            question_id=question.question_id, question=question.question,
            operation=question.operation, columns=question.columns,
            parameters={
                "analysis_type": question.analysis_type,
                "expected_result": question.expected_result,
                "visualization": question.visualization,
                "theme_ids": question.theme_ids,
                "data_scope": question.scope.model_dump(mode="json"),
                "analysis_plan": question.plan.model_dump(mode="json", exclude_none=True),
                "pandas_action": fallback_code,
                "planner_fallback": True,
            },
            result=fallback_result, calculation=fallback_code,
            source_partition=question.source_partition,
        )
    except Exception as fallback_exc:
        raise ValueError(
            f"Pandas Planner vượt quá {MAX_PANDAS_ACTIONS} action và plan dự phòng cũng lỗi: {fallback_exc}"
        ) from fallback_exc


def _execute_with_revision(llm, question: FramedQuestion, partitions,
                           profile_context: dict[str, Any], instructions: str) -> ComputedQuestionResult:
    last_error: Exception | None = None
    for attempt in range(MAX_PLAN_REVISIONS + 1):
        _sync_question_plan(question)
        try:
            if question.source_partition not in partitions:
                raise ValueError(f"source_partition không hợp lệ: {question.source_partition!r}")
            _validate_question_scope(partitions[question.source_partition], question, instructions)
            _validate_time_scope(partitions[question.source_partition], question, instructions)
            partition_df = partitions[question.source_partition]
            logger.info(
                "Question %s executing ACTIVE Pandas Planner action loop (partition=%s).",
                question.question_id, question.source_partition,
            )
            return _execute_question_with_actions(llm, partition_df, question)
        except (ValueError, TypeError, KeyError) as exc:
            last_error = exc
            if attempt >= MAX_PLAN_REVISIONS:
                break
            logger.warning("Plan %s lỗi lần %s; yêu cầu LLM sửa: %s", question.question_id, attempt + 1, exc)
            repaired = _repair_plan(llm, question, exc, profile_context)
            question.plan = repaired.plan
            if repaired.scope:
                question.scope = repaired.scope
            if repaired.question:
                question.question = repaired.question
    raise ValueError(f"Plan vẫn lỗi sau {MAX_PLAN_REVISIONS} lần sửa: {last_error}")


def _plan_signature(question: FramedQuestion) -> str:
    return question.plan.model_dump_json(exclude_none=True)


def _split_independent_group_dimensions(questions: List[FramedQuestion]) -> List[FramedQuestion]:
    """Turn a multi-category cross-tab into one independent analysis per dimension."""
    expanded: List[FramedQuestion] = []
    used_ids = {question.question_id for question in questions}
    dependency_expansions: dict[str, list[str]] = {}
    for question in questions:
        dimensions = list(dict.fromkeys(question.plan.group_by))
        if len(dimensions) <= 1 or question.plan.time_grain is not None:
            expanded.append(question)
            continue
        split_ids = []
        metrics_label = " và ".join(question.plan.metrics) or "Số lượng"
        for index, dimension in enumerate(dimensions, start=1):
            split = question.model_copy(deep=True)
            if index > 1:
                base_id = f"{question.question_id}_{index}"
                split.question_id = base_id
                suffix = 1
                while split.question_id in used_ids:
                    split.question_id = f"{base_id}_{suffix}"
                    suffix += 1
                used_ids.add(split.question_id)
            split.plan.group_by = [dimension]
            split.question = f"Hiệu suất {metrics_label} theo {dimension}?"
            filter_columns = {item.column for item in split.plan.filters}
            split.scope.dimension_columns = [
                column for column in split.scope.dimension_columns
                if column == dimension or column in filter_columns
            ]
            _sync_question_plan(split)
            split_ids.append(split.question_id)
            expanded.append(split)
        dependency_expansions[question.question_id] = split_ids
        logger.info("Tách %s thành các phân tích độc lập theo dimensions: %s",
                    question.question_id, dimensions)
    for question in expanded:
        question.depends_on_question_ids = list(dict.fromkeys(
            dependency_id
            for original_id in question.depends_on_question_ids
            for dependency_id in dependency_expansions.get(original_id, [original_id])
        ))
    return expanded


def _build_coverage_map(llm, profile_context: dict[str, Any], instructions: str) -> AnalysisCoverageMap:
    parser = JsonOutputParser(pydantic_object=AnalysisCoverageMap)
    compact_profiles: dict[str, Any] = {}
    detail_keys = {
        "type", "semantic_type", "dtype", "unique_values_count",
        "missing_values_count", "min", "max", "mean", "std",
        "top_5_values", "years", "is_sensitive_person_name",
    }
    for partition, profile in profile_context.items():
        details = profile.get("column_details") or {}
        compact_details = {
            column: {key: value for key, value in (detail or {}).items() if key in detail_keys}
            for column, detail in details.items()
        }
        scope = profile.get("data_scope")
        if isinstance(scope, dict):
            scope = dict(scope)
            dimensions = []
            for dimension in scope.get("dimensions") or []:
                item = dict(dimension)
                if isinstance(item.get("values"), list):
                    item["values"] = item["values"][:10]
                if isinstance(item.get("periods"), list):
                    item["periods"] = item["periods"][:24]
                dimensions.append(item)
            scope["dimensions"] = dimensions
        compact_profiles[partition] = {
            "num_rows": profile.get("num_rows"),
            "column_details": compact_details,
            "data_scope": scope,
        }
    prompt = PromptTemplate(
        template="""
Bạn là **Data Semantics Analyst**. Nhiệm vụ là xây dựng **Analysis Coverage Map** cho toàn bộ dữ liệu trước khi sinh bất kỳ câu hỏi phân tích nào.

Dữ liệu có thể gồm một hoặc nhiều file, sheet hoặc bảng. Mỗi nguồn độc lập được xem là một `source_partition`. Chỉ được suy luận từ **tên cột, dtype, thống kê dữ liệu, cấu trúc partition và yêu cầu người dùng**. Không tự suy diễn lĩnh vực, ý nghĩa nghiệp vụ hoặc quan hệ không có bằng chứng.

Không gán ý nghĩa nghiệp vụ cụ thể cho metric nếu tên cột, tên nguồn hoặc yêu cầu người dùng không cung cấp bằng chứng trực tiếp. Nếu partition không có cột được phân loại `time`, tuyệt đối không tạo theme hoặc phạm vi thời gian và không tự sinh cột thời gian.

`instructions` là yêu cầu phân tích do người dùng nhập. Hãy bóc tách yêu cầu này thành các quan hệ phân tích cần coverage; giữ nguyên mục tiêu, metric và dimension mà người dùng nêu, nhưng chỉ ánh xạ chúng vào những cột thực sự tồn tại và có bằng chứng từ hồ sơ. Không tự thêm mục tiêu ngoài yêu cầu khi dữ liệu không hỗ trợ.

### 1. Xác định source_partition

- Xác định toàn bộ partition xuất hiện trong `{profiles}` và giữ nguyên tên gốc.
- Nếu chỉ có một nguồn, chỉ tạo một `source_partition`.
- Nếu có nhiều nguồn, xử lý từng partition độc lập.
- Không mặc định hai cột cùng tên ở hai partition biểu diễn cùng một thực thể.
- Không kết hợp dữ liệu giữa các partition nếu hồ sơ không cung cấp bằng chứng rõ ràng về khả năng join.

### 2. Phân loại cột

Trong từng partition, gán mỗi cột vào đúng một vai trò:

`time`, `measure`, `outcome`, `driver`, `category`, `geography`, `operation`, `identifier`, `free_text`, `sensitive`, `exclude`.

Quy tắc:

- `identifier`: ID, mã bản ghi hoặc khóa kỹ thuật; không dùng làm dimension nếu gần như unique.
- `category`: giá trị có tính phân nhóm/so sánh và có lặp lại.
- Cardinality cao không phải lý do để loại một cột khỏi `category`.
- `free_text`: chỉ dành cho văn bản tự do, mô tả, ghi chú hoặc nội dung phi cấu trúc; không dùng chỉ vì cột có nhiều giá trị.
- `sensitive`: dữ liệu định danh cá nhân hoặc nhạy cảm; không dùng làm dimension nếu người dùng không yêu cầu rõ ràng.
- `exclude`: cột hằng, gần như trống, metadata kỹ thuật hoặc không có giá trị phân tích.
- Không dùng `exclude` như phương án mặc định khi khó xác định vai trò.

### 3. Xây dựng analytical themes

Từ các cột hợp lệ, tạo bộ **analytical themes nhỏ nhất nhưng đủ bao phủ các quan hệ phân tích quan trọng**.

Ưu tiên yêu cầu phân tích trong `instructions`: nếu người dùng đã nêu các mục tiêu cụ thể, chỉ tạo
theme cho các mục tiêu đó và các dimension được nêu hoặc bắt buộc để trả lời chúng. Không tự mở rộng
thành danh sách phân tích phổ biến của lĩnh vực. Mỗi theme chỉ có **một dimension phân tích chính**
(một cột time/category/geography/operation/driver) và một hoặc nhiều metric liên quan; không tạo
theme có tiêu đề hoặc columns gộp nhiều dimension độc lập. Nếu một yêu cầu có nhiều dimension, tạo
một theme riêng cho từng dimension ngay từ đầu, không tạo theme lai rồi chờ bước sau tách.

Mỗi theme phải:

- thuộc đúng một `source_partition`;
- biểu diễn một quan hệ phân tích nguyên tử;
- sử dụng một dimension hoặc một nhóm dimension thực sự gắn kết;
- có thể liên kết với một hoặc nhiều metric liên quan trực tiếp;
- chỉ sử dụng tên cột tồn tại trong partition đó.

**Tách theme khi dimension khác nhau về bản chất.**

Không:

- tạo theme quá rộng không chỉ rõ dimension và metric;
- tạo theme cho từng cột một cách máy móc;
- tạo nhiều theme biểu diễn cùng một quan hệ;
- tạo theme chỉ để tăng coverage.

Ngược lại, nhiều metric có thể cùng thuộc một theme nếu chúng cùng đánh giá một dimension. Không gộp các dimension khác bản chất chỉ vì chúng sử dụng chung metric.

Nếu dimension có cardinality cao nhưng vẫn có ý nghĩa phân tích, giữ theme đó. Việc giới hạn **Top N** được xử lý ở bước phân tích/visualization sau này.

### 4. Xác định required

Đặt `required=true` khi theme thỏa ít nhất một điều kiện:

- trực tiếp phục vụ `{instructions}`;
- hoặc là quan hệ tối thiểu bắt buộc để trả lời một yêu cầu trực tiếp trong `{instructions}`.

Không đặt `required=true` chỉ vì cột tồn tại, có thể tạo biểu đồ, có nhiều metric, hoặc có khả năng
tạo insight chung chung. Nếu không có yêu cầu trực tiếp và quan hệ không đủ bằng chứng, đặt
`required=false` hoặc không tạo theme.

Coverage phải được đánh giá trên **toàn bộ các source_partition**, không được chỉ tập trung vào partition lớn nhất hoặc đầu tiên.

### 5. Ràng buộc partition

Mỗi theme chỉ thuộc một `source_partition`, chỉ dùng cột của partition đó, giữ chính xác tên partition và tên cột gốc, không dịch/viết tắt/tự tạo tên cột, và không cross-partition nếu chưa có bằng chứng rõ ràng cho phép join.

`column_details` và `data_scope.dimensions[].column` là danh sách tên cột hợp lệ duy nhất. Nội dung trong `top_5_values`, `values` hoặc `periods` là **giá trị dữ liệu**, không phải tên cột và tuyệt đối không được đưa vào `column_roles` hoặc `theme.columns`. Nếu hồ sơ không có dimension phù hợp, không được tạo theme bằng cách biến giá trị mẫu thành cột.

Nếu nhiều partition có schema tương tự, chỉ coi chúng là một nguồn phân tích chung khi `{profiles}` xác nhận rõ chúng có thể hợp nhất. Nếu không, giữ riêng.

### 6. Mục tiêu của Coverage Map

Coverage Map chỉ trả lời: **“Những khía cạnh nào của dữ liệu cần được phân tích?”**

Không sinh câu hỏi phân tích, viết Pandas query, thực hiện phép tính, đưa ra insight, đề xuất biểu đồ hoặc kết luận từ dữ liệu.

Ưu tiên **ít theme nhưng đủ coverage**, thay vì tạo nhiều theme nhỏ, trùng lặp hoặc không có giá trị.

### INPUT

Hồ sơ dữ liệu:
{profiles}

Yêu cầu người dùng:
{instructions}

{format_instructions}

### OUTPUT

Chỉ trả về **JSON hợp lệ** đúng schema `AnalysisCoverageMap`. Không Markdown, không giải thích, không thêm nội dung ngoài JSON.
""",
        input_variables=["profiles", "instructions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    compact_profiles_json = json.dumps(compact_profiles, ensure_ascii=False, indent=2, default=str)
    compact_profiles_json = compact_profiles_json[:60000]
    response = text_from_response(llm.invoke(prompt.invoke({
        "profiles": compact_profiles_json,
        "instructions": instructions,
    }), config={"request_options": {"timeout": 60}}))
    return AnalysisCoverageMap.model_validate_json(_clean_json_response(response))


def _validate_coverage_map(coverage_map: AnalysisCoverageMap, partitions) -> None:
    role_keys = {(item.source_partition, item.column) for item in coverage_map.column_roles}
    for partition, frame in partitions.items():
        for column in frame.columns:
            column = str(column)
            key = (partition, column)
            if column.startswith("_") or key in role_keys or is_person_name_column(column, frame[column]):
                continue
            semantic = infer_column_semantic(frame[column], column)
            role = {
                "datetime": "time",
                "numeric_measure": "measure",
                "categorical": "category",
                "boolean/status": "category",
                "identifier": "identifier",
            }.get(semantic, "free_text")
            coverage_map.column_roles.append(ColumnAnalysisRole(
                column=column,
                source_partition=partition,
                role=role,
                include_in_analysis=role not in {"free_text", "identifier"},
                rationale="Bổ sung tự động từ schema/dtype vì LLM không liệt kê cột này.",
            ))
    role_keys = [(item.source_partition, item.column) for item in coverage_map.column_roles]
    if len(role_keys) != len(set(role_keys)):
        raise ValueError("Coverage map phân loại trùng một cột.")
    expected = {
        (partition, str(column)) for partition, frame in partitions.items() for column in frame.columns
        if not str(column).startswith("_") and not is_person_name_column(column, frame[column])
    }
    missing_roles = expected - set(role_keys)
    if missing_roles:
        raise ValueError(f"Coverage map chưa phân loại các cột: {sorted(missing_roles)}")
    for theme in coverage_map.themes:
        if theme.source_partition not in partitions:
            raise ValueError(f"Theme {theme.theme_id} dùng source_partition không tồn tại.")
        unknown = [column for column in theme.columns
                   if column not in partitions[theme.source_partition].columns]
        if unknown:
            raise ValueError(f"Theme {theme.theme_id} dùng cột không tồn tại: {unknown}")


def _split_multidimension_themes(coverage_map: AnalysisCoverageMap) -> None:
    """Ensure independent business dimensions remain independent coverage units."""
    role_by_column = {
        (item.source_partition, item.column): item.role
        for item in coverage_map.column_roles
    }
    dimension_roles = {"time", "category", "geography", "operation", "driver"}
    used_ids = {theme.theme_id for theme in coverage_map.themes}
    normalized = []
    for theme in coverage_map.themes:
        dimensions = [
            column for column in theme.columns
            if role_by_column.get((theme.source_partition, column)) in dimension_roles
        ]
        if len(dimensions) <= 1:
            normalized.append(theme)
            continue
        shared_columns = [column for column in theme.columns if column not in dimensions]
        for index, dimension in enumerate(dimensions, start=1):
            split = theme.model_copy(deep=True)
            if index > 1:
                base_id = f"{theme.theme_id}_{index}"
                split.theme_id = base_id
                suffix = 1
                while split.theme_id in used_ids:
                    split.theme_id = f"{base_id}_{suffix}"
                    suffix += 1
                used_ids.add(split.theme_id)
            split.title = f"{theme.title} theo {dimension}"
            split.columns = [dimension, *shared_columns]
            split.suggested_analyses = [f"Phân tích các metric theo {dimension}"]
            normalized.append(split)
        logger.info("Tách theme %s theo các dimension độc lập: %s", theme.theme_id, dimensions)
    coverage_map.themes = normalized


def _covered_columns_by_theme(coverage_map: AnalysisCoverageMap,
                              questions: List[FramedQuestion]) -> dict[str, set[str]]:
    known = {theme.theme_id: theme for theme in coverage_map.themes}
    role_by_column = {
        (item.source_partition, item.column): item.role
        for item in coverage_map.column_roles
    }
    dimension_roles = {"time", "category", "geography", "operation", "driver"}
    for question in questions:
        unknown = set(question.theme_ids) - set(known)
        if unknown:
            raise ValueError(f"Câu {question.question_id} tham chiếu theme không tồn tại: {sorted(unknown)}")
        mismatched = [theme_id for theme_id in question.theme_ids
                      if known[theme_id].source_partition != question.plan.source_partition]
        if mismatched:
            raise ValueError(
                f"Câu {question.question_id} khác source_partition với theme: {mismatched}"
            )
    columns_by_theme: dict[str, set[str]] = {theme_id: set() for theme_id in known}
    for question in questions:
        plan_columns = set(_question_columns(question))
        for theme_id in question.theme_ids:
            theme = known[theme_id]
            dimension_columns = {
                column for column in theme.columns
                if role_by_column.get((theme.source_partition, column)) in dimension_roles
            }
            target_columns = dimension_columns or set(theme.columns)
            relevant_columns = plan_columns & target_columns
            if not relevant_columns:
                logger.warning(
                    "Question %s khai báo theme %s nhưng plan không dùng cột nào của theme.",
                    question.question_id, theme_id,
                )
                continue
            columns_by_theme[theme_id].update(relevant_columns)
    return columns_by_theme


def _missing_required_themes(coverage_map: AnalysisCoverageMap,
                             questions: List[FramedQuestion]) -> list[str]:
    known = {theme.theme_id: theme for theme in coverage_map.themes}
    required = {theme.theme_id for theme in coverage_map.themes if theme.required}
    columns_by_theme = _covered_columns_by_theme(coverage_map, questions)
    covered = {theme_id for theme_id, columns in columns_by_theme.items() if columns}
    missing = sorted(required - covered)
    if missing:
        logger.warning("Question plan chưa có phân tích thực tế cho themes: %s", missing)
    return missing


def _validate_plan_scope_coverage(questions: List[FramedQuestion], partitions,
                                  instructions: str) -> None:
    """Validate requested partition coverage and only requested temporal coverage."""
    instruction_text = (instructions or "").casefold()
    temporal_request = bool(re.search(
        r"\b(?:time|trend|trend_over_time|temporal|year|month|quarter|week|day|date|growth|"
        r"thời gian|xu hướng|theo năm|theo tháng|theo quý|theo tuần|theo ngày|tăng trưởng)\b",
        instruction_text,
    ))
    mentioned = {
        name for name in partitions if str(name).casefold() in instruction_text
    }
    required_partitions = mentioned or set(partitions)
    covered_partitions = {question.plan.source_partition for question in questions}
    missing = required_partitions - covered_partitions
    if missing:
        raise ValueError(f"Scope Coverage còn thiếu source_partition: {sorted(missing)}")

    if not temporal_request:
        return

    for partition_name in required_partitions:
        frame = partitions[partition_name]
        numeric_columns = [
            str(column) for column in frame.columns
            if infer_column_semantic(frame[column], str(column)) == "numeric_measure"
        ]
        if not numeric_columns:
            continue
        for column in frame.columns:
            name = str(column)
            if infer_column_semantic(frame[column], name) != "datetime":
                continue
            dates = to_datetime_series(frame[column]).dropna()
            if dates.dt.year.nunique() < 2:
                continue
            temporal_questions = [
                question for question in questions
                if question.plan.source_partition == partition_name
                and name in question.plan.group_by
                and question.plan.time_grain is not None
            ]
            if not temporal_questions:
                years = sorted(int(year) for year in dates.dt.year.unique())
                raise ValueError(
                    f"Scope Coverage thiếu phân tích/so sánh thời gian cho {partition_name}.{name} "
                    f"trên các năm {years}."
                )


def _supplement_questions(llm, coverage_map: AnalysisCoverageMap,
                          questions: List[FramedQuestion], missing_theme_ids: list[str],
                          profile_context: dict[str, Any], instructions: str) -> List[FramedQuestion]:
    parser = JsonOutputParser(pydantic_object=CoverageSupplementOutput)
    themes = {theme.theme_id: theme.model_dump(mode="json") for theme in coverage_map.themes}
    prompt = PromptTemplate(
        template="""
Bạn là Question Planner. Kế hoạch hiện tại chưa có câu hỏi phân tích cho các analytical themes bên
dưới. Hãy bổ sung số câu hỏi mà bạn cảm thấy đủ để bao phủ toàn bộ theme còn thiếu. Plan của câu hỏi phải
thực sự sử dụng ít nhất một cột thuộc theme; chỉ ghi theme_id nhưng không dùng cột theme không được
tính là bao phủ. Một câu hỏi được phép bao
phủ nhiều theme liên quan và nhiều metric trong cùng source_partition. Không lặp câu hiện có,
không sinh Python; mỗi câu phải có theme_ids và Structured Analysis Plan thực thi được. Nếu dùng
depends_on_question_ids, chỉ tham chiếu question_id trong danh sách câu hỏi hiện có.
Mỗi câu phải khai báo `scope`: mặc định coverage_mode=full_dataset, source_partitions chứa đúng
source_partition của plan, và không dùng filter. Chỉ dùng user_requested_subset khi filter được yêu
cầu rõ trong yêu cầu người dùng; analytical_subset chỉ dành cho drill-down có dependency và rationale.
Khai báo mọi cột filter trong time_columns hoặc dimension_columns tương ứng.

Các theme còn thiếu: {missing_themes}
Toàn bộ coverage map: {coverage_map}
Câu hỏi hiện có: {questions}
Hồ sơ dữ liệu: {profiles}
Yêu cầu người dùng: {instructions}
{format_instructions}
Chỉ trả về JSON hợp lệ.
""",
        input_variables=["missing_themes", "coverage_map", "questions", "profiles", "instructions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    response = text_from_response(llm.invoke(prompt.invoke({
        "missing_themes": json.dumps({key: themes[key] for key in missing_theme_ids}, ensure_ascii=False),
        "coverage_map": coverage_map.model_dump_json(),
        "questions": json.dumps([item.model_dump(mode="json") for item in questions], ensure_ascii=False),
        "profiles": json.dumps(profile_context, ensure_ascii=False),
        "instructions": instructions,
    }), config={"request_options": {"timeout": 60}}))
    return CoverageSupplementOutput.model_validate_json(_clean_json_response(response)).questions


def frame_questions(state: GraphState) -> GraphState:
    if not state.get("dataframe_profile"):
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

    profiles = state.get("sheet_profiles") or {}
    profile_context: dict[str, Any] = {}
    for partition_name, df in partitions.items():
        raw_profile = profiles.get(partition_name)
        partition_profile = DataProfile(**raw_profile) if raw_profile else DataProfile(
            num_rows=len(df), num_columns=len(df.columns), column_details={}, key_observations=""
        )
        profile_context[partition_name] = {
            "num_rows": partition_profile.num_rows,
            "column_details": safe_column_details(partition_profile.column_details),
            "data_scope": (
                partition_profile.data_scope.model_dump(mode="json")
                if partition_profile.data_scope else
                (state.get("data_scope_profiles") or {}).get(partition_name)
            ),
        }

    combined_instructions = effective_instructions(
        state.get("instructions", ""), state.get("workbook_instruction_context")
    )
    try:
        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
        coverage_map = _build_coverage_map(llm, profile_context, combined_instructions)
        _split_multidimension_themes(coverage_map)
        _validate_coverage_map(coverage_map, partitions)
        state["analysis_coverage_map"] = coverage_map
        print("\n=== ANALYSIS COVERAGE MAP ===")
        for theme in coverage_map.themes:
            print(
                f"[LLM] [THEME] {theme.theme_id}: {theme.title} | required={theme.required} "
                f"| source={theme.source_partition} | columns={theme.columns}"
            )
        print("=== KẾT THÚC COVERAGE MAP ===\n")
    except Exception as exc:
        logger.error("LLM không tạo được Analysis Coverage Map hợp lệ: %s", exc)
        state["status"] = "error"
        state["error_message"] = f"LLM không tạo được Analysis Coverage Map hợp lệ: {exc}"
        return state

    parser = JsonOutputParser(pydantic_object=FramedQuestionsOutput)
    prompt = PromptTemplate(
        template="""
Bạn là Question Planner. Dựa trên Analysis Coverage Map đã được lập trước, hãy tạo bộ câu hỏi nhỏ
nhất nhưng bao phủ tất cả theme có required=true. Không đặt câu hỏi theo từng cột máy móc. Một câu
hỏi có thể bao phủ nhiều theme liên quan và nhiều metric trong cùng source_partition, chẳng hạn so
sánh đồng thời hai chỉ số kết quả theo một dimension. Mỗi câu bắt buộc khai báo `theme_ids`.
`instructions` là yêu cầu phân tích chính của người dùng. Hãy bóc tách yêu cầu thành các câu hỏi nhỏ,
mỗi câu trả lời một quan hệ phân tích nguyên tử hoặc một nhóm metric dùng chung một dimension. Giữ
đúng phạm vi, metric, dimension và điều kiện mà người dùng yêu cầu; không gom các dimension độc lập
vào một câu và không tự thêm mục tiêu không có trong yêu cầu. Nếu yêu cầu không nêu mục tiêu phân tích
cụ thể, chỉ tạo các câu hỏi tối thiểu cần thiết để bao phủ các theme được đánh dấu `required`.
Chỉ sử dụng ý nghĩa đã được Coverage Map và hồ sơ chứng minh; không tự gán ý nghĩa nghiệp vụ cho metric.
Nếu partition không có cột mang vai trò `time` thì không được tạo phân tích, phạm vi, filter hoặc cột thời gian.
Chỉ gộp nhiều METRIC khi chúng dùng cùng một dimension. Mỗi câu hỏi/plan chỉ dùng một dimension phân
tích chính; các dimension phân tích độc lập phải thành câu hỏi và plan riêng, không đưa nhiều dimension
độc lập vào cùng `group_by` để tạo phân tích tổ hợp.
Không được bỏ một dimension có ý nghĩa chỉ vì cardinality cao; dùng `sort` và `limit` khi cần tạo kết quả dễ đọc.
Không tạo câu trùng ý. Sắp câu hỏi theo mạch từ tổng quan đến chi tiết. Chỉ đặt
`depends_on_question_ids` cho câu đào sâu trong CÙNG THEME thực sự cần quan sát kết quả câu trước;
không nối tuần tự các theme độc lập. Dependency phải tham chiếu question_id tồn tại và không tạo vòng.
Mỗi câu bắt buộc khai báo `scope` có cấu trúc. Mặc định dùng coverage_mode=full_dataset,
source_partitions=[source_partition của plan], không có filter, và khai báo các time_columns /
dimension_columns liên quan. Chỉ dùng user_requested_subset nếu chính người dùng yêu cầu rõ subset;
chỉ dùng analytical_subset cho drill-down có dependency và rationale cụ thể. Mọi cột filter phải
được khai báo trong scope; không tự giới hạn bất kỳ dimension, partition hoặc nhóm dữ liệu nào.
Phạm vi thời gian là một phần của tính đúng đắn: đọc `min_date`, `max_date`, `years`, `periods` trong data_scope.
Nếu người dùng không yêu cầu một giai đoạn cụ thể, câu hỏi và plan phải dùng toàn bộ phạm vi có
trong dữ liệu, tuyệt đối không tự chọn riêng một năm/tháng. Nếu có filter thời gian, câu hỏi phải
nêu đúng phạm vi đó và không được bỏ qua các kỳ khác ngoài yêu cầu người dùng.
Nếu có từ hai năm trở lên và có metric phù hợp, phải có phân tích theo thời gian bao phủ các năm để
thể hiện và so sánh thay đổi giữa các năm.

Với mỗi câu hỏi, không sinh Python và không chọn operation hardcode. Sinh `plan` JSON cho Generic
Pandas Executor bằng cách kết hợp:
- filters: eq/ne/gt/gte/lt/lte/in/not_in/contains/between/is_null/not_null;
- group_by, metrics, aggregation: sum/mean/count/median/min/max/std/nunique;
- time_grain: day/week/month/quarter/year;
- transforms tuần tự: pct_change, difference, cumulative, share_of_total, ratio, rank,
  rolling_mean, correlation, distribution, outlier_iqr, round;
- sort và limit cho top/bottom N.

Mỗi plan chỉ dùng một source_partition và tên cột chính xác trong hồ sơ. Chỉ dùng numeric_measure
cho tổng hợp số/correlation. Không dùng identifier, số điện thoại, mã giao dịch, tên người, text
hoặc unknown làm measurement. Không chọn phép tính khi dữ liệu hợp lệ không đủ. Đặt analysis_type
là nhãn ngắn mô tả đúng phép phân tích, visualization nếu hữu ích, và
Chỉ đưa vào `questions` những câu hỏi thực sự phục vụ báo cáo và bắt buộc
`selected_question_ids` chứa toàn bộ question_id trong `questions`.
Với thời gian, chỉ dùng kỳ thực sự có trong hồ sơ hoặc được người dùng yêu cầu; không tự bịa năm.
Filter trọn một năm phải dùng hai điều kiện gte YYYY-01-01 và lt (YYYY+1)-01-01, không dùng so
sánh chuỗi tùy ý và không dùng lte YYYY-12-31 vì có thể bỏ sót timestamp cuối ngày.

Analysis Coverage Map: {coverage_map}
Hồ sơ theo source_partition: {profiles}
Yêu cầu người dùng: {instructions}
{format_instructions}
Chỉ trả về JSON hợp lệ.
""",
        input_variables=["coverage_map", "profiles", "instructions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        response = text_from_response(llm.invoke(prompt.invoke({
            "coverage_map": coverage_map.model_dump_json(),
            "profiles": json.dumps(profile_context, ensure_ascii=False, indent=2),
            "instructions": combined_instructions,
        }), config={"request_options": {"timeout": 60}}))
        parsed = FramedQuestionsOutput.model_validate_json(_clean_json_response(response))
    except Exception as exc:
        logger.error("LLM không tạo được bộ câu hỏi/plan hợp lệ: %s", exc)
        state["status"] = "error"
        state["error_message"] = f"LLM không tạo được bộ câu hỏi/plan hợp lệ: {exc}"
        return state

    selected_ids = set(parsed.selected_question_ids)
    selected = [question for question in parsed.questions if question.question_id in selected_ids]
    selected = _split_independent_group_dimensions(selected)
    seen_plans = {_plan_signature(question) for question in selected}
    known_question_ids = {question.question_id for question in parsed.questions}
    known_question_ids.update(question.question_id for question in selected)
    try:
        missing_themes = _missing_required_themes(coverage_map, selected)
        while missing_themes:
            logger.warning("Question plan còn thiếu analytical themes: %s", missing_themes)
            missing_before = len(missing_themes)
            supplements = _supplement_questions(
                llm, coverage_map, selected, missing_themes,
                profile_context, combined_instructions
            )
            supplements = _split_independent_group_dimensions(supplements)
            new_questions = []
            for question in supplements:
                if _plan_signature(question) in seen_plans:
                    continue
                base_id = question.question_id
                suffix = 1
                while question.question_id in known_question_ids:
                    question.question_id = f"{base_id}_{suffix}"
                    suffix += 1
                known_question_ids.add(question.question_id)
                seen_plans.add(_plan_signature(question))
                new_questions.append(question)
            if not new_questions:
                raise ValueError(
                    f"Planner chỉ tạo plan trùng khi còn thiếu themes: {missing_themes}"
                )
            selected.extend(new_questions)
            remaining = _missing_required_themes(coverage_map, selected)
            missing_after = len(remaining)
            if missing_after >= missing_before:
                raise ValueError(
                    "Câu hỏi bổ sung không tạo phân tích thực tế cho themes được yêu cầu: "
                    f"{missing_themes}"
                )
            missing_themes = remaining
        _remove_spurious_dependencies(selected)
        selected = _order_questions(selected)
        _validate_plan_scope_coverage(selected, partitions, combined_instructions)
    except ValueError as exc:
        state["status"] = "error"
        state["error_message"] = f"Question plan không bao phủ Coverage Map: {exc}"
        return state
    unanswered = [(question, "Question Planner không chọn vào kế hoạch bao phủ theme cuối cùng")
                  for question in parsed.questions if question.question_id not in selected_ids]
    results: List[ComputedQuestionResult] = []
    answered: List[FramedQuestion] = []
    answered_ids: set[str] = set()
    results_by_id: dict[str, ComputedQuestionResult] = {}
    selected_failures: list[tuple[FramedQuestion, str]] = []
    for question in selected:
        missing_dependencies = set(question.depends_on_question_ids) - answered_ids
        if missing_dependencies:
            reason = f"Câu hỏi tiền đề chưa trả lời thành công: {sorted(missing_dependencies)}"
            unanswered.append((question, reason))
            selected_failures.append((question, reason))
            continue
        try:
            _refine_dependent_plan(llm, question, results_by_id, profile_context)
            computed = _execute_with_revision(
                llm, question, partitions, profile_context, combined_instructions
            )
            results.append(computed)
            answered.append(question)
            answered_ids.add(question.question_id)
            results_by_id[question.question_id] = computed
        except Exception as exc:
            logger.warning("Không thể trả lời %s: %s", question.question_id, exc)
            reason = str(exc)
            unanswered.append((question, reason))
            selected_failures.append((question, reason))

    _print_question_outcomes(answered, unanswered)
    if selected_failures:
        failures = "; ".join(
            f"{question.question_id}: {reason}"
            for question, reason in selected_failures
        )
        state["status"] = "error"
        state["error_message"] = (
            f"Không thể tạo báo cáo đầy đủ: chỉ thực thi thành công {len(answered)}/{len(selected)} "
            f"câu hỏi đã chọn. Các câu hỏi thất bại: {failures}"
        )
        return state
    if not results:
        state["status"] = "error"
        state["error_message"] = "LLM không tạo/sửa được analysis plan có thể thực thi."
        return state
    state["framed_questions"] = answered
    state["computed_question_results"] = results
    state["status"] = "questions_computed"
    return state
