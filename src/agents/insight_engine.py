import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, List

import requests
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError

from agents.question_framer import compact_computed_results
from data_io import effective_instructions
from graph.state import GraphState
from llm import get_llm, text_from_response
from schemas.messages import AnalysisInsight
from privacy import is_person_name_column

logger = logging.getLogger(__name__)


class GeneratedInsightsOutput(BaseModel):
    insights: List[AnalysisInsight] = Field(description="Các phát hiện từ kết quả pandas đã kiểm chứng.")


def _results_by_partition(results, analysis_partitions=None) -> Dict[str, list]:
    """Keep every LLM request scoped to one source sheet/partition."""
    default_partition = (analysis_partitions or ["Dữ liệu"])[0]
    grouped: Dict[str, list] = OrderedDict()
    for item in results:
        grouped.setdefault(item.source_partition or default_partition, []).append(item)
    return grouped


def _insight_batches(grouped: Dict[str, list]):
    """Bound insight prompts by target column after per-column question expansion."""
    for partition, results in grouped.items():
        batches: Dict[str, list] = OrderedDict()
        for index, item in enumerate(results):
            target = str(item.parameters.get("target_column") or f"batch_{index // 8}")
            batches.setdefault(target, []).append(item)
        for batch in batches.values():
            yield partition, batch


def _profile_summary(state: GraphState, partition: str) -> str:
    details = ((state.get("sheet_profiles") or {}).get(partition) or {}).get("column_details")
    if details is None:
        profile = state.get("dataframe_profile")
        details = profile.column_details if profile else {}
    lines = [f"Sheet/phần dữ liệu: {partition}", "Schema (chỉ tên cột và kiểu dữ liệu):"]
    for name, detail in details.items():
        if detail.get("is_sensitive_person_name") or is_person_name_column(name):
            continue
        dtype = str(detail.get("type", "")).lower()
        label = ("numeric" if any(x in dtype for x in ("int", "float", "decimal"))
                 else "datetime" if "date" in dtype
                 else "boolean" if "bool" in dtype else "text")
        lines.append(f"- {name} (Type: {label})")
    return "\n".join(lines)


def _visuals_context(visuals, partition: str) -> str:
    selected = [v for v in (visuals or []) if v.suggested_section == partition
                or v.description.startswith(f"[{partition}]")]
    if not selected:
        return "Không có biểu đồ dành riêng cho phần dữ liệu này."
    return "\n".join(
        ["Các biểu đồ của phần dữ liệu này:"]
        + [f"- ID={v.visual_id}; loại={v.type}; mô tả={v.description}; tệp={v.file_path}"
           for v in selected]
    )


def extract_insights(state: GraphState) -> GraphState:
    """Generate insights in bounded, sheet-scoped LLM calls."""
    request_id = state["request_id"]
    instructions = effective_instructions(
        state.get("instructions", ""), state.get("workbook_instruction_context")
    )
    if not state.get("dataframe_profile") or not state.get("computed_question_results"):
        state["status"] = "error"
        state["error_message"] = "Cannot generate insights: Missing audited pandas results."
        return state
    if state.get("llm_provider", "gemini") == "gemini" and not os.getenv("GEMINI_API_KEY"):
        state["status"] = "error"
        state["error_message"] = "API key for Gemini not found. Please set GEMINI_API_KEY in your .env file."
        return state
    try:
        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
    except Exception as exc:
        state["status"] = "error"
        state["error_message"] = f"Failed to initialize LLM for insight generation: {exc}"
        return state

    parser = JsonOutputParser(pydantic_object=GeneratedInsightsOutput)
    prompt = PromptTemplate(
        template="""
Bạn là chuyên gia phân tích dữ liệu đa lĩnh vực. Yêu cầu này chỉ dành cho sheet/phần dữ
liệu `{partition}`. Nhận diện lĩnh vực từ schema, yêu cầu và tên cột; nếu chưa rõ, dùng
văn phong trung tính. Không trộn hoặc so sánh với sheet khác.

`result` và `parameters` trong KẾT QUẢ PANDAS là nguồn duy nhất cho mọi khẳng định định
lượng. Không tự tính thêm, ước lượng hay đưa vào số không có trong nguồn. Profile và biểu
đồ chỉ là ngữ cảnh định tính.
Không tạo insight nếu kết quả rỗng, null hoặc ghi nhận không đủ dữ liệu. Không diễn giải
correlation null/n=0/n=1 thành "không có mối liên hệ". Không suy luận quan hệ nhân quả từ
tương quan. Với thời gian, chỉ nhận xét xu hướng khi có ít nhất 2 kỳ hợp lệ và nêu giới hạn
khi chỉ có 2 kỳ; ưu tiên kết luận xu hướng khi có từ 3 kỳ.

Mỗi phát hiện gồm insight_id, title, finding, narrative, limitations,
supporting_visual_ids và source_partition.
Luôn đặt source_partition chính xác là `{partition}`. Tạo 1-2 phát hiện khác biệt; nêu bằng
chứng, ý nghĩa và chỉ dẫn biểu đồ thực sự hỗ trợ. `limitations` để [] trừ khi nguồn cho thấy
một giới hạn cụ thể như thiếu dữ liệu, cỡ mẫu/mẫu số không tương đương hoặc quá ít kỳ quan sát.
Narrative theo logic: Phát hiện → Bằng chứng số liệu → Ý nghĩa kinh doanh/chuyên môn → Hành động
đề xuất. Chỉ thêm giới hạn cụ thể khi có bằng chứng. Viết như báo cáo điều hành: câu ngắn,
trực tiếp, ưu tiên động từ,
đặt kết luận quan trọng trước; tránh mở đầu kiểu “dữ liệu cho thấy”, tránh lặp lại title và không
dùng các cụm rỗng như “cần lưu ý”, “đáng quan tâm” nếu không nói rõ người đọc nên làm gì.
Khuyến nghị phải cụ thể nhưng thận trọng: nêu đối tượng cần ưu tiên, chỉ số cần theo dõi hoặc
thử nghiệm cần thực hiện; không biến tương quan thành nguyên nhân.
Không dùng câu cảnh báo mẫu như “cần lưu ý rằng ... có thể ảnh hưởng đến kết quả”.
Mỗi narrative bắt buộc hiển thị value cụ thể từ result. Khi nói về tỷ lệ phải nêu numerator,
denominator/sample size nếu chúng có trong result. Không dùng “phần lớn”, “khá cao”, “đáng kể”.
`title` chỉ mô tả mục đích phân tích hoặc phát hiện; không đưa tên file, tên sheet hay
`source_partition` vào title. Với thời gian, phải giữ nguyên chính xác kỳ/ngày xuất hiện trong
kết quả pandas, không tự đổi mốc, dịch kỳ hoặc suy ra khoảng thời gian mới.

SCHEMA:
{profile_summary}

BIỂU ĐỒ:
{visuals_context}

KẾT QUẢ PANDAS ĐÃ KIỂM CHỨNG:
{computed_facts}

YÊU CẦU NGƯỜI DÙNG VÀ QUY TẮC TỪ WORKBOOK:
{instructions}

{format_instructions}
Chỉ trả về JSON hợp lệ.
""",
        input_variables=["partition", "profile_summary", "visuals_context", "computed_facts", "instructions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    grouped = _results_by_partition(
        state["computed_question_results"], state.get("analysis_partitions")
    )
    generated: List[AnalysisInsight] = []
    for partition, results in _insight_batches(grouped):
        values = {
            "partition": partition,
            "profile_summary": _profile_summary(state, partition),
            "visuals_context": _visuals_context(state.get("generated_visuals"), partition),
            "computed_facts": json.dumps(
                compact_computed_results(results, max_items=8), ensure_ascii=False, indent=2
            ),
            "instructions": instructions,
        }
        parsed = None
        raw = ""
        for attempt in range(3):
            try:
                logger.info("Insight LLM for %r, attempt %s/3 (%s results).",
                            partition, attempt + 1, len(results))
                response = llm.invoke(prompt.invoke(values), config={"request_options": {"timeout": 60}})
                raw = text_from_response(response).strip()
                if raw.startswith("```json") and raw.endswith("```"):
                    raw = raw[len("```json"):-len("```")].strip()
                parsed = GeneratedInsightsOutput.model_validate_json(raw)
                break
            except (requests.exceptions.RequestException, TimeoutError) as exc:
                if attempt == 2:
                    state["status"] = "error"
                    state["error_message"] = f"Không thể tạo insight cho sheet '{partition}' sau 3 lần thử: {exc}"
                    return state
                time.sleep(2 * (2 ** attempt))
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.error("Invalid insight JSON for %r: %s; raw=%s", partition, exc, raw[:500])
                if attempt == 2:
                    state["status"] = "error"
                    state["error_message"] = f"LLM trả JSON insight không hợp lệ cho sheet '{partition}' sau 3 lần thử: {exc}"
                    return state
                continue
            except Exception as exc:
                logger.error("Insight generation failed for %r: %s", partition, exc, exc_info=True)
                state["status"] = "error"
                state["error_message"] = f"Không thể tạo insight cho sheet '{partition}': {exc}"
                return state
        if parsed:
            valid_visual_ids = {
                visual.visual_id for visual in (state.get("generated_visuals") or [])
                if visual.suggested_section == partition
                or visual.description.startswith(f"[{partition}]")
            }
            for insight in parsed.insights:
                insight.source_partition = partition
                insight.insight_id = f"insight_{len(generated) + 1}"
                insight.evidence_question_ids = [item.question_id for item in results]
                insight.question = " | ".join(item.question for item in results)
                insight.finding = insight.finding or insight.title
                insight.source_sheet = partition
                insight.source_columns = list(dict.fromkeys(
                    column for item in results for column in item.columns
                ))
                insight.metrics = {item.question_id: item.result for item in results}
                insight.evidence_valid = False
                insight.supporting_visual_ids = [
                    visual_id for visual_id in insight.supporting_visual_ids
                    if visual_id in valid_visual_ids
                ]
                generated.append(insight)

    if not generated:
        state["status"] = "error"
        state["error_message"] = "LLM không tạo được insight từ kết quả pandas đã kiểm chứng."
        return state
    state["analysis_insights"] = generated
    state["status"] = "insights_generated"
    logger.info("Generated %s insights for request %s.", len(generated), request_id)
    return state
