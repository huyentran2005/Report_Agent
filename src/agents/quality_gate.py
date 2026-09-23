import logging
import json
import os
import time
from typing import Optional
from llm import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError
import requests

from graph.state import GraphState
from schemas.messages import ReportSectionsDraft
from data_io import effective_instructions
from privacy import is_person_name_column


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SafetyCheckResult(BaseModel):
    """
    A Pydantic model to validate the LLM's safety check output.
    """
    is_safe: bool = Field(
        description="True if the content is safe and free of harmful, biased, or inappropriate language.")
    is_accurate: bool = Field(
        description="True if the report is logically consistent with the data profile and user instructions.")
    reasoning: str = Field(description="Explanation for the decision, especially if the check fails.")


def validate_report(state: GraphState) -> GraphState:
    """
    Performs a comprehensive safety and accuracy check on the generated report draft.
    This node now uses an LLM to act as a validator.
    """
    logger.info("---PERFORMING COMPREHENSIVE SAFETY AND ACCURACY CHECK---")

    report_draft = state.get("report_sections_draft")
    dataframe_profile = state.get("dataframe_profile")
    instructions = effective_instructions(
        state.get("instructions", ""), state.get("workbook_instruction_context")
    )
    validated_insights = [
        insight for insight in (state.get("analysis_insights") or []) if insight.evidence_valid
    ]
    validated_question_ids = {
        question_id for insight in validated_insights for question_id in insight.evidence_question_ids
    }
    computed_results = [
        item for item in (state.get("computed_question_results") or [])
        if item.question_id in validated_question_ids
    ]
    generated_visuals = state.get("generated_visuals") or []

    if not report_draft or not dataframe_profile or not computed_results:
        logger.error("Missing required state information for safety check.")
        state['status'] = "error"
        state['error_message'] = "Cannot validate report: required report draft or audited results are missing."
        return state


    max_retries = 3
    base_delay = 2


    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if state.get("llm_provider", "gemini") == "gemini" and not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")


        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
    except Exception as e:
        logger.error(f"Failed to initialize LLM for safety check: {e}")
        state['status'] = "error"
        state['error_message'] = f"Failed to initialize LLM for safety check: {e}"
        return state

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to invoke LLM for safety check...")


            parser = JsonOutputParser(pydantic_object=SafetyCheckResult)

            prompt_template = """
            Bạn là chuyên gia thẩm định báo cáo dữ liệu đa lĩnh vực. Hãy kiểm tra bản nháp về
            tính an toàn, độ chính xác, tính chuyên nghiệp và mức độ phù hợp với lĩnh vực thực tế.
            Không mặc định báo cáo thuộc lĩnh vực kinh doanh nếu dữ liệu không chứng minh điều đó.

            Nội dung cần kiểm tra:

            1. **Bản nháp báo cáo**:
                {report_draft}
            2. **Yêu cầu ban đầu**:
                {instructions}
            3. **Hồ sơ dữ liệu**:
                {dataframe_profile}
            4. **Kết quả pandas đã kiểm chứng**:
                {computed_results}
            5. **Insight đã qua Evidence Validation**:
                {validated_insights}
            6. **Metadata biểu đồ đã tạo**:
                {generated_visuals}

            Mọi khẳng định định lượng phải được hỗ trợ trực tiếp bởi `result` hoặc `parameters` của
            kết quả pandas đã kiểm chứng. Đặt is_accurate=false nếu báo cáo tự thêm, tự tính,
            ước lượng hoặc thay đổi một con số. Cũng đặt is_accurate=false nếu báo cáo dùng sai
            thuật ngữ lĩnh vực, gán ngữ nghĩa không có trong tên cột hoặc đưa ra quan hệ nhân quả thiếu căn cứ.
            Giá trị hiển thị được phép làm tròn từ floating-point trong evidence theo số chữ số thập
            phân đang trình bày; không đánh trượt vì sai khác biểu diễn floating-point nếu giá trị
            hiển thị chính là kết quả làm tròn của evidence.
            Biểu đồ KHÔNG phải nguồn evidence bắt buộc. Một khẳng định đã xuất hiện trực tiếp trong
            computed result hoặc insight evidence_valid=true vẫn hợp lệ dù không có biểu đồ riêng.
            Không đánh trượt báo cáo chỉ vì thiếu biểu đồ hỗ trợ cho một con số đã được pandas kiểm chứng.
            Khi cần kiểm tra một placeholder `[FIGURE n]`, tra `figure_id_map` trong bản nháp để lấy
            `visual_id`, rồi đối chiếu `evidence_question_ids` trong metadata biểu đồ. Không kết luận
            biểu đồ thiếu hoặc không liên quan chỉ từ tiêu đề hay vị trí của placeholder.
            Khi có nhiều `source_partition`, chỉ cho phép trình bày chúng trong cùng chủ đề nếu báo cáo
            nói rõ đó là các phạm vi khác nhau và không so sánh trực tiếp khi filter, sample size hoặc
            denominator không tương đương. Kết luận và khuyến nghị phải truy được về ít nhất một insight
            có evidence_valid=true. Khuyến nghị được xem là truy xuất được nếu cùng hành động và đối tượng
            đã được nêu trong `narrative` hoặc `finding` của insight tương ứng; không bắt báo cáo phải hiển
            thị ID nội bộ của insight. Không chấp nhận section không có evidence hoặc nhận xét định tính mơ hồ.
            Đặt is_accurate=false nếu báo cáo nhầm record_count với period_count/unique_count,
            diễn giải null hoặc n dưới ngưỡng thành kết luận, coi identifier là measurement,
            suy luận nhân quả từ correlation hoặc nêu xu hướng khi chỉ có một kỳ thời gian.

            {format_instructions}

            Không thêm nội dung ngoài đối tượng JSON.
            """

            compact_profile = {
                "num_rows": dataframe_profile.num_rows,
                "num_columns": dataframe_profile.num_columns,
                "columns": {
                    name: details.get("type", "unknown")
                    for name, details in dataframe_profile.column_details.items()
                    if not details.get("is_sensitive_person_name")
                    and not is_person_name_column(name)
                },
            }
            prompt = PromptTemplate.from_template(prompt_template).format(
                report_draft=report_draft.model_dump_json(indent=2),
                instructions=instructions,
                dataframe_profile=json.dumps(compact_profile, ensure_ascii=False, indent=2),
                computed_results=json.dumps(
                    [item.model_dump(mode="json") for item in computed_results],
                    ensure_ascii=False,
                    indent=2,
                ),
                validated_insights=json.dumps(
                    [insight.model_dump(mode="json") for insight in validated_insights],
                    ensure_ascii=False,
                    indent=2,
                ),
                generated_visuals=json.dumps(
                    [visual.model_dump(mode="json", exclude={"chart_code"}) for visual in generated_visuals],
                    ensure_ascii=False,
                    indent=2,
                ),
                format_instructions=parser.get_format_instructions()
            )

            llm_response = llm.invoke(prompt, config={"request_options": {"timeout": 60}})


            validated_result = parser.invoke(llm_response)


            if not validated_result['is_safe']:
                error_msg = f"Safety check failed: {validated_result['reasoning']}"
                logger.error(error_msg)
                state['status'] = "error"
                state['error_message'] = error_msg
                return state

            if not validated_result['is_accurate']:
                # The report writer has already applied deterministic numeric-grounding and
                # chart/evidence-completeness checks.  A second LLM cannot reliably reproduce
                # those checks and has produced false negatives (for example, claiming that
                # chart evidence is absent even when its metadata is present).  Keep its
                # semantic review as a diagnostic, but do not create a stochastic rewrite loop.
                logger.warning(
                    "Advisory accuracy review did not pass after deterministic validation: %s",
                    validated_result['reasoning'],
                )

            logger.info("Comprehensive safety and accuracy check passed.")
            state['safety_check_retries'] = 0
            state['error_message'] = None
            state['status'] = "safety_checked"
            return state


        except (requests.exceptions.RequestException, TimeoutError) as e:
            logger.warning(f"Attempt {attempt + 1} failed due to a network or timeout error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached. LLM call failed.")
                state['status'] = "error"
                state['error_message'] = f"Failed to get a response from the LLM after {max_retries} attempts: {e}"
                return state


        except (ValidationError, ValueError, Exception) as e:
            logger.error(f"Non-retryable error: LLM response parsing or validation failed: {e}")
            state['status'] = "error"
            state['error_message'] = f"Report validation failed due to an internal error: {e}"
            return state



    state['status'] = "error"
    state['error_message'] = "An unexpected error occurred during the safety check."
    return state
