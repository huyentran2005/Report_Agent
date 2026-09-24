import logging
import json
import os
import time
import requests
import re
from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llm import get_llm, text_from_response
from pydantic import BaseModel, Field, ValidationError
from graph.state import GraphState
from schemas.messages import ReportSectionsDraft
from data_io import effective_instructions
from privacy import is_person_name_column
from agents.report_execution_state import build_report_execution_state

logger = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?")


class UngroundedNumbersError(ValueError):
    pass


class IncompleteReportError(ValueError):
    pass


def _report_completeness_issues(report_draft, report_plan, insights, visuals):
    """Verify that every planned theme has meaningful prose and the correct charts."""
    if not report_plan:
        return []
    issues = []
    narratives = report_draft.analysis_narratives
    if len(narratives) != len(report_plan.themes):
        issues.append(
            f"có {len(narratives)} narrative nhưng REPORT PLAN có {len(report_plan.themes)} theme"
        )
    insight_by_id = {item.insight_id: item for item in insights}
    visual_by_id = {item.visual_id: item for item in visuals or []}
    mapped_ids = []
    generic_titles = {"tiêu đề phát hiện", "phân tích biểu đồ", "finding", "chart analysis"}
    for index, theme in enumerate(report_plan.themes):
        if index >= len(narratives):
            issues.append(f"thiếu narrative cho theme {theme.title!r}")
            continue
        narrative = narratives[index]
        title, separator, body = narrative.partition(":-")
        if not separator or title.strip().casefold() in generic_titles or len(body.strip()) < 60:
            issues.append(f"narrative của theme {theme.title!r} thiếu tiêu đề/nội dung phân tích")
        theme_question_ids = {
            question_id
            for insight_id in theme.insight_ids
            if (insight := insight_by_id.get(insight_id))
            for question_id in insight.evidence_question_ids
        }
        placeholders = re.findall(r"\[FIGURE\s+(\d+)\]", narrative, flags=re.IGNORECASE)
        narrative_visual_ids = {
            report_draft.figure_id_map.get(f"[FIGURE {number}]") for number in placeholders
        } - {None}
        mapped_ids.extend(narrative_visual_ids)
        wrong = [
            visual_id for visual_id in narrative_visual_ids
            if visual_id not in visual_by_id
            or not theme_question_ids.intersection(visual_by_id[visual_id].evidence_question_ids)
        ]
        if wrong:
            issues.append(f"theme {theme.title!r} chứa biểu đồ không cùng evidence: {wrong}")
        expected = {
            visual.visual_id for visual in visuals or []
            if theme_question_ids.intersection(visual.evidence_question_ids)
        }
        missing = expected - narrative_visual_ids
        if missing:
            issues.append(f"theme {theme.title!r} thiếu biểu đồ: {sorted(missing)}")
    duplicate_ids = sorted({item for item in mapped_ids if mapped_ids.count(item) > 1})
    if duplicate_ids:
        issues.append(f"biểu đồ bị gắn vào nhiều narrative: {duplicate_ids}")
    return issues


def _strip_source_names_from_headings(report_draft, file_path, partitions):
    """Remove file/sheet labels and routing separators from every visible heading."""
    source_path = Path(str(file_path))
    labels = [source_path.name, source_path.stem, *(partitions or [])]

    def clean(value):
        result = value
        for label in labels:
            if label:
                result = re.sub(re.escape(str(label)), "", result, flags=re.IGNORECASE)
        result = re.sub(r"\b(?:xlsx?|csv|tsv)\b", "", result, flags=re.IGNORECASE)
        result = result.replace("|", " - ")
        result = re.sub(r"\s{2,}", " ", result)
        return result.strip(" _-:|.")

    report_draft.dataset_title = clean(report_draft.dataset_title) or "Báo cáo chuyên đề"
    normalized = []
    for narrative in report_draft.analysis_narratives:
        title, separator, body = narrative.partition(":-")
        if "|" in title:
            _, finding = title.split("|", 1)
            title = clean(finding) or "Các chỉ báo và mối liên hệ chính"
        else:
            title = clean(title) or "Các chỉ báo và mối liên hệ chính"
        normalized.append(f"{title}:-{body}" if separator else title)
    report_draft.analysis_narratives = normalized
    return report_draft


def _numeric_tokens(value):
    """Return numeric tokens from nested JSON-like values."""
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numeric_tokens(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numeric_tokens(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        return [float(token.replace(",", "")) for token in NUMBER_PATTERN.findall(value)]
    return []


def _displayed_numbers(text):
    """Return (value, decimal places) so audited values may be presented with normal rounding."""
    displayed = []
    for token in NUMBER_PATTERN.findall(text):
        normalized = token.replace(",", "")
        decimal_places = len(normalized.rsplit(".", 1)[1]) if "." in normalized else 0
        displayed.append((float(normalized), decimal_places))
    return displayed


def _matches_audited_value(displayed, decimal_places, source):

    tolerance = 0.5 * (10 ** -decimal_places) + 1e-12
    return abs(displayed - source) <= tolerance


def _unsupported_report_numbers(
    report_draft, computed_results, validated_insights, dataframe_profile
):


    def allowed_numbers(results):
        return [
            number for item in results
            for number in (*_numeric_tokens(item.result), *_numeric_tokens(item.parameters),
                           *_numeric_tokens(item.columns), *_numeric_tokens(item.source_partition))
        ]

    def unsupported_in_text(text, allowed):
        clean_text = re.sub(r"\[FIGURE\s+\d+\]", "", text, flags=re.IGNORECASE)
        return [
            number for number, decimal_places in _displayed_numbers(clean_text)
            if not any(_matches_audited_value(number, decimal_places, source) for source in allowed)
        ]

    allowed = allowed_numbers(computed_results)
    allowed.extend(
        number
        for insight in validated_insights
        for number in _numeric_tokens({
            "metrics": insight.metrics,
            "sample_size": insight.sample_size,
            "denominator": insight.denominator,
            "missing_values": insight.missing_values,
        })
    )
    allowed.extend(_numeric_tokens({
        "num_rows": dataframe_profile.num_rows,
        "num_columns": dataframe_profile.num_columns,
    }))
    global_text = "\n".join([
        report_draft.dataset_title,
        report_draft.introduction_text,
        report_draft.data_quality_text,
        *report_draft.key_takeaways_bullet_points,
        *report_draft.notable_issues,
        report_draft.conclusion_text,
        *report_draft.clarification_questions,
    ])
    unsupported = unsupported_in_text(global_text, allowed)
    for narrative in report_draft.analysis_narratives:
        unsupported.extend(unsupported_in_text(narrative, allowed))
    return unsupported


def _strip_unsupported_report_numbers(report_draft, computed_results,
                                      validated_insights, dataframe_profile) -> list[float]:
    """Remove only ungrounded numeric tokens so one hallucinated number cannot abort export."""
    allowed = [
        number for item in computed_results
        for number in (*_numeric_tokens(item.result), *_numeric_tokens(item.parameters))
    ]
    allowed.extend(
        number for insight in validated_insights
        for number in _numeric_tokens({
            "metrics": insight.metrics, "sample_size": insight.sample_size,
            "denominator": insight.denominator, "missing_values": insight.missing_values,
        })
    )
    allowed.extend(_numeric_tokens({
        "num_rows": dataframe_profile.num_rows, "num_columns": dataframe_profile.num_columns,
    }))

    removed: list[float] = []

    def clean(text: str) -> str:
        def replace(match):
            token = match.group(0)
            value = float(token.replace(",", ""))
            decimals = len(token.rsplit(".", 1)[1]) if "." in token else 0
            if any(_matches_audited_value(value, decimals, source) for source in allowed):
                return token
            removed.append(value)
            return ""
        return NUMBER_PATTERN.sub(replace, text or "")

    report_draft.report_subtitle = clean(report_draft.report_subtitle)
    report_draft.introduction_text = clean(report_draft.introduction_text)
    report_draft.data_quality_text = clean(report_draft.data_quality_text)
    report_draft.analysis_narratives = [clean(text) for text in report_draft.analysis_narratives]
    report_draft.key_takeaways_bullet_points = [clean(text) for text in report_draft.key_takeaways_bullet_points]
    report_draft.notable_issues = [clean(text) for text in report_draft.notable_issues]
    report_draft.conclusion_text = clean(report_draft.conclusion_text)
    report_draft.dataset_title = clean(report_draft.dataset_title)
    return removed


def _remove_generic_caveats(text: str) -> str:
    """Drop boilerplate cautions that do not name a measured data limitation."""
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text or "")
    generic_patterns = (
        r"\bcần lưu ý rằng\b.*\bcó thể ảnh hưởng (?:đến|tới) kết quả\b",
        r"\bsố lượng .+ khác nhau\b.*\bcó thể ảnh hưởng (?:đến|tới) kết quả\b",
        r"\bkết quả (?:này )?có thể bị ảnh hưởng bởi\b(?!.*(?:thiếu|cỡ mẫu|mẫu số|phạm vi))",
    )
    kept = [
        part.strip() for part in parts
        if part.strip() and not any(
            re.search(pattern, part, flags=re.IGNORECASE) for pattern in generic_patterns
        )
    ]
    return " ".join(kept)


def _sanitize_generic_caveats(report_draft):
    """Remove repeated generic caveats from every reader-facing prose field."""
    report_draft.introduction_text = _remove_generic_caveats(report_draft.introduction_text)
    report_draft.data_quality_text = _remove_generic_caveats(report_draft.data_quality_text)
    report_draft.conclusion_text = _remove_generic_caveats(report_draft.conclusion_text)
    report_draft.analysis_narratives = [
        cleaned for narrative in report_draft.analysis_narratives
        if (cleaned := _remove_generic_caveats(narrative))
    ]
    report_draft.key_takeaways_bullet_points = [
        cleaned for item in report_draft.key_takeaways_bullet_points
        if (cleaned := _remove_generic_caveats(item))
    ]
    report_draft.notable_issues = [
        cleaned for item in report_draft.notable_issues
        if (cleaned := _remove_generic_caveats(item))
    ]
    return report_draft


def draft_report(state: GraphState) -> GraphState:
    """
    Generates the initial draft of the report sections (introduction, narratives, takeaways, conclusion)
    based on the data profile, generated insights, and visuals.
    """
    request_id = state['request_id']




    dataset_name = state.get('dataset_name', 'Unnamed Dataset')

    instructions = effective_instructions(
        state.get('instructions', ""), state.get("workbook_instruction_context")
    )
    dataframe_profile = state.get('dataframe_profile', None)
    analysis_insights = [
        insight for insight in (state.get('analysis_insights') or []) if insight.evidence_valid
    ]
    generated_visuals = state.get('generated_visuals', None)
    validated_question_ids = {
        question_id for insight in analysis_insights for question_id in insight.evidence_question_ids
    }
    computed_results = [
        item for item in (state.get('computed_question_results') or [])
        if item.question_id in validated_question_ids
    ]

    if not dataframe_profile or not analysis_insights or not computed_results:
        logger.error(f"Missing data profile, insights, or audited results for request {request_id}.")
        state['status'] = "error"
        state['error_message'] = "Cannot draft report: Missing audited pandas results."
        return state



    logger.info(f"ReportDraftingNode processing request: {request_id}")
    logger.info("Report drafting started with status: %s", state["status"])

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if state.get("llm_provider", "gemini") == "gemini" and not gemini_api_key:
        logger.error(f"GEMINI_API_KEY not found for request {request_id}. Please ensure it's set in your .env file.")
        state['status'] = "error"
        state['error_message'] = "API key for Gemini not found. Please set GEMINI_API_KEY in your .env file."
        return state

    try:
        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
    except Exception as e:
        logger.error(f"Failed to initialize LLM for report drafting: {e}", exc_info=True)
        state['status'] = "error"
        state['error_message'] = f"Failed to initialize LLM for report drafting: {e}"
        return state


    profile_summary = ""
    if dataframe_profile:


        profile_summary = (
            f"Số bản ghi đã kiểm chứng: {dataframe_profile.num_rows}\n"
            f"Số cột đã kiểm chứng: {dataframe_profile.num_columns}\n"
            "Schema dữ liệu (chỉ gồm tên cột và kiểu dữ liệu):\n"
        )
        sheets = state.get("workbook_sheets") or []
        if sheets:
            sheet_schema = [
                {"sheet_name": sheet.get("sheet_name"), "columns": [
                    column for column in sheet.get("columns", [])
                    if not is_person_name_column(column)
                ]}
                for sheet in sheets if str(sheet.get("role", "")).upper() == "DATA"
            ]
            profile_summary += f"Tên sheet và tiêu đề cột trong workbook: {sheet_schema}\n"
        for col_name, details in dataframe_profile.column_details.items():
            if details.get("is_sensitive_person_name") or is_person_name_column(col_name):
                continue
            dtype = str(details.get("type", "")).lower()
            type_label = ("numeric" if any(token in dtype for token in ("int", "float", "decimal"))
                          else "datetime" if "date" in dtype
                          else "boolean" if "bool" in dtype else "text")
            profile_summary += f"- {col_name} (Type: {type_label})\n"

    insights_context = ""
    if analysis_insights:
        insights_context = "\n\nCác phát hiện phân tích:\n"
        for i, insight in enumerate(analysis_insights):
            insights_context += f"Phát hiện {i + 1} (ID: {insight.insight_id}):\n"
            insights_context += f"Tiêu đề: {insight.title}\n"
            insights_context += f"Diễn giải: {insight.narrative}\n"
            compact_metrics = {}
            for question_id, metric_result in (insight.metrics or {}).items():
                if isinstance(metric_result, list) and len(metric_result) > 20:
                    compact_metrics[question_id] = metric_result[:20]
                else:
                    compact_metrics[question_id] = metric_result
            insights_context += (
                "Evidence đã duyệt: "
                f"{json.dumps(compact_metrics, ensure_ascii=False, default=str)}\n"
            )
            insights_context += f"Cỡ mẫu: {insight.sample_size}; Mẫu số: {insight.denominator}\n"
            insights_context += f"Cột nguồn: {insight.source_columns}; Bộ lọc: {insight.filters}\n"
            if insight.limitations:
                insights_context += f"Giới hạn đã kiểm chứng: {insight.limitations}\n"
            if insight.source_partition:
                insights_context += f"Phần dữ liệu nguồn: {insight.source_partition}\n"
            if insight.supporting_visual_ids:
                insights_context += f"Biểu đồ hỗ trợ: {', '.join(insight.supporting_visual_ids)}\n"
            insights_context += "---\n"

    visuals_context = ""
    visual_reference_for_llm = []
    if generated_visuals:
        visuals_context = "\n\nCác biểu đồ đã tạo để tham chiếu trong báo cáo:\n"
        for i, visual in enumerate(generated_visuals):
            visuals_context += f"ID biểu đồ: {visual.visual_id}\n"
            visuals_context += f"Mô tả: {visual.description}\n"
            visuals_context += f"Evidence question IDs: {visual.evidence_question_ids}\n"
            visuals_context += "---\n"
            visual_reference_for_llm.append({
                "visual_id": visual.visual_id,
                "description": visual.description,
                "evidence_question_ids": visual.evidence_question_ids,
            })
    compact_results = []
    for item in computed_results:
        payload = item.model_dump(mode="python")
        value = payload.get("result")
        if isinstance(value, list) and len(value) > 20:
            payload["result"] = value[:20]
            payload["context_note"] = "Kết quả rút gọn cho report LLM; evidence đầy đủ vẫn nằm trong state."
        compact_results.append(payload)
    computed_facts = json.dumps(compact_results, ensure_ascii=False, indent=2, default=str)
    report_plan = state.get("report_plan")
    execution_state = build_report_execution_state(state)
    state["report_execution_state"] = execution_state.model_dump(mode="python")
    report_plan_payload = report_plan.model_dump(mode="python") if report_plan else {}
    for table in report_plan_payload.get("evidence_tables", []):
        if isinstance(table.get("rows"), list) and len(table["rows"]) > 20:
            table["rows"] = table["rows"][:20]
            table["context_note"] = "Bảng rút gọn cho report LLM; bảng đầy đủ vẫn nằm trong state."
    report_plan_context = json.dumps(
        execution_state.model_dump(mode="python"), ensure_ascii=False, indent=2, default=str
    )
    max_retries = 3
    base_delay = 2
    llm_raw_output_str = ""
    report_draft = None
    grounding_feedback = state.get("error_message") or "Chưa có bản nháp nào bị từ chối."

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to invoke LLM for report drafting...")
            parser = JsonOutputParser(pydantic_object=ReportSectionsDraft)
            prompt = PromptTemplate(
                template="""
                Bạn là chuyên gia tạo báo cáo dữ liệu đa lĩnh vực. Nhiệm vụ chính là soạn một báo cáo
                hoàn chỉnh theo yêu cầu người dùng; các phép tính, phát hiện và biểu đồ chỉ là bằng chứng
                đầu vào để viết báo cáo, không phải sản phẩm cuối cùng độc lập.

                Trước khi viết, hãy nhận diện lĩnh vực từ tên cột, nội dung phát hiện và yêu cầu người dùng.
                Sau đó dùng thuật ngữ, giọng văn, cách giải thích và loại khuyến nghị phù hợp với lĩnh vực đó.
                Không mặc định dữ liệu liên quan đến bán hàng, khách hàng, doanh thu, lợi nhuận hoặc hoạt động
                kinh doanh nếu các khái niệm này không xuất hiện trong dữ liệu. Nếu chưa đủ căn cứ nhận diện
                lĩnh vực, hãy dùng văn phong phân tích trung tính và không tự gán bối cảnh.

                Quy tắc căn cứ định lượng:
                - KẾT QUẢ PANDAS ĐÃ KIỂM CHỨNG là nguồn duy nhất cho mọi con số trong phần diễn giải.
                  Các trường audit của insight evidence_valid=true (`metrics`, `sample_size`,
                  `denominator`, `missing_values`) cũng hợp lệ vì được Evidence Validator dẫn xuất
                  trực tiếp từ chính kết quả pandas.
                - Không tự tính, ước lượng, ngoại suy hoặc thêm số không có trong `result` hay `parameters`.
                - Có thể làm tròn giá trị đã có để dễ đọc nhưng không được tạo ra giá trị mới.
                - Không cộng/trừ các kết quả, không tính tỷ trọng/chênh lệch và không đổi số thập phân
                  sang phần trăm nếu giá trị chuyển đổi không xuất hiện trực tiếp trong kết quả pandas.
                - Mọi ngày, tháng, năm và kỳ thời gian phải khớp chính xác chuỗi thời gian trong
                  kết quả pandas; không tự dịch mốc, đổi kỳ, nội suy hoặc bổ sung khoảng thời gian.
                - Văn bản profile, văn bản phát hiện và mô tả biểu đồ chỉ là ngữ cảnh, không phải
                  nguồn số liệu bổ sung. Chỉ `num_rows`, `num_columns` của profile và các trường audit
                  của insight nêu trên được dùng làm evidence định lượng.
                - Không biến null, n=0 hoặc n=1 thành một kết luận. Khi evidence không đủ, ghi đúng
                  "Không đủ dữ liệu để đánh giá." và không suy diễn thêm.
                - Phân biệt record_count, period_count và unique_count; không gọi một kỳ thời gian
                  là một bản ghi. Không suy luận quan hệ nhân quả từ correlation.

                Phản hồi BẮT BUỘC là đối tượng JSON hợp lệ, đúng chính xác schema được cung cấp.

                JSON gồm:
                - `report_subtitle`: Một dòng ngắn mô tả domain, phạm vi và khoảng thời gian nếu có;
                  không chứa tên file, tên sheet hoặc metadata nội bộ.
                - `introduction_text`: Giới thiệu mục tiêu, phạm vi dữ liệu và nội dung người đọc sẽ nhận được;
                  không thêm thông tin nền không có trong dữ liệu.
                - `data_quality_text`: Luôn trả về chuỗi rỗng; báo cáo không có mục chất lượng dữ liệu riêng.
                - `analysis_narratives`: Danh sách đoạn phân tích chi tiết. Mỗi đoạn phải có tiêu đề,
                  sau đó là chuỗi `:-` rồi mới đến nội dung. Dùng placeholder `[FIGURE 1]`, `[FIGURE 2]`...
                  theo đúng thứ tự khi nhắc đến biểu đồ.
                  Mỗi biểu đồ phải được đặt trong narrative của theme/insight có cùng evidence và ngay
                  cạnh placeholder phải có nhận xét giải thích biểu đồ thể hiện điều gì, bằng chứng nào
                  nổi bật và hàm ý gì. Không tạo mục trực quan hóa bổ sung tách khỏi nội dung phân tích.
                  Tiêu đề phải có dạng `Tiêu đề phát hiện:-Nội dung`. Tuyệt đối không đặt tên file,
                  tên sheet/source_partition hoặc ký tự `|` trong tiêu đề.
                - `key_takeaways_bullet_points`: Các kết luận hoặc hàm ý ngắn gọn, có thể hành động và phù hợp lĩnh vực.
                  Chỉ được dùng hành động/khuyến nghị đã xuất hiện trong insight evidence_valid=true;
                  không tự đề xuất “tập trung”, “tối ưu”, “cải thiện” hoặc chiến lược mới nếu insight
                  không nêu hành động đó từ evidence tương ứng.
                - `notable_issues`: Các vấn đề hoặc bất thường đáng chú ý có bằng chứng; để [] nếu không có.
                - `conclusion_text`: Kết luận tổng hợp và bước tiếp theo hợp lý. Chỉ nêu giới hạn nếu
                  context có một giới hạn cụ thể đã kiểm chứng.
                - `dataset_title`: Tiêu đề ngắn phản ánh đúng mục đích phân tích, không chứa tên file,
                  đường dẫn, tên sheet/source_partition và không tự gán lĩnh vực.
                - `figure_id_map`: Ánh xạ placeholder như `[FIGURE 1]` tới `visual_id` thực tế.
                - `clarification_questions`: Câu hỏi cần làm rõ nếu thiếu ngữ cảnh; để trống khi không cần.

                Tiêu chuẩn văn phong:
                - Viết theo phong cách báo cáo điều hành: súc tích, chắc câu, ưu tiên kết luận và tác động;
                  dùng từ phổ thông, chính xác, hạn chế danh từ hóa và câu bị động.
                - Mỗi đoạn chỉ triển khai một thông điệp chính. Câu đầu nêu kết luận, câu sau đưa bằng
                  chứng, rồi giải thích hàm ý và hành động. Không lặp cùng một con số ở nhiều phần.
                - Dùng tiêu đề giàu thông tin, mô tả điều thực sự xảy ra thay vì tiêu đề chung như
                  “Phân tích dữ liệu”, “Kết quả chính” hoặc “Một số nhận xét”.
                - Khi có đề xuất, mở đầu bằng động từ hành động phù hợp lĩnh vực như “Ưu tiên”,
                  “Rà soát”, “Theo dõi”, “Thử nghiệm”, “Duy trì”; nêu rõ đối tượng và căn cứ.
                - Phân biệt rõ kết quả quan sát, diễn giải và khuyến nghị. Giới hạn là tùy chọn, không
                  bắt buộc trong từng đoạn.
                - Mỗi phát hiện theo logic: Phát hiện → Bằng chứng số liệu → Ý nghĩa → Hành động.
                  Chỉ thêm giới hạn khi input nêu rõ thiếu dữ liệu, cỡ mẫu/mẫu số không tương đương,
                  phạm vi khác nhau hoặc số kỳ quan sát quá ít. Không tự tạo câu cảnh báo chung.
                - Chỉ tạo một mục cho insight có evidence_valid=true. Không có evidence thì bỏ mục hoàn toàn.
                - Nếu tiêu đề/nội dung nói tỷ lệ, số lượng, trung bình, tổng hoặc chênh lệch, phải ghi value,
                  sample size và denominator khi evidence cung cấp. Không dùng “phần lớn”, “khá cao”, “đáng kể”.
                - Không hiển thị tên sheet/source_partition, task_id, sheet_id hoặc mã nội bộ trong báo cáo.
                - Khi các metric giống tên nhưng khác nguồn/cột/filter/cỡ mẫu, phải nói rõ chúng thuộc phạm vi
                  khác nhau; không gộp hoặc so sánh nếu không tương đương.
                - Mỗi kết luận và khuyến nghị phải gắn trực tiếp với evidence của một phát hiện đã duyệt;
                  bỏ các khuyến nghị và cảnh báo chung chung không có metric cụ thể đi kèm.
                - Không dùng các câu mẫu như “cần lưu ý rằng ... có thể ảnh hưởng đến kết quả”. Nếu
                  không có limitation cụ thể trong context, kết thúc đoạn sau hàm ý hoặc hành động.
                - Tổ chức analysis_narratives theo đúng analytical themes trong REPORT PLAN; gộp các
                  insight cùng theme thành một narrative thay vì tạo section cho từng câu hỏi nhỏ.
                  Bắt buộc tạo đủ một narrative cho MỌI theme trong REPORT PLAN và đưa nội dung của
                  MỌI insight evidence_valid=true vào đúng narrative; không bỏ bớt vì nội dung dài
                  hoặc vì insight khác có chủ đề gần giống. Khi đã có biểu đồ cho evidence, diễn giải
                  bằng biểu đồ và nhận xét, không yêu cầu lặp lại cùng evidence dưới dạng bảng.
                - Phần mở đầu đóng vai trò executive summary, dài tối đa một đoạn ngắn: nêu quy mô,
                  kết quả nổi bật, bất thường và phạm vi thời gian bằng số cụ thể có trong evidence.
                  Bảng và biểu đồ sẽ được exporter dựng từ REPORT PLAN; không tự tính lại số liệu.
                - Không dùng lời dẫn meta như “báo cáo này sẽ”, “phần dưới đây”, “dựa trên dữ liệu được
                  cung cấp” khi có thể đi thẳng vào nội dung. Không dùng giọng quảng cáo hoặc cường điệu.
                - Tránh thuật ngữ kinh doanh chung chung hoặc khẳng định không được dữ liệu hỗ trợ.
                - Viết đúng ngôn ngữ báo cáo mà người dùng yêu cầu, với tiêu đề phần rõ ràng.
                - Tiêu đề các phát hiện chỉ nêu mục đích hoặc nội dung phân tích; không chứa tên file,
                  tên sheet/source_partition hoặc dấu `|`.

                ---
                Schema dữ liệu:
                {profile_summary}

                ---
                Các phát hiện đã tạo (dùng để xây dựng diễn giải):
                {insights_context}

                ---
                Các biểu đồ đã tạo:
                {visuals_context}
                Visual ID và mô tả có sẵn: {visual_reference_for_llm}

                ---
                KẾT QUẢ PANDAS ĐÃ KIỂM CHỨNG (kèm nguồn gốc operation/phép tính):
                {computed_facts}

                ---
                REPORT EXECUTION STATE ĐÃ KIỂM CHỨNG (structured state hiện tại; evidence đầy đủ được giữ ngoài prompt):
                {report_plan_context}

                ---
                Yêu cầu của người dùng:
                {instructions}

                Phản hồi từ lần kiểm tra căn cứ trước:
                {grounding_feedback}

                ---
                Hãy tạo bản nháp báo cáo đầy đủ. Dùng placeholder `[FIGURE N]` trong phần diễn giải
                và điền `figure_id_map` chính xác. MỌI biểu đồ trong danh sách đã tạo phải xuất hiện
                trong figure_id_map, nằm trong narrative tương ứng và có nhận xét ngay tại đó.

                {format_instructions}

                Chỉ trả về JSON hợp lệ.
                """,
                input_variables=["profile_summary", "insights_context", "visuals_context", "instructions",
                                 "visual_reference_for_llm", "computed_facts", "report_plan_context",
                                 "grounding_feedback", "dataset_name"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )


            llm_response = llm.invoke(prompt.invoke({
                "profile_summary": profile_summary,
                "insights_context": insights_context,
                "visuals_context": visuals_context,
                "instructions": instructions,
                "visual_reference_for_llm": visual_reference_for_llm,
                "computed_facts": computed_facts,
                "report_plan_context": report_plan_context,
                "grounding_feedback": grounding_feedback,
                "dataset_name": dataset_name
            }), config={"request_options": {"timeout": 60}})
            llm_raw_output_str = text_from_response(llm_response)

            stripped_str = llm_raw_output_str.strip()
            if stripped_str.startswith("```json") and stripped_str.endswith("```"):
                json_str = stripped_str[len("```json"):-len("```")].strip()
            else:
                json_str = stripped_str

            parsed_draft_output_dict = json.loads(json_str)

            report_draft = ReportSectionsDraft.model_validate(parsed_draft_output_dict)
            report_draft = _strip_source_names_from_headings(
                report_draft, state.get("file_path", ""), state.get("analysis_partitions")
            )
            report_draft = _sanitize_generic_caveats(report_draft)

            completeness_issues = _report_completeness_issues(
                report_draft, report_plan, analysis_insights, generated_visuals
            )
            if completeness_issues:
                raise IncompleteReportError("; ".join(completeness_issues))

            unsupported_numbers = _unsupported_report_numbers(
                report_draft, computed_results, analysis_insights, dataframe_profile
            )
            if unsupported_numbers:
                removed_numbers = _strip_unsupported_report_numbers(
                    report_draft, computed_results, analysis_insights, dataframe_profile
                )
                logger.warning(
                    "Report draft chứa số ngoài evidence; đã loại token và tiếp tục: %s",
                    removed_numbers,
                )

            logger.info("LLM returned report draft sections.")

            break

        except (UngroundedNumbersError, IncompleteReportError) as e:
            logger.warning("Report draft bị từ chối lần %s/%s: %s", attempt + 1, max_retries, e)
            grounding_feedback = (
                f"Bản nháp trước bị từ chối: {e}. Hãy sửa đúng mọi lỗi được nêu, tạo đủ một "
                "narrative cho từng theme theo đúng thứ tự REPORT PLAN, dùng tiêu đề có nghĩa, "
                "nhận xét evidence và gắn đúng biểu đồ; không thêm số tự tính mới."
            )
            if attempt == max_retries - 1:
                state['status'] = "error"
                state['error_message'] = str(e)
                return state
        except (requests.exceptions.RequestException, TimeoutError) as e:
            logger.warning(f"LLM call failed on attempt {attempt + 1}/{max_retries} due to network/timeout: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying LLM call for request {request_id} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached. LLM call failed for request {request_id}.")
                state['status'] = "error"
                state['error_message'] = f"Failed to get a response from the LLM after {max_retries} attempts: {e}"
                return state

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Report draft JSON/schema không hợp lệ lần %s/%s cho request %s: %s",
                attempt + 1,
                max_retries,
                request_id,
                e,
            )
            logger.debug("Raw LLM Output: %s", llm_raw_output_str[:1000])
            grounding_feedback = (
                f"Bản nháp trước không phải JSON hợp lệ theo schema: {e}. "
                "Hãy xuất lại duy nhất một đối tượng JSON hợp lệ, không có dấu phẩy thừa, "
                "không có markdown và giữ đầy đủ nội dung báo cáo theo REPORT PLAN."
            )
            if attempt == max_retries - 1:
                state['status'] = "error"
                state['error_message'] = (
                    f"LLM output for report draft remained invalid JSON or schema after "
                    f"{max_retries} attempts: {e}"
                )
                return state
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call for report drafting for request {request_id}: {e}",
                         exc_info=True)
            state['status'] = "error"
            state['error_message'] = f"An unexpected error occurred during report drafting LLM call: {e}"
            return state
    if report_draft:
        report_draft.data_quality_text = ""
        state['report_sections_draft'] = report_draft
        state['error_message'] = None
        state['status'] = "report_drafted"

        logger.info(f"ReportDraftingNode completed for request: {request_id}. Report drafted.")
    else:

        state['status'] = "error"
        state['error_message'] = "An unexpected failure occurred after all LLM retries or no report draft was generated."
    return state



