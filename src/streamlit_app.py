"""Streamlit interface for the AI report workspace."""
from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import PATHS
from graph.builder import create_graph_workflow
from graph.state import GraphState

load_dotenv()
PATHS.create()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

PROVIDERS = {
    "Google Gemini": ("gemini", "GEMINI_MODEL", "gemini-2.5-flash"),
    "ProxyLLM / OpenAI": ("proxyllm", "PROXYLLM_MODEL", "gpt-4o-mini"),
}
WORKFLOW_LABELS = {
    "data_analysis": "Đọc và lập hồ sơ dữ liệu",
    "question_framing": "Lập câu hỏi và tính số liệu kiểm chứng",
    "visualization": "Tạo biểu đồ phù hợp",
    "insight_generation": "Rút ra các phát hiện chính",
    "evidence_validation": "Kiểm chứng bằng chứng và loại phát hiện trùng lặp",
    "report_planning": "Lập cấu trúc KPI và chủ đề báo cáo",
    "report_drafting": "Soạn nội dung báo cáo",
    "safety_check": "Kiểm tra tính nhất quán",
    "report_finalization": "Xuất báo cáo hoàn chỉnh",
}


def safe_name(filename: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", Path(filename).name, flags=re.UNICODE)
    return cleaned.strip(" .") or "dataset.csv"


def save_upload(uploaded_file) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = PATHS.uploads / f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_name(uploaded_file.name)}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


@st.cache_data(max_entries=8, show_spinner=False)
def preview_csv(content: bytes) -> pd.DataFrame:

    return pd.read_csv(BytesIO(content), nrows=20)


def make_state(file_path: Path, instructions: str, provider: str, model: str, language: str) -> GraphState:
    language_name = "tiếng Việt" if language == "Tiếng Việt" else "tiếng Anh"
    localized_instructions = (
        f"{instructions or 'Tạo báo cáo tổng quan từ dữ liệu và nêu các nội dung quan trọng.'}\n\n"
        f"QUAN TRỌNG: Viết toàn bộ nội dung, diễn giải biểu đồ và báo cáo cuối cùng bằng {language_name}."
    )
    return {
        "request_id": uuid.uuid4().hex,
        "file_path": str(file_path),
        "instructions": localized_instructions,
        "dataframe_profile": None,
        "framed_questions": [],
        "computed_question_results": [],
        "workbook_sheets": [],
        "workbook_instruction_context": "",
        "analysis_partitions": [],
        "sheet_profiles": {},
        "analysis_insights": None,
        "validated_insights": [],
        "report_plan": None,
        "generated_visuals": None,
        "report_sections_draft": None,
        "final_report": None,
        "feedback_history": None,
        "status": "initial",
        "error_message": None,
        "safety_check_retries": 0,
        "llm_provider": provider,
        "llm_model": model,
        "report_language": language,
        "report_output_dir": str(PATHS.reports),
        "chart_output_dir": str(PATHS.charts),
    }


def run_workflow(state: GraphState) -> GraphState:
    latest = state
    workflow = create_graph_workflow()
    with st.status("Đang tạo báo cáo...", expanded=True) as status:
        for event in workflow.stream(state, config={"recursion_limit": 50}):
            for node, update in event.items():
                if node == "__end__":
                    continue
                latest = update
                st.write(f":material/check_circle: {WORKFLOW_LABELS.get(node, node)}")
        if latest.get("status") == "error":
            status.update(label="Không thể hoàn thành báo cáo", state="error")
        else:
            status.update(label="Báo cáo đã sẵn sàng", state="complete", expanded=False)
    return latest


def show_result(state: GraphState) -> None:
    if state.get("status") == "error":
        st.error(state.get("error_message") or "Đã xảy ra lỗi không xác định.", icon=":material/error:")
        return
    if state.get("status") == "invalid_instructions":
        st.warning("Yêu cầu chưa phù hợp với dữ liệu. Hãy mô tả rõ mục tiêu phân tích.", icon=":material/warning:")
        return
    report = state.get("final_report")
    if not report:
        st.warning("Workflow kết thúc nhưng chưa tạo được nội dung báo cáo.")
        return

    st.success("Đã tạo báo cáo thành công.", icon=":material/check_circle:")
    workbook_sheets = state.get("workbook_sheets") or []
    if workbook_sheets:
        with st.expander("Phân loại các sheet trong workbook", icon=":material/table_view:"):
            st.dataframe(pd.DataFrame(workbook_sheets), hide_index=True, width="stretch")
    pdf_path = Path(report.pdf_file_path) if report.pdf_file_path else None
    if not pdf_path or not pdf_path.exists():
        st.error("Không tìm thấy file PDF của báo cáo.", icon=":material/error:")
        return

    pdf_bytes = pdf_path.read_bytes()
    st.download_button(
        "Tải PDF", pdf_bytes, pdf_path.name, "application/pdf",
        icon=":material/picture_as_pdf:", type="primary",
    )
    with st.expander("Xem báo cáo PDF", icon=":material/picture_as_pdf:", expanded=True):
        st.pdf(pdf_bytes, height=800)
    visuals = state.get("generated_visuals") or []
    if visuals:
        with st.expander("Các biểu đồ trong báo cáo", icon=":material/bar_chart:"):
            for visual in visuals:
                image_path = Path(visual.file_path)
                if image_path.exists():
                    st.image(str(image_path), caption=visual.description)


st.set_page_config(page_title="Report Agent", page_icon=":material/analytics:", layout="wide")
st.title("Report Agent", icon=":material/analytics:")
st.caption("Tải dữ liệu CSV, mô tả mục tiêu và nhận báo cáo phân tích có biểu đồ trong một quy trình.")

with st.sidebar:
    st.header("Cấu hình mô hình", icon=":material/tune:")
    provider_label = st.selectbox("Nhà cung cấp", list(PROVIDERS), key="provider")
    provider, model_env, fallback_model = PROVIDERS[provider_label]
    model = st.text_input("Tên model", value=os.getenv(model_env, fallback_model), key=f"model_{provider}")
    language = st.selectbox("Ngôn ngữ", ["Tiếng Việt", "English"], key="report_language")

with st.form("report_request", border=True):
    st.subheader("Tạo báo cáo mới", icon=":material/upload_file:")
    uploaded = st.file_uploader("Tệp dữ liệu", type=("csv", "xlsx", "xls"), help="Hỗ trợ CSV và Excel; nên dùng ít nhất 5 dòng và 2 cột.")
    instructions = st.text_area(
        "Mục tiêu phân tích",
        placeholder="Ví dụ: Xác định các xu hướng chính, nhóm nổi bật, điểm bất thường và hàm ý phù hợp với lĩnh vực của dữ liệu.",
        height=120,
        max_chars=1_000,
    )
    submitted = st.form_submit_button("Tạo báo cáo", type="primary", icon=":material/auto_awesome:")

if uploaded:
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            workbook = pd.read_excel(BytesIO(uploaded.getvalue()), sheet_name=None)
            preview_frames = []
            for sheet_name, frame in workbook.items():
                if not frame.empty:
                    frame = frame.head(20).copy()
                    frame.insert(0, "_sheet_name", str(sheet_name))
                    preview_frames.append(frame)
            sample = pd.concat(preview_frames, ignore_index=True, sort=False) if preview_frames else pd.DataFrame()
        else:
            sample = preview_csv(uploaded.getvalue())
        with st.expander("Xem trước dữ liệu", icon=":material/table_view:"):
            st.dataframe(sample, height=280)
            st.caption(f"Hiển thị tối đa 20 dòng · {len(sample.columns)} cột")
    except Exception as exc:
        st.warning(f"Không thể xem trước CSV: {exc}")

if submitted:
    if not uploaded:
        st.warning("Hãy chọn một tệp CSV trước khi tạo báo cáo.", icon=":material/upload_file:")
    elif not model.strip():
        st.warning("Tên model không được để trống.", icon=":material/warning:")
    else:
        try:
            file_path = save_upload(uploaded)
            result = run_workflow(make_state(file_path, instructions.strip(), provider, model.strip(), language))
            st.session_state["last_report"] = result
        except Exception as exc:
            logger.exception("Report workflow failed")
            st.error(f"Không thể tạo báo cáo: {exc}", icon=":material/error:")

if "last_report" in st.session_state:
    show_result(st.session_state["last_report"])
