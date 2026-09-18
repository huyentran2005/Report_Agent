"""LangGraph workflow definition."""
import logging

from langgraph.graph import END, StateGraph

from agents.data_profiler import profile_dataset
from agents.question_framer import frame_questions
from agents.insight_engine import extract_insights
from agents.evidence_validator import validate_evidence
from agents.report_planner import build_report_plan
from agents.report_writer import draft_report
from agents.report_exporter import export_report
from agents.quality_gate import validate_report
from agents.chart_generator import generate_visuals
from graph.state import GraphState

logger = logging.getLogger(__name__)


def after_analysis(state: GraphState) -> str:
    profile = state.get("dataframe_profile")
    if not profile or profile.num_rows < 5 or profile.num_columns < 2:
        return END
    return "question_framing"


def after_question_framing(state: GraphState) -> str:
    return "insight_generation" if state.get("status") == "questions_computed" else END


def after_visualization(state: GraphState) -> str:
    return "report_drafting" if state.get("status") == "visuals_generated" else END


def after_insight_generation(state: GraphState) -> str:
    return "evidence_validation" if state.get("status") == "insights_generated" else END


def after_evidence_validation(state: GraphState) -> str:
    return "report_planning" if state.get("status") == "evidence_validated" else END


def after_report_planning(state: GraphState) -> str:
    return "visualization" if state.get("status") == "report_planned" else END


def after_report_drafting(state: GraphState) -> str:
    if state.get("status") == "report_drafted" and state.get("report_sections_draft"):
        return "safety_check"
    logger.error("Stopping workflow because report drafting did not produce a valid draft.")
    return END


def after_safety_check(state: GraphState) -> str:
    if state.get("status") != "error":
        return "report_finalization"

    if not state.get("report_sections_draft"):
        logger.error("Stopping workflow because there is no report draft to export.")
        return END

    state["status"] = "safety_warning"
    state["error_message"] = (state.get("error_message") or "") + "\nBáo cáo chưa vượt qua bước kiểm tra tự động; vui lòng xem lại các phát hiện quan trọng."
    logger.warning("Safety review did not pass; exporting with a validation warning.")
    return "report_finalization"


def create_graph_workflow():
    graph = StateGraph(GraphState)
    for name, node in {
        "data_analysis": profile_dataset,
        "question_framing": frame_questions,
        "visualization": generate_visuals,
        "insight_generation": extract_insights,
        "evidence_validation": validate_evidence,
        "report_planning": build_report_plan,
        "report_drafting": draft_report,
        "safety_check": validate_report,
        "report_finalization": export_report,
    }.items():
        graph.add_node(name, node)
    graph.set_entry_point("data_analysis")
    graph.add_conditional_edges("data_analysis", after_analysis, {"question_framing": "question_framing", END: END})
    graph.add_conditional_edges("question_framing", after_question_framing, {"insight_generation": "insight_generation", END: END})
    graph.add_conditional_edges("insight_generation", after_insight_generation, {"evidence_validation": "evidence_validation", END: END})
    graph.add_conditional_edges("evidence_validation", after_evidence_validation, {"report_planning": "report_planning", END: END})
    graph.add_conditional_edges("report_planning", after_report_planning, {"visualization": "visualization", END: END})
    graph.add_conditional_edges("visualization", after_visualization, {"report_drafting": "report_drafting", END: END})
    graph.add_conditional_edges("report_drafting", after_report_drafting, {"safety_check": "safety_check", END: END})
    graph.add_conditional_edges(
        "safety_check", after_safety_check,
        {"report_finalization": "report_finalization", "report_drafting": "report_drafting", END: END},
    )
    graph.add_edge("report_finalization", END)
    return graph.compile()
