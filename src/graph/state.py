from typing import List, NotRequired, Optional, TypedDict
from schemas.messages import (DataProfile, AnalysisInsight, GeneratedVisual, ReportSectionsDraft,
                              ReportFormat, UserFeedback, FramedQuestion, ComputedQuestionResult, ReportPlan)

class GraphState(TypedDict):
    request_id: str
    file_path: str
    instructions: str
    dataframe_profile: Optional[DataProfile]
    framed_questions: NotRequired[List[FramedQuestion]]
    computed_question_results: NotRequired[List[ComputedQuestionResult]]
    analysis_insights: Optional[List[AnalysisInsight]]
    validated_insights: NotRequired[List[AnalysisInsight]]
    report_plan: NotRequired[ReportPlan]
    generated_visuals: Optional[List[GeneratedVisual]]
    report_sections_draft: Optional[ReportSectionsDraft]
    final_report: Optional[ReportFormat]
    feedback_history: Optional[List[UserFeedback]]
    status: str
    error_message: Optional[str]
    safety_check_retries: int
    llm_provider: str
    llm_model: str
    report_output_dir: NotRequired[str]
    chart_output_dir: NotRequired[str]
    report_language: NotRequired[str]
    workbook_sheets: NotRequired[list[dict[str, object]]]
    workbook_instruction_context: NotRequired[str]
    analysis_partitions: NotRequired[List[str]]
    sheet_profiles: NotRequired[dict[str, dict[str, object]]]
