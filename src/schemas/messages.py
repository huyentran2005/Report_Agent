from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime

class DataProfile(BaseModel):
    
    num_rows: int = Field(description="Number of rows in the dataset.")
    num_columns: int = Field(description="Number of columns in the dataset.")
    column_details: Dict[str, Dict[str, Any]] = Field(description="Dictionary of column names to their details (e.g., 'type', 'unique_values_count', 'missing_values_count', 'mean', 'std', 'min', 'max').")
    key_observations: str = Field(description="Key observations about the dataset's structure, quality, and potential issues (e.g., missing values, outliers, data types that need conversion).")
    data_scope: Optional["PartitionDataScope"] = Field(
        default=None, description="Deterministic scope of the source partition."
    )


class DimensionScope(BaseModel):
    column: str
    kind: Literal["time", "category", "geography", "operation", "dimension"]
    valid_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    years: List[int] = Field(default_factory=list)
    periods: List[str] = Field(default_factory=list)
    values: List[Any] = Field(default_factory=list)
    values_complete: bool = False


class PartitionDataScope(BaseModel):
    source_partition: str
    row_count: int = Field(ge=0)
    dimensions: List[DimensionScope] = Field(default_factory=list)


class QuestionDataScope(BaseModel):
    coverage_mode: Literal["full_dataset", "user_requested_subset", "analytical_subset"] = "full_dataset"
    source_partitions: List[str] = Field(default_factory=list)
    time_columns: List[str] = Field(default_factory=list)
    dimension_columns: List[str] = Field(default_factory=list)
    rationale: str = ""


AnalysisOperation = str


class ColumnAnalysisRole(BaseModel):
    column: str
    source_partition: str
    role: Literal[
        "time", "measure", "outcome", "driver", "category", "geography",
        "operation", "identifier", "free_text", "sensitive", "exclude",
    ]
    include_in_analysis: bool = True
    rationale: str


class AnalyticalTheme(BaseModel):
    theme_id: str
    title: str
    purpose: str
    source_partition: str
    columns: List[str]
    required: bool = True
    suggested_analyses: List[str] = Field(default_factory=list)


class AnalysisCoverageMap(BaseModel):
    column_roles: List[ColumnAnalysisRole]
    themes: List[AnalyticalTheme] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_theme_ids(self):
        theme_ids = [theme.theme_id for theme in self.themes]
        if len(theme_ids) != len(set(theme_ids)):
            raise ValueError("Mỗi analytical theme phải có theme_id duy nhất.")
        return self


class AnalysisFilter(BaseModel):
    column: str
    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in",
        "contains", "between", "is_null", "not_null",
    ]
    value: Any = None


class AnalysisTransform(BaseModel):
    type: Literal[
        "pct_change", "difference", "cumulative", "share_of_total",
        "ratio", "rank", "rolling_mean", "correlation", "distribution",
        "outlier_iqr", "round",
    ]
    column: Optional[str] = None
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    output_column: Optional[str] = None
    periods: int = Field(default=1, ge=1)
    window: int = Field(default=3, ge=2)
    decimals: int = Field(default=2, ge=0, le=10)


class AnalysisSort(BaseModel):
    by: str
    ascending: bool = True


class StructuredAnalysisPlan(BaseModel):
    """Declarative analysis plan compiled by the generic Pandas executor."""

    source_partition: str
    group_by: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    aggregation: Literal["sum", "mean", "count", "median", "min", "max", "std", "nunique"] = "sum"
    time_grain: Optional[Literal["day", "week", "month", "quarter", "year"]] = None
    filters: List[AnalysisFilter] = Field(default_factory=list)
    transforms: List[AnalysisTransform] = Field(default_factory=list)
    sort: Optional[AnalysisSort] = None
    limit: Optional[int] = Field(default=None, ge=1)
    dropna: bool = True
    include_sample_size: bool = True

    @field_validator("sort", mode="before")
    @classmethod
    def normalize_empty_sort(cls, value):
        """Treat an empty LLM sort object as an omitted optional sort specification."""
        if value is None or value == {}:
            return None
        if isinstance(value, str) and value.strip().casefold() in {"", "null", "none", "n/a"}:
            return None
        return value


class FramedQuestion(BaseModel):
    question_id: str = Field(description="Stable identifier such as question_1.")
    question: str = Field(description="The report question to answer.")
    operation: AnalysisOperation = Field(default="structured_plan", description="Reader-facing analysis label.")
    columns: List[str] = Field(default_factory=list, description="Exact source columns used by the plan.")
    analysis_type: Optional[str] = Field(default=None, description="Reader-facing analysis category.")
    expected_result: str = Field(
        min_length=1,
        description="Required reader-facing description of the expected result and its meaning.",
    )
    visualization: Optional[str] = Field(default=None, description="Suggested visualization type, if useful.")
    theme_ids: List[str] = Field(
        min_length=1,
        description="Analytical themes from the coverage map that this question covers.",
    )
    scope: QuestionDataScope = Field(
        default_factory=QuestionDataScope,
        description="Explicit input-data scope; full_dataset is the default.",
    )
    depends_on_question_ids: List[str] = Field(
        default_factory=list,
        description="Questions whose computed results must be observed before this question is planned/executed.",
    )
    plan: StructuredAnalysisPlan
    source_partition: Optional[str] = Field(default=None, description="Sheet or compatible sheet group to analyze.")


class ComputedQuestionResult(BaseModel):
    question_id: str
    question: str
    operation: AnalysisOperation
    columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Any
    calculation: str = Field(description="Human-readable provenance for the pandas calculation.")
    source_partition: Optional[str] = None

class AnalysisInsight(BaseModel):
    
    insight_id: str = Field(description="Unique identifier for the insight.")
    title: str = Field(description="A concise, descriptive title for the insight.")
    narrative: str = Field(description="A detailed narrative explaining the insight, its context, and implications.")
    supporting_visual_ids: List[str] = Field(default_factory=list, description="List of IDs of generated visuals that support this insight.")
    source_partition: Optional[str] = Field(default=None, description="Independent sheet or compatible sheet group supporting the insight.")
    question: str = Field(default="", description="Audited question answered by this insight.")
    finding: str = Field(default="", description="Concise evidence-backed finding.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Exact audited metrics used by the finding.")
    source_sheet: Optional[str] = Field(default=None, description="Internal source sheet provenance.")
    source_columns: List[str] = Field(default_factory=list, description="Exact source columns used by pandas.")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters applied before calculation.")
    sample_size: Optional[int] = Field(default=None, ge=0, description="Number of observations supporting the metric.")
    denominator: Optional[int] = Field(default=None, ge=0, description="Denominator for rates or proportions.")
    missing_values: Optional[int] = Field(default=None, ge=0, description="Missing source values affecting this insight.")
    evidence_valid: bool = Field(default=False, description="Whether deterministic evidence validation passed.")
    limitations: List[str] = Field(default_factory=list, description="Scope and evidence limitations.")
    evidence_question_ids: List[str] = Field(default_factory=list, description="Internal links to audited results.")

    @field_validator("supporting_visual_ids", mode="before")
    @classmethod
    def normalize_supporting_visual_ids(cls, value):
        """Normalize a single visual ID, a frequent LLM JSON formatting mistake."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        raise ValueError("supporting_visual_ids must be an array or one visual ID string")


class ReportKPI(BaseModel):
    name: str
    value: Any
    unit: str = ""
    sample_size: Optional[int] = None
    denominator: Optional[int] = None
    evidence_question_id: str


class EvidenceTable(BaseModel):
    title: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    evidence_question_ids: List[str]


class ReportTheme(BaseModel):
    title: str
    purpose: str
    insight_ids: List[str]


class ReportPlan(BaseModel):
    kpis: List[ReportKPI] = Field(default_factory=list, max_length=6)
    themes: List[ReportTheme] = Field(default_factory=list)
    evidence_tables: List[EvidenceTable] = Field(default_factory=list)

class VisualGenerationInstruction(BaseModel):
    
    type: str = Field(description="Reader-friendly chart type such as bar, horizontal_bar, line, area, pie, scatter, histogram, or boxplot.")
    columns: List[str] = Field(description="List of column names to be used for the chart.")
    title: Optional[str] = Field(default=None, description="Title for the chart.")
    description: str = Field(description="A brief explanation of what the chart should convey or highlight.")
    suggested_section: Optional[str] = Field(default=None, description="Where in a report this visual would best fit (e.g., 'Introduction', 'Sales Analysis', 'Customer Demographics', 'Conclusion').")
    evidence_question_ids: List[str] = Field(default_factory=list, description="Audited questions directly supported by this chart.")

class GeneratedVisual(BaseModel):
    visual_id: str = Field(description="Unique identifier for the generated visual.")
    type: str = Field(description="Reader-friendly chart type used for the generated visual.")
    description: str = Field(description="Description of what the visual depicts.")
    file_path: str = Field(description="Local file path where the generated chart image is saved.")
    suggested_section: str = Field(description="Suggested section in the report where this visual should be placed.")
    chart_code: Optional[str] = Field(default=None, description="The Python code used to generate the chart.")
    evidence_question_ids: List[str] = Field(default_factory=list, description="Audited questions directly supported by this chart.")

class ReportSectionsDraft(BaseModel):
    report_subtitle: str = Field(default="", description="Short domain-specific scope and time-period subtitle.")
    introduction_text: str = Field(description="A comprehensive introduction to the report.")
    data_quality_text: str = Field(default="", description="Data quality, scope, and analytical limitations.")
    analysis_narratives: List[str] = Field(description="Detailed analysis narratives, each describing a key finding.")
    notable_issues: List[str] = Field(default_factory=list, description="Material anomalies or issues supported by audited results.")
    key_takeaways_bullet_points: List[str] = Field(description="Concise, actionable key takeaways or main conclusions.")
    conclusion_text: str = Field(description="A summary conclusion for the entire report.")
    dataset_title: str = Field(description="A concise, descriptive title for the dataset, generated by the LLM.")
    figure_id_map: Dict[str, str] = Field(default_factory=dict, description="A mapping from generic figure placeholders (e.g., '[FIGURE 1]') used in narratives to their actual 'visual_id's. The LLM should create this map based on the order it refers to figures.")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions for the user if more information is needed.")

class ReportFormat(BaseModel):
    content: str = Field(description="The final report content, e.g., in Markdown format.")
    format_type: str = Field(description="The format of the report content (e.g., 'markdown', 'html', 'pdf_path').")
    pdf_file_path: Optional[str] = Field(default=None, description="Path to the generated PDF file, if applicable.")


class UserFeedback(BaseModel):
    feedback_id: str = Field(description="Unique identifier for the feedback.")
    message: str = Field(description="The text content of the feedback.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp when the feedback was provided.")

