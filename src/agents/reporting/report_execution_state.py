"""Bounded structured state for report generation (SKILL.state inspired)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ThemeEvidenceState(BaseModel):
    theme_id: str
    title: str
    purpose: str
    insight_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    visual_ids: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


class ReportExecutionState(BaseModel):
    request_id: str
    user_goal: str
    status: str = "planning"
    pending_theme_ids: list[str] = Field(default_factory=list)
    completed_theme_ids: list[str] = Field(default_factory=list)
    themes: list[ThemeEvidenceState] = Field(default_factory=list)
    visual_assignments: dict[str, list[str]] = Field(default_factory=dict)
    validation_issues: list[str] = Field(default_factory=list)


def build_report_execution_state(state: dict[str, Any]) -> ReportExecutionState:
    """Create the canonical bounded state; full evidence remains in GraphState."""
    insights = [item for item in state.get("analysis_insights") or [] if item.evidence_valid]
    insight_by_id = {item.insight_id: item for item in insights}
    visuals = state.get("generated_visuals") or []
    report_plan = state.get("report_plan")
    themes: list[ThemeEvidenceState] = []
    assignments: dict[str, list[str]] = {}
    for index, theme in enumerate(report_plan.themes if report_plan else [], start=1):
        theme_insights = [insight_by_id[item] for item in theme.insight_ids if item in insight_by_id]
        question_ids = list(dict.fromkeys(
            question_id for insight in theme_insights for question_id in insight.evidence_question_ids
        ))
        visual_ids = [
            visual.visual_id for visual in visuals
            if set(question_ids).intersection(visual.evidence_question_ids)
        ]
        theme_id = f"theme_{index}"
        assignments[theme_id] = visual_ids
        themes.append(ThemeEvidenceState(
            theme_id=theme_id,
            title=theme.title,
            purpose=theme.purpose,
            insight_ids=[item.insight_id for item in theme_insights],
            question_ids=question_ids,
            visual_ids=visual_ids,
            evidence_references=[f"computed_result:{item}" for item in question_ids],
            findings=[item.finding or item.narrative for item in theme_insights],
        ))
    return ReportExecutionState(
        request_id=state["request_id"],
        user_goal=state.get("instructions", ""),
        pending_theme_ids=[item.theme_id for item in themes],
        themes=themes,
        visual_assignments=assignments,
        validation_issues=[state.get("error_message")] if state.get("error_message") else [],
    )
