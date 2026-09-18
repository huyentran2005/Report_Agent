import logging
import pandas as pd
import json
import os
import requests
import time
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llm import get_llm, text_from_response
from pydantic import BaseModel, Field, ValidationError
from graph.state import GraphState
from schemas.messages import VisualGenerationInstruction, GeneratedVisual
from data_io import (effective_instructions, infer_column_semantic, is_date_like_series, read_dataset_partitions,
                     split_partitions_by_source, to_datetime_series)
from privacy import is_person_name_column
from agents.question_framer import compact_computed_results

logger = logging.getLogger(__name__)

CHART_OUTPUT_DIR = os.path.join("local_app_data", "charts")
os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)



CHART_NAVY = "#1C3155"
CHART_TEAL = "#008F87"
CHART_ORANGE = "#E87524"
CHART_TEXT = "#26354A"
CHART_MUTED = "#74849B"
CHART_GRID = "#E3E9EF"
MAX_CATEGORY_LABELS = 12
MAX_CATEGORY_LABEL_LENGTH = 28



class SuggestedVisualizations(BaseModel):
    suggestions: List[VisualGenerationInstruction] = Field(
        description="A list of suggested visualizations, including chart type, columns, title, and description.")


def _is_readable_category(series: pd.Series) -> bool:
    """Return False when a categorical axis would be too crowded to read."""
    values = series.dropna()
    if values.empty:
        return False
    unique_values = values.astype(str).drop_duplicates()
    if len(unique_values) > MAX_CATEGORY_LABELS:
        return False
    return int(unique_values.str.len().max()) <= MAX_CATEGORY_LABEL_LENGTH


def _is_readable_instruction(item: VisualGenerationInstruction, df: pd.DataFrame) -> bool:
    if not item.columns or not all(column in df.columns for column in item.columns):
        return False
    if any(is_person_name_column(column, df[column]) for column in item.columns):
        return False
    if any(infer_column_semantic(df[column], column) in {"identifier", "text", "unknown"}
           for column in item.columns):
        return False
    chart_type = item.type.lower()
    if chart_type == "pie":
        return len(item.columns) == 1 and df[item.columns[0]].nunique(dropna=True) <= 6
    if chart_type == "bar":
        axis_column = item.columns[0]
        if not pd.api.types.is_numeric_dtype(df[axis_column]):
            return _is_readable_category(df[axis_column])
    if chart_type == "line" and item.columns:
        axis_column = item.columns[0]
        if not _looks_like_date(df[axis_column]) and not pd.api.types.is_numeric_dtype(df[axis_column]):
            return _is_readable_category(df[axis_column])
    return True


def _fallback_suggestions(df: pd.DataFrame) -> List[VisualGenerationInstruction]:
    """Choose charts that a non-technical reader can understand quickly."""
    numeric = [column for column in df.select_dtypes(include="number").columns.tolist()
               if not is_person_name_column(column, df[column])
               and not is_date_like_series(df[column], str(column))]
    safe_columns = [column for column in df.columns if not str(column).startswith("_")
                    and not is_person_name_column(column, df[column])]
    date_column = next((column for column in safe_columns if _looks_like_date(df[column])), None)
    categorical = [
        column for column in safe_columns
        if column != date_column and column not in numeric and _is_readable_category(df[column])
    ]
    suggestions: List[VisualGenerationInstruction] = []
    if date_column and numeric:
        for value_column in numeric[:2]:
            suggestions.append(VisualGenerationInstruction(type="line", columns=[date_column, value_column], title=f"Xu hướng {value_column} theo thời gian", description=f"Diễn biến {value_column} theo các kỳ tháng được chuẩn hóa từ {date_column}."))
    if categorical and numeric:
        for group_column in categorical[:2]:
            suggestions.append(VisualGenerationInstruction(type="bar", columns=[group_column, numeric[0]], title=f"{numeric[0]} theo {group_column}", description="So sánh giá trị giữa các nhóm bằng các cột dễ đọc."))
    if categorical:
        suggestions.append(VisualGenerationInstruction(type="bar", columns=[categorical[0]], title=f"Số lượng theo {categorical[0]}", description="Đếm số bản ghi trong từng nhóm."))
    if not suggestions and numeric:
        suggestions.append(VisualGenerationInstruction(type="bar", columns=[numeric[0]], title=f"Giá trị {numeric[0]}", description="Tóm tắt các giá trị bằng biểu đồ cột."))
    return suggestions[:4]


def _suggestions_from_results(df: pd.DataFrame, results) -> List[VisualGenerationInstruction]:
    """Create charts only for selected questions whose audited operation benefits from a chart."""
    suggestions = []
    seen = set()
    for result in results or []:
        if isinstance(result.result, list):
            plotted_values = []
            for record in result.result:
                if isinstance(record, dict):
                    plotted_values.extend(
                        value for key, value in record.items()
                        if key not in {result.columns[0], "period"}
                        and isinstance(value, (int, float)) and pd.notna(value)
                    )
            if len(plotted_values) < 2 or pd.Series(plotted_values).nunique() < 2:
                continue
        if (result.operation == "trend_over_time" and len(result.columns) == 2
                and result.parameters.get("aggregation") == "sum"):
            chart_type = "line"
        elif result.operation in {"mean_by_group", "count_by_category"}:
            chart_type = "bar"
        elif (result.operation == "top_n" and len(result.columns) == 1
              and result.parameters.get("aggregation") == "count"):
            chart_type = "bar"
        elif result.operation == "distribution_summary" and len(result.columns) == 1:
            chart_type = "histogram"
        elif result.operation == "correlation" and len(result.columns) == 2:
            chart_type = "scatter"
        else:
            continue
        key = (chart_type, tuple(result.columns))
        if key in seen:
            continue
        seen.add(key)
        if result.operation == "trend_over_time":
            description = f"Diễn biến {result.columns[-1]} theo các kỳ thời gian hợp lệ."
        elif result.operation in {"mean_by_group", "count_by_category", "top_n"}:
            description = f"So sánh kết quả giữa các nhóm của {result.columns[0]}."
        elif result.operation == "correlation":
            description = f"Mối liên hệ quan sát được giữa {result.columns[0]} và {result.columns[1]}."
        else:
            description = f"Phân phối các giá trị hợp lệ của {result.columns[0]}."
        item = VisualGenerationInstruction(
            type=chart_type,
            columns=result.columns,
            title=_chart_title_from_result(result),
            description=description,
            evidence_question_ids=[result.question_id],
        )
        if _is_readable_instruction(item, df):
            suggestions.append(item)
    return suggestions[:4]


def _chart_title_from_result(result) -> str:
    """Create a declarative chart title without exposing an internal question."""
    if result.operation == "trend_over_time":
        return f"Xu hướng {result.columns[-1]} theo {result.columns[0]}"
    if result.operation == "mean_by_group" and len(result.columns) >= 2:
        return f"Trung bình {result.columns[1]} theo {result.columns[0]}"
    if result.operation == "count_by_category":
        return f"Số lượng theo {result.columns[0]}"
    if result.operation == "top_n":
        return f"Các nhóm dẫn đầu theo {result.columns[0]}"
    if result.operation == "distribution_summary":
        return f"Phân phối {result.columns[0]}"
    if result.operation == "correlation" and len(result.columns) >= 2:
        return f"Mối liên hệ giữa {result.columns[0]} và {result.columns[1]}"
    return f"Phân tích {' và '.join(result.columns)}"


def _looks_like_date(series: pd.Series) -> bool:
    return is_date_like_series(series)


def _reader_friendly(items: List[VisualGenerationInstruction], df: pd.DataFrame) -> List[VisualGenerationInstruction]:
    """Remove specialist chart types from a report intended for general readers."""
    friendly = [
        item for item in items
        if (item.type.lower() in {"bar", "line", "histogram", "boxplot", "scatter"}
            or (item.type.lower() == "pie" and len(item.columns) == 1))
        and _is_readable_instruction(item, df)
    ]
    return friendly[:6]


def _generate_chart_from_audited_result(result, output_path: str) -> Optional[str]:
    """Render group/trend charts from the exact audited records, not raw re-aggregation."""
    if not isinstance(result.result, list) or len(result.result) < 2:
        return None
    records = [record for record in result.result if isinstance(record, dict)]
    if len(records) < 2:
        return None
    dimension = "period" if result.operation == "trend_over_time" else result.columns[0]
    if any(dimension not in record for record in records):
        return None
    metric_keys = [
        key for key in records[0]
        if key not in {dimension, "sample_size"}
        and all(isinstance(record.get(key), (int, float)) for record in records)
    ]
    if not metric_keys:
        return None
    metric = metric_keys[0]
    values = pd.Series([record[metric] for record in records], dtype="float64")
    if values.nunique(dropna=True) < 2:
        return None

    labels = [str(record[dimension]) for record in records]
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="white")
    ax.set_facecolor("white")
    if result.operation == "trend_over_time":
        ax.plot(labels, values, color=CHART_NAVY, marker="o", linewidth=2.2)
        chart_code = f"audited_result.plot.line(x={dimension!r}, y={metric!r})"
    else:
        bars = ax.bar(labels, values, color=CHART_TEAL)
        ax.bar_label(bars, fmt="{:,.2f}", padding=3, fontsize=8, color=CHART_TEXT)
        chart_code = f"audited_result.plot.bar(x={dimension!r}, y={metric!r})"
    ax.set_title(_chart_title_from_result(result), color=CHART_NAVY, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(str(dimension).replace("_", " ").title(), fontweight="bold", color=CHART_TEXT)
    ax.set_ylabel(str(metric).replace("_", " ").title(), fontweight="bold", color=CHART_TEXT)
    ax.tick_params(axis="x", rotation=30, labelsize=8, colors=CHART_TEXT)
    ax.tick_params(axis="y", labelsize=9, colors=CHART_TEXT)
    ax.grid(axis="y", color=CHART_GRID, linewidth=0.7, alpha=0.85)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_code


def generate_chart(df: pd.DataFrame, instruction: VisualGenerationInstruction, output_path: str) -> Optional[str]:
    """
    Generates a chart based on the instruction and saves it to the output path.
    Returns the file_path if successful, None otherwise.
    """
    if any(column in df.columns and is_person_name_column(column, df[column])
           for column in instruction.columns):
        logger.warning("Skipping chart that could expose or group personal names: %s", instruction.columns)
        return None
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="white")
    ax.set_facecolor("white")
    chart_code_str = ""

    try:
        if not instruction.columns or not all(col in df.columns for col in instruction.columns):
            missing_cols = [col for col in instruction.columns if col not in df.columns]
            logger.warning(
                f"Skipping chart generation: Missing or invalid columns {missing_cols} for instruction: {instruction.model_dump_json()}")
            plt.close(fig)
            return None

        if not _is_readable_instruction(instruction, df):
            logger.warning(
                "Skipping unreadable chart with too many or overly long category labels: %s",
                instruction.model_dump_json(),
            )
            plt.close(fig)
            return None

        for index, col in enumerate(instruction.columns):
            numeric_required = instruction.type in ["histogram", "boxplot", "scatter"] or (instruction.type == "bar" and len(instruction.columns) == 2 and index == 1) or (instruction.type == "line" and index == 1)
            if numeric_required and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(
                    f"Skipping chart {instruction.type}: Column '{col}' is not numeric for numeric plot type. Instruction: {instruction.model_dump_json()}")
                plt.close(fig)
                return None

        if instruction.type == "pie":
            column = instruction.columns[0] if instruction.columns else None
            if column and len(instruction.columns) == 1:
                counts = df[column].value_counts().head(8)
                if len(counts) < 2 or counts.sum() <= 0:
                    logger.warning("Skipping pie chart without at least two populated groups: %s", column)
                    plt.close(fig)
                    return None
                ax.pie(
                    counts.values, labels=counts.index.astype(str), autopct="%1.0f%%",
                    startangle=90, colors=sns.color_palette([CHART_NAVY, CHART_TEAL, CHART_ORANGE], len(counts)),
                    wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
                    textprops={"color": CHART_TEXT, "fontsize": 9},
                )
                ax.set_title(instruction.title or instruction.description)
                chart_code_str = f"df['{column}'].value_counts().head(8).plot.pie(autopct='%1.0f%%')"
            else:
                plt.close(fig)
                return None
        elif instruction.type == "bar":
            if len(instruction.columns) == 2:
                x_col, y_col = instruction.columns[0], instruction.columns[1]
                chart_data = df[[x_col, y_col]].dropna()
                grouped_values = chart_data.groupby(x_col)[y_col].mean()
                if len(grouped_values) < 2 or grouped_values.nunique() < 2:
                    logger.warning("Skipping bar chart without at least two distinct plotted values: %s", instruction.columns)
                    plt.close(fig)
                    return None
                sns.barplot(x=x_col, y=y_col, data=chart_data, ax=ax, color=CHART_NAVY, errorbar=None)
                chart_code_str = f"sns.barplot(x='{x_col}', y='{y_col}', data=df, ax=ax)"
            elif len(instruction.columns) == 1:
                column = instruction.columns[0]
                if pd.api.types.is_numeric_dtype(df[column]):

                    logger.warning("Skipping one-point numeric bar chart: %s", column)
                    plt.close(fig)
                    return None
                else:
                    counts = df[column].value_counts(dropna=True)
                    if len(counts) < 2:
                        logger.warning("Skipping count chart without at least two populated groups: %s", column)
                        plt.close(fig)
                        return None
                    sns.countplot(x=column, data=df, ax=ax, color=CHART_TEAL)
                    chart_code_str = f"sns.countplot(x='{column}', data=df, ax=ax)"
            else:
                logger.warning(
                    f"Bar chart with {len(instruction.columns)} columns not fully supported without more specific instruction: {instruction.model_dump_json()}")
                plt.close(fig)
                return None
        elif instruction.type == "line":
            if len(instruction.columns) == 2:
                x_col, y_col = instruction.columns[0], instruction.columns[1]
                if _looks_like_date(df[x_col]):
                    df_temp = df.copy()
                    df_temp[x_col] = to_datetime_series(df_temp[x_col])
                    df_temp = df_temp.dropna(subset=[x_col, y_col]).sort_values(by=x_col)


                    df_sorted = (
                        df_temp.set_index(x_col)[y_col]
                        .resample("MS").sum(min_count=1).dropna().reset_index()
                    )
                else:
                    df_sorted = df.sort_values(by=x_col).dropna(subset=[x_col, y_col])

                if df_sorted.empty:
                    logger.warning(
                        f"Skipping line chart due to no valid data after date conversion/sorting for columns: {instruction.columns}. Instruction: {instruction.model_dump_json()}")
                    plt.close(fig)
                    return None
                numeric_y = pd.to_numeric(df_sorted[y_col], errors="coerce").dropna()
                if len(numeric_y) < 2 or numeric_y.nunique() < 2:
                    logger.warning("Skipping line chart without at least two distinct finite points: %s", instruction.columns)
                    plt.close(fig)
                    return None

                sns.lineplot(
                    x=x_col, y=y_col, data=df_sorted, ax=ax, color=CHART_NAVY,
                    marker="o", markersize=5, linewidth=2.2, errorbar=None,
                )
                chart_code_str = (
                    f"df_temp = df.copy()\n"
                    f"df_temp[{x_col!r}] = to_datetime_series(df_temp[{x_col!r}])\n"
                    f"df_sorted = df_temp.set_index({x_col!r})[{y_col!r}].resample('MS').sum(min_count=1).reset_index()"
                    if _looks_like_date(df[x_col]) else
                    f"df_sorted = df.sort_values(by={x_col!r})"
                )
            else:
                logger.warning(
                    f"Line chart with {len(instruction.columns)} columns not supported: {instruction.model_dump_json()}")
                plt.close(fig)
                return None
        elif instruction.type == "scatter":
            if len(instruction.columns) == 2:
                x_col, y_col = instruction.columns[0], instruction.columns[1]
                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
                chart_code_str = f"sns.scatterplot(x='{x_col}', y='{y_col}', data=df, ax=ax)"
            else:
                logger.warning(
                    f"Scatter chart requires 2 columns, got {len(instruction.columns)}: {instruction.model_dump_json()}")
                plt.close(fig)
                return None
        elif instruction.type == "histogram":
            if len(instruction.columns) == 1:
                sns.histplot(df[instruction.columns[0]], kde=True, ax=ax)
                chart_code_str = f"sns.histplot(df['{instruction.columns[0]}'], kde=True, ax=ax)"
            else:
                logger.warning(
                    f"Histogram requires 1 column, got {len(instruction.columns)}: {instruction.model_dump_json()}")
                plt.close(fig)
                return None
        elif instruction.type == "boxplot":
            if len(instruction.columns) == 1:
                sns.boxplot(y=df[instruction.columns[0]], ax=ax)
                chart_code_str = f"sns.boxplot(y=df['{instruction.columns[0]}'], ax=ax)"
            elif len(instruction.columns) == 2:
                x_col, y_col = instruction.columns[0], instruction.columns[1]
                sns.boxplot(x=x_col, y=y_col, data=df, ax=ax)
                chart_code_str = f"sns.boxplot(x='{x_col}', y='{y_col}', data=df, ax=ax)"
            else:
                logger.warning(
                    f"Boxplot with {len(instruction.columns)} columns not fully supported: {instruction.model_dump_json()}")
                plt.close(fig)
                return None
        else:
            logger.warning(f"Unsupported chart type: {instruction.type}")
            plt.close(fig)
            return None

        if instruction.title:
            ax.set_title(instruction.title)
        else:
            ax.set_title(instruction.description)

        ax.set_title(ax.get_title(), color=CHART_NAVY, fontsize=14, fontweight="bold", pad=14)
        ax.tick_params(axis="both", colors=CHART_TEXT, labelsize=9)
        ax.xaxis.label.set_color(CHART_TEXT)
        ax.yaxis.label.set_color(CHART_TEXT)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CHART_MUTED)
            ax.spines[side].set_linewidth(0.8)
        if instruction.type != "pie":
            ax.grid(axis="y", color=CHART_GRID, linewidth=0.7, alpha=0.85)
            ax.set_axisbelow(True)
        if instruction.type == "bar":
            for container in ax.containers:
                try:
                    ax.bar_label(container, fmt="{:,.1f}", padding=3, fontsize=8,
                                 color=CHART_TEXT, fontweight="bold")
                except (TypeError, ValueError):
                    pass
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            legend.get_frame().set_edgecolor(CHART_GRID)
            legend.get_frame().set_facecolor("white")
            for text_item in legend.get_texts():
                text_item.set_color(CHART_TEXT)
        if not ax.has_data():
            logger.warning("Skipping empty chart after rendering: %s", instruction.columns)
            plt.close(fig)
            return None

        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Chart saved to: {output_path}")
        return chart_code_str

    except Exception as e:
        logger.error(f"Error generating {instruction.type} chart for columns {instruction.columns}: {e}", exc_info=True)
        plt.close(fig)
        return None


def generate_visuals(state: GraphState) -> GraphState:
    """
    Suggests and generates data visualizations based on the data profile and user instructions.
    """
    request_id = state['request_id']
    file_path = state['file_path']
    instructions = effective_instructions(
        state['instructions'], state.get("workbook_instruction_context")
    )
    dataframe_profile = state['dataframe_profile']
    computed_results = state.get("computed_question_results") or []
    analysis_insights = state.get("analysis_insights") or []
    validated_question_ids = {
        question_id for insight in analysis_insights if insight.evidence_valid
        for question_id in insight.evidence_question_ids
    }
    computed_results = [
        item for item in computed_results if item.question_id in validated_question_ids
    ]




    logger.info(f"VisualizationNode processing request: {request_id}")
    logger.info("Visualization started with status: %s", state["status"])


    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if state.get("llm_provider", "gemini") == "gemini" and not gemini_api_key:
        logger.error(f"GEMINI_API_KEY not found for request {request_id}. Please ensure it's set in your .env file.")
        state['status'] = "error"
        state['error_message'] = "API key for Gemini not found. Please set GEMINI_API_KEY in your .env file."
        return state

    llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
    try:
        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))

    except Exception as e:
        logger.error(f"Failed to initialize LLM for visualization: {e}", exc_info=True)
        state['status'] = "error"
        state['error_message'] = f"Failed to initialize LLM for visualization: {e}"
        return state


    try:
        partitions = split_partitions_by_source(read_dataset_partitions(
            file_path, state.get("workbook_sheets"), state.get("instructions", "")
        ))
        df = next(iter(partitions.values()))
        if not partitions or any(frame.empty for frame in partitions.values()):
            raise ValueError("Uploaded CSV is empty.")
    except Exception as e:
        logger.error(f"Error loading data for visualization for request {request_id}: {e}", exc_info=True)
        state['status'] = "error"
        state['error_message'] = f"Failed to load or process CSV for visualization: {e}"
        return state




    generated_visuals_list: List[GeneratedVisual] = []
    seen_evidence = set()
    for result in computed_results:
        if result.operation not in {"mean_by_group", "count_by_category", "top_n", "trend_over_time"}:
            continue
        signature = (result.source_partition, result.operation, tuple(result.columns))
        if signature in seen_evidence:
            continue
        seen_evidence.add(signature)
        visual_id = f"chart_{request_id}_{len(generated_visuals_list) + 1}"
        output_file_path = os.path.join(CHART_OUTPUT_DIR, f"{visual_id}.png")
        chart_code = _generate_chart_from_audited_result(result, output_file_path)
        if not chart_code:
            continue
        description = (
            f"Diễn biến {result.columns[-1]} theo thời gian."
            if result.operation == "trend_over_time"
            else f"So sánh {result.columns[-1]} giữa các nhóm {result.columns[0]}."
        )
        generated_visuals_list.append(GeneratedVisual(
            visual_id=visual_id, type="line" if result.operation == "trend_over_time" else "bar",
            description=description, file_path=output_file_path,
            suggested_section=result.source_partition or "Analysis", chart_code=chart_code,
            evidence_question_ids=[result.question_id],
        ))
        if len(generated_visuals_list) >= 6:
            break
    if generated_visuals_list:
        state["generated_visuals"] = generated_visuals_list
        state["status"] = "visuals_generated"
        logger.info("Generated %s charts directly from audited result records.", len(generated_visuals_list))
        return state

    if len(partitions) > 1:
        generated_visuals_list: List[GeneratedVisual] = []
        visual_index = 0
        for partition_name, partition_df in partitions.items():
            partition_results = [item for item in computed_results
                                 if item.source_partition == partition_name]
            instructions_for_partition = _suggestions_from_results(partition_df, partition_results)
            for instruction in instructions_for_partition[:4]:
                if visual_index >= 6:
                    break
                visual_index += 1
                visual_id = f"chart_{request_id}_{visual_index}"
                output_file_path = os.path.join(CHART_OUTPUT_DIR, f"{visual_id}.png")
                chart_code = generate_chart(partition_df, instruction, output_file_path)
                if chart_code:
                    generated_visuals_list.append(GeneratedVisual(
                        visual_id=visual_id, type=instruction.type,
                        description=instruction.description,
                        file_path=output_file_path,
                        suggested_section=partition_name, chart_code=chart_code,
                        evidence_question_ids=instruction.evidence_question_ids,
                    ))
            if visual_index >= 6:
                break
        state['generated_visuals'] = generated_visuals_list
        state['status'] = "visuals_generated"
        return state

    max_retries = 3
    base_delay = 2
    llm_raw_output_str = ""

    for attempt in range(max_retries):

        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to invoke LLM for visualization suggestions...")

            parser = JsonOutputParser(pydantic_object=SuggestedVisualizations)
            prompt = PromptTemplate(
                template="""
                Bạn là chuyên gia trực quan hóa dữ liệu cho nhiều lĩnh vực.
                Hãy nhận diện lĩnh vực từ tên cột và yêu cầu người dùng, rồi đề xuất các biểu đồ
                phù hợp với ngữ nghĩa của dữ liệu. Không mặc định các cột là doanh thu, khách hàng,
                sản phẩm hoặc lợi nhuận nếu dữ liệu không thể hiện những khái niệm đó.

                Với mỗi đề xuất, cung cấp:
                - `type`: Dùng 'bar' cho so sánh và 'line' cho xu hướng. Chỉ dùng 'pie' khi thể hiện
                  tỷ trọng của một số ít nhóm.
                - `columns`: Danh sách 1 hoặc 2 tên cột chính xác trong dữ liệu. Cột phải tồn tại và
                  phù hợp với loại biểu đồ. Với xu hướng thời gian, cột đầu là thời gian và cột sau là số.
                  Với biểu đồ cột một biến phân loại, hệ thống sẽ đếm số bản ghi.
                Không đề xuất bar/pie/line theo cột phân loại có hơn 12 giá trị khác nhau hoặc nhãn quá dài
                  (ví dụ mã định danh, tên sản phẩm chi tiết, tên người hoặc nội dung văn bản tự do).
                  Không dùng cột được đánh dấu `is_sensitive_person_name`; không hiển thị, đếm,
                  xếp hạng hoặc phân nhóm tên người trên biểu đồ.
                - `title`: Tiêu đề ngắn, chuyên nghiệp và phù hợp với lĩnh vực.
                - `description`: Mô tả ngắn về điều biểu đồ giúp người đọc đánh giá.
                - `suggested_section`: Tên phần báo cáo phù hợp với nội dung, không dùng tên phần
                  mang tính kinh doanh nếu lĩnh vực không phải kinh doanh.

                Cột và thông tin dữ liệu (bỏ qua cột metadata bắt đầu bằng `_`):
                {column_details_json}

                Yêu cầu người dùng: {instructions}
                Các insight đã được xác thực trước khi chọn biểu đồ: {insights}
                Các câu hỏi/kết quả pandas đã chọn: {computed_results}

                Đề xuất 2-6 biểu đồ có giá trị. Ưu tiên dễ hiểu hơn kỹ thuật phức tạp.
                Nếu tồn tại ít nhất một tổ hợp cột hợp lệ, bắt buộc đề xuất biểu đồ thay vì trả danh sách rỗng.
                Không dùng scatter, histogram, boxplot, heatmap hoặc biểu đồ chuyên biệt.
                Cân nhắc kiểu dữ liệu và phân phối; nếu một cột giống thời gian, có thể đề xuất biểu đồ xu hướng.
                Không đổi, dịch hoặc tự suy ra mốc thời gian; biểu đồ phải sử dụng trực tiếp cột thời gian đã cho.
                Tên cột phải khớp chính xác với `column_details_json`.

                {format_instructions}

                Chỉ trả về JSON hợp lệ.
                """,
                input_variables=["column_details_json", "instructions", "insights", "computed_results"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            suggested_visuals: List[VisualGenerationInstruction] = []
            logger.info("Requesting visualization suggestions from the LLM.")

            profile_payload = dataframe_profile.model_dump(exclude_unset=True)
            profile_payload["column_details"] = {
                name: details for name, details in profile_payload.get("column_details", {}).items()
                if not details.get("is_sensitive_person_name") and not is_person_name_column(name)
            }
            column_details_json = json.dumps(profile_payload, ensure_ascii=False, indent=2)
            llm_response = llm.invoke(prompt.invoke({
                "column_details_json": column_details_json,
                "instructions": instructions,
                "insights": json.dumps([item.model_dump(mode="json") for item in analysis_insights], ensure_ascii=False),
                "computed_results": json.dumps(
                    compact_computed_results(computed_results, max_items=8), ensure_ascii=False
                ),
            }))
            llm_raw_output_str = text_from_response(llm_response)

            stripped_str = llm_raw_output_str.strip()
            if stripped_str.startswith("```json") and stripped_str.endswith("```"):
                json_str = stripped_str[len("```json"):-len("```")].strip()
            else:
                json_str = stripped_str

            parsed_suggestions_output = SuggestedVisualizations.model_validate_json(json_str)
            audited_suggestions = _suggestions_from_results(df, computed_results)
            audited_by_key = {
                (tuple(item.columns), item.type): item for item in audited_suggestions
            }
            grounded_suggestions = [
                item for item in parsed_suggestions_output.suggestions
                if (tuple(item.columns), item.type.lower()) in audited_by_key
            ]
            for item in grounded_suggestions:
                audited = audited_by_key[(tuple(item.columns), item.type.lower())]
                item.title = audited.title
                item.description = audited.description
                item.evidence_question_ids = audited.evidence_question_ids
            suggested_visuals = _reader_friendly(grounded_suggestions, df)
            if not suggested_visuals:
                suggested_visuals = _suggestions_from_results(df, computed_results)
            logger.info("LLM returned %s visualization suggestions.", len(suggested_visuals))
            break

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
            logger.error(f"LLM output for request {request_id} was invalid JSON or failed Pydantic validation: {e}",
                        exc_info=True)
            logger.error(f"Raw LLM Output: {llm_raw_output_str[:500]}...")
            state['status'] = "error"
            state[
                'error_message'] = f"LLM output for visualization suggestions was invalid JSON or schema: {e}. Raw LLM Output: {llm_raw_output_str[:500]}..."
            return state
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during LLM call for visualization suggestion for request {request_id}: {e}",
                exc_info=True)
            state['status'] = "error"
            state['error_message'] = f"An unexpected error occurred during visualization suggestion LLM call: {e}"
            return state
    if 'suggested_visuals' not in locals():
        state['status'] = "error"
        state['error_message'] = "An unexpected failure occurred after all LLM retries."
        return state

    generated_visuals_list: List[GeneratedVisual] = []
    logger.info("Starting chart generation for %s suggestions.", len(suggested_visuals))
    for i, instruction in enumerate(suggested_visuals):
        visual_id = f"chart_{request_id}_{i + 1}"
        output_filename = f"{visual_id}.png"
        output_file_path = os.path.join(CHART_OUTPUT_DIR, output_filename)

        logger.info(
            "Generating chart %s: type=%s, columns=%s, description=%r",
            i + 1, instruction.type, instruction.columns, instruction.description)

        chart_code = generate_chart(df, instruction, output_file_path)

        if chart_code:
            generated_visuals_list.append(GeneratedVisual(
                visual_id=visual_id,
                type=instruction.type,
                description=instruction.description,
                file_path=output_file_path,
                suggested_section=instruction.suggested_section if instruction.suggested_section else "Analysis",
                chart_code=chart_code,
                evidence_question_ids=instruction.evidence_question_ids,
            ))
            logger.info("Chart %s generated successfully: %s", i + 1, output_file_path)
        else:
            logger.warning("Chart %s failed to generate; skipping it.", i + 1)

    state['generated_visuals'] = generated_visuals_list
    state['status'] = "visuals_generated"
    logger.info(
        f"VisualizationNode completed for request: {request_id}. Generated {len(generated_visuals_list)} visuals.")
    return state

