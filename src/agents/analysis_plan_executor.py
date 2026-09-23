"""Validate and execute declarative analysis plans with Pandas."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from data_io import infer_column_semantic, to_datetime_series
from privacy import is_person_name_column
from schemas.messages import AnalysisFilter, AnalysisTransform, StructuredAnalysisPlan


class InvalidAnalysisPlan(ValueError):
    """Raised when a plan is unsafe, inconsistent, or cannot answer the question."""


_NUMERIC_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "std"}
_TIME_FREQUENCIES = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        clean = value.copy()
        numeric_columns = clean.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            clean[column] = clean[column].mask(~np.isfinite(clean[column]))
        clean = clean.astype(object).where(pd.notna(clean), None)
        return clean.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return _json_safe(value.rename(value.name or "value").reset_index())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64, pd.Period)):
        return str(value)
    return value


def _referenced_columns(plan: StructuredAnalysisPlan) -> list[str]:
    columns = [*plan.group_by, *plan.metrics]
    columns.extend(item.column for item in plan.filters if item.column)
    for transform in plan.transforms:
        columns.extend(value for value in (transform.column, transform.numerator, transform.denominator) if value)
    return list(dict.fromkeys(columns))


def validate_analysis_plan(df: pd.DataFrame, plan: StructuredAnalysisPlan) -> list[str]:
    columns = _referenced_columns(plan)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise InvalidAnalysisPlan(f"Cột không tồn tại: {missing}")
    metadata = [column for column in columns if str(column).startswith("_")]
    if metadata:
        raise InvalidAnalysisPlan(f"Không được phân tích cột metadata: {metadata}")
    sensitive = [column for column in columns if is_person_name_column(column, df[column])]
    if sensitive:
        raise InvalidAnalysisPlan(f"Không được phân tích cột tên cá nhân: {sensitive}")
    if not plan.group_by and not plan.metrics and not plan.transforms:
        raise InvalidAnalysisPlan("Plan phải có group_by, metrics hoặc transform phân tích.")
    if plan.time_grain:
        if not plan.group_by:
            raise InvalidAnalysisPlan("time_grain yêu cầu ít nhất một cột group_by thời gian.")
        parsed = to_datetime_series(df[plan.group_by[0]])
        if parsed.notna().mean() < 0.7:
            raise InvalidAnalysisPlan(f"Cột {plan.group_by[0]!r} không đủ dữ liệu thời gian hợp lệ.")
    if plan.aggregation in _NUMERIC_AGGREGATIONS:
        invalid = [column for column in plan.metrics
                   if infer_column_semantic(df[column], column) != "numeric_measure"]
        if invalid:
            raise InvalidAnalysisPlan(f"Aggregation {plan.aggregation} yêu cầu numeric_measure: {invalid}")
    return columns


def _apply_filter(work: pd.DataFrame, spec: AnalysisFilter) -> pd.DataFrame:
    series = work[spec.column]
    op, value = spec.operator, spec.value
    is_datetime = (
        pd.api.types.is_datetime64_any_dtype(series)
        or infer_column_semantic(series, spec.column) == "datetime"
        or series.dropna().map(
            lambda item: isinstance(item, (date, datetime, pd.Timestamp, np.datetime64))
        ).mean() >= 0.7
    )
    if is_datetime:
        series = to_datetime_series(series)
        if op == "between" and isinstance(value, list):
            value = [pd.to_datetime(item, errors="coerce") for item in value]
        elif op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
            value = pd.to_datetime(value, errors="coerce")
            if pd.isna(value):
                raise InvalidAnalysisPlan(
                    f"Giá trị filter thời gian không hợp lệ cho {spec.column!r}: {spec.value!r}"
                )
    if op == "eq":
        mask = series == value
    elif op == "ne":
        mask = series != value
    elif op == "gt":
        mask = series > value
    elif op == "gte":
        mask = series >= value
    elif op == "lt":
        mask = series < value
    elif op == "lte":
        mask = series <= value
    elif op in {"in", "not_in"}:
        if not isinstance(value, list):
            raise InvalidAnalysisPlan(f"Filter {op} yêu cầu value là một danh sách.")
        mask = series.isin(value)
        if op == "not_in":
            mask = ~mask
    elif op == "contains":
        mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
    elif op == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise InvalidAnalysisPlan("Filter between yêu cầu value=[min, max].")
        mask = series.between(value[0], value[1])
    elif op == "is_null":
        mask = series.isna()
    else:
        mask = series.notna()
    return work.loc[mask]


def _metric_columns(frame: pd.DataFrame, plan: StructuredAnalysisPlan) -> list[str]:
    excluded = set(plan.group_by) | {"sample_size"}
    return [column for column in frame.columns if column not in excluded]


def _transform(frame: pd.DataFrame, spec: AnalysisTransform, plan: StructuredAnalysisPlan) -> pd.DataFrame:
    metrics = _metric_columns(frame, plan)
    column = spec.column or (metrics[0] if metrics else None)
    output = spec.output_column
    if spec.type == "correlation":
        left, right = spec.numerator, spec.denominator
        if not left or not right:
            if len(plan.metrics) != 2:
                raise InvalidAnalysisPlan("correlation yêu cầu đúng hai metrics.")
            left, right = plan.metrics
        value = frame[[left, right]].dropna()[left].corr(frame[[left, right]].dropna()[right])
        return pd.DataFrame([{"metric_x": left, "metric_y": right, "pearson_correlation": value,
                              "sample_size": int(frame[[left, right]].dropna().shape[0])}])
    if spec.type == "distribution":
        if not column:
            raise InvalidAnalysisPlan("distribution yêu cầu column.")
        values = frame[column].dropna()
        return pd.DataFrame([{"column": column, "count": values.count(), "mean": values.mean(),
                              "std": values.std(), "min": values.min(), "q25": values.quantile(.25),
                              "median": values.median(), "q75": values.quantile(.75), "max": values.max()}])
    if spec.type == "outlier_iqr":
        if not column:
            raise InvalidAnalysisPlan("outlier_iqr yêu cầu column.")
        values = frame[column].dropna()
        q1, q3 = values.quantile(.25), values.quantile(.75)
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return pd.DataFrame([{"column": column, "count": values.count(), "q25": q1, "q75": q3,
                              "lower_bound": lower, "upper_bound": upper,
                              "outlier_count": ((values < lower) | (values > upper)).sum()}])
    if spec.type == "ratio":
        if not spec.numerator or not spec.denominator:
            raise InvalidAnalysisPlan("ratio yêu cầu numerator và denominator.")
        denominator = frame[spec.denominator].replace(0, np.nan)
        frame[output or f"ratio_{spec.numerator}_to_{spec.denominator}"] = frame[spec.numerator] / denominator
    elif spec.type == "share_of_total":
        if not column:
            raise InvalidAnalysisPlan("share_of_total yêu cầu một metric.")
        total = frame[column].sum()
        if total == 0:
            raise InvalidAnalysisPlan("Không thể tính tỷ trọng khi tổng bằng 0.")
        frame[output or f"share_percent_{column}"] = frame[column] / total * 100
    elif spec.type == "pct_change":
        if not column:
            raise InvalidAnalysisPlan("pct_change yêu cầu một metric.")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[output or f"pct_change_{column}"] = numeric.pct_change(
            periods=spec.periods, fill_method=None
        ) * 100
    elif spec.type == "difference":
        if not column:
            raise InvalidAnalysisPlan("difference yêu cầu một metric.")
        frame[output or f"difference_{column}"] = frame[column].diff(spec.periods)
    elif spec.type == "cumulative":
        if not column:
            raise InvalidAnalysisPlan("cumulative yêu cầu một metric.")
        frame[output or f"cumulative_{column}"] = frame[column].cumsum()
    elif spec.type == "rank":
        if not column:
            raise InvalidAnalysisPlan("rank yêu cầu một metric.")
        frame[output or f"rank_{column}"] = frame[column].rank(method="dense", ascending=False)
    elif spec.type == "rolling_mean":
        if not column:
            raise InvalidAnalysisPlan("rolling_mean yêu cầu một metric.")
        frame[output or f"rolling_mean_{column}"] = frame[column].rolling(spec.window, min_periods=1).mean()
    elif spec.type == "round":
        targets = [column] if column else metrics
        frame[targets] = frame[targets].round(spec.decimals)
    return frame


def execute_analysis_plan(df: pd.DataFrame, plan: StructuredAnalysisPlan) -> tuple[Any, str, list[str]]:
    columns = validate_analysis_plan(df, plan)
    work = df.copy(deep=True)
    for filter_spec in plan.filters:
        work = _apply_filter(work, filter_spec)
    if work.empty:
        raise InvalidAnalysisPlan("Bộ lọc làm kết quả rỗng.")

    group_by = list(plan.group_by)
    if plan.time_grain:
        time_column = group_by[0]
        work[time_column] = to_datetime_series(work[time_column]).dt.to_period(
            _TIME_FREQUENCIES[plan.time_grain]
        ).astype("string")

    direct_transforms = {"correlation", "distribution", "outlier_iqr"}
    if plan.transforms and plan.transforms[0].type in direct_transforms:
        result = work
    elif group_by:
        required = list(dict.fromkeys([*group_by, *plan.metrics]))
        grouped_work = work[required].dropna() if plan.dropna else work[required]
        if plan.metrics:
            result = grouped_work.groupby(group_by, dropna=plan.dropna)[plan.metrics].agg(plan.aggregation).reset_index()
        else:
            result = grouped_work.groupby(group_by, dropna=plan.dropna).size().rename("count").reset_index()
        if plan.include_sample_size:
            sizes = grouped_work.groupby(group_by, dropna=plan.dropna).size().rename("sample_size").reset_index()
            result = result.merge(sizes, on=group_by, how="left")
    elif plan.metrics:
        if plan.transforms:
            result = work[plan.metrics].copy()
        else:
            result = (
                work[plan.metrics].agg(plan.aggregation)
                .rename_axis("metric").reset_index(name="value")
            )
    else:
        result = work

    for transform in plan.transforms:
        result = _transform(result, transform, plan)
    if plan.sort and plan.sort.by in result.columns:
        result = result.sort_values(plan.sort.by, ascending=plan.sort.ascending)
    elif plan.sort:
        raise InvalidAnalysisPlan(f"Không thể sort theo cột kết quả {plan.sort.by!r}.")
    if plan.limit:
        result = result.head(plan.limit)
    payload = _json_safe(result)
    if payload in (None, [], {}):
        raise InvalidAnalysisPlan("Kết quả phân tích rỗng.")
    return payload, plan.model_dump_json(exclude_none=True), columns
