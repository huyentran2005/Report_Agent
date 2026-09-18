from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import pandas as pd
from privacy import is_person_name_column


SHEET_ROLES = {"DATA", "INSTRUCTION", "METADATA", "INVALID"}
GENERIC_COLUMN_PATTERN = re.compile(
    r"^(?:unnamed(?::\s*\d+)?|(?:col(?:umn)?|cot|cột|field|var|ma|mã|code|id|"
    r"gia[\s_-]*tri|giá[\s_-]*trị|du[\s_-]*lieu|dữ[\s_-]*liệu|truong|trường)"
    r"[\s_.-]*\d+|\d+)$",
    re.IGNORECASE,
)
WEAK_SEMANTIC_PATTERN = re.compile(
    r"^(?:cột|cot|column|field|trường|truong|mã|ma|code|id|giá trị|gia tri|"
    r"dữ liệu|du lieu|thông tin|thong tin|nhóm|nhom)(?:[\s_.-]*\d+)?$",
    re.IGNORECASE,
)


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _numeric_cell_ratio(frame: pd.DataFrame) -> float:
    values = frame.stack(future_stack=True).dropna()
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def _text_length(frame: pd.DataFrame) -> float:
    values = frame.stack(future_stack=True).dropna()
    texts = values[~pd.to_numeric(values, errors="coerce").notna()].astype(str)
    return float(texts.str.len().mean()) if not texts.empty else 0.0


def to_datetime_series(series: pd.Series) -> pd.Series:
    """Parse ordinary dates and compact YYYYMMDD values without treating arbitrary integers as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact_mask = text.str.fullmatch(r"(?:19|20)\d{6}", na=False)
    if compact_mask.mean() >= 0.70:
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return pd.to_datetime(text, errors="coerce", format="mixed")


def is_date_like_series(series: pd.Series, column_name: str = "") -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    parsed = to_datetime_series(series)
    return bool(len(series) and parsed.notna().mean() >= 0.70)


def infer_column_semantic(series: pd.Series, column_name: str = "") -> str:
    """Classify analytical meaning, not merely the pandas storage dtype."""
    name = re.sub(r"[^a-z0-9]+", "_", str(column_name).lower()).strip("_")
    values = series.dropna()
    if _is_generic_column_name(column_name):
        return "unknown"
    if is_date_like_series(series, column_name):
        return "datetime"
    if values.empty:
        return "unknown"
    identifier_tokens = (
        "id", "identifier", "code", "ma_", "_ma", "mã", "phone", "telephone",
        "mobile", "sdt", "transaction", "account", "customer_id", "user_id",
    )
    if any(token in name for token in identifier_tokens):
        return "identifier"
    if pd.api.types.is_bool_dtype(series):
        return "boolean/status"
    text_values = values.astype(str).str.strip()
    normalized = set(text_values.str.casefold().unique())
    if normalized and normalized <= {
        "true", "false", "yes", "no", "y", "n", "có", "không", "co", "khong",
        "active", "inactive", "enabled", "disabled",
    }:
        return "boolean/status"
    digits = text_values.str.replace(r"\D", "", regex=True)
    if (digits.str.len().between(9, 15).mean() >= 0.90
            and text_values.nunique() / max(len(text_values), 1) >= 0.80):
        return "identifier"
    if pd.api.types.is_numeric_dtype(series) or pd.to_numeric(values, errors="coerce").notna().mean() >= 0.90:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.nunique() < 2:
            return "numeric_measure"
        return "numeric_measure"
    unique_count = int(text_values.nunique())
    unique_ratio = unique_count / max(len(text_values), 1)
    if unique_count <= 30 or unique_ratio <= 0.30:
        return "categorical"
    if text_values.str.len().mean() >= 30:
        return "text"
    return "text"


def _fallback_sheet_role(frame: pd.DataFrame) -> tuple[str, str]:
    """Classify by structure/content only; sheet names are deliberately ignored."""
    if frame.empty or len(frame.columns) == 0:
        return "INVALID", "Sheet rỗng hoặc không có cột dữ liệu."
    rows, columns = frame.shape
    non_empty_ratio = float(frame.notna().sum().sum() / max(rows * columns, 1))
    numeric_ratio = _numeric_cell_ratio(frame)
    average_text_length = _text_length(frame)
    active_columns = sum(
        frame[column].notna().sum() >= max(2, rows * 0.05) for column in frame.columns
    )
    if columns <= 6 and rows <= 200 and average_text_length >= 30 and numeric_ratio < 0.25:
        return "INSTRUCTION", "Cấu trúc ít cột, chủ yếu là văn bản dài mang tính hướng dẫn."
    if columns >= 20 and non_empty_ratio < 0.30 and numeric_ratio >= 0.40:
        return "METADATA", "Bảng rộng, thưa và chủ yếu là số tổng hợp/metadata."
    if rows <= 3 and columns >= 2 and numeric_ratio >= 0.25:
        return "METADATA", "Bảng nhỏ có tỷ lệ số cao, phù hợp với metadata hoặc kết quả tổng hợp có sẵn."
    if rows >= 3 and active_columns >= 2:
        return "DATA", "Bảng có ít nhất hai cột hoạt động và đủ hàng để phân tích."
    return "INVALID", "Không đủ dấu hiệu cấu trúc để phân loại chắc chắn."


def _is_generic_column_name(name: Any) -> bool:
    text = str(name).strip()
    return not text or bool(GENERIC_COLUMN_PATTERN.fullmatch(text)) or bool(WEAK_SEMANTIC_PATTERN.fullmatch(text))


def _apply_column_names(frame: pd.DataFrame, proposed: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    copy = frame.copy()
    original_names = [str(column) for column in copy.columns]
    proposed = proposed or {}
    used: set[str] = set()
    renames: dict[str, str] = {}
    final_names = []
    for index, original in enumerate(original_names):
        if not _is_generic_column_name(original):
            candidate = original
            suffix = 2
            while candidate in used:
                candidate = f"{original} ({suffix})"
                suffix += 1
            used.add(candidate)
            final_names.append(candidate)
            continue
        candidate = str(proposed.get(original, "")).strip()
        if not candidate or _is_generic_column_name(candidate):

            candidate = original
            suffix = 2
            while candidate in used:
                candidate = f"{original} ({suffix})"
                suffix += 1
            final_names.append(candidate)
            used.add(candidate)
            continue
        base = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{base} ({suffix})"
            suffix += 1
        used.add(candidate)
        final_names.append(candidate)
        renames[original] = candidate
    copy.columns = final_names
    return copy, renames


def _frame_with_detected_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Detect a plausible table header while preserving uncertain columns as UNKNOWN/generic."""
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()
    width = max(len(raw.columns), 1)
    candidates = []
    for position in range(min(15, len(raw))):
        row = raw.iloc[position]
        values = row.dropna().astype(str).str.strip()
        if len(values) < 2:
            continue
        short_text_ratio = float(values.str.len().between(1, 60).mean())
        unique_ratio = float(values.nunique() / max(len(values), 1))
        coverage = len(values) / width
        following = raw.iloc[position + 1:position + 4]
        following_density = float(following.notna().mean().mean()) if not following.empty else 0.0
        score = 3 * coverage + 2 * short_text_ratio + unique_ratio + following_density
        candidates.append((score, -position, position))
    header_position = max(candidates)[2] if candidates else 0
    header_row = raw.iloc[header_position]



    parent_row = raw.iloc[header_position - 1] if header_position > 0 else None
    parent_non_null = parent_row.dropna().astype(str).str.strip() if parent_row is not None else pd.Series(dtype="string")
    use_parent = bool(parent_row is not None and (
        2 <= parent_row.notna().sum() < header_row.notna().sum()
        or (len(parent_non_null) >= 2 and parent_non_null.nunique() < len(parent_non_null))
    ))
    parent_values = parent_row.ffill() if use_parent else None
    headers = []
    for index, value in enumerate(header_row):
        child = str(value).strip() if pd.notna(value) else ""
        parent = (str(parent_values.iloc[index]).strip()
                  if use_parent and pd.notna(parent_values.iloc[index]) else "")
        if parent and child and parent.casefold() != child.casefold():
            header = f"{parent} - {child}"
        else:
            header = child or parent or f"Unnamed: {index}"
        headers.append(header)



    unique_headers = []
    used_headers: set[str] = set()
    for header in headers:
        candidate = header
        suffix = 2
        while candidate in used_headers:
            candidate = f"{header} ({suffix})"
            suffix += 1
        used_headers.add(candidate)
        unique_headers.append(candidate)
    frame = raw.iloc[header_position + 1:].copy()
    frame.columns = unique_headers
    return frame.dropna(axis=0, how="all").reset_index(drop=True)


def _read_data_sheet_with_detected_header(path: str | Path, sheet_name: str) -> pd.DataFrame:
    return _frame_with_detected_header(pd.read_excel(path, sheet_name=sheet_name, header=None))


def _cross_sheet_column_renames(
    sheets: dict[str, pd.DataFrame], data_names: list[str]
) -> dict[str, dict[str, str]]:
    """Infer a generic header only from consensus at the same position across DATA sheets."""
    suggestions: dict[str, dict[str, str]] = {name: {} for name in data_names}
    for sheet_name in data_names:
        frame = sheets[sheet_name]
        for position, column in enumerate(frame.columns):
            original = str(column)
            if not _is_generic_column_name(original):
                continue
            candidates = set()
            family = _dtype_family(frame.iloc[:, position])
            for other_name in data_names:
                if other_name == sheet_name or position >= len(sheets[other_name].columns):
                    continue
                other_frame = sheets[other_name]
                other_column = str(other_frame.columns[position])
                if (not _is_generic_column_name(other_column)
                        and _dtype_family(other_frame.iloc[:, position]) == family):
                    candidates.add(other_column)
            if len(candidates) == 1:
                suggestions[sheet_name][original] = candidates.pop()
    return suggestions


def inspect_workbook(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() not in {".xlsx", ".xls"}:
        return []
    catalog: list[dict[str, Any]] = []
    for sheet_name, frame in pd.read_excel(source, sheet_name=None).items():
        fallback_role, fallback_reason = _fallback_sheet_role(frame)
        sensitive_columns = {
            str(column) for column in frame.columns
            if is_person_name_column(column, frame[column])
            or infer_column_semantic(frame[column], str(column)) == "identifier"
        }
        sample = [
            {str(column): ("[REDACTED]" if str(column) in sensitive_columns and pd.notna(value)
                           else _json_scalar(value))
             for column, value in row.items()}
            for row in frame.head(3).to_dict(orient="records")
        ]
        catalog.append({
            "sheet_name": str(sheet_name),
            "columns": [str(column) for column in frame.columns],
            "rows": int(len(frame)),
            "non_empty_ratio": round(float(frame.notna().sum().sum() / max(frame.size, 1)), 4),
            "numeric_cell_ratio": round(_numeric_cell_ratio(frame), 4),
            "average_text_length": round(_text_length(frame), 2),
            "sample": sample,
            "fallback_role": fallback_role,
            "fallback_reason": fallback_reason,
        })
    return catalog


def fallback_sheet_classifications(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sheet_name": item["sheet_name"],
            "role": item["fallback_role"],
            "reason": item["fallback_reason"],
            "columns": item["columns"],
            "rows": item["rows"],
            "column_renames": {},
        }
        for item in catalog
    ]


def _dtype_family(series: pd.Series) -> str:
    if is_date_like_series(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    non_null = series.dropna()
    if not non_null.empty and pd.to_numeric(non_null, errors="coerce").notna().mean() >= 0.80:
        return "numeric"
    return "text"


def _schemas_compatible(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_columns = {str(column) for column in left.columns}
    right_columns = {str(column) for column in right.columns}
    shared = left_columns & right_columns
    if not shared:
        return False
    overlap = len(shared) / max(min(len(left_columns), len(right_columns)), 1)
    if overlap < 0.80:
        return False
    for column in shared:
        if _dtype_family(left[column]) != _dtype_family(right[column]):
            return False
    return True


def _requested_sheet_names(sheet_names: list[str], instructions: str) -> list[str]:
    """Select sheets explicitly mentioned by the user, without assuming naming conventions."""
    normalized_instructions = instructions.casefold()
    requested = []
    for name in sheet_names:
        escaped = re.escape(name.casefold())
        explicit_pattern = rf"(?:sheet|tab|worksheet)\s*(?:tên\s*)?[:=\-]?\s*['\"]?{escaped}(?![\w])"
        quoted_pattern = rf"['\"]{escaped}['\"]"
        if re.search(explicit_pattern, normalized_instructions) or re.search(quoted_pattern, normalized_instructions):
            requested.append(name)
    return requested


def read_dataset_partitions(
    path: str | Path,
    workbook_sheets: list[dict[str, Any]] | None = None,
    instructions: str = "",
) -> dict[str, pd.DataFrame]:
    """Return compatible DATA sheets merged into independent schema groups."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = _frame_with_detected_header(pd.read_csv(source, header=None))
        frame.columns = [str(column) for column in frame.columns]
        frame, _ = _apply_column_names(frame)
        return {source.stem: frame}
    if source.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("Chỉ hỗ trợ tệp CSV, XLSX hoặc XLS.")

    sheets = pd.read_excel(source, sheet_name=None)
    for frame in sheets.values():
        frame.columns = [str(column) for column in frame.columns]
    classification_by_name = {
        str(item["sheet_name"]): item for item in (workbook_sheets or [])
    }
    if workbook_sheets:
        roles = {str(item["sheet_name"]): str(item.get("role", "UNKNOWN")).upper() for item in workbook_sheets}
    else:
        catalog = inspect_workbook(source)
        roles = {item["sheet_name"]: item["fallback_role"] for item in catalog}
    for sheet_name in list(sheets):
        if roles.get(str(sheet_name)) == "DATA":
            sheets[sheet_name] = _read_data_sheet_with_detected_header(source, str(sheet_name))
    all_data_names = [
        str(name) for name, frame in sheets.items()
        if roles.get(str(name)) == "DATA" and not frame.empty
    ]
    cross_sheet_renames = _cross_sheet_column_renames(sheets, all_data_names)
    for sheet_name, frame in list(sheets.items()):
        classification = classification_by_name.get(str(sheet_name), {})
        proposed = dict(cross_sheet_renames.get(str(sheet_name), {}))
        proposed.update({
            original: candidate
            for original, candidate in (classification.get("column_renames") or {}).items()
            if candidate and not _is_generic_column_name(candidate)
        })
        renamed, applied = _apply_column_names(frame, proposed)
        sheets[sheet_name] = renamed
        if applied and classification:
            classification["applied_column_renames"] = applied
    data_names = [str(name) for name, frame in sheets.items() if roles.get(str(name)) == "DATA" and not frame.empty]
    requested = _requested_sheet_names(data_names, instructions)
    selected_names = requested or data_names
    if not selected_names:
        raise ValueError("Workbook không có sheet DATA phù hợp với yêu cầu.")

    groups: list[list[tuple[str, pd.DataFrame]]] = []
    for name in selected_names:
        frame = sheets[name]
        target = next((group for group in groups if _schemas_compatible(group[0][1], frame)), None)
        if target is None:
            groups.append([(name, frame)])
        else:
            target.append((name, frame))

    partitions: dict[str, pd.DataFrame] = {}
    for group in groups:
        names = [name for name, _ in group]
        frames = []
        for name, frame in group:
            copy = frame.copy()
            copy.insert(0, "_source_sheet", name)
            frames.append(copy)
        key = " + ".join(names)
        partitions[key] = pd.concat(frames, ignore_index=True, sort=False)
    return partitions


def split_partitions_by_source(partitions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Expand compatible groups so questions and calculations stay scoped to original sheets."""
    per_sheet: dict[str, pd.DataFrame] = {}
    for partition_name, frame in partitions.items():
        if "_source_sheet" not in frame.columns:
            per_sheet[partition_name] = frame.copy()
            continue
        for sheet_name, sheet_frame in frame.groupby("_source_sheet", sort=False, dropna=False):
            per_sheet[str(sheet_name)] = sheet_frame.reset_index(drop=True)
    return per_sheet


def read_dataset(path: str | Path, workbook_sheets: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    partitions = read_dataset_partitions(path, workbook_sheets)
    if len(partitions) != 1:
        raise ValueError("Dữ liệu gồm nhiều nhóm schema độc lập; hãy dùng read_dataset_partitions().")
    return next(iter(partitions.values()))


def extract_instruction_context(path: str | Path, workbook_sheets: list[dict[str, Any]], max_chars: int = 12000) -> str:
    source = Path(path)
    if source.suffix.lower() not in {".xlsx", ".xls"}:
        return ""
    instruction_names = {
        str(item["sheet_name"]) for item in workbook_sheets if str(item.get("role", "")).upper() == "INSTRUCTION"
    }
    if not instruction_names:
        return ""
    sheets = pd.read_excel(source, sheet_name=None)
    sections = []
    for sheet_name, frame in sheets.items():
        if str(sheet_name) not in instruction_names:
            continue
        lines = []
        for row in frame.itertuples(index=False, name=None):
            values = [str(value).strip() for value in row if pd.notna(value) and str(value).strip()]
            if values:
                lines.append(" | ".join(values))
        if lines:
            sections.append(f"[Sheet: {sheet_name}]\n" + "\n".join(lines))
    return "\n\n".join(sections)[:max_chars]


def describe_workbook(path: str | Path, workbook_sheets: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if workbook_sheets is not None:
        return workbook_sheets
    catalog = inspect_workbook(path)
    return fallback_sheet_classifications(catalog)


def effective_instructions(user_instructions: str, workbook_instruction_context: str | None) -> str:
    if not workbook_instruction_context:
        return user_instructions
    return (
        f"{user_instructions}\n\n"
        "NGỮ CẢNH HƯỚNG DẪN TỪ WORKBOOK (chỉ dùng để hiểu thuật ngữ, quy tắc đánh giá và ý nghĩa dữ liệu; "
        "không coi các con số trong phần này là kết quả phân tích):\n"
        f"{workbook_instruction_context}"
    )
