from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import pandas as pd


def _json_scalar(value: Any) -> Any:
    """Chuyển một giá trị pandas/numpy thành kiểu đơn giản có thể ghi vào JSON."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _numeric_cell_ratio(frame: pd.DataFrame) -> float:
    """Tính tỷ lệ ô không rỗng có thể chuyển đổi thành số trong toàn bộ bảng."""
    values = frame.stack(future_stack=True).dropna()
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def _text_length(frame: pd.DataFrame) -> float:
    """Tính độ dài trung bình của các ô văn bản không thể chuyển thành số."""
    values = frame.stack(future_stack=True).dropna()
    texts = values[~pd.to_numeric(values, errors="coerce").notna()].astype(str)
    return float(texts.str.len().mean()) if not texts.empty else 0.0


def to_datetime_series(series: pd.Series) -> pd.Series:
    """Chuyển cột ngày thông thường hoặc dạng YYYYMMDD thành datetime một cách thận trọng."""
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
    """Kiểm tra cột có kiểu ngày hoặc có ít nhất 70% giá trị chuyển được thành ngày."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    parsed = to_datetime_series(series)
    return bool(len(series) and parsed.notna().mean() >= 0.70)


def attach_column_semantics(
    frames: dict[str, pd.DataFrame],
    semantics: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, pd.DataFrame]:
    """Gắn ý nghĩa và quyền sử dụng do LLM phân loại vào từng DataFrame."""
    for partition, frame in frames.items():
        partition_semantics = semantics.get(partition)
        if partition_semantics is None:
            raise ValueError(f"Thiếu phân loại cột của partition {partition!r}.")
        expected = {str(column) for column in frame.columns if not str(column).startswith("_")}
        received = set(partition_semantics)
        if expected != received:
            raise ValueError(
                f"Phân loại cột không khớp cho {partition!r}: "
                f"thiếu={sorted(expected - received)}, thừa={sorted(received - expected)}."
            )
        incomplete = [
            column for column, details in partition_semantics.items()
            if not details.get("semantic_type") or not details.get("usage_permission")
        ]
        if incomplete:
            raise ValueError(
                f"Phân loại cột của {partition!r} thiếu semantic_type hoặc "
                f"usage_permission: {sorted(incomplete)}."
            )
        frame.attrs["column_semantics"] = partition_semantics
    return frames


def column_semantic(frame: pd.DataFrame, column: str) -> str:
    """Lấy semantic type bắt buộc của một cột từ kết quả phân loại LLM đã gắn vào bảng."""
    details = frame.attrs.get("column_semantics", {}).get(str(column))
    if not details or not details.get("semantic_type"):
        raise ValueError(f"Cột {column!r} chưa có phân loại semantic từ LLM.")
    return str(details["semantic_type"])


def column_usage_permission(frame: pd.DataFrame, column: str) -> str:
    """Lấy quyền sử dụng cột do LLM quyết định: đầy đủ, chỉ phân nhóm hoặc bị chặn."""
    details = frame.attrs.get("column_semantics", {}).get(str(column))
    if not details or not details.get("usage_permission"):
        raise ValueError(f"Cột {column!r} chưa có phân loại quyền sử dụng từ LLM.")
    return str(details["usage_permission"])


def _frame_with_detected_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Phát hiện dòng header phù hợp và giữ nguyên các header chưa xác định hoặc chung chung."""
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
    """Đọc một sheet không chỉ định header rồi tự phát hiện dòng tiêu đề của bảng."""
    return _frame_with_detected_header(pd.read_excel(path, sheet_name=sheet_name, header=None))


def inspect_workbook(path: str | Path) -> list[dict[str, Any]]:
    """Thu thập cấu trúc, thống kê và ba dòng mẫu của từng sheet Excel."""
    source = Path(path)
    if source.suffix.lower() not in {".xlsx", ".xls"}:
        return []
    catalog: list[dict[str, Any]] = []
    for sheet_name, frame in pd.read_excel(source, sheet_name=None).items():
        sample = [
            {str(column): _json_scalar(value)
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
        })
    return catalog


def _dtype_family(series: pd.Series) -> str:
    """Xếp cột vào nhóm kiểu kỹ thuật datetime, numeric, boolean hoặc text."""
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
    """Kiểm tra hai bảng có đủ cột chung và kiểu tương thích để ghép theo hàng hay không."""
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
    """Chọn các sheet được người dùng nhắc rõ trong yêu cầu mà không suy đoán theo tên."""
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
    """Đọc các sheet DATA và ghép những sheet tương thích thành các nhóm schema độc lập."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = _frame_with_detected_header(pd.read_csv(source, header=None))
        frame.columns = [str(column) for column in frame.columns]
        return {source.stem: frame}
    if source.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("Chỉ hỗ trợ tệp CSV, XLSX hoặc XLS.")

    sheets = pd.read_excel(source, sheet_name=None)
    for frame in sheets.values():
        frame.columns = [str(column) for column in frame.columns]
    if workbook_sheets:
        roles = {str(item["sheet_name"]): str(item.get("role", "UNKNOWN")).upper() for item in workbook_sheets}
    else:
        raise ValueError("Cần kết quả phân loại sheet từ LLM trước khi đọc workbook Excel.")
    for sheet_name in list(sheets):
        if roles.get(str(sheet_name)) == "DATA":
            sheets[sheet_name] = _read_data_sheet_with_detected_header(source, str(sheet_name))
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
            copy = add_duration_metrics(frame)
            copy.insert(0, "_source_sheet", name)
            frames.append(copy)
        key = " + ".join(names)
        partitions[key] = pd.concat(frames, ignore_index=True, sort=False)
    return partitions


def split_partitions_by_source(partitions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Tách các nhóm đã ghép về từng sheet gốc để câu hỏi và phép tính giữ đúng phạm vi."""
    per_sheet: dict[str, pd.DataFrame] = {}
    for partition_name, frame in partitions.items():
        if "_source_sheet" not in frame.columns:
            per_sheet[partition_name] = frame.copy()
            continue
        for sheet_name, sheet_frame in frame.groupby("_source_sheet", sort=False, dropna=False):
            per_sheet[str(sheet_name)] = sheet_frame.reset_index(drop=True)
    return per_sheet


def _duration_role(column: str) -> str | None:
    """Nhận diện header ngày có vai trò bắt đầu hoặc kết thúc mà không giả định lĩnh vực."""
    text = re.sub(r"[^a-z0-9]+", " ", str(column).casefold()).strip()
    if re.search(r"(^| )(start|begin|created|accepted|opened|issued|from)( |$)", text):
        return "start"
    if re.search(
        r"(^| )(end|finish|completed|closed|resolved|solved|received|delivered|processed|to)( |$)",
        text,
    ):
        return "end"
    return None


def _duration_metric_name(start: str, end: str, existing: pd.Index) -> str:
    """Tạo tên duy nhất và ổn định cho chỉ số thời lượng từ hai header nguồn."""
    start_text = re.sub(r"[^a-z0-9]+", " ", str(start).casefold()).strip()
    end_text = re.sub(r"[^a-z0-9]+", " ", str(end).casefold()).strip()
    pairs = (
        ("accept", "process", "Acceptance Processing Duration Days"),
        ("order", "deliver", "Order Delivery Duration Days"),
        ("create", "complete", "Completion Duration Days"),
        ("open", "close", "Open to Close Duration Days"),
        ("start", "end", "Process Duration Days"),
    )
    name = "Duration Days"
    for left, right, candidate in pairs:
        if left in start_text and right in end_text:
            name = candidate
            break
    if name not in existing:
        return name
    suffix = 2
    while f"{name} {suffix}" in existing:
        suffix += 1
    return f"{name} {suffix}"


def add_duration_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Tạo chỉ số thời lượng và ẩn hai cột datetime nguồn khỏi bảng phân tích.

    Hàm không sửa workbook gốc. Bảng trả về chỉ giữ cột thời lượng dẫn xuất để
    các bước sau không vô tình phân tích hai timestamp nguồn như chiều lịch.
    """
    result = frame.copy()
    datetime_columns = [
        str(column) for column in result.columns
        if not str(column).startswith("_") and is_date_like_series(result[column], str(column))
    ]
    starts = [column for column in datetime_columns if _duration_role(column) == "start"]
    ends = [column for column in datetime_columns if _duration_role(column) == "end"]
    if not starts or not ends:
        return result
    used: set[str] = set()
    for start in starts:
        end = next((candidate for candidate in ends if candidate not in used), None)
        if end is None:
            continue
        start_values = to_datetime_series(result[start])
        end_values = to_datetime_series(result[end])
        duration = (end_values - start_values).dt.total_seconds() / 86400.0
        valid = duration.notna()
        if not valid.any():
            continue
        duration_name = _duration_metric_name(start, end, result.columns)
        result[duration_name] = duration.where(duration >= 0)
        used.add(end)
        result = result.drop(columns=[start, end])
        result.attrs.setdefault("derived_columns", {})[duration_name] = {
            "source_columns": [start, end], "unit": "days", "formula": "end - start",
        }
    return result


def read_dataset(path: str | Path, workbook_sheets: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Đọc dữ liệu và trả về một DataFrame khi tệp chỉ có đúng một nhóm schema."""
    partitions = read_dataset_partitions(path, workbook_sheets)
    if len(partitions) != 1:
        raise ValueError("Dữ liệu gồm nhiều nhóm schema độc lập; hãy dùng read_dataset_partitions().")
    return next(iter(partitions.values()))


def extract_instruction_context(path: str | Path, workbook_sheets: list[dict[str, Any]], max_chars: int = 12000) -> str:
    """Trích văn bản từ các sheet INSTRUCTION và giới hạn độ dài để đưa vào ngữ cảnh LLM."""
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
    """Trả về phân loại sheet đã có hoặc catalog cấu trúc cơ bản của workbook."""
    if workbook_sheets is not None:
        return workbook_sheets
    return inspect_workbook(path)


def effective_instructions(user_instructions: str, workbook_instruction_context: str | None) -> str:
    """Ghép yêu cầu người dùng với hướng dẫn trong workbook thành ngữ cảnh phân tích cuối."""
    if not workbook_instruction_context:
        return user_instructions
    return (
        f"{user_instructions}\n\n"
        "NGỮ CẢNH HƯỚNG DẪN TỪ WORKBOOK (chỉ dùng để hiểu thuật ngữ, quy tắc đánh giá và ý nghĩa dữ liệu; "
        "không coi các con số trong phần này là kết quả phân tích):\n"
        f"{workbook_instruction_context}"
    )
