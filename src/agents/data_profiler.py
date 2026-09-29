import logging
import pandas as pd
import json
import os
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Literal
import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llm import get_llm, text_from_response
from pydantic import BaseModel, Field, ValidationError
from graph.state import GraphState
from schemas.messages import (ColumnSemanticClassificationResponse, DataProfile,
                              DimensionScope, PartitionDataScope)
from data_io import (attach_column_semantics, column_semantic, column_usage_permission,
                     effective_instructions,
                     extract_instruction_context, inspect_workbook,
                     read_dataset_partitions, split_partitions_by_source, to_datetime_series)
load_dotenv()

logger = logging.getLogger(__name__)
SHEET_CLASSIFICATION_BATCH_SIZE = 8
COLUMN_CLASSIFICATION_BATCH_SIZE = 40


def _basic_file_info(file_path: str, sheet_count: int = 0) -> Dict[str, Any]:
    """Lấy metadata thực tế của file mà không đưa ra bất kỳ quyết định phân loại nào."""
    source = Path(file_path)
    return {
        "file_name": source.name,
        "extension": source.suffix.lower(),
        "size_bytes": source.stat().st_size,
        "sheet_count": sheet_count,
    }

class DatasetProfileResponse(BaseModel):
    """LLM only writes observations; all structural statistics stay deterministic."""
    key_observations: str = Field(
        description="Nhận xét chính về cấu trúc, chất lượng và vấn đề tiềm ẩn của dữ liệu.")


class SheetClassification(BaseModel):
    sheet_name: str
    role: Literal["DATA", "INSTRUCTION", "METADATA", "INVALID"]
    reason: str

class WorkbookClassificationResponse(BaseModel):
    sheets: List[SheetClassification]


def _strip_json_fence(raw: str) -> str:
    """Loại bỏ Markdown fence tùy chọn mà không làm thay đổi nội dung JSON bên trong."""
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_dataset_profile_response(raw: str) -> DatasetProfileResponse:
    """Parse JSON của LLM và chấp nhận ký tự điều khiển thô nằm trong chuỗi.

    Một số gateway tương thích OpenAI có thể trả JSON chứa trực tiếp ký tự xuống
    dòng, tab hoặc NUL. Chế độ ``strict=False`` xử lý các ký tự này, sau đó
    Pydantic vẫn kiểm tra cấu trúc dữ liệu đã giải mã như bình thường.
    """
    payload = json.loads(_strip_json_fence(raw), strict=False)
    parsed = DatasetProfileResponse.model_validate(payload)
    parsed.key_observations = "".join(
        " " if ord(character) < 32 and character not in "\n\t" else character
        for character in parsed.key_observations
    ).replace("\x00", " ")
    return parsed


def _column_catalog(partitions: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Tạo thống kê thực tế của từng cột để LLM phân loại """
    catalog: List[Dict[str, Any]] = []
    for partition, frame in partitions.items():
        for column in frame.columns:
            name = str(column)
            if name.startswith("_"):
                continue
            series = frame[column]
            values = series.dropna()
            text_values = values.astype(str).str.strip()
            numeric = pd.to_numeric(values, errors="coerce")
            parsed_dates = to_datetime_series(series)
            unique_count = int(values.nunique())
            valid_count = int(values.size)
            item: Dict[str, Any] = {
                "source_partition": partition,
                "column": name,
                "dtype": str(series.dtype),
                "row_count": int(len(series)),
                "valid_count": valid_count,
                "missing_count": int(series.isna().sum()),
                "unique_count": unique_count,
                "unique_ratio": round(unique_count / max(valid_count, 1), 4),
                "numeric_parse_ratio": round(float(numeric.notna().mean()), 4) if valid_count else 0.0,
                "datetime_parse_ratio": round(float(parsed_dates.notna().mean()), 4) if len(series) else 0.0,
                "average_text_length": round(float(text_values.str.len().mean()), 2) if valid_count else 0.0,
            }
            if valid_count:
                item["sample_values"] = [str(value) for value in values.drop_duplicates().head(5)]
            catalog.append(item)
    return catalog


def _classify_columns(
    llm,
    partitions: Dict[str, pd.DataFrame],
    instructions: str,
    file_info: Dict[str, Any],
    workbook_sheets: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Yêu cầu LLM phân loại đúng một lần cho mọi cột phân tích trong các partition."""
    catalog = _column_catalog(partitions)
    parser = JsonOutputParser(pydantic_object=ColumnSemanticClassificationResponse)
    prompt = PromptTemplate(
        template="""
Bạn đang phân loại ý nghĩa phân tích của các cột dữ liệu. Với mỗi cột, chọn đúng một nhãn:
- datetime: mốc ngày/giờ có thể dùng làm chiều thời gian;
- identifier: mã, số điện thoại, tài khoản hoặc khóa nhận diện không có ý nghĩa cộng/trung bình;
- numeric_measure: đại lượng số mà tổng hợp, so sánh hoặc thống kê có ý nghĩa;
- categorical: nhãn phân nhóm;
- boolean_status: trạng thái nhị phân hoặc cờ;
- free_text: văn bản tự do;
- person_name: tên người hoặc dữ liệu trực tiếp nhận diện một cá nhân;
- unknown: chưa đủ bằng chứng.

Phân loại riêng quyền sử dụng của mỗi cột bằng `usage_permission`:
- full_analysis: có thể dùng làm chiều phân tích, bộ lọc, phép tính hoặc số liệu phù hợp với semantic_type;
- group_only: chỉ được dùng làm nhãn phân nhóm/group_by và hiển thị nhãn, không được dùng làm metric,
  phép tính hoặc biến đầu vào cho transform;
- blocked: không được đưa vào câu hỏi, phép phân tích, biểu đồ hoặc nội dung báo cáo.

`semantic_type` mô tả cột là gì; `usage_permission` mô tả cột được phép dùng ra sao. Hai kết quả
phải được quyết định độc lập. Ví dụ, tên nhân viên có thể là `person_name` + `group_only` nếu việc
so sánh theo nhân viên phù hợp với yêu cầu; thông tin liên hệ hoặc định danh cần bảo vệ có thể là
`identifier`/`person_name` + `blocked`. Không tự động chặn mọi `person_name`.

Giữ nguyên tuyệt đối `source_partition` và `column`. Trả về đúng một mục cho mỗi cột, không thêm,
không bỏ sót và không đổi tên cột. Không mặc định cột số là numeric_measure: nếu phép cộng hoặc
trung bình không có ý nghĩa thì chọn identifier hoặc unknown. Chỉ dựa trên tên cột, thống kê kỹ
thuật và mẫu an toàn được cung cấp. `confidence` nằm trong [0, 1] và `reason` phải ngắn gọn.

Yêu cầu người dùng: {instructions}
Thông tin cơ bản của file: {file_info}
Kết quả phân loại sheet từ LLM: {workbook_sheets}
Các cột cần phân loại: {catalog}
{format_instructions}
Chỉ trả về JSON hợp lệ.
""",
        input_variables=["instructions", "file_info", "workbook_sheets", "catalog"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    classified: Dict[tuple[str, str], Dict[str, Any]] = {}
    for start in range(0, len(catalog), COLUMN_CLASSIFICATION_BATCH_SIZE):
        batch = catalog[start:start + COLUMN_CLASSIFICATION_BATCH_SIZE]
        expected = {(item["source_partition"], item["column"]) for item in batch}
        raw = text_from_response(llm.invoke(prompt.invoke({
            "instructions": instructions,
            "file_info": json.dumps(file_info, ensure_ascii=False, indent=2),
            "workbook_sheets": json.dumps(workbook_sheets, ensure_ascii=False, indent=2),
            "catalog": json.dumps(batch, ensure_ascii=False, indent=2),
        }), config={"request_options": {"timeout": 60}}))
        parsed = ColumnSemanticClassificationResponse.model_validate_json(_strip_json_fence(raw))
        returned = [(item.source_partition, item.column) for item in parsed.columns]
        if len(returned) != len(set(returned)) or set(returned) != expected:
            raise ValueError(
                "LLM phân loại cột không khớp: "
                f"thiếu={sorted(expected - set(returned))}, "
                f"thừa={sorted(set(returned) - expected)}."
            )
        for item in parsed.columns:
            classified[(item.source_partition, item.column)] = item.model_dump(mode="json")

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (partition, column), details in classified.items():
        result.setdefault(partition, {})[column] = details
    return result


def _profile_frame(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Tạo hồ sơ thống kê cho từng cột dựa trên semantic type đã được LLM xác định."""
    details_by_column: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        col_name = str(col)
        if col_name.startswith("_"):
            continue
        unique_values = int(df[col].nunique())
        missing_values_count = int(df[col].isnull().sum())
        semantic_type = column_semantic(df, col_name)
        usage_permission = column_usage_permission(df, col_name)
        detail: Dict[str, Any] = {
            "type": str(df[col].dtype),
            "semantic_type": semantic_type,
            "usage_permission": usage_permission,
            "valid_count": int(df[col].notna().sum()),
            "unique_values_count": unique_values,
            "missing_values_count": missing_values_count,
            "missing_values_percentage": f"{(missing_values_count / len(df) * 100):.2f}%",
        }
        if semantic_type == "datetime":
            detail["type"] = "datetime"
            parsed_dates = to_datetime_series(df[col])
            detail["min"] = str(parsed_dates.min()) if parsed_dates.notna().any() else None
            detail["max"] = str(parsed_dates.max()) if parsed_dates.notna().any() else None
            detail["period_count"] = int(parsed_dates.dropna().dt.to_period("M").nunique())
            detail["years"] = sorted(int(year) for year in parsed_dates.dropna().dt.year.unique())
            detail["periods"] = sorted(str(period) for period in parsed_dates.dropna().dt.to_period("M").unique())
        elif pd.api.types.is_numeric_dtype(df[col]) and semantic_type == "numeric_measure":
            for name, value in (("mean", df[col].mean()), ("std", df[col].std())):
                detail[name] = float(value) if pd.notna(value) else None
            for name, value in (("min", df[col].min()), ("max", df[col].max())):
                detail[name] = (float(value) if isinstance(value, (np.floating, float)) else int(value)) if pd.notna(value) else None
            quantiles = df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            detail["quantiles"] = {str(key): float(value) if pd.notna(value) else None for key, value in quantiles.items()}
            detail["variance"] = float(df[col].var()) if df[col].notna().sum() >= 2 else None
        elif ((pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]))
              and usage_permission != "blocked"
              and semantic_type in {"categorical", "boolean_status", "person_name"}):
            counts = df[col].value_counts()
            counts.index = counts.index.infer_objects()
            top_values = counts.nlargest(5).to_dict()
            detail["top_5_values"] = {str(key): int(value) for key, value in top_values.items()}
        details_by_column[col_name] = detail
    return details_by_column


def _profile_data_scope(source_partition: str, df: pd.DataFrame) -> PartitionDataScope:
    """Ghi nhận phạm vi giá trị thực tế của các dimension mà không yêu cầu LLM suy đoán."""
    dimensions: list[DimensionScope] = []
    for column in df.columns:
        name = str(column)
        if name.startswith("_"):
            continue
        semantic = column_semantic(df, name)
        usage_permission = column_usage_permission(df, name)
        if usage_permission == "blocked":
            continue
        if semantic == "datetime":
            dates = to_datetime_series(df[column]).dropna()
            dimensions.append(DimensionScope(
                column=name, kind="time", valid_count=int(dates.size),
                unique_count=int(dates.nunique()),
                min_date=dates.min().isoformat() if not dates.empty else None,
                max_date=dates.max().isoformat() if not dates.empty else None,
                years=sorted(int(year) for year in dates.dt.year.unique()),
                periods=sorted(str(period) for period in dates.dt.to_period("M").unique()),
            ))
        elif semantic in {"categorical", "boolean_status", "person_name"}:
            values = df[column].dropna().unique().tolist()
            complete = len(values) <= 100
            dimensions.append(DimensionScope(
                column=name, kind="category", valid_count=int(df[column].notna().sum()),
                unique_count=int(df[column].nunique()),
                values=[str(value) for value in values] if complete else [],
                values_complete=complete,
            ))
    return PartitionDataScope(
        source_partition=source_partition, row_count=len(df), dimensions=dimensions
    )


def _classify_workbook(llm, file_path: str, user_instructions: str) -> List[Dict[str, Any]]:
    """Gửi thông tin workbook cho LLM để phân loại đầy đủ vai trò của từng sheet."""
    catalog = inspect_workbook(file_path)
    if not catalog:
        return []
    file_info = _basic_file_info(file_path, len(catalog))
    catalog_for_llm = catalog
    parser = JsonOutputParser(pydantic_object=WorkbookClassificationResponse)
    prompt = PromptTemplate(
        template="""
        Bạn đang phân loại các sheet trong workbook để chuẩn bị tạo báo cáo dữ liệu.
        Với từng sheet, chọn đúng một nhãn trong danh sách đóng:
        - DATA: bảng dữ liệu quan sát/bản ghi cần được Pandas phân tích.
        - INSTRUCTION: văn bản mô tả quy tắc, định nghĩa, hướng dẫn hoặc tiêu chí đánh giá.
        - METADATA: thông tin bổ sung, bảng tra cứu hoặc kết quả tổng hợp có sẵn; không tính như DATA.
        - INVALID: sheet rỗng, không xác định được bảng hoặc không đủ dữ liệu để phân tích.

        Không suy luận vai trò chỉ từ tên sheet và không hard-code tên sheet. Hãy dựa vào cấu trúc,
        mật độ dữ liệu, kiểu nội dung và mẫu giá trị. Phải trả về đúng một mục cho mỗi sheet,
        giữ nguyên `sheet_name`. Không viết mã và không thêm nhãn khác.

        Yêu cầu phân tích của người dùng: {instructions}
        Thông tin cơ bản của file: {file_info}
        Thông tin các sheet: {catalog}
        {format_instructions}
        Chỉ trả về JSON hợp lệ.
        """,
        input_variables=["instructions", "file_info", "catalog"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    metadata = {item["sheet_name"]: item for item in catalog}
    reconciled_by_name: Dict[str, Dict[str, Any]] = {}

    for start in range(0, len(catalog_for_llm), SHEET_CLASSIFICATION_BATCH_SIZE):
        batch = catalog_for_llm[start:start + SHEET_CLASSIFICATION_BATCH_SIZE]
        expected = [item["sheet_name"] for item in batch]
        try:
            raw = text_from_response(llm.invoke(prompt.invoke({
                "instructions": user_instructions,
                "file_info": json.dumps(file_info, ensure_ascii=False, indent=2),
                "catalog": json.dumps(batch, ensure_ascii=False, indent=2),
            }), config={"request_options": {"timeout": 60}}))
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = WorkbookClassificationResponse.model_validate_json(cleaned)
            returned = [item.sheet_name for item in parsed.sheets]
            if len(returned) != len(set(returned)) or set(returned) != set(expected):
                missing = sorted(set(expected) - set(returned))
                unexpected = sorted(set(returned) - set(expected))
                raise ValueError(
                    f"Danh sách sheet không khớp (thiếu={missing}, ngoài_lô={unexpected})."
                )

            for item in parsed.sheets:
                reconciled_by_name[item.sheet_name] = {
                    "sheet_name": item.sheet_name,
                    "role": item.role,
                    "reason": item.reason,
                    "columns": metadata[item.sheet_name]["columns"],
                    "rows": metadata[item.sheet_name]["rows"],
                }
        except Exception as exc:
            raise RuntimeError(
                f"LLM không phân loại được lô sheet {expected}: {exc}"
            ) from exc



    return [reconciled_by_name[item["sheet_name"]] for item in catalog]


def profile_dataset(state: GraphState) -> GraphState:
    """Điều phối việc phân loại workbook, phân loại cột và tạo hồ sơ dữ liệu tổng thể."""
    request_id = state.get('request_id', 'unknown_request')
    file_path = state.get('file_path')
    instructions = state.get('instructions', '')

    if not file_path:
        logger.error(f"Missing file_path for request {request_id}.")
        state['status'] = "error"
        state['error_message'] = "Cannot perform data analysis: A file path was not provided."
        return state

    logger.info(f"DataAnalysisNode processing request: {request_id}")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if state.get("llm_provider", "gemini") == "gemini" and not gemini_api_key:
        logger.error(f"GEMINI_API_KEY not found for request {request_id}. Please ensure it's set in your .env file.")
        state['status'] = "error"
        state['error_message'] = "API key for Gemini not found. Please set GEMINI_API_KEY in your .env file."
        return state

    try:
        llm = get_llm(state.get("llm_provider"), state.get("llm_model"))
    except Exception as e:
        logger.error(f"Failed to initialize LLM for data analysis: {e}", exc_info=True)
        state['status'] = "error"
        state['error_message'] = f"Failed to initialize LLM for data analysis: {e}"
        return state

    try:
        workbook_sheets = _classify_workbook(llm, file_path, instructions)
    except Exception as exc:
        logger.error("LLM không phân loại được workbook: %s", exc, exc_info=True)
        state["status"] = "error"
        state["error_message"] = f"LLM không phân loại được workbook: {exc}"
        return state
    file_info = _basic_file_info(file_path, len(workbook_sheets))
    state["file_info"] = file_info
    print("\n=== THÔNG TIN FILE VÀ PHÂN LOẠI SHEET ===")
    print(json.dumps({
        "file_info": file_info,
        "workbook_sheets": workbook_sheets,
    }, ensure_ascii=False, indent=2))
    print("=== KẾT THÚC PHÂN LOẠI SHEET ===\n")
    if workbook_sheets:
        state["workbook_sheets"] = workbook_sheets
        workbook_context = extract_instruction_context(file_path, workbook_sheets)
        state["workbook_instruction_context"] = workbook_context
        instructions = effective_instructions(instructions, workbook_context)

    try:
        partitions = read_dataset_partitions(file_path, workbook_sheets, instructions)
        if not partitions or any(frame.empty for frame in partitions.values()):
            raise ValueError("Tệp dữ liệu không có bản ghi DATA để phân tích.")
        analysis_frames = split_partitions_by_source(partitions)
    except Exception as e:
        logger.error(f"Error loading data for request {request_id}: {e}", exc_info=True)
        state['status'] = "error"
        state['error_message'] = f"Không thể đọc hoặc chuẩn bị dữ liệu: {e}"
        return state

    try:
        column_semantics = _classify_columns(
            llm, analysis_frames, instructions, file_info, workbook_sheets
        )
        attach_column_semantics(analysis_frames, column_semantics)
        state["column_semantics"] = column_semantics
        print("\n=== PHÂN LOẠI Ý NGHĨA VÀ QUYỀN SỬ DỤNG CỘT ===")
        print(json.dumps(column_semantics, ensure_ascii=False, indent=2))
        print("=== KẾT THÚC PHÂN LOẠI CỘT ===\n")
    except Exception as exc:
        logger.error("LLM không phân loại được ý nghĩa và quyền sử dụng cột: %s", exc, exc_info=True)
        state["status"] = "error"
        state["error_message"] = f"LLM không phân loại được ý nghĩa và quyền sử dụng cột: {exc}"
        return state

    sheet_profiles = {}
    data_scope_profiles = {}
    for name, frame in analysis_frames.items():
        column_details = _profile_frame(frame)
        data_scope = _profile_data_scope(name, frame)
        data_scope_profiles[name] = data_scope.model_dump(mode="json")
        sheet_profiles[name] = {
            "num_rows": len(frame), "num_columns": len(column_details),
            "column_details": column_details, "key_observations": "",
            "data_scope": data_scope.model_dump(mode="json"),
        }
    state["analysis_partitions"] = list(analysis_frames)
    state["sheet_profiles"] = sheet_profiles
    state["data_scope_profiles"] = data_scope_profiles
    if len(sheet_profiles) == 1:
        only_profile = next(iter(sheet_profiles.values()))
        aggregate_details = {
            column: detail for column, detail in only_profile["column_details"].items()
            if not column.startswith("_")
        }
    else:
        aggregate_details = {
            f"{partition} :: {column}": detail
            for partition, profile in sheet_profiles.items()
            for column, detail in profile["column_details"].items()
            if not column.startswith("_")
        }
    profile_data = {
        "num_rows": sum(item["num_rows"] for item in sheet_profiles.values()),
        "num_columns": sum(len([c for c in item["column_details"] if not c.startswith("_")]) for item in sheet_profiles.values()),
        "column_details": aggregate_details,
        "key_observations": "",
    }


    max_retries = 3
    base_delay = 2
    llm_raw_output_str = ""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to invoke LLM for data profiling...")

            parser = JsonOutputParser(pydantic_object=DatasetProfileResponse)
            prompt = PromptTemplate(
                template="""
                Bạn là chuyên gia lập hồ sơ dữ liệu, có khả năng làm việc với dữ liệu thuộc mọi lĩnh vực.

                Trước tiên, hãy đánh giá yêu cầu của người dùng. Nếu yêu cầu không liên quan đến việc
                phân tích hoặc lập báo cáo từ bộ dữ liệu được cung cấp, chỉ trả về đúng thông báo sau:
                "Tôi là trợ lý tạo báo cáo dữ liệu và không có thông tin về chủ đề đó. Vui lòng cung cấp yêu cầu liên quan đến việc tạo báo cáo từ dữ liệu."
                Không thêm bất kỳ giải thích nào khi từ chối.

                Nếu yêu cầu phù hợp, hãy thực hiện các việc sau:
                - Nhận diện lĩnh vực của dữ liệu từ tên cột, kiểu dữ liệu và yêu cầu người dùng; không mặc định
                  đây là dữ liệu kinh doanh, bán hàng hay bất kỳ lĩnh vực cụ thể nào.
                - Tóm tắt cấu trúc dữ liệu, vấn đề chất lượng và các bước chuẩn bị dữ liệu ban đầu.
                - Tập trung vào giá trị thiếu, ngoại lệ, kiểu dữ liệu chưa phù hợp và điểm thiếu nhất quán.
                - Dùng thuật ngữ chuyên nghiệp phù hợp với lĩnh vực được nhận diện. Nếu chưa đủ bằng chứng
                  để xác định lĩnh vực, hãy dùng ngôn ngữ báo cáo trung tính.

                Hồ sơ dữ liệu:
                {profile_data}

                Khi có workbook_sheets, hãy dùng tên sheet và tiêu đề cột chính xác để xác định nguồn
                của phát hiện. Không bao giờ xem `_source_sheet` là một chỉ số phân tích.

                Yêu cầu của người dùng: {instructions}

                {format_instructions}

                Chỉ trả về JSON hợp lệ.
                """,
                input_variables=["profile_data", "instructions"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )



            profile_payload = dict(profile_data)
            if workbook_sheets:
                profile_payload["workbook_sheets"] = workbook_sheets
            profile_data_str = json.dumps(profile_payload, ensure_ascii=False, indent=2)

            llm_response = llm.invoke(prompt.invoke({
                "profile_data": profile_data_str,
                "instructions": instructions
            }),config={"request_options": {"timeout": 60}})
            llm_raw_output_str = text_from_response(llm_response)
            stripped_str = llm_raw_output_str.strip()
            if "Tôi là trợ lý tạo báo cáo dữ liệu và không có thông tin về chủ đề đó" in stripped_str:
                state['status'] = "error"
                state['error_message'] = "User instructions were not related to data report generation."
                return state

            parsed_profile_output = _parse_dataset_profile_response(stripped_str)


            if parsed_profile_output.key_observations.strip() == (
                    "Tôi là trợ lý tạo báo cáo dữ liệu và không có thông tin về chủ đề đó. "
                    "Vui lòng cung cấp yêu cầu liên quan đến phân tích hoặc tạo báo cáo dữ liệu."
            ):
                logger.warning(f"User instructions unrelated to data analysis for request {request_id}")
                state['status'] = "invalid_instructions"

                state['error_message'] = parsed_profile_output.key_observations
                return state


            profile_data["key_observations"] = parsed_profile_output.key_observations

            state['dataframe_profile'] = DataProfile(**profile_data)
            state['status'] = "data_profiled"
            logger.info(f"DataAnalysisNode completed for request: {request_id}")
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
            logger.error(f"LLM output for request {request_id} was invalid JSON or failed Pydantic validation: {e}", exc_info=True)
            logger.error(f"Raw LLM Output: {llm_raw_output_str[:500]}...")
            state['status'] = "error"
            state['error_message'] = f"LLM output was invalid JSON or malformed. Validation Error: {e}"
            return state

        except Exception as e:
            logger.error(f"An unexpected non-retryable error occurred in DataAnalysisNode for request {request_id}: {e}",
                         exc_info=True)
            state['status'] = "error"
            state['error_message'] = f"An unexpected error occurred during data analysis: {e}"
            return state
    state['status'] = "error"
    state['error_message'] = "An unexpected failure occurred after all LLM retries."
    return state
