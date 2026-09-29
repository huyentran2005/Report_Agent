"""Làm sạch thông tin cột sau khi LLM đã phân loại dữ liệu nhạy cảm."""
from __future__ import annotations

def safe_column_details(column_details: dict) -> dict:
    """Loại mẫu giá trị của cột mà LLM không cho phép sử dụng."""
    safe = {}
    for column, details in column_details.items():
        item = dict(details)
        if item.get("usage_permission") == "blocked":
            item.pop("top_5_values", None)
        safe[column] = item
    return safe
