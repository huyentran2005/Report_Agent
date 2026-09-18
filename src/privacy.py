"""Privacy guards for columns that may contain people's names."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_PERSON_TERMS = {
    "person", "people", "customer", "client", "employee", "staff", "student",
    "patient", "doctor", "agent", "owner", "contact", "respondent", "user", "reviewer",
    "evaluator", "assessor", "caller", "interviewer", "teacher", "manager",
    "nguoi", "nhan_vien", "khach_hang", "benh_nhan", "hoc_sinh", "sinh_vien",
    "bac_si", "tu_van_vien", "dien_vien",
}
_NON_PERSON_TERMS = {
    "product", "item", "company", "organization", "organisation", "brand", "category",
    "sheet", "file", "model", "campaign", "project", "san_pham", "cong_ty", "thuong_hieu",
}


def is_person_name_column(column: object, series: pd.Series | None = None) -> bool:
    """Conservatively flag columns likely to contain identifiable personal names."""
    name = _normalized(column)
    tokens = set(name.split("_"))
    if tokens & _NON_PERSON_TERMS:
        return False
    direct = {
        "name", "full_name", "fullname", "person_name", "contact_name", "display_name",
        "ho_ten", "hoten", "ten_nguoi", "ten_nhan_vien", "ten_khach_hang",
        "ten_benh_nhan", "ten_hoc_sinh", "ten_sinh_vien",
    }
    if name in direct or ("name" in tokens and bool(tokens & _PERSON_TERMS)):
        return True
    if "ten" in tokens and bool(tokens & _PERSON_TERMS):
        return True

    if name in {"ten", "name"}:
        return True
    role_headers = {
        "reviewer", "evaluator", "assessor", "caller", "interviewer", "contact_person",
        "nguoi_danh_gia", "nguoi_goi", "nguoi_phu_trach",
    }
    if name in role_headers and (series is None or not pd.api.types.is_numeric_dtype(series)):
        return True
    return False


def person_name_columns(df: pd.DataFrame) -> set[str]:
    return {str(column) for column in df.columns if is_person_name_column(column, df[column])}


def safe_column_details(column_details: dict) -> dict:
    """Remove potentially identifying sample values before an object is sent to an LLM."""
    safe = {}
    for column, details in column_details.items():
        item = dict(details)
        if item.get("is_sensitive_person_name") or is_person_name_column(column):
            item.pop("top_5_values", None)
            item["is_sensitive_person_name"] = True
        safe[column] = item
    return safe
