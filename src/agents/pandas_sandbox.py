"""Restricted Pandas action execution for the planner observation loop."""
from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd


class UnsafePandasAction(ValueError):
    pass


_BLOCKED_NODES = (
    ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
    ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Global,
    ast.Nonlocal, ast.Delete,
)
_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
}
_BLOCKED_ATTRIBUTES = {
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet",
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql",
    "system", "popen", "remove", "unlink", "rmdir", "mkdir", "makedirs",
}


def _validate_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            raise UnsafePandasAction(f"Cú pháp không được phép: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise UnsafePandasAction(f"Tên không được phép: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _BLOCKED_ATTRIBUTES:
                raise UnsafePandasAction(f"Thuộc tính không được phép: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in {"len", "min", "max", "sum", "round", "sorted", "list", "dict", "set", "tuple", "print"}:
                raise UnsafePandasAction(f"Hàm global không được phép: {node.func.id}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.astype(object).where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return _json_safe(value.rename(value.name or "value").reset_index())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, pd.Period, np.datetime64)):
        return str(value)
    return value


def execute_pandas(code: str, dataframe: pd.DataFrame) -> Any:
    """Execute a read-only Pandas snippet that assigns its output to ``result``."""
    tree = ast.parse(code, mode="exec")
    _validate_tree(tree)
    local_scope = {"df": dataframe.copy(deep=True), "pd": pd, "np": np}
    safe_builtins = {
        "len": len, "min": min, "max": max, "sum": sum, "round": round,
        "sorted": sorted, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "print": print,
    }
    exec(compile(tree, "<pandas_action>", "exec"), {"__builtins__": safe_builtins}, local_scope)
    if "result" not in local_scope:
        raise ValueError("Action phải gán kết quả cuối vào biến `result`.")
    return _json_safe(local_scope["result"])
