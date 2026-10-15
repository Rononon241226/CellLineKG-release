"""Validate model-generated Python before restricted execution."""

from __future__ import annotations

import ast


FORBIDDEN_NODE_TYPES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
)

FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "builtins",
    "compile",
    "ctypes",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "importlib",
    "input",
    "inspect",
    "locals",
    "memoryview",
    "object",
    "open",
    "os",
    "pathlib",
    "requests",
    "setattr",
    "shutil",
    "socket",
    "subprocess",
    "super",
    "sys",
    "type",
    "urllib",
    "vars",
}

FORBIDDEN_ATTRIBUTES = {
    "ExcelFile",
    "HDFStore",
    "io",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
}


class CodeValidationError(ValueError):
    """Report Python code that violates the restricted execution policy."""


def validate_code(code: str) -> ast.Module:
    """Validate Python syntax and reject dangerous operations.

    Arg:
        code: Model-generated Python source.

    Returns:
        Parsed and validated module syntax tree.
    """
    if not isinstance(code, str) or not code.strip():
        raise CodeValidationError("Python code must be a non-empty string.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeValidationError(f"Invalid Python syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise CodeValidationError(
                f"Python syntax is not allowed: {type(node).__name__}"
            )
        if isinstance(node, ast.Name):
            _validate_name(node.id)
        if isinstance(node, ast.Attribute):
            _validate_attribute(node.attr)
    return tree


def _validate_name(name: str) -> None:
    """Validate one referenced Python name.

    Arg:
        name: Name found in the parsed syntax tree.

    Returns:
        None.
    """
    if name.startswith("_") or name in FORBIDDEN_NAMES:
        raise CodeValidationError(f"Python name is not allowed: {name}")


def _validate_attribute(attribute: str) -> None:
    """Validate one referenced object attribute.

    Arg:
        attribute: Attribute found in the parsed syntax tree.

    Returns:
        None.
    """
    if attribute.startswith("_") or attribute in FORBIDDEN_ATTRIBUTES:
        raise CodeValidationError(f"Python attribute is not allowed: {attribute}")
