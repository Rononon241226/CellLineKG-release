"""Execute validated pandas code in a persistent child process."""

from __future__ import annotations

import ast
import contextlib
import io
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.baseline.Agent.src.tools.data_files import DataAccess
from benchmark.baseline.Agent.src.tools.cell_scope import CellScopeReader


SAFE_BUILTINS = {
    "Exception": Exception,
    "ValueError": ValueError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def worker_main(connection: Connection, workspace_root: str) -> None:
    """Run the persistent restricted execution loop.

    Arg:
        connection: Parent-child communication pipe.
        workspace_root: Project root containing the data directory.

    Returns:
        None.
    """
    namespace = _build_namespace(Path(workspace_root))
    connection.send({"status": "ready"})
    try:
        while True:
            command = connection.recv()
            action = command.get("action")
            if action == "close":
                break
            if action != "execute":
                connection.send(
                    {
                        "ok": False,
                        "error_type": "WorkerProtocolError",
                        "error_message": f"Unknown worker action: {action}",
                    }
                )
                continue
            connection.send(_execute_code(command["code"], namespace))
    except EOFError:
        pass
    finally:
        connection.close()


def _build_namespace(workspace_root: Path) -> dict[str, Any]:
    """Build the restricted persistent execution namespace.

    Arg:
        workspace_root: Project root containing the data directory.

    Returns:
        Namespace containing pandas and approved data readers.
    """
    data_access = DataAccess(workspace_root)
    cell_scope = CellScopeReader(workspace_root)
    return {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "list_data_files": data_access.list_data_files,
        "read_csv": data_access.read_csv,
        "read_excel": data_access.read_excel,
        "read_json": data_access.read_json,
        "read_text": data_access.read_text,
        "search_text": data_access.search_text,
        "read_cell_scope": cell_scope.read_cell_scope,
    }


def _execute_code(code: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Execute code and capture its visible output.

    Arg:
        code: Validated Python source.
        namespace: Persistent restricted execution namespace.

    Returns:
        Serializable execution result payload.
    """
    started = time.perf_counter()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
        result: Any = None
        with (
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stderr_buffer),
        ):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
                expression = ast.Expression(body=tree.body[-1].value)
                if prefix.body:
                    exec(
                        compile(ast.fix_missing_locations(prefix), "<llm-table>", "exec"),
                        namespace,
                        namespace,
                    )
                result = eval(
                    compile(ast.fix_missing_locations(expression), "<llm-table>", "eval"),
                    namespace,
                    namespace,
                )
            else:
                exec(
                    compile(ast.fix_missing_locations(tree), "<llm-table>", "exec"),
                    namespace,
                    namespace,
                )

        result_text, result_type, result_metadata = _format_result(result)
        return {
            "ok": True,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": result_text,
            "result_type": result_type,
            "result_metadata": result_metadata,
            "execution_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "ok": False,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=4),
            "execution_seconds": time.perf_counter() - started,
        }


def _format_result(value: Any) -> tuple[str, str, dict[str, Any]]:
    """Format the final expression without changing the stored value.

    Arg:
        value: Value produced by the final expression.

    Returns:
        Display text, result type, and compact metadata.
    """
    if isinstance(value, pd.DataFrame):
        text = value.to_string(max_rows=100, max_cols=50)
        return (
            text,
            "DataFrame",
            {
                "shape": list(value.shape),
                "columns": [str(column) for column in value.columns[:200]],
                "columns_truncated": len(value.columns) > 200,
            },
        )
    if isinstance(value, pd.Series):
        text = value.to_string(max_rows=100)
        return (
            text,
            "Series",
            {
                "length": len(value),
                "name": str(value.name) if value.name is not None else None,
            },
        )
    return repr(value), type(value).__name__, {}
