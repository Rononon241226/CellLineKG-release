"""Expose one restricted pandas session as an agent tool."""

from __future__ import annotations

from typing import Any

from benchmark.baseline.Agent.src.sandbox.session import PandasSession


PANDAS_TOOL_NAME = "run_python"
PANDAS_TOOL_DESCRIPTION = """
Run Python and pandas code in the current question's persistent, read-only data session.
Available names include pd, list_data_files, read_csv, read_excel, read_json,
read_text, search_text, and read_cell_scope. Read only project paths beginning
with data/. read_cell_scope returns the fixed benchmark DepMap ModelIDs as a Series
for membership filtering, such as frame["ModelID"].isin(read_cell_scope()).
Imports, direct file access, network access, subprocesses, and file writes are blocked.
""".strip()


class PandasTool:
    """Wrap a persistent pandas session for later provider integration."""

    def __init__(self, session: PandasSession):
        """Initialize the pandas agent tool.

        Arg:
            session: Active or startable one-question pandas session.

        Returns:
            None.
        """
        self.session = session

    @property
    def name(self) -> str:
        """Return the provider-neutral tool name.

        Arg:
            None.

        Returns:
            Tool name used in model tool calls.
        """
        return PANDAS_TOOL_NAME

    @property
    def description(self) -> str:
        """Return the provider-neutral tool description.

        Arg:
            None.

        Returns:
            Tool instructions shown to each model.
        """
        return PANDAS_TOOL_DESCRIPTION

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the provider-neutral JSON input schema.

        Arg:
            None.

        Returns:
            JSON schema requiring one Python code string.
        """
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python and pandas code to execute.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        }

    def run(self, code: str) -> dict[str, Any]:
        """Execute one Python block through the restricted session.

        Arg:
            code: Model-generated Python source.

        Returns:
            Serializable execution result.
        """
        return self.session.execute(code).to_dict()
