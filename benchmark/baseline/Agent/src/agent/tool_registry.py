"""Register local tools and safely serialize their execution results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from benchmark.baseline.Agent.src.providers.base import (
    ProviderError,
    ToolCall,
    ToolDefinition,
)


class AgentTool(Protocol):
    """Describe a local tool that can be exposed to a model."""

    @property
    def name(self) -> str:
        """Return the model-visible tool name."""
        ...

    @property
    def description(self) -> str:
        """Return the model-visible tool description."""
        ...

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the model-visible JSON input schema."""
        ...

    def run(self, **arguments: Any) -> Any:
        """Execute the tool with model-generated arguments."""
        ...


@dataclass(frozen=True)
class ToolExecution:
    """Store one serialized local tool result."""

    content: str
    is_error: bool = False


class ToolRegistry:
    """Expose unique local tools and dispatch model tool calls."""

    def __init__(self, tools: Sequence[AgentTool]):
        """Initialize the tool registry.

        Arg:
            tools: Local tool objects to expose to the model.

        Returns:
            None.
        """
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            if not tool.name:
                raise ProviderError("Tool names must not be empty.")
            if tool.name in self._tools:
                raise ProviderError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return provider-neutral definitions for all registered tools.

        Arg:
            None.

        Returns:
            Tool definitions in registration order.
        """
        return tuple(
            ToolDefinition.from_tool(tool)
            for tool in self._tools.values()
        )

    def execute(self, call: ToolCall) -> ToolExecution:
        """Execute one requested tool and serialize its result.

        Arg:
            call: Provider-neutral model tool call.

        Returns:
            JSON tool output and an error flag.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return self._error(
                "UnknownToolError",
                f"Unknown tool: {call.name}",
            )

        try:
            result = tool.run(**call.arguments)
            content = json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return self._error(type(exc).__name__, str(exc))
        return ToolExecution(content=content)

    @staticmethod
    def _error(error_type: str, message: str) -> ToolExecution:
        """Create a structured tool error for the model.

        Arg:
            error_type: Exception or registry error category.
            message: Concise error explanation.

        Returns:
            Serialized error marked as failed.
        """
        content = json.dumps(
            {
                "ok": False,
                "error_type": error_type,
                "error_message": message,
            },
            ensure_ascii=False,
        )
        return ToolExecution(content=content, is_error=True)
