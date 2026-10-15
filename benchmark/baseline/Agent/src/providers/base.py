"""Define the provider-neutral messages used by every model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence


class ProviderError(RuntimeError):
    """Report invalid provider configuration, requests, or responses."""


@dataclass(frozen=True)
class ToolDefinition:
    """Describe one local tool independently of any model provider."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_tool(cls, tool: Any) -> "ToolDefinition":
        """Create a definition from one registered tool object.

        Arg:
            tool: Object exposing name, description, and input_schema properties.

        Returns:
            Provider-neutral tool definition.
        """
        return cls(
            name=str(tool.name),
            description=str(tool.description),
            input_schema=dict(tool.input_schema),
        )


@dataclass(frozen=True)
class ToolCall:
    """Represent one tool invocation requested by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    """Represent one provider-neutral conversation message."""

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: Optional[str] = None
    is_error: bool = False

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Sequence[ToolCall] = (),
    ) -> "ChatMessage":
        """Create an assistant message, optionally containing tool calls.

        Arg:
            content: Text returned by the model.
            tool_calls: Tool calls returned in the same model response.

        Returns:
            Provider-neutral assistant message.
        """
        return cls(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls),
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ) -> "ChatMessage":
        """Create the result of one local tool execution.

        Arg:
            tool_call_id: Identifier from the matching model tool call.
            content: Serialized tool output or error message.
            is_error: Whether local tool execution failed.

        Returns:
            Provider-neutral tool-result message.
        """
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )


@dataclass(frozen=True)
class Usage:
    """Store normalized token counts from one provider response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ProviderResponse:
    """Store one normalized model response."""

    message: ChatMessage
    finish_reason: str
    usage: Usage = field(default_factory=Usage)
    response_id: Optional[str] = None


class LLMProvider(Protocol):
    """Describe the common interface implemented by all providers."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        """Request one model response.

        Arg:
            messages: Provider-neutral conversation history.
            tools: Local tools available to the model.
            tool_choice: Whether the model may call tools.

        Returns:
            Normalized text, tool calls, finish reason, and token usage.
        """
        ...
