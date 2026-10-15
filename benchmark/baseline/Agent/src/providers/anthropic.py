"""Adapt the Anthropic Messages API to provider-neutral messages."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from benchmark.baseline.Agent.src.providers.base import (
    ChatMessage,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)
from benchmark.baseline.Agent.src.providers.http import JsonTransport, UrllibJsonTransport


class AnthropicProvider:
    """Call the Anthropic Messages API and normalize its response."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 300.0,
        transport: Optional[JsonTransport] = None,
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        temperature: float = 0.0,
    ):
        """Initialize the Anthropic provider.

        Arg:
            api_key: Anthropic API key.
            model: Exact Claude model identifier.
            max_tokens: Maximum output tokens per response.
            timeout_seconds: HTTP request timeout.
            transport: Optional injectable JSON transport for tests.
            base_url: Overridable Anthropic API base URL.
            anthropic_version: Anthropic API version header.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        if not api_key:
            raise ProviderError("API key must not be empty.")
        if not model:
            raise ProviderError("Model name must not be empty.")
        if max_tokens <= 0:
            raise ProviderError("max_tokens must be positive.")
        if temperature < 0:
            raise ProviderError("temperature must be non-negative.")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transport = transport or UrllibJsonTransport()
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.temperature = float(temperature)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        """Request and normalize one Anthropic Messages response.

        Arg:
            messages: Provider-neutral conversation history.
            tools: Local tools available to Claude.
            tool_choice: Auto to allow tools or none to force text output.

        Returns:
            Normalized assistant message and usage information.
        """
        system_text, api_messages = self._serialize_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if system_text:
            payload["system"] = system_text
        if tool_choice not in {"auto", "none"}:
            raise ProviderError("tool_choice must be 'auto' or 'none'.")
        if tools and tool_choice == "auto":
            payload["tools"] = [self._serialize_tool(tool) for tool in tools]
            payload["tool_choice"] = {"type": "auto"}

        raw_response = self.transport.post_json(
            url=f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        return self._parse_response(raw_response)

    @staticmethod
    def _serialize_tool(tool: ToolDefinition) -> dict[str, Any]:
        """Convert one neutral tool to Anthropic format.

        Arg:
            tool: Provider-neutral tool definition.

        Returns:
            Anthropic client-tool object.
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    @staticmethod
    def _serialize_messages(
        messages: Sequence[ChatMessage],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert neutral history to Anthropic system and message fields.

        Arg:
            messages: Provider-neutral conversation history.

        Returns:
            Combined system text and Anthropic message list.
        """
        system_parts: list[str] = []
        serialized: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            """Append accumulated tool results as one Anthropic user message.

            Arg:
                None.

            Returns:
                None.
            """
            if pending_results:
                serialized.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for message in messages:
            if message.role == "system":
                flush_results()
                if message.content:
                    system_parts.append(message.content)
                continue
            if message.role == "tool":
                if not message.tool_call_id:
                    raise ProviderError("A tool result requires tool_call_id.")
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                if message.is_error:
                    result["is_error"] = True
                pending_results.append(result)
                continue

            flush_results()
            if message.role == "user":
                serialized.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                serialized.append({"role": "assistant", "content": content})
            else:
                raise ProviderError(f"Unsupported message role: {message.role}")
        flush_results()
        return "\n\n".join(system_parts), serialized

    @staticmethod
    def _parse_response(raw_response: dict[str, Any]) -> ProviderResponse:
        """Normalize one Anthropic Messages response object.

        Arg:
            raw_response: Parsed Anthropic JSON response.

        Returns:
            Provider-neutral response.
        """
        raw_content = raw_response.get("content")
        if not isinstance(raw_content, list):
            raise ProviderError("Anthropic response is missing content blocks.")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in raw_content:
            if not isinstance(block, dict):
                raise ProviderError("Anthropic returned an invalid content block.")
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "tool_use":
                call_input = block.get("input")
                if not isinstance(call_input, dict):
                    raise ProviderError("Anthropic tool input must be a JSON object.")
                try:
                    tool_calls.append(
                        ToolCall(
                            id=str(block["id"]),
                            name=str(block["name"]),
                            arguments=call_input,
                        )
                    )
                except KeyError as exc:
                    raise ProviderError(
                        "Anthropic returned an invalid tool-use block."
                    ) from exc

        raw_usage = raw_response.get("usage") or {}
        input_tokens = int(raw_usage.get("input_tokens") or 0)
        output_tokens = int(raw_usage.get("output_tokens") or 0)
        stop_reason = str(raw_response.get("stop_reason") or "")
        finish_reason = "tool_calls" if stop_reason == "tool_use" else stop_reason
        return ProviderResponse(
            message=ChatMessage.assistant("".join(text_parts), tool_calls),
            finish_reason=finish_reason,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            response_id=raw_response.get("id"),
        )
