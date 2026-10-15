"""Adapt OpenAI-compatible Chat Completions APIs."""

from __future__ import annotations

import json
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


class OpenAICompatibleProvider:
    """Call a Chat Completions API and normalize its response."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 300.0,
        transport: Optional[JsonTransport] = None,
        max_tokens_parameter: str = "max_tokens",
        temperature: float = 0.0,
    ):
        """Initialize one OpenAI-compatible provider.

        Arg:
            api_key: Provider API key.
            model: Exact model identifier selected for evaluation.
            base_url: API base URL, with or without a trailing slash.
            max_tokens: Maximum output tokens per response.
            timeout_seconds: HTTP request timeout.
            transport: Optional injectable JSON transport for tests.
            max_tokens_parameter: Provider-specific output-limit field name.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        if not api_key:
            raise ProviderError("API key must not be empty.")
        if not model:
            raise ProviderError("Model name must not be empty.")
        if not base_url:
            raise ProviderError("Base URL must not be empty.")
        if max_tokens <= 0:
            raise ProviderError("max_tokens must be positive.")
        if temperature < 0:
            raise ProviderError("temperature must be non-negative.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transport = transport or UrllibJsonTransport()
        self.max_tokens_parameter = max_tokens_parameter
        self.temperature = float(temperature)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        """Request and normalize one Chat Completions response.

        Arg:
            messages: Provider-neutral conversation history.
            tools: Local tools available to the model.
            tool_choice: Auto to allow tools or none to force text output.

        Returns:
            Normalized assistant message and usage information.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._serialize_message(message) for message in messages],
            self.max_tokens_parameter: self.max_tokens,
            "temperature": self.temperature,
        }
        if tool_choice not in {"auto", "none"}:
            raise ProviderError("tool_choice must be 'auto' or 'none'.")
        if tools:
            payload["tools"] = [self._serialize_tool(tool) for tool in tools]
            payload["tool_choice"] = tool_choice

        raw_response = self.transport.post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        return self._parse_response(raw_response)

    @staticmethod
    def _serialize_tool(tool: ToolDefinition) -> dict[str, Any]:
        """Convert one neutral tool to Chat Completions format.

        Arg:
            tool: Provider-neutral tool definition.

        Returns:
            Chat Completions function-tool object.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        """Convert one neutral message to Chat Completions format.

        Arg:
            message: Provider-neutral conversation message.

        Returns:
            Chat Completions message object.
        """
        if message.role == "tool":
            if not message.tool_call_id:
                raise ProviderError("A tool result requires tool_call_id.")
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if message.role not in {"system", "user", "assistant"}:
            raise ProviderError(f"Unsupported message role: {message.role}")

        serialized: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.role == "assistant" and message.tool_calls:
            serialized["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        return serialized

    @staticmethod
    def _parse_response(raw_response: dict[str, Any]) -> ProviderResponse:
        """Normalize one Chat Completions response object.

        Arg:
            raw_response: Parsed provider JSON response.

        Returns:
            Provider-neutral response.
        """
        try:
            choice = raw_response["choices"][0]
            raw_message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Chat response is missing choices[0].message.") from exc

        content = raw_message.get("content") or ""
        raw_calls = raw_message.get("tool_calls") or []
        tool_calls = tuple(
            OpenAICompatibleProvider._parse_tool_call(raw_call)
            for raw_call in raw_calls
        )
        raw_usage = raw_response.get("usage") or {}
        input_tokens = int(raw_usage.get("prompt_tokens") or 0)
        output_tokens = int(raw_usage.get("completion_tokens") or 0)
        total_tokens = int(
            raw_usage.get("total_tokens") or input_tokens + output_tokens
        )
        return ProviderResponse(
            message=ChatMessage.assistant(content, tool_calls),
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=Usage(input_tokens, output_tokens, total_tokens),
            response_id=raw_response.get("id"),
        )

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> ToolCall:
        """Normalize one Chat Completions tool call.

        Arg:
            raw_call: Tool-call object from the provider response.

        Returns:
            Provider-neutral tool call with parsed JSON arguments.
        """
        try:
            call_id = str(raw_call["id"])
            function = raw_call["function"]
            name = str(function["name"])
            raw_arguments = function.get("arguments") or "{}"
            arguments = json.loads(raw_arguments)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError("Provider returned an invalid tool call.") from exc
        if not isinstance(arguments, dict):
            raise ProviderError("Tool-call arguments must decode to a JSON object.")
        return ToolCall(id=call_id, name=name, arguments=arguments)


class OpenAIProvider(OpenAICompatibleProvider):
    """Configure the OpenAI Chat Completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 300.0,
        transport: Optional[JsonTransport] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
    ):
        """Initialize the OpenAI adapter.

        Arg:
            api_key: OpenAI API key.
            model: Exact OpenAI model identifier.
            max_tokens: Maximum output tokens per response.
            timeout_seconds: HTTP request timeout.
            transport: Optional injectable JSON transport for tests.
            base_url: Overridable OpenAI-compatible API base URL.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        super().__init__(
            api_key,
            model,
            base_url,
            max_tokens,
            timeout_seconds,
            transport,
            max_tokens_parameter="max_completion_tokens",
            temperature=temperature,
        )


class DeepSeekProvider(OpenAICompatibleProvider):
    """Configure the DeepSeek Chat Completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 300.0,
        transport: Optional[JsonTransport] = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
    ):
        """Initialize the DeepSeek adapter.

        Arg:
            api_key: DeepSeek API key.
            model: Exact DeepSeek model identifier.
            max_tokens: Maximum output tokens per response.
            timeout_seconds: HTTP request timeout.
            transport: Optional injectable JSON transport for tests.
            base_url: Overridable DeepSeek-compatible API base URL.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        super().__init__(
            api_key,
            model,
            base_url,
            max_tokens,
            timeout_seconds,
            transport,
            temperature=temperature,
        )


class KimiProvider(OpenAICompatibleProvider):
    """Configure the Kimi Chat Completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 300.0,
        transport: Optional[JsonTransport] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        temperature: float = 0.0,
    ):
        """Initialize the Kimi adapter.

        Arg:
            api_key: Moonshot API key.
            model: Exact Kimi model identifier.
            max_tokens: Maximum output tokens per response.
            timeout_seconds: HTTP request timeout.
            transport: Optional injectable JSON transport for tests.
            base_url: Overridable Kimi-compatible API base URL.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        super().__init__(
            api_key,
            model,
            base_url,
            max_tokens,
            timeout_seconds,
            transport,
            temperature=temperature,
        )

class TestProvider(OpenAICompatibleProvider):
    """Configure the test Chat Completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        transport: Optional[JsonTransport] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        """Initialize the test adapter.

        Arg:
            api_key: test API key.
            model: Exact test model identifier.
            max_tokens: Maximum output tokens per response.
            timeout_seconds: HTTP request timeout.
            transport: Optional injectable JSON transport for tests.
            base_url: Overridable test-compatible API base URL.

        Returns:
            None.
        """
        super().__init__(
            api_key,
            model,
            base_url,
            max_tokens,
            timeout_seconds,
            transport,
        )
