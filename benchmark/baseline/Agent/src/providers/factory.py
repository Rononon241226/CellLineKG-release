"""Construct a provider adapter from an explicit provider name."""

from __future__ import annotations

from typing import Optional

from benchmark.baseline.Agent.src.providers.anthropic import AnthropicProvider
from benchmark.baseline.Agent.src.providers.base import LLMProvider, ProviderError
from benchmark.baseline.Agent.src.providers.http import JsonTransport
from benchmark.baseline.Agent.src.providers.openai_compatible import (
    DeepSeekProvider,
    KimiProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)


def create_provider(
    provider_name: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    timeout_seconds: float = 300.0,
    temperature: float = 0.0,
    transport: Optional[JsonTransport] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """Create one of the supported model providers.

    Arg:
        provider_name: One of openai, deepseek, kimi, or anthropic.
        api_key: API key passed directly by the later configuration layer.
        model: Exact provider model identifier.
        max_tokens: Maximum output tokens per model response.
        timeout_seconds: HTTP request timeout.
        temperature: Sampling temperature sent to the model provider.
        transport: Optional injectable JSON transport for tests.
        base_url: Optional provider endpoint override.

    Returns:
        Configured provider implementing the shared completion interface.
    """
    normalized_name = provider_name.strip().lower()
    common = {
        "api_key": api_key,
        "model": model,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "temperature": temperature,
        "transport": transport,
    }
    if normalized_name in {"openai", "gpt"}:
        if base_url:
            common["base_url"] = base_url
        return OpenAIProvider(**common)
    if normalized_name in {"openai-compatible", "openai_compatible"}:
        if not base_url:
            raise ProviderError(
                "An OpenAI-compatible provider requires base_url."
            )
        return OpenAICompatibleProvider(base_url=base_url, **common)
    if normalized_name == "deepseek":
        if base_url:
            common["base_url"] = base_url
        return DeepSeekProvider(**common)
    if normalized_name in {"kimi", "moonshot"}:
        if base_url:
            common["base_url"] = base_url
        return KimiProvider(**common)
    if normalized_name in {"anthropic", "claude"}:
        if base_url:
            common["base_url"] = base_url
        return AnthropicProvider(**common)
    raise ProviderError(f"Unsupported provider: {provider_name}")
