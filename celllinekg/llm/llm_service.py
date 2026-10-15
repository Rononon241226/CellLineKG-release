"""Shared LLM service for CellLineKG modules."""

from __future__ import annotations

import os
from typing import Any

from celllinekg.config import load_dotenv


def build_chat_llm(
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = 4096,
    timeout_seconds: float = 300.0,
    preserve_max_tokens_parameter: bool = False,
    require_supported_parameters: bool = False,
    reasoning_effort: str | None = None,
) -> Any:
    """Build one OpenAI-compatible chat client from explicit settings.

    Arg:
        api_key: API credential for the selected endpoint.
        model: Exact model identifier sent to the endpoint.
        base_url: Optional OpenAI-compatible API base URL.
        temperature: Optional sampling temperature.
        max_tokens: Maximum output tokens per response.
        timeout_seconds: Request timeout in seconds.
        preserve_max_tokens_parameter: Send max_tokens without SDK renaming.
        require_supported_parameters: Restrict OpenRouter to compatible endpoints.
        reasoning_effort: Optional explicit reasoning effort.

    Returns:
        A configured ChatOpenAI client for synchronous chat calls.
    """
    if not api_key:
        raise RuntimeError("API key is not configured.")
    if not model:
        raise RuntimeError("Model is not configured.")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "LangChain OpenAI dependency is missing. Install it with: "
            "`python -m pip install -U langchain-openai`."
        ) from exc

    llm_kwargs: dict[str, object] = {
        "api_key": api_key,
        "model": model,
        "timeout": timeout_seconds,
    }
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    extra_body: dict[str, object] = {}
    if preserve_max_tokens_parameter:
        extra_body["max_tokens"] = max_tokens
    else:
        llm_kwargs["max_tokens"] = max_tokens
    if require_supported_parameters:
        extra_body["provider"] = {"require_parameters": True}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    if extra_body:
        llm_kwargs["extra_body"] = extra_body
    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatOpenAI(**llm_kwargs)


def get_chat_llm(
    *,
    max_tokens: int = 4096,
    preserve_max_tokens_parameter: bool = False,
    require_supported_parameters: bool = False,
    reasoning_effort: str | None = None,
) -> Any:
    """
    Create a chat LLM client from environment variables.

    Arg:
        max_tokens: Maximum output tokens per response.
        preserve_max_tokens_parameter: Send max_tokens without SDK renaming.
        require_supported_parameters: Restrict OpenRouter to compatible endpoints.
        reasoning_effort: Optional explicit reasoning effort.

    Returns:
        A configured ChatOpenAI client for synchronous chat calls.
    """
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")
    if not model:
        raise RuntimeError("LLM_MODEL is not configured.")

    return build_chat_llm(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        preserve_max_tokens_parameter=preserve_max_tokens_parameter,
        require_supported_parameters=require_supported_parameters,
        reasoning_effort=reasoning_effort,
    )
