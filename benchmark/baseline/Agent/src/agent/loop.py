"""Run a provider-neutral model and local-tool conversation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from benchmark.baseline.Agent.src.agent.tool_registry import ToolRegistry
from benchmark.baseline.Agent.src.providers.base import (
    ChatMessage,
    LLMProvider,
    ProviderError,
    Usage,
)


BUDGET_FINALIZATION_PROMPT = (
    "Retrieval is complete. Provide the final answer using only the completed "
    "tool results. Do not request additional tools."
)


class AgentLoopError(ProviderError):
    """Report a failed tool loop while preserving its partial trajectory."""

    def __init__(
        self,
        message: str,
        *,
        messages: Sequence[ChatMessage] = (),
        usage: Usage = Usage(),
        accepted_retrieval_usage: Usage = Usage(),
        attempted_retrieval_usage: Usage = Usage(),
        finalization_usage: Usage = Usage(),
        budget_limited: bool = False,
        budget_event: dict[str, Any] | None = None,
        model_calls: int = 0,
        tool_calls: int = 0,
    ):
        """Initialize an error with completed work before failure.

        Arg:
            message: Human-readable failure reason.
            messages: Conversation accumulated before failure.
            usage: Token usage accumulated before failure.
            accepted_retrieval_usage: Retrieval usage retained in the checkpoint.
            attempted_retrieval_usage: Retrieval usage including a discarded call.
            finalization_usage: Usage from the final answer call.
            budget_limited: Whether retrieval stopped because of the budget.
            budget_event: Audit metadata for a discarded over-budget response.
            model_calls: Number of completed model requests.
            tool_calls: Number of executed local tool calls.

        Returns:
            None.
        """
        super().__init__(message)
        self.messages = tuple(messages)
        self.usage = usage
        self.accepted_retrieval_usage = accepted_retrieval_usage
        self.attempted_retrieval_usage = attempted_retrieval_usage
        self.finalization_usage = finalization_usage
        self.budget_limited = budget_limited
        self.budget_event = budget_event
        self.model_calls = model_calls
        self.tool_calls = tool_calls


class TokenBudgetExceeded(AgentLoopError):
    """Report that another model or tool round would exceed the token budget."""


class BudgetFinalizationError(AgentLoopError):
    """Report a failed answer call after retrieval budget rollback."""


@dataclass(frozen=True)
class AgentRunResult:
    """Store the complete result of one question."""

    answer: str
    messages: tuple[ChatMessage, ...]
    usage: Usage
    model_calls: int
    tool_calls: int
    accepted_retrieval_usage: Usage = Usage()
    attempted_retrieval_usage: Usage = Usage()
    finalization_usage: Usage = Usage()
    budget_limited: bool = False
    budget_event: dict[str, Any] | None = None


class AgentLoop:
    """Continue model requests until a final text answer is returned."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_tool_rounds: int = 12,
        max_retrieval_tokens: int | None = None,
    ):
        """Initialize the automatic tool loop.

        Arg:
            provider: Configured model provider.
            registry: Local tools available to the model.
            max_tool_rounds: Maximum model turns that may request tools.
            max_retrieval_tokens: Optional cumulative retrieval budget.

        Returns:
            None.
        """
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive.")
        if max_retrieval_tokens is not None and max_retrieval_tokens <= 0:
            raise ValueError("max_retrieval_tokens must be positive when provided.")
        self.provider = provider
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds
        self.max_retrieval_tokens = max_retrieval_tokens

    def run(
        self,
        question: str,
        system_prompt: str = "",
    ) -> AgentRunResult:
        """Answer one question through zero or more local tool calls.

        Arg:
            question: Benchmark question shown to the model.
            system_prompt: Optional shared evaluation instructions.

        Returns:
            Final answer, full conversation, usage, and call counts.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")

        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage("system", system_prompt))
        messages.append(ChatMessage("user", question))

        actual_usage = Usage()
        accepted_retrieval_usage = Usage()
        attempted_retrieval_usage = Usage()
        model_calls = 0
        tool_call_count = 0
        tool_rounds = 0

        while True:
            response = self.provider.complete(messages, self.registry.definitions)
            model_calls += 1
            calls = response.message.tool_calls
            if not calls:
                actual_usage = _add_usage(actual_usage, response.usage)
                finalization_usage = response.usage
                messages.append(response.message)
                if response.finish_reason == "tool_calls":
                    raise _build_loop_error(
                        "Provider ended for tool calls without returning any calls.",
                        messages,
                        actual_usage,
                        model_calls,
                        tool_call_count,
                        accepted_retrieval_usage=accepted_retrieval_usage,
                        attempted_retrieval_usage=attempted_retrieval_usage,
                        finalization_usage=finalization_usage,
                    )
                if not response.message.content.strip():
                    raise _build_loop_error(
                        "Model returned an empty final answer.",
                        messages,
                        actual_usage,
                        model_calls,
                        tool_call_count,
                        accepted_retrieval_usage=accepted_retrieval_usage,
                        attempted_retrieval_usage=attempted_retrieval_usage,
                        finalization_usage=finalization_usage,
                    )
                return AgentRunResult(
                    answer=response.message.content,
                    messages=tuple(messages),
                    usage=actual_usage,
                    model_calls=model_calls,
                    tool_calls=tool_call_count,
                    accepted_retrieval_usage=accepted_retrieval_usage,
                    attempted_retrieval_usage=attempted_retrieval_usage,
                    finalization_usage=finalization_usage,
                )

            attempted_retrieval_usage = _add_usage(
                attempted_retrieval_usage,
                response.usage,
            )
            actual_usage = _add_usage(actual_usage, response.usage)
            if (
                self.max_retrieval_tokens is not None
                and attempted_retrieval_usage.total_tokens
                > self.max_retrieval_tokens
            ):
                budget_event = {
                    "reason": "TokenBudgetReached",
                    "retrieval_token_limit": self.max_retrieval_tokens,
                    "accepted_retrieval_tokens": (
                        accepted_retrieval_usage.total_tokens
                    ),
                    "attempted_retrieval_tokens": (
                        attempted_retrieval_usage.total_tokens
                    ),
                    "discarded_response_usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "discarded_tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                            "executed": False,
                        }
                        for call in calls
                    ],
                }
                finalization_messages = [
                    *messages,
                    ChatMessage("user", BUDGET_FINALIZATION_PROMPT),
                ]
                try:
                    final_response = self.provider.complete(
                        finalization_messages,
                        self.registry.definitions,
                        tool_choice="none",
                    )
                except Exception as exc:
                    raise BudgetFinalizationError(
                        f"Final answer generation after budget rollback failed: "
                        f"{type(exc).__name__}: {exc}",
                        messages=finalization_messages,
                        usage=actual_usage,
                        accepted_retrieval_usage=accepted_retrieval_usage,
                        attempted_retrieval_usage=attempted_retrieval_usage,
                        budget_limited=True,
                        budget_event=budget_event,
                        model_calls=model_calls,
                        tool_calls=tool_call_count,
                    ) from exc
                model_calls += 1
                actual_usage = _add_usage(actual_usage, final_response.usage)
                finalization_usage = final_response.usage
                final_messages = [*finalization_messages, final_response.message]
                if final_response.message.tool_calls:
                    raise BudgetFinalizationError(
                        "Model requested tools during tool-free budget finalization.",
                        messages=final_messages,
                        usage=actual_usage,
                        accepted_retrieval_usage=accepted_retrieval_usage,
                        attempted_retrieval_usage=attempted_retrieval_usage,
                        finalization_usage=finalization_usage,
                        budget_limited=True,
                        budget_event=budget_event,
                        model_calls=model_calls,
                        tool_calls=tool_call_count,
                    )
                if _contains_serialized_tool_call(final_response.message.content):
                    raise BudgetFinalizationError(
                        "Model returned serialized tool calls instead of a final answer.",
                        messages=final_messages,
                        usage=actual_usage,
                        accepted_retrieval_usage=accepted_retrieval_usage,
                        attempted_retrieval_usage=attempted_retrieval_usage,
                        finalization_usage=finalization_usage,
                        budget_limited=True,
                        budget_event=budget_event,
                        model_calls=model_calls,
                        tool_calls=tool_call_count,
                    )
                if not final_response.message.content.strip():
                    raise BudgetFinalizationError(
                        "Model returned an empty answer after budget rollback.",
                        messages=final_messages,
                        usage=actual_usage,
                        accepted_retrieval_usage=accepted_retrieval_usage,
                        attempted_retrieval_usage=attempted_retrieval_usage,
                        finalization_usage=finalization_usage,
                        budget_limited=True,
                        budget_event=budget_event,
                        model_calls=model_calls,
                        tool_calls=tool_call_count,
                    )
                return AgentRunResult(
                    answer=final_response.message.content,
                    messages=tuple(final_messages),
                    usage=actual_usage,
                    model_calls=model_calls,
                    tool_calls=tool_call_count,
                    accepted_retrieval_usage=accepted_retrieval_usage,
                    attempted_retrieval_usage=attempted_retrieval_usage,
                    finalization_usage=finalization_usage,
                    budget_limited=True,
                    budget_event=budget_event,
                )

            accepted_retrieval_usage = _add_usage(
                accepted_retrieval_usage,
                response.usage,
            )
            messages.append(response.message)

            tool_rounds += 1
            if tool_rounds > self.max_tool_rounds:
                raise _build_loop_error(
                    f"Model exceeded {self.max_tool_rounds} tool-call rounds.",
                    messages,
                    actual_usage,
                    model_calls,
                    tool_call_count,
                    accepted_retrieval_usage=accepted_retrieval_usage,
                    attempted_retrieval_usage=attempted_retrieval_usage,
                )

            for call in calls:
                execution = self.registry.execute(call)
                messages.append(
                    ChatMessage.tool_result(
                        tool_call_id=call.id,
                        content=execution.content,
                        is_error=execution.is_error,
                    )
                )
                tool_call_count += 1


def _build_loop_error(
    message: str,
    messages: Sequence[ChatMessage],
    usage: Usage,
    model_calls: int,
    tool_calls: int,
    error_type: type[AgentLoopError] = AgentLoopError,
    accepted_retrieval_usage: Usage = Usage(),
    attempted_retrieval_usage: Usage = Usage(),
    finalization_usage: Usage = Usage(),
) -> AgentLoopError:
    """Create a loop error containing the accumulated run state.

    Arg:
        message: Human-readable failure reason.
        messages: Conversation accumulated before failure.
        usage: Accumulated actual token usage.
        model_calls: Number of completed model requests.
        tool_calls: Number of executed local tool calls.
        error_type: Agent loop exception class to construct.
        accepted_retrieval_usage: Retrieval usage retained in the checkpoint.
        attempted_retrieval_usage: Retrieval usage including discarded work.
        finalization_usage: Final answer generation usage.

    Returns:
        Agent loop error with partial trajectory metadata.
    """
    return error_type(
        message,
        messages=messages,
        usage=usage,
        accepted_retrieval_usage=accepted_retrieval_usage,
        attempted_retrieval_usage=attempted_retrieval_usage,
        finalization_usage=finalization_usage,
        model_calls=model_calls,
        tool_calls=tool_calls,
    )


def _add_usage(left: Usage, right: Usage) -> Usage:
    """Add two provider-neutral usage records.

    Arg:
        left: Existing accumulated usage.
        right: Usage from one completed model call.

    Returns:
        Component-wise usage total.
    """
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _contains_serialized_tool_call(content: str) -> bool:
    """Detect DeepSeek tool-call markup returned as ordinary text.

    Arg:
        content: Candidate final answer text.

    Returns:
        True when the text contains serialized DSML tool-call markup.
    """
    normalized = str(content or "").casefold()
    return "dsml" in normalized and "tool_calls" in normalized
