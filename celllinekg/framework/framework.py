"""CellLineKG framework controller."""

from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace
from typing import Any, Sequence

from celllinekg.framework.tools import (
    run_evidence_path_retrieval_tool,
    run_text2cypher_tool,
)
from celllinekg.framework.workflow_selector import select_workflow
from celllinekg.llm import TrackedChatLLM, UsageTracker, get_chat_llm


ROUTE_DISPLAY_NAMES = {
    "evidence_path_retrieval": "Evidence-path Retrieval",
    "text2cypher": "Text2Cypher",
}
FINALIZATION_PHASES = ("answer_synthesis", "evidence_path_summary")


class RetrievalTokenBudgetExceeded(RuntimeError):
    """Report that a retrieval response crossed the configured token budget."""


class _BudgetedTrackedChatLLM(TrackedChatLLM):
    """Track calls and discard retrieval responses that cross the token budget."""

    def __init__(
        self,
        llm: Any,
        tracker: UsageTracker,
        retrieval_token_limit: int,
    ) -> None:
        """
        Initialize the budget-aware tracked model.

        Arg:
            llm: Underlying LangChain-style chat model.
            tracker: Question-level usage tracker.
            retrieval_token_limit: Maximum accepted retrieval tokens.

        Returns:
            None.
        """
        if retrieval_token_limit <= 0:
            raise ValueError("retrieval_token_limit must be positive.")
        super().__init__(llm, tracker)
        self.retrieval_token_limit = retrieval_token_limit
        self.accepted_retrieval_tokens = 0
        self.attempted_retrieval_tokens = 0
        self.finalization_tokens = 0
        self.budget_limited = False
        self.rollback_used = False
        self._accepted_responses: dict[str, Any] = {}

    def invoke_for_phase(
        self,
        phase: str,
        messages: Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke one model phase and enforce the retrieval token budget.

        Arg:
            phase: Pipeline stage responsible for the call.
            messages: Model input messages.
            *args: Additional positional model arguments.
            **kwargs: Additional keyword model arguments.

        Returns:
            Accepted response or a safe checkpoint response after rollback.
        """
        if self.budget_limited and phase not in FINALIZATION_PHASES:
            raise RetrievalTokenBudgetExceeded(
                "No further retrieval calls are allowed after budget rollback."
            )

        started = time.perf_counter()
        try:
            response = self._llm.invoke(messages, *args, **kwargs)
        except Exception as exc:
            self.tracker.record_error(
                phase=phase,
                duration_seconds=time.perf_counter() - started,
                error=exc,
            )
            raise

        self.tracker.record_success(
            phase=phase,
            duration_seconds=time.perf_counter() - started,
            response=response,
        )
        call_tokens = _response_total_tokens(response)

        if phase in FINALIZATION_PHASES:
            self.finalization_tokens += call_tokens
            return response

        self.attempted_retrieval_tokens += call_tokens
        if self.attempted_retrieval_tokens > self.retrieval_token_limit:
            self.budget_limited = True
            checkpoint_response = self._checkpoint_response(phase, messages)
            if checkpoint_response is None:
                raise RetrievalTokenBudgetExceeded(
                    f"The {phase} call crossed the retrieval token budget before "
                    "an answerable checkpoint was available."
                )
            self.rollback_used = True
            return checkpoint_response

        self.accepted_retrieval_tokens += call_tokens
        self._accepted_responses[phase] = response
        return response

    def budget_metrics(self) -> dict[str, int | bool]:
        """
        Return token-budget metrics for the completed question.

        Arg:
            None.

        Returns:
            Accepted, attempted, finalization, and budget state metrics.
        """
        return {
            "accepted_retrieval_tokens": self.accepted_retrieval_tokens,
            "attempted_retrieval_tokens": self.attempted_retrieval_tokens,
            "finalization_tokens": self.finalization_tokens,
            "budget_limited": self.budget_limited,
        }

    def _checkpoint_response(
        self,
        phase: str,
        messages: Sequence[Any],
    ) -> Any | None:
        """
        Build a safe response from the last accepted workflow checkpoint.

        Arg:
            phase: Retrieval phase that crossed the token budget.
            messages: Input messages for the discarded call.

        Returns:
            Prior or synthetic response, or None without a safe checkpoint.
        """
        if phase == "cypher_verification":
            return self._accepted_responses.get("cypher_generation")
        if phase == "path_filtering":
            indexes = _path_indexes(messages)
            if indexes:
                return SimpleNamespace(
                    content=json.dumps(indexes),
                    usage_metadata=None,
                    response_metadata={"finish_reason": "budget_rollback"},
                )
        return None


class CellLineKGFramework:
    """Route a user question to the selected CellLineKG workflow."""

    def ask(self, question: str) -> str:
        """
        Answer a CellLineKG user question.

        Arg:
            question: User question.

        Returns:
            The final answer string from the selected workflow.
        """
        result = self.ask_with_route(question)
        return result["answer"]

    def ask_with_route(self, question: str) -> dict[str, str]:
        """
        Answer a CellLineKG user question and keep the selected route.

        Arg:
            question: User question.

        Returns:
            A dictionary with route and answer fields.
        """
        llm = get_chat_llm()
        workflow = select_workflow(llm=llm, question=question)
        route = ROUTE_DISPLAY_NAMES.get(workflow, workflow)
        if workflow == "evidence_path_retrieval":
            answer = run_evidence_path_retrieval_tool(llm=llm, question=question)
        else:
            answer = run_text2cypher_tool(question=question)
        return {
            "route": route,
            "answer": answer,
        }

    def ask_with_metrics(
        self,
        question: str,
        retrieval_token_limit: int | None = None,
    ) -> dict[str, Any]:
        """Answer one question and return aggregate usage and timing metrics.

        Arg:
            question: User question.
            retrieval_token_limit: Optional retrieval token budget for the question.

        Returns:
            Route, answer, status, optional error, and question-level metrics.
        """
        tracker = UsageTracker(
            retrieval_token_limit=retrieval_token_limit,
            finalization_phases=FINALIZATION_PHASES,
        )
        route = ""
        answer = ""
        llm: TrackedChatLLM | None = None
        try:
            base_llm = get_chat_llm()
            tracker.set_model(_model_identifier(base_llm))
            if retrieval_token_limit is None:
                llm = TrackedChatLLM(base_llm, tracker)
            else:
                llm = _BudgetedTrackedChatLLM(
                    base_llm,
                    tracker,
                    retrieval_token_limit,
                )
            workflow = select_workflow(llm=llm, question=question)
            route = ROUTE_DISPLAY_NAMES.get(workflow, workflow)
            if workflow == "evidence_path_retrieval":
                answer = run_evidence_path_retrieval_tool(llm=llm, question=question)
            else:
                answer = run_text2cypher_tool(question=question, llm=llm)
            metrics = tracker.finish()
            if isinstance(llm, _BudgetedTrackedChatLLM):
                metrics.update(llm.budget_metrics())
            return {
                "route": route,
                "answer": answer,
                "status": "success",
                "error": None,
                "budget_limited": bool(
                    isinstance(llm, _BudgetedTrackedChatLLM)
                    and llm.budget_limited
                ),
                "metrics": metrics,
            }
        except Exception as exc:
            metrics = tracker.finish()
            if isinstance(llm, _BudgetedTrackedChatLLM):
                metrics.update(llm.budget_metrics())
            return {
                "route": route,
                "answer": answer,
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "budget_limited": bool(
                    isinstance(llm, _BudgetedTrackedChatLLM)
                    and llm.budget_limited
                ),
                "metrics": metrics,
            }


def _model_identifier(llm: Any) -> str:
    """Read a stable model identifier from a LangChain-style client.

    Arg:
        llm: Configured chat model client.

    Returns:
        Model identifier or an empty string when unavailable.
    """
    for attribute in ("model_name", "model"):
        value = getattr(llm, attribute, "")
        if value:
            return str(value)
    return ""


def _response_total_tokens(response: Any) -> int:
    """
    Read total token usage from a LangChain-style response.

    Arg:
        response: Model response containing token usage metadata.

    Returns:
        Non-negative total token count or zero when unavailable.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict) or not usage:
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)) and not isinstance(total, bool) and total >= 0:
        return int(total)
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    if not isinstance(input_tokens, (int, float)) or isinstance(input_tokens, bool):
        input_tokens = 0
    if not isinstance(output_tokens, (int, float)) or isinstance(output_tokens, bool):
        output_tokens = 0
    return max(0, int(input_tokens)) + max(0, int(output_tokens))


def _path_indexes(messages: Sequence[Any]) -> list[int]:
    """
    Read raw path indexes from a path-filtering user message.

    Arg:
        messages: Path-filtering prompt messages.

    Returns:
        Ordered path indexes used as the rollback checkpoint.
    """
    user_content = ""
    for message in reversed(messages):
        if isinstance(message, (tuple, list)) and len(message) >= 2:
            if str(message[0]).strip().casefold() == "user":
                user_content = str(message[1])
                break
        elif str(getattr(message, "type", "")).casefold() in {"human", "user"}:
            user_content = str(getattr(message, "content", ""))
            break
    if "Paths:\n" in user_content:
        user_content = user_content.split("Paths:\n", 1)[1]
    return [
        int(match.group(1))
        for match in re.finditer(r"(?m)^\s*(\d+)\.\s", user_content)
    ]
