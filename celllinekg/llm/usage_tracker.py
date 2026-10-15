"""Track per-question LLM token usage and elapsed time."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


class TokenBudgetExceeded(RuntimeError):
    """Report that a further model call is blocked by the token budget."""


@dataclass(frozen=True)
class LLMCallMetrics:
    """Store usage and timing for one attempted LLM call."""

    phase: str
    status: str
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_available: bool
    error_type: str = ""
    error_message: str = ""


class UsageTracker:
    """Accumulate LLM calls and full wall-clock time for one question."""

    def __init__(
        self,
        model: str = "",
        retrieval_token_limit: int | None = None,
        finalization_phases: Sequence[str] = (),
    ):
        """Initialize one question-level tracker.

        Arg:
            model: Configured model identifier when available.
            retrieval_token_limit: Optional retrieval token budget for the question.
            finalization_phases: Phase names excluded from the retrieval budget.

        Returns:
            None.
        """
        if retrieval_token_limit is not None and retrieval_token_limit <= 0:
            raise ValueError(
                "retrieval_token_limit must be positive when provided."
            )
        self.model = model
        self.retrieval_token_limit = retrieval_token_limit
        self.finalization_phases = frozenset(finalization_phases)
        self.started_at = datetime.now(timezone.utc)
        self._started_counter = time.perf_counter()
        self._finished_at: datetime | None = None
        self._duration_seconds: float | None = None
        self._calls: list[LLMCallMetrics] = []

    def ensure_next_call_allowed(self, phase: str) -> None:
        """Reject another retrieval call after completed usage exceeds the budget.

        Arg:
            phase: Pipeline stage responsible for the next call.

        Returns:
            None.
        """
        if phase in self.finalization_phases:
            return
        retrieval_tokens = sum(
            call.total_tokens
            for call in self._calls
            if call.phase not in self.finalization_phases
        )
        if (
            self.retrieval_token_limit is not None
            and retrieval_tokens > self.retrieval_token_limit
        ):
            raise TokenBudgetExceeded(
                f"Retrieval token usage exceeded the "
                f"{self.retrieval_token_limit} token budget before another "
                "retrieval call."
            )

    def set_model(self, model: str) -> None:
        """Set the model identifier after client construction.

        Arg:
            model: Configured model identifier.

        Returns:
            None.
        """
        self.model = str(model or "")

    def record_success(
        self,
        phase: str,
        duration_seconds: float,
        response: Any,
    ) -> None:
        """Record one successful LLM response without affecting the answer.

        Arg:
            phase: Pipeline stage responsible for the call.
            duration_seconds: Monotonic elapsed call time.
            response: LangChain-style model response.

        Returns:
            None.
        """
        try:
            usage, available = _extract_usage(response)
        except Exception:
            usage = _empty_usage()
            available = False
        self._calls.append(
            LLMCallMetrics(
                phase=phase,
                status="success",
                duration_seconds=float(duration_seconds),
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                usage_available=available,
            )
        )

    def record_error(
        self,
        phase: str,
        duration_seconds: float,
        error: Exception,
    ) -> None:
        """Record one failed LLM attempt with zero reported usage.

        Arg:
            phase: Pipeline stage responsible for the call.
            duration_seconds: Monotonic elapsed call time.
            error: Exception raised by the model client.

        Returns:
            None.
        """
        self._calls.append(
            LLMCallMetrics(
                phase=phase,
                status="error",
                duration_seconds=float(duration_seconds),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                usage_available=False,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )

    def finish(self) -> dict[str, Any]:
        """Finalize and serialize the current question metrics.

        Arg:
            None.

        Returns:
            Aggregate metrics and one record per attempted LLM call.
        """
        if self._finished_at is None:
            self._finished_at = datetime.now(timezone.utc)
            self._duration_seconds = time.perf_counter() - self._started_counter
        successful_calls = [call for call in self._calls if call.status == "success"]
        return {
            "model": self.model,
            "started_at": _isoformat(self.started_at),
            "finished_at": _isoformat(self._finished_at),
            "duration_seconds": float(self._duration_seconds or 0.0),
            "llm_duration_seconds": sum(
                call.duration_seconds for call in self._calls
            ),
            "input_tokens": sum(call.input_tokens for call in self._calls),
            "output_tokens": sum(call.output_tokens for call in self._calls),
            "total_tokens": sum(call.total_tokens for call in self._calls),
            "retrieval_token_limit": self.retrieval_token_limit,
            "model_calls": len(self._calls),
            "successful_model_calls": len(successful_calls),
            "failed_model_calls": len(self._calls) - len(successful_calls),
            "usage_complete": bool(successful_calls)
            and len(successful_calls) == len(self._calls)
            and all(call.usage_available for call in successful_calls),
            "calls": [asdict(call) for call in self._calls],
        }


class TrackedChatLLM:
    """Proxy synchronous LLM calls while preserving the wrapped client API."""

    def __init__(self, llm: Any, tracker: UsageTracker):
        """Initialize a tracked client proxy.

        Arg:
            llm: Underlying LangChain-style chat model.
            tracker: Question-level usage tracker.

        Returns:
            None.
        """
        self._llm = llm
        self.tracker = tracker

    def invoke_for_phase(
        self,
        phase: str,
        messages: Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke the wrapped model and record one pipeline phase.

        Arg:
            phase: Pipeline stage responsible for the call.
            messages: Model input messages.
            *args: Additional positional arguments for the wrapped client.
            **kwargs: Additional keyword arguments for the wrapped client.

        Returns:
            Unchanged model response.
        """
        self.tracker.ensure_next_call_allowed(phase)
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
        return response

    def invoke(
        self,
        messages: Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke an unlabelled call using a generic phase name.

        Arg:
            messages: Model input messages.
            *args: Additional positional arguments for the wrapped client.
            **kwargs: Additional keyword arguments for the wrapped client.

        Returns:
            Unchanged model response.
        """
        return self.invoke_for_phase("llm", messages, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes not implemented by the tracking proxy.

        Arg:
            name: Requested attribute name.

        Returns:
            Attribute from the wrapped model client.
        """
        return getattr(self._llm, name)


def invoke_with_tracking(
    llm: Any,
    phase: str,
    messages: Sequence[Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke either a tracked proxy or an unchanged model client.

    Arg:
        llm: Tracked or ordinary LangChain-style model client.
        phase: Pipeline stage responsible for the call.
        messages: Model input messages.
        *args: Additional positional arguments for the model client.
        **kwargs: Additional keyword arguments for the model client.

    Returns:
        Unchanged model response.
    """
    if isinstance(llm, TrackedChatLLM):
        return llm.invoke_for_phase(phase, messages, *args, **kwargs)
    return llm.invoke(messages, *args, **kwargs)


def _extract_usage(response: Any) -> tuple[dict[str, int], bool]:
    """Read token usage from current and legacy LangChain response fields.

    Arg:
        response: LangChain-style model response.

    Returns:
        Normalized token counts and whether usage metadata was available.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict) or not usage:
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, dict) or not usage:
        return _empty_usage(), False

    input_tokens = _token_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _token_value(
        usage,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _token_value(usage, "total_tokens")
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }, True


def _token_value(usage: dict[str, Any], *names: str) -> int:
    """Return the first valid non-negative token value.

    Arg:
        usage: Provider usage mapping.
        *names: Candidate usage field names.

    Returns:
        Non-negative integer token count or zero.
    """
    for name in names:
        value = usage.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return 0


def _empty_usage() -> dict[str, int]:
    """Create a zero-valued usage mapping.

    Arg:
        None.

    Returns:
        Zero input, output, and total token counts.
    """
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _isoformat(value: datetime) -> str:
    """Format one timezone-aware datetime as UTC ISO 8601 text.

    Arg:
        value: Timezone-aware datetime.

    Returns:
        UTC timestamp ending in Z.
    """
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
