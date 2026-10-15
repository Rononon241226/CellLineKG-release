"""Write reproducible single-question results without storing API keys."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmark.baseline.Agent.src.agent import AgentLoopError, AgentRunResult
from benchmark.baseline.Agent.src.config import ModelConfig
from benchmark.baseline.Agent.src.providers import ChatMessage


SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class RunRecord:
    """Store one successful or failed real-model run."""

    run_id: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    model_key: str
    provider: str
    model: str
    question: str
    system_prompt: str
    answer: str
    usage: dict[str, int]
    model_calls: int
    tool_calls: int
    messages: list[dict[str, Any]]
    question_type: str = ""
    max_tool_rounds: int = 0
    python_timeout_seconds: float = 0.0
    retrieval_token_limit: int | None = None
    accepted_retrieval_usage: dict[str, int] | None = None
    attempted_retrieval_usage: dict[str, int] | None = None
    finalization_usage: dict[str, int] | None = None
    budget_limited: bool = False
    budget_event: Optional[dict[str, Any]] = None
    error: Optional[dict[str, str]] = None


def create_success_record(
    config: ModelConfig,
    question: str,
    system_prompt: str,
    result: AgentRunResult,
    started_at: datetime,
    finished_at: datetime,
    question_type: str = "",
    max_tool_rounds: int = 0,
    python_timeout_seconds: float = 0.0,
    retrieval_token_limit: int | None = None,
) -> RunRecord:
    """Create a successful run record.

    Arg:
        config: Non-secret model configuration.
        question: Question sent to the model.
        system_prompt: Evaluation instructions sent to the model.
        result: Completed agent run.
        started_at: UTC run start time.
        finished_at: UTC run finish time.
        question_type: Single, multi, or evidence-path evaluation category.
        max_tool_rounds: Tool-call round limit applied to the run.
        python_timeout_seconds: Per-call Python execution timeout.
        retrieval_token_limit: Optional retrieval token budget.

    Returns:
        Serializable successful run record.
    """
    return RunRecord(
        run_id=_new_run_id(started_at),
        status="success",
        started_at=_isoformat(started_at),
        finished_at=_isoformat(finished_at),
        duration_seconds=(finished_at - started_at).total_seconds(),
        model_key=config.key,
        provider=config.provider,
        model=config.model,
        question=question,
        system_prompt=system_prompt,
        answer=result.answer,
        usage=asdict(result.usage),
        model_calls=result.model_calls,
        tool_calls=result.tool_calls,
        messages=[_serialize_message(message) for message in result.messages],
        question_type=question_type,
        max_tool_rounds=max_tool_rounds,
        python_timeout_seconds=python_timeout_seconds,
        retrieval_token_limit=retrieval_token_limit,
        accepted_retrieval_usage=asdict(result.accepted_retrieval_usage),
        attempted_retrieval_usage=asdict(result.attempted_retrieval_usage),
        finalization_usage=asdict(result.finalization_usage),
        budget_limited=result.budget_limited,
        budget_event=result.budget_event,
    )


def create_error_record(
    config: ModelConfig,
    question: str,
    system_prompt: str,
    error: Exception,
    started_at: datetime,
    finished_at: datetime,
    question_type: str = "",
    max_tool_rounds: int = 0,
    python_timeout_seconds: float = 0.0,
    retrieval_token_limit: int | None = None,
) -> RunRecord:
    """Create a failed run record without exposing credentials.

    Arg:
        config: Non-secret model configuration.
        question: Question intended for the model.
        system_prompt: Evaluation instructions intended for the model.
        error: Exception raised during the run.
        started_at: UTC run start time.
        finished_at: UTC run finish time.
        question_type: Single, multi, or evidence-path evaluation category.
        max_tool_rounds: Tool-call round limit applied to the run.
        python_timeout_seconds: Per-call Python execution timeout.
        retrieval_token_limit: Optional retrieval token budget.

    Returns:
        Serializable failed run record.
    """
    if isinstance(error, AgentLoopError):
        usage = asdict(error.usage)
        accepted_retrieval_usage = asdict(error.accepted_retrieval_usage)
        attempted_retrieval_usage = asdict(error.attempted_retrieval_usage)
        finalization_usage = asdict(error.finalization_usage)
        budget_limited = error.budget_limited
        budget_event = error.budget_event
        model_calls = error.model_calls
        tool_calls = error.tool_calls
        messages = [_serialize_message(message) for message in error.messages]
    else:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        accepted_retrieval_usage = dict(usage)
        attempted_retrieval_usage = dict(usage)
        finalization_usage = dict(usage)
        budget_limited = False
        budget_event = None
        model_calls = 0
        tool_calls = 0
        messages = []
    return RunRecord(
        run_id=_new_run_id(started_at),
        status="error",
        started_at=_isoformat(started_at),
        finished_at=_isoformat(finished_at),
        duration_seconds=(finished_at - started_at).total_seconds(),
        model_key=config.key,
        provider=config.provider,
        model=config.model,
        question=question,
        system_prompt=system_prompt,
        answer="",
        usage=usage,
        model_calls=model_calls,
        tool_calls=tool_calls,
        messages=messages,
        question_type=question_type,
        max_tool_rounds=max_tool_rounds,
        python_timeout_seconds=python_timeout_seconds,
        retrieval_token_limit=retrieval_token_limit,
        accepted_retrieval_usage=accepted_retrieval_usage,
        attempted_retrieval_usage=attempted_retrieval_usage,
        finalization_usage=finalization_usage,
        budget_limited=budget_limited,
        budget_event=budget_event,
        error={
            "type": type(error).__name__,
            "message": str(error),
        },
    )


def write_run_record(record: RunRecord, output_directory: Path) -> Path:
    """Write one run record to a unique JSON file.

    Arg:
        record: Successful or failed run record.
        output_directory: Directory for single-question JSON outputs.

    Returns:
        Path to the written JSON file.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    safe_model_key = SAFE_NAME_PATTERN.sub("_", record.model_key)
    output_path = output_directory / f"{record.run_id}_{safe_model_key}.json"
    output_path.write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time.

    Arg:
        None.

    Returns:
        Current UTC datetime.
    """
    return datetime.now(timezone.utc)


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    """Convert one conversation message to JSON-compatible data.

    Arg:
        message: Provider-neutral conversation message.

    Returns:
        Message role, content, tool calls, and tool-result metadata.
    """
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "is_error": message.is_error,
    }


def _new_run_id(started_at: datetime) -> str:
    """Create a sortable unique run identifier.

    Arg:
        started_at: UTC run start time.

    Returns:
        Timestamp and random suffix.
    """
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _isoformat(value: datetime) -> str:
    """Format one datetime as UTC ISO 8601 text.

    Arg:
        value: Timezone-aware datetime.

    Returns:
        UTC timestamp ending in Z.
    """
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
