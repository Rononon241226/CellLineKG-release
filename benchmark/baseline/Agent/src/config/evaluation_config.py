"""Load question-type settings for raw-table evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


QUESTION_TYPES = ("single", "multi", "evidence_path")


class EvaluationConfigError(ValueError):
    """Report invalid question-type evaluation settings."""


@dataclass(frozen=True)
class ToolRoundLimits:
    """Store maximum tool-call rounds for each question type."""

    single: int
    multi: int
    evidence_path: int

    def for_question_type(self, question_type: str) -> int:
        """Return the configured limit for one question type.

        Arg:
            question_type: Supported question type.

        Returns:
            Positive maximum number of tool-call rounds.
        """
        if question_type not in QUESTION_TYPES:
            raise EvaluationConfigError(
                f"Unsupported question type: {question_type}"
            )
        return int(getattr(self, question_type))


@dataclass(frozen=True)
class RetrievalTokenLimits:
    """Store retrieval token budgets for each question type."""

    single: int
    multi: int
    evidence_path: int

    def for_question_type(self, question_type: str) -> int:
        """Return the configured token budget for one question type.

        Arg:
            question_type: Supported question type.

        Returns:
            Positive cumulative token budget.
        """
        if question_type not in QUESTION_TYPES:
            raise EvaluationConfigError(
                f"Unsupported question type: {question_type}"
            )
        return int(getattr(self, question_type))


def load_tool_round_limits(path: Path) -> ToolRoundLimits:
    """Load and validate question-type tool-round limits.

    Arg:
        path: YAML file containing a tool_round_limits mapping.

    Returns:
        Validated single, multi, and evidence-path limits.
    """
    payload = _load_evaluation_mapping(path)
    settings = payload.get("tool_round_limits")
    if not isinstance(settings, dict):
        raise EvaluationConfigError(
            "Evaluation configuration requires a 'tool_round_limits' mapping."
        )
    if set(settings) != set(QUESTION_TYPES):
        raise EvaluationConfigError(
            "tool_round_limits must contain exactly 'single', 'multi', "
            "and 'evidence_path'."
        )
    values: dict[str, int] = {}
    for question_type in QUESTION_TYPES:
        value = settings[question_type]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvaluationConfigError(
                f"Tool-round limit for '{question_type}' must be a positive integer."
            )
        values[question_type] = value
    return ToolRoundLimits(**values)


def load_python_timeout_seconds(path: Path) -> float:
    """Load and validate the per-call Python execution timeout.

    Arg:
        path: YAML file containing python_timeout_seconds.

    Returns:
        Positive timeout in seconds.
    """
    payload = _load_evaluation_mapping(path)
    value = payload.get("python_timeout_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise EvaluationConfigError(
            "python_timeout_seconds must be a positive number."
        )
    return float(value)


def load_retrieval_token_limits(path: Path) -> RetrievalTokenLimits | None:
    """Load optional retrieval token budgets by question type.

    Arg:
        path: YAML file that may contain a retrieval_token_limits mapping.

    Returns:
        Validated limits, or None when unrestricted behavior is configured.
    """
    payload = _load_evaluation_mapping(path)
    settings = payload.get("retrieval_token_limits")
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != set(QUESTION_TYPES):
        raise EvaluationConfigError(
            "retrieval_token_limits must contain exactly 'single', 'multi', "
            "and 'evidence_path'."
        )
    values: dict[str, int] = {}
    for question_type in QUESTION_TYPES:
        value = settings[question_type]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvaluationConfigError(
                f"Retrieval token limit for '{question_type}' must be a positive integer."
            )
        values[question_type] = value
    return RetrievalTokenLimits(**values)


def _load_evaluation_mapping(path: Path) -> dict[str, Any]:
    """Read the shared evaluation YAML mapping.

    Arg:
        path: Evaluation configuration path.

    Returns:
        Parsed top-level mapping.
    """
    if not path.is_file():
        raise EvaluationConfigError(f"Evaluation configuration not found: {path}")
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationConfigError(
            f"Invalid evaluation configuration YAML: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvaluationConfigError("Evaluation configuration must be a mapping.")
    return payload
