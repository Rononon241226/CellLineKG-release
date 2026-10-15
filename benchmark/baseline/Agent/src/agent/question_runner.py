"""Create fresh raw-data tools and a pandas session for each question."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from benchmark.baseline.Agent.src.agent.loop import AgentLoop, AgentRunResult
from benchmark.baseline.Agent.src.agent.tool_registry import ToolRegistry
from benchmark.baseline.Agent.src.providers.base import LLMProvider
from benchmark.baseline.Agent.src.sandbox.session import (
    DEFAULT_MAX_OUTPUT_CHARACTERS,
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_TIMEOUT_SECONDS,
    PandasSession,
)
from benchmark.baseline.Agent.src.tools.pandas_tool import PandasTool
from benchmark.baseline.Agent.src.tools.skill_reader import (
    ListSkillsTool,
    ReadSkillTool,
    SkillReader,
)


SessionFactory = Callable[..., PandasSession]


class QuestionRunner:
    """Run each benchmark question with a fresh restricted data session."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace_root: Path,
        *,
        max_tool_rounds: int = 12,
        max_retrieval_tokens: int | None = None,
        python_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        python_memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        max_python_output_characters: int = DEFAULT_MAX_OUTPUT_CHARACTERS,
        session_factory: SessionFactory = PandasSession,
    ):
        """Initialize the per-question runner.

        Arg:
            provider: Configured model provider.
            workspace_root: Project root containing data and benchmark resources.
            max_tool_rounds: Maximum model turns that may request tools.
            max_retrieval_tokens: Optional cumulative retrieval budget.
            python_timeout_seconds: Maximum duration of one Python execution.
            python_memory_limit_mb: Maximum pandas worker memory in megabytes.
            max_python_output_characters: Maximum visible Python output per call.
            session_factory: Injectable pandas session constructor for tests.

        Returns:
            None.
        """
        self.provider = provider
        self.workspace_root = workspace_root.resolve()
        self.max_tool_rounds = max_tool_rounds
        self.max_retrieval_tokens = max_retrieval_tokens
        self.python_timeout_seconds = python_timeout_seconds
        self.python_memory_limit_mb = python_memory_limit_mb
        self.max_python_output_characters = max_python_output_characters
        self.session_factory = session_factory

    def run(
        self,
        question: str,
        system_prompt: str = "",
    ) -> AgentRunResult:
        """Run one question and always close its pandas session.

        Arg:
            question: Benchmark question shown to the model.
            system_prompt: Optional shared evaluation instructions.

        Returns:
            Final answer, conversation, usage, and call counts.
        """
        session = self.session_factory(
            self.workspace_root,
            timeout_seconds=self.python_timeout_seconds,
            memory_limit_mb=self.python_memory_limit_mb,
            max_output_characters=self.max_python_output_characters,
        )
        try:
            skill_reader = SkillReader(
                self.workspace_root
                / "benchmark"
                / "baseline"
                / "Agent"
                / "skills"
            )
            registry = ToolRegistry(
                [
                    ListSkillsTool(skill_reader),
                    ReadSkillTool(skill_reader),
                    PandasTool(session),
                ]
            )
            loop = AgentLoop(
                provider=self.provider,
                registry=registry,
                max_tool_rounds=self.max_tool_rounds,
                max_retrieval_tokens=self.max_retrieval_tokens,
            )
            return loop.run(question=question, system_prompt=system_prompt)
        finally:
            session.close()
