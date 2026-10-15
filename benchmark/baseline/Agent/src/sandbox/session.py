"""Manage one persistent restricted pandas worker per question."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from benchmark.baseline.Agent.src.sandbox.code_guard import (
    CodeValidationError,
    validate_code,
)
from benchmark.baseline.Agent.src.sandbox.worker import worker_main


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_LIMIT_MB = 3072
DEFAULT_MAX_OUTPUT_CHARACTERS = 30000


@dataclass
class ExecutionResult:
    """Store one restricted Python execution result."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    result: str = ""
    result_type: str = ""
    result_metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""
    traceback: str = ""
    execution_seconds: float = 0.0
    output_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the execution result to a serializable dictionary.

        Arg:
            None.

        Returns:
            Dictionary containing all execution result fields.
        """
        return asdict(self)


class PandasSession:
    """Own one persistent child process for a single question."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        max_output_characters: int = DEFAULT_MAX_OUTPUT_CHARACTERS,
    ):
        """Initialize a one-question pandas session.

        Arg:
            workspace_root: Project root containing the data directory.
            timeout_seconds: Maximum duration of one code execution.
            memory_limit_mb: Maximum worker resident memory in megabytes.
            max_output_characters: Maximum visible text returned per execution.

        Returns:
            None.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive.")
        if max_output_characters <= 0:
            raise ValueError("max_output_characters must be positive.")

        self.workspace_root = workspace_root.resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_mb = int(memory_limit_mb)
        self.max_output_characters = int(max_output_characters)
        self._context = mp.get_context("spawn")
        self._parent_connection: Any = None
        self._process: Any = None
        self._closed = False

    def start(self) -> None:
        """Start the child process and wait for its ready signal.

        Arg:
            None.

        Returns:
            None.
        """
        if self._closed:
            raise RuntimeError("This pandas session has already been closed.")
        if self.is_running:
            return

        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=worker_main,
            args=(child_connection, str(self.workspace_root)),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._parent_connection = parent_connection
        self._process = process

        startup_deadline = time.monotonic() + 30.0
        while not parent_connection.poll(0.05):
            if not process.is_alive():
                self._stop_worker()
                raise RuntimeError("Pandas worker exited during startup.")
            if time.monotonic() >= startup_deadline:
                self._stop_worker()
                raise RuntimeError("Pandas worker did not start within 30 seconds.")
        message = parent_connection.recv()
        if message.get("status") != "ready":
            self._stop_worker()
            raise RuntimeError(f"Unexpected pandas worker startup response: {message}")

    @property
    def is_running(self) -> bool:
        """Report whether the worker process is alive.

        Arg:
            None.

        Returns:
            True when the child process is running.
        """
        return bool(self._process is not None and self._process.is_alive())

    def execute(self, code: str) -> ExecutionResult:
        """Validate and execute one model-generated Python block.

        Arg:
            code: Python source to run in the persistent session.

        Returns:
            Structured execution result.
        """
        if self._closed:
            return ExecutionResult(
                ok=False,
                error_type="SessionTerminated",
                error_message="The pandas session has already been closed.",
            )

        try:
            validate_code(code)
        except CodeValidationError as exc:
            return ExecutionResult(
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        try:
            self.start()
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        self._parent_connection.send({"action": "execute", "code": code})
        payload = self._wait_for_execution()
        result = ExecutionResult(**payload)
        self._truncate_output(result)
        return result

    def close(self) -> None:
        """Close the child process and release session resources.

        Arg:
            None.

        Returns:
            None.
        """
        if self._process is None:
            self._closed = True
            return

        if self._process.is_alive() and self._parent_connection is not None:
            try:
                self._parent_connection.send({"action": "close"})
                self._process.join(timeout=2.0)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._parent_connection is not None:
            self._parent_connection.close()
        self._closed = True

    def __enter__(self) -> "PandasSession":
        """Enter the one-question session context.

        Arg:
            None.

        Returns:
            Active pandas session.
        """
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> None:
        """Exit the session context and close its worker.

        Arg:
            exc_type: Exception type raised in the context, if any.
            exc_value: Exception value raised in the context, if any.
            traceback_value: Exception traceback raised in the context, if any.

        Returns:
            None.
        """
        del exc_type, exc_value, traceback_value
        self.close()

    def _wait_for_execution(self) -> dict[str, Any]:
        """Wait for a worker result while enforcing resource limits.

        Arg:
            None.

        Returns:
            Serializable execution result payload.
        """
        started = time.monotonic()
        memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
        process_monitor = psutil.Process(self._process.pid)

        while True:
            if self._parent_connection.poll(0.05):
                return self._parent_connection.recv()

            if not self._process.is_alive():
                self._stop_worker()
                return {
                    "ok": False,
                    "error_type": "WorkerExited",
                    "error_message": (
                        "The pandas worker exited unexpectedly. The next run_python "
                        "call will start an empty session; reload required data."
                    ),
                }

            if time.monotonic() - started > self.timeout_seconds:
                self._stop_worker()
                return {
                    "ok": False,
                    "error_type": "TimeoutError",
                    "error_message": (
                        f"Python execution exceeded {self.timeout_seconds:g} seconds. "
                        "The session was reset; the next run_python call will start "
                        "an empty session, so reload required data."
                    ),
                }

            try:
                resident_memory = process_monitor.memory_info().rss
            except psutil.Error:
                continue
            if resident_memory > memory_limit_bytes:
                self._stop_worker()
                return {
                    "ok": False,
                    "error_type": "MemoryLimitError",
                    "error_message": (
                        f"Pandas worker exceeded {self.memory_limit_mb} MB of resident "
                        "memory. The session was reset; the next run_python call will "
                        "start an empty session, so reload required data."
                    ),
                }

    def _stop_worker(self) -> None:
        """Stop the current worker while allowing a fresh worker to start later.

        Arg:
            None.

        Returns:
            None.
        """
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._parent_connection is not None:
            self._parent_connection.close()
        self._parent_connection = None
        self._process = None

    def _truncate_output(self, result: ExecutionResult) -> None:
        """Limit visible output without changing worker-side variables.

        Arg:
            result: Execution result to modify for display.

        Returns:
            None.
        """
        remaining = self.max_output_characters
        for field_name in ("stdout", "stderr", "result", "traceback"):
            value = getattr(result, field_name)
            if not value:
                continue
            if remaining <= 0:
                setattr(result, field_name, "")
                result.output_truncated = True
                continue
            if len(value) <= remaining:
                remaining -= len(value)
                continue
            suffix = "\n...[output truncated]"
            if remaining >= len(suffix):
                visible = remaining - len(suffix)
                setattr(result, field_name, value[:visible] + suffix)
            else:
                setattr(result, field_name, value[:remaining])
            result.output_truncated = True
            remaining = 0
