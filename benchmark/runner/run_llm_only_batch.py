"""Run an LLM-only baseline in batches and save usage metrics.

Input:
    An Excel or CSV file. QA150 requires number, category, sub_category,
    and question. QA multi and QA path require number and question.

Output:
    A results workbook and one complete JSON record per question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celllinekg.llm import (
    TrackedChatLLM,
    UsageTracker,
    build_chat_llm,
    get_chat_llm,
    invoke_with_tracking,
)
from benchmark.baseline.Agent.src.config import (
    EvaluationConfigError,
    ModelConfig,
    ModelConfigError,
    load_env_file,
    load_model_configs,
    load_retrieval_token_limits,
)


QUESTION_TYPES = ("single", "multi", "evidence_path")
DEFAULT_EVALUATION_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "runner" / "config" / "evaluation.yaml"
)
DEFAULT_MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "runner" / "config" / "models.yaml"
)
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
REQUIRED_COLUMNS = {
    "single": ("number", "category", "sub_category", "question"),
    "multi": ("number", "question"),
    "evidence_path": ("number", "question"),
}
RESULT_COLUMNS = (
    "answer",
    "status",
    "error_type",
    "error_message",
    "finish_reason",
    "duration_seconds",
    "llm_duration_seconds",
    "model_calls",
    "successful_model_calls",
    "failed_model_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "accepted_retrieval_tokens",
    "attempted_retrieval_tokens",
    "finalization_tokens",
    "budget_enforcement_version",
    "token_limit_parameter",
    "reasoning_effort",
    "usage_complete",
    "model_key",
    "provider",
    "model",
    "started_at",
    "finished_at",
)
LLM_ONLY_SYSTEM_PROMPT = (
    "Answer the biomedical cell-line question using only the model's internal "
    "knowledge and without external tools or retrieved data. "
    "Use one concise sentence or one concise paragraph."
)
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
LLM_ONLY_BUDGET_ENFORCEMENT_VERSION = "server_max_tokens_high_reasoning_v2"
LLM_ONLY_TOKEN_LIMIT_PARAMETER = "max_tokens"
LLM_ONLY_REASONING_EFFORT = "high"


class BatchInputError(ValueError):
    """Report an invalid LLM-only batch input."""


def build_parser() -> argparse.ArgumentParser:
    """Build the LLM-only batch command-line parser.

    Arg:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run LLM-only questions from an Excel or CSV file."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model key from models.yaml. When omitted, the legacy "
            "LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL settings are used."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--question-type", choices=QUESTION_TYPES, required=True)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument(
        "--run",
        choices=("run1", "run2", "run3"),
        default="run1",
        help="Repeat label used for single-question evaluation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="Path to models.yaml for optional model selection.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to a dotenv-style credential file.",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG_PATH,
        help="Path to the shared evaluation YAML configuration.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip matching questions with an existing successful record.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the input without calling the model.",
    )
    return parser


def load_questions(
    input_path: Path,
    question_type: str,
    sheet_name: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and validate the columns required for one question type.

    Arg:
        input_path: Excel or CSV question file.
        question_type: Single, multi, or evidence-path question type.
        sheet_name: Optional Excel worksheet name.
        limit: Optional maximum number of rows.

    Returns:
        Normalized question rows in input order.
    """
    if question_type not in REQUIRED_COLUMNS:
        raise BatchInputError(f"Unsupported question type: {question_type}")
    if not input_path.is_file():
        raise BatchInputError(f"Input file not found: {input_path}")
    if limit is not None and limit < 0:
        raise BatchInputError("limit must be non-negative.")

    required = REQUIRED_COLUMNS[question_type]
    header = _read_table(input_path, sheet_name, nrows=0)
    column_map = _resolve_columns(header.columns, required)
    frame = _read_table(
        input_path,
        sheet_name,
        usecols=list(column_map.values()),
        dtype=str,
    )
    frame = frame.rename(
        columns={actual: canonical for canonical, actual in column_map.items()}
    )
    frame = frame.loc[:, list(required)].copy()
    for column in required:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    empty_details = []
    for column in required:
        empty_rows = frame.index[frame[column] == ""].tolist()
        if empty_rows:
            excel_rows = ", ".join(str(index + 2) for index in empty_rows[:10])
            empty_details.append(f"{column} at row(s) {excel_rows}")
    if empty_details:
        raise BatchInputError(
            "Required input values are empty: " + "; ".join(empty_details)
        )

    duplicate_numbers = frame.loc[
        frame["number"].duplicated(keep=False), "number"
    ].unique()
    if len(duplicate_numbers):
        raise BatchInputError(
            "Duplicate number values: " + ", ".join(duplicate_numbers)
        )
    if limit is not None:
        frame = frame.head(limit)
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise BatchInputError("Input contains no questions to run.")
    return frame


def run_batch(
    *,
    questions: pd.DataFrame,
    question_type: str,
    output_directory: Path,
    resume: bool = False,
    llm_factory: Callable[[], Any] = get_chat_llm,
    retrieval_token_limit: int | None = None,
    model_key: str | None = None,
    provider: str = "",
) -> pd.DataFrame:
    """Run all questions through the direct LLM-only baseline.

    Arg:
        questions: Validated input questions.
        question_type: Single, multi, or evidence-path question type.
        output_directory: Directory for Excel and JSON results.
        resume: Whether to skip matching successful records.
        llm_factory: Model client factory, injectable for tests.
        retrieval_token_limit: Optional retrieval token budget per question.
        model_key: Optional configuration key used to isolate resumed runs.
        provider: Provider label stored for reproducibility.

    Returns:
        Flat result table in input order.
    """
    individual_directory = output_directory / "individual"
    individual_directory.mkdir(parents=True, exist_ok=True)
    records, record_paths = _load_records(individual_directory)

    total = len(questions)
    for position, row in questions.iterrows():
        number = row["number"]
        existing = records.get(number)
        if resume and _record_matches(
            existing,
            row,
            question_type,
            retrieval_token_limit,
            model_key=model_key,
        ):
            print(
                f"[{position + 1}/{total}] {number}: skipped (already successful)",
                flush=True,
            )
            continue

        print(f"[{position + 1}/{total}] {number}: {row['question']}", flush=True)
        answer = ""
        error: dict[str, str] | None = None
        status = "success"
        finish_reason = ""
        tracker = UsageTracker()
        try:
            base_llm = llm_factory()
            tracker.set_model(_model_identifier(base_llm))
            llm = TrackedChatLLM(base_llm, tracker)
            response = invoke_with_tracking(
                llm,
                "llm_only",
                [
                    ("system", LLM_ONLY_SYSTEM_PROMPT),
                    ("user", row["question"]),
                ],
            )
            answer = str(getattr(response, "content", response)).strip()
            finish_reason = _response_finish_reason(response)
        except Exception as exc:
            status = "error"
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

        metrics = tracker.finish()
        if status == "success" and not answer:
            status = "error"
            error = {
                "type": "EmptyAnswer",
                "message": "The model returned an empty final answer.",
            }

        payload = {
            "question_type": question_type,
            **row.to_dict(),
            "answer": answer,
            "status": status,
            "error": error,
            "metrics": metrics,
            "system_prompt": LLM_ONLY_SYSTEM_PROMPT,
            "retrieval_token_limit": retrieval_token_limit,
            "finish_reason": finish_reason,
            "budget_enforcement_version": (
                LLM_ONLY_BUDGET_ENFORCEMENT_VERSION
                if retrieval_token_limit is not None
                else ""
            ),
            "token_limit_parameter": (
                LLM_ONLY_TOKEN_LIMIT_PARAMETER
                if retrieval_token_limit is not None
                else ""
            ),
            "reasoning_effort": LLM_ONLY_REASONING_EFFORT,
            "model_key": model_key,
            "provider": provider,
        }
        record_path = _write_record(payload, individual_directory, number)
        records[number] = payload
        record_paths[number] = record_path
        output = build_results_table(questions, records, record_paths)
        write_results_workbook(output, output_directory / "results.xlsx")

    output = build_results_table(questions, records, record_paths)
    write_results_workbook(output, output_directory / "results.xlsx")
    return output


def build_results_table(
    questions: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    record_paths: dict[str, Path],
) -> pd.DataFrame:
    """Flatten LLM-only records into an Excel-ready table.

    Arg:
        questions: Validated input questions.
        records: Result payloads indexed by question number.
        record_paths: JSON paths indexed by question number.

    Returns:
        Table containing inputs, answers, and aggregate metrics.
    """
    rows = []
    for _, question_row in questions.iterrows():
        row = question_row.to_dict()
        number = row["number"]
        payload = records.get(number)
        if payload is None:
            row.update({column: "" for column in RESULT_COLUMNS})
            row["status"] = "pending"
            rows.append(row)
            continue

        error = payload.get("error") or {}
        metrics = payload.get("metrics") or {}
        row.update(
            {
                "answer": payload.get("answer", ""),
                "status": payload.get("status", ""),
                "error_type": error.get("type", ""),
                "error_message": error.get("message", ""),
                "finish_reason": payload.get("finish_reason", ""),
                "duration_seconds": metrics.get("duration_seconds", 0.0),
                "llm_duration_seconds": metrics.get("llm_duration_seconds", 0.0),
                "model_calls": metrics.get("model_calls", 0),
                "successful_model_calls": metrics.get("successful_model_calls", 0),
                "failed_model_calls": metrics.get("failed_model_calls", 0),
                "input_tokens": metrics.get("input_tokens", 0),
                "output_tokens": metrics.get("output_tokens", 0),
                "total_tokens": metrics.get("total_tokens", 0),
                "accepted_retrieval_tokens": 0,
                "attempted_retrieval_tokens": 0,
                "finalization_tokens": metrics.get("total_tokens", 0),
                "budget_enforcement_version": payload.get(
                    "budget_enforcement_version", ""
                ),
                "token_limit_parameter": payload.get(
                    "token_limit_parameter", ""
                ),
                "reasoning_effort": payload.get("reasoning_effort", ""),
                "usage_complete": metrics.get("usage_complete", False),
                "model_key": payload.get("model_key") or "",
                "provider": payload.get("provider", ""),
                "model": metrics.get("model", ""),
                "started_at": metrics.get("started_at", ""),
                "finished_at": metrics.get("finished_at", ""),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*questions.columns, *RESULT_COLUMNS])


def write_results_workbook(results: pd.DataFrame, output_path: Path) -> None:
    """Write the current result table to an atomic Excel workbook.

    Arg:
        results: Flat LLM-only result table.
        output_path: Final Excel workbook path.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.xlsx")
    results.to_excel(
        temporary_path,
        index=False,
        sheet_name="results",
        freeze_panes=(1, 0),
    )
    temporary_path.replace(output_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run or validate one LLM-only question batch.

    Arg:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero on complete success and one when any question fails.
    """
    args = build_parser().parse_args(argv)
    selected_config: ModelConfig | None = None
    selected_api_key = ""
    selected_base_url: str | None = None
    try:
        token_limits = load_retrieval_token_limits(args.evaluation_config)
        retrieval_token_limit = (
            token_limits.for_question_type(args.question_type)
            if token_limits is not None
            else None
        )
        questions = load_questions(
            args.input,
            args.question_type,
            sheet_name=args.sheet,
            limit=args.limit,
        )
        if args.model:
            load_env_file(args.env_file)
            model_configs = load_model_configs(args.config)
            if args.model not in model_configs:
                available = ", ".join(sorted(model_configs))
                raise BatchInputError(
                    f"Unknown model key '{args.model}'. Available: {available}"
                )
            selected_config = model_configs[args.model]
            selected_api_key = selected_config.resolve_api_key()
            selected_base_url = _resolve_chat_base_url(selected_config)
    except (
        OSError,
        BatchInputError,
        EvaluationConfigError,
        ModelConfigError,
        ValueError,
    ) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    output_directory = args.output_directory or default_output_directory(
        question_type=args.question_type,
        model_key=args.model,
        run_label=args.run,
    )

    if args.check:
        model_detail = (
            f"; model={selected_config.model}; output={output_directory}"
            if selected_config is not None
            else ""
        )
        print(
            f"Batch is ready: {len(questions)} question(s); "
            f"question_type={args.question_type}"
            f"; retrieval_token_limit={retrieval_token_limit or 'unrestricted'}"
            f"{model_detail}"
        )
        return 0

    llm_factory: Callable[[], Any] = get_chat_llm
    if retrieval_token_limit is not None:
        llm_factory = partial(
            get_chat_llm,
            max_tokens=retrieval_token_limit,
            preserve_max_tokens_parameter=True,
            reasoning_effort=LLM_ONLY_REASONING_EFFORT,
        )
    provider = ""
    if selected_config is not None:
        is_openrouter = (
            selected_base_url is not None
            and "openrouter.ai" in selected_base_url.casefold()
        )
        effective_max_tokens = (
            retrieval_token_limit
            if retrieval_token_limit is not None
            else selected_config.max_tokens
        )
        llm_factory = partial(
            build_chat_llm,
            api_key=selected_api_key,
            model=selected_config.model,
            base_url=selected_base_url,
            temperature=(
                None
                if retrieval_token_limit is not None and is_openrouter
                else selected_config.temperature
            ),
            max_tokens=effective_max_tokens,
            timeout_seconds=selected_config.request_timeout_seconds,
            preserve_max_tokens_parameter=retrieval_token_limit is not None,
            require_supported_parameters=(
                retrieval_token_limit is not None and is_openrouter
            ),
            reasoning_effort=LLM_ONLY_REASONING_EFFORT,
        )
        provider = _provider_label(selected_config, selected_base_url)
    try:
        results = run_batch(
            questions=questions,
            question_type=args.question_type,
            output_directory=output_directory,
            resume=args.resume,
            llm_factory=llm_factory,
            retrieval_token_limit=retrieval_token_limit,
            model_key=args.model,
            provider=provider,
        )
    except (OSError, ImportError, ValueError) as exc:
        print(f"Could not write batch output: {exc}", file=sys.stderr)
        return 1

    failed = int((results["status"] != "success").sum())
    print(f"Saved LLM-only results: {output_directory / 'results.xlsx'}")
    if failed:
        print(f"Batch completed with {failed} unsuccessful question(s).", file=sys.stderr)
        return 1
    return 0


def _read_table(
    input_path: Path,
    sheet_name: str | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel or CSV input table.

    Arg:
        input_path: Excel or CSV question file.
        sheet_name: Optional Excel worksheet name.
        **kwargs: Reader arguments forwarded to pandas.

    Returns:
        Loaded input table.
    """
    suffix = input_path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(input_path, **kwargs)
    if suffix == ".xlsx":
        selected_sheet: str | int = sheet_name if sheet_name is not None else 0
        return pd.read_excel(input_path, sheet_name=selected_sheet, **kwargs)
    raise BatchInputError("Input must be an .xlsx or .csv file.")


def _resolve_columns(
    actual_columns: Sequence[Any],
    required_columns: Sequence[str],
) -> dict[str, Any]:
    """Resolve required column names case-insensitively.

    Arg:
        actual_columns: Column labels found in the input file.
        required_columns: Canonical required column names.

    Returns:
        Mapping from canonical names to actual labels.
    """
    normalized: dict[str, list[Any]] = {}
    for column in actual_columns:
        normalized.setdefault(str(column).strip().casefold(), []).append(column)
    ambiguous = [key for key, values in normalized.items() if len(values) > 1]
    if ambiguous:
        raise BatchInputError(
            "Ambiguous duplicate column names: " + ", ".join(ambiguous)
        )
    missing = [column for column in required_columns if column not in normalized]
    if missing:
        raise BatchInputError("Missing required columns: " + ", ".join(missing))
    return {column: normalized[column][0] for column in required_columns}


def _model_identifier(llm: Any) -> str:
    """Read a stable model identifier from a chat client.

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


def _response_finish_reason(response: Any) -> str:
    """Read the provider finish reason from a model response.

    Arg:
        response: LangChain-style model response.

    Returns:
        Normalized finish reason or an empty string.
    """
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    for key in ("finish_reason", "stop_reason", "native_finish_reason"):
        value = metadata.get(key)
        if value:
            return str(value).strip().casefold()
    return ""


def _record_path(individual_directory: Path, number: str) -> Path:
    """Build a safe stable JSON path for one question number.

    Arg:
        individual_directory: Directory containing question records.
        number: Question identifier.

    Returns:
        Safe JSON file path.
    """
    safe_number = SAFE_NAME_PATTERN.sub("_", str(number)).strip("._") or "question"
    return individual_directory / f"{safe_number}.json"


def _write_record(
    payload: dict[str, Any],
    individual_directory: Path,
    number: str,
) -> Path:
    """Atomically write one complete question record.

    Arg:
        payload: Complete question result payload.
        individual_directory: Directory containing JSON records.
        number: Question identifier.

    Returns:
        Final JSON record path.
    """
    final_path = _record_path(individual_directory, number)
    temporary_path = final_path.with_suffix(".tmp.json")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(final_path)
    return final_path


def _load_records(
    individual_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Load existing JSON records for resume and workbook rebuilding.

    Arg:
        individual_directory: Directory containing JSON records.

    Returns:
        Payload and file-path dictionaries indexed by question number.
    """
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for path in individual_directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        number = str(payload.get("number", "")).strip()
        if number:
            records[number] = payload
            paths[number] = path
    return records, paths


def _record_matches(
    payload: dict[str, Any] | None,
    row: pd.Series,
    question_type: str,
    retrieval_token_limit: int | None = None,
    model_key: str | None = None,
) -> bool:
    """Check whether an existing successful record matches the input row.

    Arg:
        payload: Existing question record, if present.
        row: Current normalized question row.
        question_type: Current question type.
        retrieval_token_limit: Retrieval budget required for the resumed run.
        model_key: Optional selected model key required for the resumed run.

    Returns:
        True when the existing successful record can be reused.
    """
    if not payload or payload.get("status") != "success":
        return False
    if payload.get("question_type") != question_type:
        return False
    if payload.get("retrieval_token_limit") != retrieval_token_limit:
        return False
    if (
        retrieval_token_limit is not None
        and payload.get("budget_enforcement_version")
        != LLM_ONLY_BUDGET_ENFORCEMENT_VERSION
    ):
        return False
    if payload.get("reasoning_effort") != LLM_ONLY_REASONING_EFFORT:
        return False
    if model_key is not None and payload.get("model_key") != model_key:
        return False
    return all(
        str(payload.get(column, "")).strip() == str(row[column]).strip()
        for column in REQUIRED_COLUMNS[question_type]
    )


def default_output_directory(
    *,
    question_type: str,
    model_key: str | None,
    run_label: str,
) -> Path:
    """Build the default LLM-only benchmark output directory.

    Arg:
        question_type: Single, multi, or evidence-path question type.
        model_key: Optional selected model configuration key.
        run_label: Repeat label used for single-question evaluation.

    Returns:
        Default directory for one LLM-only batch.
    """
    safe_model = SAFE_NAME_PATTERN.sub("_", model_key or "default").strip("._")
    task_root = PROJECT_ROOT / "results" / "benchmark" / "runs" / question_type
    if question_type == "single":
        task_root = task_root / run_label
    return task_root / safe_model / "llm_only"


def _resolve_chat_base_url(config: ModelConfig) -> str | None:
    """Resolve an endpoint supported by the ChatOpenAI client.

    Arg:
        config: Selected model configuration.

    Returns:
        Explicit compatible base URL, or None for the native OpenAI endpoint.
    """
    if config.base_url:
        return config.base_url
    normalized_provider = config.provider.strip().casefold()
    if normalized_provider in {"openai", "gpt"}:
        return None
    if normalized_provider == "deepseek":
        return "https://api.deepseek.com"
    if normalized_provider in {"kimi", "moonshot"}:
        return "https://api.moonshot.cn/v1"
    raise BatchInputError(
        f"Model '{config.key}' requires an OpenAI-compatible base_url for "
        "LLM-only evaluation."
    )


def _provider_label(config: ModelConfig, base_url: str | None) -> str:
    """Create a reproducible provider label for result records.

    Arg:
        config: Selected model configuration.
        base_url: Resolved API base URL.

    Returns:
        OpenRouter for its endpoint, otherwise the configured provider name.
    """
    if base_url and "openrouter.ai" in base_url.casefold():
        return "openrouter"
    return config.provider


if __name__ == "__main__":
    raise SystemExit(main())
