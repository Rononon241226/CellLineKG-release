"""Run a workbook or CSV batch through the raw-table question agent.

Input:
    A single-question file with number, category, sub_category, and question
    columns, or a multi/evidence-path file with number and question columns.

Output:
    One batch results workbook and one complete JSON record per question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.baseline.Agent.src.agent import QuestionRunner
from benchmark.baseline.Agent.src.config import (
    EvaluationConfigError,
    ModelConfigError,
    QUESTION_TYPES,
    load_env_file,
    load_model_configs,
    load_python_timeout_seconds,
    load_retrieval_token_limits,
    load_tool_round_limits,
)
from benchmark.baseline.Agent.src.evaluation.result_writer import (
    create_error_record,
    create_success_record,
    utc_now,
)
from benchmark.baseline.Agent.src.providers import ProviderError


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "benchmark" / "runner" / "config" / "models.yaml"
DEFAULT_EVALUATION_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "runner" / "config" / "evaluation.yaml"
)
DEFAULT_PROMPT_PATH = (
    PROJECT_ROOT / "benchmark" / "runner" / "config" / "system_prompt.txt"
)
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "benchmark" / "runs"
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")
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
    "duration_seconds",
    "model_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "accepted_retrieval_tokens",
    "attempted_retrieval_tokens",
    "finalization_tokens",
    "budget_limited",
    "run_id",
)


class BatchInputError(ValueError):
    """Report an invalid batch question file."""


def build_parser() -> argparse.ArgumentParser:
    """Build the batch command-line parser.

    Arg:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run raw-table questions from an Excel or CSV file."
    )
    parser.add_argument("--model", required=True, help="Model key from models.yaml.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Excel or CSV file containing the batch questions.",
    )
    parser.add_argument(
        "--question-type",
        required=True,
        choices=QUESTION_TYPES,
        help="Question type used for input validation and tool-round limits.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel worksheet name; the first sheet is used by default.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to models.yaml.",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG_PATH,
        help="Path to the evaluation YAML configuration.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the shared system prompt.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to a dotenv-style credential file.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Exact batch output directory; the publication results tree is the default.",
    )
    parser.add_argument(
        "--run",
        choices=("run1", "run2", "run3"),
        default="run1",
        help="Repeat label used for single-question evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of input questions to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip numbers that already have a successful JSON record.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and input without calling the model.",
    )
    return parser


def load_questions(
    input_path: Path,
    question_type: str,
    sheet_name: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and validate only the required batch input columns.

    Arg:
        input_path: Excel or CSV question file.
        question_type: Single, multi, or evidence-path question type.
        sheet_name: Optional Excel worksheet name.
        limit: Optional maximum number of rows to return.

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
    frame = frame.rename(columns={actual: canonical for canonical, actual in column_map.items()})
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


def build_system_prompt(base_prompt: str, question_type: str) -> str:
    """Return the same shared prompt for every question type.

    Arg:
        base_prompt: Shared raw-table agent instructions.
        question_type: Single, multi, or evidence-path question type.

    Returns:
        Shared system prompt for one batch question.
    """
    if question_type not in REQUIRED_COLUMNS:
        raise BatchInputError(f"Unsupported question type: {question_type}")
    return base_prompt.strip()


def run_batch(
    *,
    questions: pd.DataFrame,
    config: Any,
    provider: Any,
    system_prompt: str,
    question_type: str,
    max_tool_rounds: int,
    python_timeout_seconds: float,
    output_directory: Path,
    resume: bool = False,
    retrieval_token_limit: int | None = None,
) -> pd.DataFrame:
    """Run all validated questions and persist each result immediately.

    Arg:
        questions: Validated input questions.
        config: Selected non-secret model configuration.
        provider: Configured model provider.
        system_prompt: Complete prompt sent with every question.
        question_type: Single, multi, or evidence-path question type.
        max_tool_rounds: Maximum tool rounds for each question.
        python_timeout_seconds: Maximum duration of one Python tool call.
        output_directory: Directory for the workbook and individual JSON files.
        resume: Whether to skip numbers with successful existing records.
        retrieval_token_limit: Optional retrieval token budget per question.

    Returns:
        Final result table in input order.
    """
    individual_directory = output_directory / "individual"
    individual_directory.mkdir(parents=True, exist_ok=True)
    records, record_paths = _load_existing_records(individual_directory)
    runner = QuestionRunner(
        provider=provider,
        workspace_root=PROJECT_ROOT,
        max_tool_rounds=max_tool_rounds,
        python_timeout_seconds=python_timeout_seconds,
        max_retrieval_tokens=retrieval_token_limit,
    )

    total = len(questions)
    for position, row in questions.iterrows():
        number = row["number"]
        existing = records.get(number)
        if resume and _record_matches_question(
            existing,
            row,
            config.key,
            question_type,
            retrieval_token_limit,
        ):
            print(f"[{position + 1}/{total}] {number}: skipped (already successful)", flush=True)
            continue

        print(f"[{position + 1}/{total}] {number}: {row['question']}", flush=True)
        started_at = utc_now()
        try:
            result = runner.run(
                question=row["question"],
                system_prompt=system_prompt,
            )
            finished_at = utc_now()
            record = create_success_record(
                config=config,
                question=row["question"],
                system_prompt=system_prompt,
                result=result,
                started_at=started_at,
                finished_at=finished_at,
                question_type=question_type,
                max_tool_rounds=max_tool_rounds,
                python_timeout_seconds=python_timeout_seconds,
                retrieval_token_limit=retrieval_token_limit,
            )
        except Exception as exc:
            finished_at = utc_now()
            record = create_error_record(
                config=config,
                question=row["question"],
                system_prompt=system_prompt,
                error=exc,
                started_at=started_at,
                finished_at=finished_at,
                question_type=question_type,
                max_tool_rounds=max_tool_rounds,
                python_timeout_seconds=python_timeout_seconds,
                retrieval_token_limit=retrieval_token_limit,
            )

        payload = asdict(record)
        for column in REQUIRED_COLUMNS[question_type]:
            payload[column] = row[column]
        record_path = _write_batch_record(payload, individual_directory)
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
    """Combine input metadata and latest run records into one result table.

    Arg:
        questions: Validated input questions.
        records: Latest record payload indexed by question number.
        record_paths: Latest JSON path indexed by question number.

    Returns:
        Flat result table suitable for Excel output.
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
        usage = payload.get("usage") or {}
        row.update(
            {
                "answer": payload.get("answer", ""),
                "status": payload.get("status", ""),
                "error_type": error.get("type", ""),
                "error_message": error.get("message", ""),
                "duration_seconds": payload.get("duration_seconds", 0.0),
                "model_calls": payload.get("model_calls", 0),
                "tool_calls": payload.get("tool_calls", 0),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "accepted_retrieval_tokens": (
                    payload.get("accepted_retrieval_usage") or {}
                ).get("total_tokens", 0),
                "attempted_retrieval_tokens": (
                    payload.get("attempted_retrieval_usage") or {}
                ).get("total_tokens", 0),
                "finalization_tokens": (
                    payload.get("finalization_usage") or {}
                ).get("total_tokens", 0),
                "budget_limited": payload.get("budget_limited", False),
                "run_id": payload.get("run_id", ""),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*questions.columns, *RESULT_COLUMNS])


def write_results_workbook(results: pd.DataFrame, output_path: Path) -> None:
    """Write the current batch results to an atomic Excel workbook.

    Arg:
        results: Flat batch result table.
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
    """Run one question batch or validate its configuration and input.

    Arg:
        argv: Optional command-line arguments, excluding the program name.

    Returns:
        Process exit code: zero on complete success and one on any failure.
    """
    args = build_parser().parse_args(argv)
    try:
        load_env_file(args.env_file)
        configs = load_model_configs(args.config)
        tool_round_limits = load_tool_round_limits(args.evaluation_config)
        python_timeout_seconds = load_python_timeout_seconds(args.evaluation_config)
        token_limits = load_retrieval_token_limits(args.evaluation_config)
        if args.model not in configs:
            available = ", ".join(sorted(configs))
            raise ModelConfigError(
                f"Unknown model key '{args.model}'. Available: {available}"
            )
        config = configs[args.model]
        max_tool_rounds = tool_round_limits.for_question_type(args.question_type)
        retrieval_token_limit = (
            token_limits.for_question_type(args.question_type)
            if token_limits is not None
            else None
        )
        provider = config.create_provider()
        base_prompt = args.prompt.read_text(encoding="utf-8").strip()
        if not base_prompt:
            raise ModelConfigError("System prompt must not be empty.")
        questions = load_questions(
            args.input,
            args.question_type,
            sheet_name=args.sheet,
            limit=args.limit,
        )
        system_prompt = build_system_prompt(base_prompt, args.question_type)
    except (
        OSError,
        BatchInputError,
        EvaluationConfigError,
        ModelConfigError,
        ProviderError,
        ValueError,
    ) as exc:
        print(f"Configuration or input error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            f"Batch is ready: {len(questions)} question(s); {config.key} "
            f"({config.provider}/{config.model}); {args.question_type} "
            f"max_tool_rounds={max_tool_rounds}; "
            f"python_timeout_seconds={python_timeout_seconds:g}"
            f"; retrieval_token_limit={retrieval_token_limit or 'unrestricted'}"
        )
        return 0

    output_directory = args.output_directory or _default_output_directory(
        args.model,
        args.question_type,
        args.run,
    )
    try:
        results = run_batch(
            questions=questions,
            config=config,
            provider=provider,
            system_prompt=system_prompt,
            question_type=args.question_type,
            max_tool_rounds=max_tool_rounds,
            python_timeout_seconds=python_timeout_seconds,
            output_directory=output_directory,
            resume=args.resume,
            retrieval_token_limit=retrieval_token_limit,
        )
    except (OSError, ImportError, ValueError) as exc:
        print(f"Could not write batch output: {exc}", file=sys.stderr)
        return 1

    failed = int((results["status"] != "success").sum())
    print(f"Saved batch results: {output_directory / 'results.xlsx'}")
    if failed:
        print(f"Batch completed with {failed} unsuccessful question(s).", file=sys.stderr)
        return 1
    return 0


def _read_table(
    input_path: Path,
    sheet_name: str | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel or CSV question table.

    Arg:
        input_path: Excel or CSV question file.
        sheet_name: Optional Excel worksheet name.
        **kwargs: Reader arguments forwarded to pandas.

    Returns:
        Loaded table.
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
    """Resolve required names case-insensitively without accepting ambiguity.

    Arg:
        actual_columns: Column labels found in the input file.
        required_columns: Canonical required column names.

    Returns:
        Mapping from canonical names to actual input labels.
    """
    normalized: dict[str, list[Any]] = {}
    for column in actual_columns:
        key = str(column).strip().casefold()
        normalized.setdefault(key, []).append(column)
    ambiguous = [key for key, values in normalized.items() if len(values) > 1]
    if ambiguous:
        raise BatchInputError(
            "Ambiguous duplicate column names: " + ", ".join(ambiguous)
        )
    missing = [column for column in required_columns if column not in normalized]
    if missing:
        raise BatchInputError(
            "Input requires columns: " + ", ".join(required_columns)
            + ". Missing: " + ", ".join(missing)
        )
    return {column: normalized[column][0] for column in required_columns}


def _write_batch_record(
    payload: dict[str, Any],
    individual_directory: Path,
) -> Path:
    """Write one batch record with its input metadata.

    Arg:
        payload: Serializable run record and input metadata.
        individual_directory: Directory for per-question JSON files.

    Returns:
        Written JSON path.
    """
    safe_number = SAFE_NAME_PATTERN.sub("_", str(payload["number"])).strip("_") or "question"
    output_path = individual_directory / f"{safe_number}_{payload['run_id']}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _load_existing_records(
    individual_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Load the latest valid JSON record for every question number.

    Arg:
        individual_directory: Directory containing per-question JSON files.

    Returns:
        Latest payload and path mappings indexed by question number.
    """
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for path in sorted(individual_directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        number = str(payload.get("number", "")).strip()
        if not number or not payload.get("run_id"):
            continue
        records[number] = payload
        paths[number] = path
    return records, paths


def _record_matches_question(
    payload: dict[str, Any] | None,
    question_row: pd.Series,
    model_key: str,
    question_type: str,
    retrieval_token_limit: int | None = None,
) -> bool:
    """Check whether a successful record is safe to reuse.

    Arg:
        payload: Existing record for the same question number, if any.
        question_row: Current validated input row.
        model_key: Selected model configuration key.
        question_type: Selected batch question type.
        retrieval_token_limit: Retrieval budget required for the resumed run.

    Returns:
        True when the record matches the current model, type, and input values.
    """
    if payload is None or payload.get("status") != "success":
        return False
    if payload.get("model_key") != model_key:
        return False
    if payload.get("question_type") != question_type:
        return False
    if payload.get("retrieval_token_limit") != retrieval_token_limit:
        return False
    for column in REQUIRED_COLUMNS[question_type]:
        if str(payload.get(column, "")) != str(question_row[column]):
            return False
    return True


def _default_output_directory(
    model_key: str,
    question_type: str,
    run_label: str,
) -> Path:
    """Build the default Agent benchmark output directory.

    Arg:
        model_key: Selected model configuration key.
        question_type: Selected batch question type.
        run_label: Repeat label used for single-question evaluation.

    Returns:
        Output directory for one Agent benchmark run.
    """
    safe_model = SAFE_NAME_PATTERN.sub("_", model_key)
    task_root = DEFAULT_OUTPUT_ROOT / question_type
    if question_type == "single":
        task_root = task_root / run_label
    return task_root / safe_model / "agent"


if __name__ == "__main__":
    raise SystemExit(main())
