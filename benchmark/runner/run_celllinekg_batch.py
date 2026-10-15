"""Run CellLineKG questions in batches and save metrics for analysis.

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
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.baseline.Agent.src.config import (
    EvaluationConfigError,
    load_retrieval_token_limits,
)
from celllinekg.framework import CellLineKGFramework
from celllinekg.llm import UsageTracker


QUESTION_TYPES = ("single", "multi", "evidence_path")
DEFAULT_EVALUATION_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "runner" / "config" / "evaluation.yaml"
)
REQUIRED_COLUMNS = {
    "single": ("number", "category", "sub_category", "question"),
    "multi": ("number", "question"),
    "evidence_path": ("number", "question"),
}
RESULT_COLUMNS = (
    "route",
    "answer",
    "status",
    "error_type",
    "error_message",
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
    "budget_limited",
    "usage_complete",
    "model",
    "started_at",
    "finished_at",
)
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
FINALIZATION_PHASES = {"answer_synthesis", "evidence_path_summary"}
BUDGET_ENFORCEMENT_VERSION = "evaluation_checkpoint_rollback_v1"


class BatchInputError(ValueError):
    """Report an invalid CellLineKG batch input."""


def build_parser() -> argparse.ArgumentParser:
    """Build the CellLineKG batch command-line parser.

    Arg:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run CellLineKG questions from an Excel or CSV file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--question-type", choices=QUESTION_TYPES, required=True)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument(
        "--model-label",
        default="deepseek",
        help="Short model label used only in the output directory.",
    )
    parser.add_argument(
        "--run",
        choices=("run1", "run2", "run3"),
        default="run1",
        help="Repeat label used for single-question evaluation.",
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
        help="Validate the input without calling CellLineKG.",
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
    agent: Any | None = None,
    retrieval_token_limit: int | None = None,
) -> pd.DataFrame:
    """Run CellLineKG for all questions and persist every result.

    Arg:
        questions: Validated input questions.
        question_type: Single, multi, or evidence-path question type.
        output_directory: Directory for Excel and JSON results.
        resume: Whether to skip matching successful records.
        agent: Optional injected agent used by tests.
        retrieval_token_limit: Optional retrieval token budget per question.

    Returns:
        Flat result table in input order.
    """
    individual_directory = output_directory / "individual"
    individual_directory.mkdir(parents=True, exist_ok=True)
    records, record_paths = _load_records(individual_directory)
    question_agent = agent or CellLineKGFramework()

    total = len(questions)
    for position, row in questions.iterrows():
        number = row["number"]
        existing = records.get(number)
        if resume and _record_matches(
            existing, row, question_type, retrieval_token_limit
        ):
            print(
                f"[{position + 1}/{total}] {number}: skipped (already successful)",
                flush=True,
            )
            continue

        print(f"[{position + 1}/{total}] {number}: {row['question']}", flush=True)
        fallback_tracker = UsageTracker()
        try:
            result = question_agent.ask_with_metrics(
                row["question"],
                retrieval_token_limit=retrieval_token_limit,
            )
        except Exception as exc:
            result = {
                "route": "",
                "answer": "",
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "metrics": fallback_tracker.finish(),
            }
        payload = {
            "question_type": question_type,
            **row.to_dict(),
            "route": result.get("route", ""),
            "answer": result.get("answer", ""),
            "status": result.get("status", "error"),
            "error": result.get("error"),
            "metrics": result.get("metrics") or {},
            "retrieval_token_limit": retrieval_token_limit,
            "budget_limited": bool(result.get("budget_limited", False)),
            "budget_enforcement_version": (
                BUDGET_ENFORCEMENT_VERSION
                if retrieval_token_limit is not None
                else ""
            ),
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
    """Flatten CellLineKG records into an Excel-ready table.

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
        retrieval_tokens, finalization_tokens = _split_phase_tokens(metrics)
        accepted_retrieval_tokens = metrics.get(
            "accepted_retrieval_tokens",
            retrieval_tokens,
        )
        attempted_retrieval_tokens = metrics.get(
            "attempted_retrieval_tokens",
            retrieval_tokens,
        )
        row.update(
            {
                "route": payload.get("route", ""),
                "answer": payload.get("answer", ""),
                "status": payload.get("status", ""),
                "error_type": error.get("type", ""),
                "error_message": error.get("message", ""),
                "duration_seconds": metrics.get("duration_seconds", 0.0),
                "llm_duration_seconds": metrics.get("llm_duration_seconds", 0.0),
                "model_calls": metrics.get("model_calls", 0),
                "successful_model_calls": metrics.get("successful_model_calls", 0),
                "failed_model_calls": metrics.get("failed_model_calls", 0),
                "input_tokens": metrics.get("input_tokens", 0),
                "output_tokens": metrics.get("output_tokens", 0),
                "total_tokens": metrics.get("total_tokens", 0),
                "accepted_retrieval_tokens": accepted_retrieval_tokens,
                "attempted_retrieval_tokens": attempted_retrieval_tokens,
                "finalization_tokens": metrics.get(
                    "finalization_tokens",
                    finalization_tokens,
                ),
                "budget_limited": payload.get("budget_limited", False),
                "usage_complete": metrics.get("usage_complete", False),
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
        results: Flat CellLineKG result table.
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


def _split_phase_tokens(metrics: dict[str, Any]) -> tuple[int, int]:
    """Split actual model usage into retrieval and finalization tokens.

    Arg:
        metrics: Question-level metrics containing individual model calls.

    Returns:
        Accepted retrieval tokens and final answer generation tokens.
    """
    retrieval_tokens = 0
    finalization_tokens = 0
    for call in metrics.get("calls", []):
        if not isinstance(call, dict) or call.get("status") != "success":
            continue
        tokens = int(call.get("total_tokens", 0) or 0)
        if call.get("phase") in FINALIZATION_PHASES:
            finalization_tokens += tokens
        else:
            retrieval_tokens += tokens
    return retrieval_tokens, finalization_tokens


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run or validate one CellLineKG question batch.

    Arg:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero on complete success and one when any question fails.
    """
    args = build_parser().parse_args(argv)
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
    except (OSError, BatchInputError, EvaluationConfigError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            f"Batch is ready: {len(questions)} question(s); "
            f"question_type={args.question_type}"
            f"; retrieval_token_limit={retrieval_token_limit or 'unrestricted'}"
        )
        return 0

    output_directory = args.output_directory or _default_output_directory(
        question_type=args.question_type,
        model_label=args.model_label,
        run_label=args.run,
    )
    try:
        results = run_batch(
            questions=questions,
            question_type=args.question_type,
            output_directory=output_directory,
            resume=args.resume,
            retrieval_token_limit=retrieval_token_limit,
        )
    except (OSError, ImportError, ValueError) as exc:
        print(f"Could not write batch output: {exc}", file=sys.stderr)
        return 1

    failed = int((results["status"] != "success").sum())
    print(f"Saved CellLineKG results: {output_directory / 'results.xlsx'}")
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
) -> bool:
    """Check whether an existing successful record matches the input row.

    Arg:
        payload: Existing question record, if present.
        row: Current normalized question row.
        question_type: Current question type.
        retrieval_token_limit: Retrieval budget required for the resumed run.

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
        != BUDGET_ENFORCEMENT_VERSION
    ):
        return False
    return all(
        str(payload.get(column, "")).strip() == str(row[column]).strip()
        for column in REQUIRED_COLUMNS[question_type]
    )


def _default_output_directory(
    *,
    question_type: str,
    model_label: str,
    run_label: str,
) -> Path:
    """Build the default CellLineKG benchmark output directory.

    Arg:
        question_type: Single, multi, or evidence-path question type.
        model_label: Short model label for the result directory.
        run_label: Repeat label used for single-question evaluation.

    Returns:
        Output directory for one CellLineKG benchmark run.
    """
    safe_model = SAFE_NAME_PATTERN.sub("_", model_label).strip("._") or "model"
    task_root = PROJECT_ROOT / "results" / "benchmark" / "runs" / question_type
    if question_type == "single":
        task_root = task_root / run_label
    return task_root / safe_model / "celllinekg"


if __name__ == "__main__":
    raise SystemExit(main())
