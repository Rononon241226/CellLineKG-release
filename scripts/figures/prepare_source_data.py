"""Prepare publication source-data workbooks for Figures 2, 3, and 4.

Input:
    Checked benchmark workbooks and the existing KEGG enrichment CSV files.

Output:
    Three Excel workbooks with one sheet for each plotted panel group.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKED_ROOT = PROJECT_ROOT / "results" / "benchmark" / "results_checked"
DEFAULT_KEGG_ROOT = PROJECT_ROOT / "results" / "kegg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "source_data"

METHOD_ORDER = [
    "deepseek-v4-flash",
    "GLM-5.2",
    "GPT-5.6 Sol",
    "Kimi K3",
    "Agent (deepseek-v4-flash)",
    "CellLineKG (deepseek-v4-flash)",
]
SINGLE_FILE_NAMES = {
    "deepseek-v4-flash": "deepseek_single_llm_check.xlsx",
    "GLM-5.2": "glm52_single_llm_check.xlsx",
    "GPT-5.6 Sol": "gpt56_sol_single_llm_check.xlsx",
    "Kimi K3": "kimi_k3_single_llm_check.xlsx",
    "Agent (deepseek-v4-flash)": "deepseek_single_agent_check.xlsx",
    "CellLineKG (deepseek-v4-flash)": "deepseek_single_celllinekg_check.xlsx",
}
MULTI_FILE_NAMES = {
    "deepseek-v4-flash": "deepseek_multi_llm_check.xlsx",
    "GLM-5.2": "glm52_multi_llm_check.xlsx",
    "GPT-5.6 Sol": "gpt56_sol_multi_llm_check.xlsx",
    "Kimi K3": "kimi_k3_multi_llm_check.xlsx",
    "Agent (deepseek-v4-flash)": "deepseek_multi_agent_check.xlsx",
    "CellLineKG (deepseek-v4-flash)": "deepseek_multi_celllinekg_check.xlsx",
}
EVIDENCE_FILE_NAMES = {
    "deepseek-v4-flash": "deepseek_path_llm_check.xlsx",
    "GLM-5.2": "glm52_path_llm_check.xlsx",
    "GPT-5.6 Sol": "gpt56_sol_path_llm_check.xlsx",
    "Kimi K3": "kimi_k3_path_llm_check.xlsx",
    "Agent (deepseek-v4-flash)": "deepseek_path_agent_check.xlsx",
    "CellLineKG (deepseek-v4-flash)": "deepseek_path_celllinekg_check.xlsx",
}


def parse_args() -> argparse.Namespace:
    """
    Parse source-data preparation paths.

    Arg:
        None.

    Returns:
        Parsed benchmark, KEGG, and output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checked-root", type=Path, default=DEFAULT_CHECKED_ROOT)
    parser.add_argument("--kegg-root", type=Path, default=DEFAULT_KEGG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gene-kegg-file",
        type=Path,
        default=None,
        help="Optional override for the HL-60 gene KEGG CSV.",
    )
    parser.add_argument(
        "--protein-kegg-file",
        type=Path,
        default=None,
        help="Optional override for the OE19 protein KEGG CSV.",
    )
    return parser.parse_args()


def discover_single_inputs(checked_root: Path) -> dict[str, dict[str, Path]]:
    """
    Discover complete checked single-question runs.

    Arg:
        checked_root: Root directory containing checked benchmark workbooks.

    Returns:
        Checked workbook paths grouped by run and method.
    """
    single_root = checked_root / "single"
    if not single_root.is_dir():
        raise FileNotFoundError(f"Single-question directory not found: {single_root}")

    discovered: dict[str, dict[str, Path]] = {}
    for run_dir in sorted(path for path in single_root.iterdir() if path.is_dir()):
        paths = {
            method: run_dir / SINGLE_FILE_NAMES[method] for method in METHOD_ORDER
        }
        existing_count = sum(path.is_file() for path in paths.values())
        if existing_count == 0:
            continue
        missing = [method for method, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{run_dir} is incomplete; missing methods: {', '.join(missing)}"
            )
        discovered[run_dir.name] = paths

    if not discovered:
        raise FileNotFoundError(f"No complete single-question runs found: {single_root}")
    return discovered


def build_fig2_benchmark_tables(
    checked_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build run-level accuracy and token tables for Figure 2a-f.

    Arg:
        checked_root: Root directory containing checked benchmark workbooks.

    Returns:
        Figure 2a-e category table and Figure 2f overall table.
    """
    category_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []

    for run, paths in discover_single_inputs(checked_root).items():
        for method in METHOD_ORDER:
            path = paths[method]
            frame = pd.read_excel(path, sheet_name=0)
            _require_columns(frame, ["category", "Accuracy", "total_tokens"], path)

            accuracy = pd.to_numeric(frame["Accuracy"], errors="raise")
            tokens = pd.to_numeric(frame["total_tokens"], errors="raise")
            if not accuracy.isin([0, 1]).all():
                raise ValueError(f"{path} contains Accuracy values other than 0 or 1.")
            if tokens.isna().any() or not tokens.gt(0).all():
                raise ValueError(f"{path} contains missing or non-positive total_tokens.")

            category_frame = pd.DataFrame(
                {
                    "category": frame["category"].astype(str).str.strip(),
                    "Accuracy": accuracy.astype(int),
                }
            )
            grouped = (
                category_frame.groupby("category", sort=False)["Accuracy"]
                .agg(correct="sum", total="count")
                .reset_index()
            )
            for row in grouped.itertuples(index=False):
                category_rows.append(
                    {
                        "run": run,
                        "category": row.category,
                        "method": method,
                        "correct": int(row.correct),
                        "total": int(row.total),
                        "accuracy": float(row.correct / row.total),
                    }
                )

            overall_rows.append(
                {
                    "run": run,
                    "method": method,
                    "questions": int(len(frame)),
                    "correct": int(accuracy.sum()),
                    "accuracy": float(accuracy.mean()),
                    "median_tokens": float(tokens.median()),
                }
            )

    return pd.DataFrame(category_rows), pd.DataFrame(overall_rows)


def build_completeness_tables(
    checked_root: Path,
    task_directory: str,
    file_names: dict[str, str],
    score_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build question-level and overall completeness tables for one benchmark.

    Arg:
        checked_root: Root directory containing checked benchmark workbooks.
        task_directory: Checked-results subdirectory for the benchmark.
        file_names: Checked workbook names indexed by method.
        score_column: Human-scored completeness column to parse.

    Returns:
        Question-level completeness rows and overall method summaries.
    """
    question_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    expected_questions: list[str] | None = None
    task_root = checked_root / task_directory

    for method in METHOD_ORDER:
        path = task_root / file_names[method]
        if not path.is_file():
            raise FileNotFoundError(f"Checked workbook not found: {path}")
        frame = pd.read_excel(path, sheet_name=0)
        _require_columns(frame, ["number", score_column, "total_tokens"], path)

        questions = [_normalize_identifier(value) for value in frame["number"]]
        if expected_questions is None:
            expected_questions = questions
        elif questions != expected_questions:
            raise ValueError(f"Question order in {path} does not match other methods.")

        tokens = pd.to_numeric(frame["total_tokens"], errors="raise")
        if tokens.isna().any() or not tokens.gt(0).all():
            raise ValueError(f"{path} contains missing or non-positive total_tokens.")

        total_numerator = 0
        total_denominator = 0
        for question, score in zip(questions, frame[score_column]):
            numerator, denominator = _parse_fraction(score, score_column)
            total_numerator += numerator
            total_denominator += denominator
            question_rows.append(
                {
                    "question": question,
                    "method": method,
                    "numerator": numerator,
                    "denominator": denominator,
                    "completeness": numerator / denominator,
                }
            )

        summary_rows.append(
            {
                "method": method,
                "questions": int(len(frame)),
                "numerator": total_numerator,
                "denominator": total_denominator,
                "completeness": total_numerator / total_denominator,
                "median_tokens": float(tokens.median()),
            }
        )

    return pd.DataFrame(question_rows), pd.DataFrame(summary_rows)


def build_kegg_source_table(
    path: Path,
    cell_line_id: str,
    cell_line_name: str,
    measurement: str,
) -> pd.DataFrame:
    """
    Select the exact top ten KEGG pathways used in one Figure 2 panel.

    Arg:
        path: Existing KEGG enrichment CSV file.
        cell_line_id: Cell-line identifier shown in source data.
        cell_line_name: Cell-line display name shown in source data.
        measurement: Gene-expression or protein-abundance description.

    Returns:
        Ranked KEGG pathway rows used by the bar plot.
    """
    if not path.is_file():
        raise FileNotFoundError(f"KEGG enrichment CSV not found: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, ["ID", "Description", "Count", "p.adjust"], path)
    frame["p.adjust"] = pd.to_numeric(frame["p.adjust"], errors="raise")
    plot_data = frame.sort_values("p.adjust", kind="stable").head(10).copy()
    plot_data.insert(0, "rank", range(1, len(plot_data) + 1))
    plot_data.insert(1, "cell_line_id", cell_line_id)
    plot_data.insert(2, "cell_line_name", cell_line_name)
    plot_data.insert(3, "measurement", measurement)
    return plot_data


def write_source_workbooks(
    output_dir: Path,
    fig2_category: pd.DataFrame,
    fig2_overall: pd.DataFrame,
    fig2_gene_kegg: pd.DataFrame,
    fig2_protein_kegg: pd.DataFrame,
    fig3_question: pd.DataFrame,
    fig3_overall: pd.DataFrame,
    fig4_question: pd.DataFrame,
    fig4_overall: pd.DataFrame,
) -> list[Path]:
    """
    Write the three publication source-data workbooks.

    Arg:
        output_dir: Directory receiving the Excel workbooks.
        fig2_category: Run-level category accuracy data for Figure 2a-e.
        fig2_overall: Run-level accuracy and token data for Figure 2f.
        fig2_gene_kegg: Ranked gene KEGG pathways for Figure 2g.
        fig2_protein_kegg: Ranked protein KEGG pathways for Figure 2h.
        fig3_question: Question-level candidate completeness for Figure 3d.
        fig3_overall: Overall candidate completeness and tokens for Figure 3e.
        fig4_question: Question-level evidence completeness for Figure 4d.
        fig4_overall: Overall evidence completeness and tokens for Figure 4e.

    Returns:
        Paths to the three written Excel workbooks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig2_path = output_dir / "Fig2_source_data.xlsx"
    fig3_path = output_dir / "Fig3_source_data.xlsx"
    fig4_path = output_dir / "Fig4_source_data.xlsx"

    with pd.ExcelWriter(fig2_path, engine="openpyxl") as writer:
        fig2_category.to_excel(writer, sheet_name="Fig2a-e", index=False)
        fig2_overall.to_excel(writer, sheet_name="Fig2f", index=False)
        fig2_gene_kegg.to_excel(writer, sheet_name="Fig2g", index=False)
        fig2_protein_kegg.to_excel(writer, sheet_name="Fig2h", index=False)

    with pd.ExcelWriter(fig3_path, engine="openpyxl") as writer:
        fig3_question.to_excel(writer, sheet_name="Fig3d", index=False)
        fig3_overall.to_excel(writer, sheet_name="Fig3e", index=False)

    with pd.ExcelWriter(fig4_path, engine="openpyxl") as writer:
        fig4_question.to_excel(writer, sheet_name="Fig4d", index=False)
        fig4_overall.to_excel(writer, sheet_name="Fig4e", index=False)

    return [fig2_path, fig3_path, fig4_path]


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    """
    Check that a source table contains required columns.

    Arg:
        frame: Source table to validate.
        columns: Required column names.
        path: Source path used in an error message.

    Returns:
        None.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def _parse_fraction(value: Any, column: str) -> tuple[int, int]:
    """
    Parse a human-scored numerator and denominator.

    Arg:
        value: Score text such as 2/3.
        column: Column name used in an error message.

    Returns:
        Integer numerator and denominator.
    """
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(value))
    if not match:
        raise ValueError(f"Invalid {column} value: {value}")
    numerator, denominator = map(int, match.groups())
    if denominator <= 0 or numerator > denominator:
        raise ValueError(f"Invalid {column} value: {value}")
    return numerator, denominator


def _normalize_identifier(value: Any) -> str:
    """
    Normalize an Excel question identifier without changing its meaning.

    Arg:
        value: Question identifier read from Excel.

    Returns:
        Stable string identifier.
    """
    if pd.isna(value):
        raise ValueError("Question number is missing.")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def main() -> None:
    """
    Prepare all publication source-data workbooks.

    Arg:
        None.

    Returns:
        None.
    """
    args = parse_args()
    gene_kegg_file = args.gene_kegg_file or (
        args.kegg_root / "gene" / "ACH-000002_HL-60_high_kegg.csv"
    )
    protein_kegg_file = args.protein_kegg_file or (
        args.kegg_root / "protein" / "SIDM00479_OE19_high_kegg.csv"
    )

    fig2_category, fig2_overall = build_fig2_benchmark_tables(args.checked_root)
    fig3_question, fig3_overall = build_completeness_tables(
        args.checked_root,
        "multi",
        MULTI_FILE_NAMES,
        "Candidate Completeness",
    )
    fig4_question, fig4_overall = build_completeness_tables(
        args.checked_root,
        "evidence_path",
        EVIDENCE_FILE_NAMES,
        "Evidence Completeness",
    )
    fig2_gene_kegg = build_kegg_source_table(
        gene_kegg_file,
        "ACH-000002",
        "HL-60",
        "Highly expressed genes",
    )
    fig2_protein_kegg = build_kegg_source_table(
        protein_kegg_file,
        "SIDM00479",
        "OE19",
        "Highly abundant proteins",
    )

    paths = write_source_workbooks(
        args.output_dir,
        fig2_category,
        fig2_overall,
        fig2_gene_kegg,
        fig2_protein_kegg,
        fig3_question,
        fig3_overall,
        fig4_question,
        fig4_overall,
    )
    for path in paths:
        print(f"Wrote source data: {path}")


if __name__ == "__main__":
    main()
