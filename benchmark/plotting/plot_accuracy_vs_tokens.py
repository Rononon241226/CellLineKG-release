"""Plot overall single-QA accuracy against median token consumption.

Input:
    The Fig2f sheet in Fig2_source_data.xlsx.

Output:
    PNG and PDF scatter plots comparing accuracy and token consumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
import pandas as pd


DEFAULT_SOURCE_DATA = PROJECT_ROOT / "results" / "source_data" / "Fig2_source_data.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark" / "figures"
METHOD_ORDER = [
    "deepseek-v4-flash",
    "GLM-5.2",
    "GPT-5.6 Sol",
    "Kimi K3",
    "Agent (deepseek-v4-flash)",
    "CellLineKG (deepseek-v4-flash)",
]
METHOD_COLORS = {
    "deepseek-v4-flash": "#B8B5CB",
    "GLM-5.2": "#DABFBE",
    "GPT-5.6 Sol": "#E4D3B0",
    "Kimi K3": "#BBC6BE",
    "Agent (deepseek-v4-flash)": "#EABF76",
    "CellLineKG (deepseek-v4-flash)": "#5F94C4",
}
METHOD_MARKERS = {
    "deepseek-v4-flash": "o",
    "GLM-5.2": "o",
    "GPT-5.6 Sol": "o",
    "Kimi K3": "o",
    "Agent (deepseek-v4-flash)": "s",
    "CellLineKG (deepseek-v4-flash)": "D",
}
ANNOTATION_OFFSETS = {
    "deepseek-v4-flash": (10, -2),
    "GLM-5.2": (10, 12),
    "GPT-5.6 Sol": (10, 12),
    "Kimi K3": (10, -28),
    "Agent (deepseek-v4-flash)": (-10, -32),
    "CellLineKG (deepseek-v4-flash)": (-10, 12),
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the scatter plot.

    Arg:
        None.

    Returns:
        Parsed source-data workbook and figure output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_summary(source_data: Path) -> pd.DataFrame:
    """Read the Fig2f source-data sheet and summarize repeated runs.

    Arg:
        source_data: Fig2 source-data workbook path.

    Returns:
        One across-run summary row per evaluated method.
    """
    run_summary = pd.read_excel(source_data, sheet_name="Fig2f")
    required = ["run", "method", "questions", "correct", "accuracy", "median_tokens"]
    missing = [column for column in required if column not in run_summary.columns]
    if missing:
        raise ValueError(f"{source_data} is missing columns: {', '.join(missing)}")
    run_summary = run_summary[required].copy()
    run_summary["method"] = run_summary["method"].astype(str).str.strip()
    for column in ["questions", "correct", "accuracy", "median_tokens"]:
        run_summary[column] = pd.to_numeric(run_summary[column], errors="raise")
    if not run_summary["method"].isin(METHOD_ORDER).all():
        raise ValueError(f"{source_data} contains an unknown method name.")
    if (run_summary["median_tokens"] <= 0).any():
        raise ValueError(f"{source_data} contains non-positive median token values.")
    if run_summary.groupby("run")["questions"].nunique().gt(1).any():
        raise ValueError("Single-QA workbooks do not contain the same question count.")
    summary = (
        run_summary.groupby("method", sort=False)
        .agg(
            runs=("run", "nunique"),
            questions=("questions", "first"),
            correct=("correct", "mean"),
            accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            median_tokens=("median_tokens", "mean"),
            median_tokens_sd=("median_tokens", "std"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    summary[["accuracy_sd", "median_tokens_sd"]] = summary[
        ["accuracy_sd", "median_tokens_sd"]
    ].fillna(0.0)
    return summary


def plot_accuracy_vs_tokens(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create the overall accuracy and token-efficiency scatter plot.

    Arg:
        summary: Overall accuracy and median token values for all methods.
        output_dir: Directory receiving PNG and PDF outputs.

    Returns:
        None. Writes accuracy_vs_token_consumption_single.png and PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for _, row in summary.iterrows():
        method = str(row["method"])
        x_value = float(row["median_tokens"])
        y_value = float(row["accuracy"])
        ax.scatter(
            x_value,
            y_value,
            s=125,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="#3F3F3F",
            linewidth=0.8,
            zorder=3,
        )
        x_offset, y_offset = ANNOTATION_OFFSETS[method]
        horizontal_alignment = "left" if x_offset >= 0 else "right"
        ax.annotate(
            f"{method}\n{y_value:.1%}, {x_value:,.0f} tokens",
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=9,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(300, 34000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(
        r"Median total tokens per question (log$_2$ scale)",
        fontsize=11,
    )
    ax.set_ylabel("Overall accuracy", fontsize=11)
    ax.set_title("Overall Accuracy vs.\nToken Consumption", fontsize=14, pad=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
    ax.set_xticks([500, 1000, 2000, 4000, 8000, 16000, 32000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axhline(0.80, color="#777777", linewidth=1.2, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        output_dir / "accuracy_vs_token_consumption_single.png",
        dpi=300,
        facecolor="white",
    )
    fig.savefig(
        output_dir / "accuracy_vs_token_consumption_single.pdf",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    """Run the single-QA accuracy and token plotting workflow.

    Arg:
        None.

    Returns:
        None.
    """
    args = parse_args()
    summary = read_summary(args.source_data)
    plot_accuracy_vs_tokens(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote accuracy and token figures to {args.output_dir}")


if __name__ == "__main__":
    main()
