"""Plot category accuracy for LLMs, Agent, and CellLineKG.

Input:
    The Fig2a-e sheet in Fig2_source_data.xlsx.

Output:
    PNG and PDF figures with five category panels and one legend panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
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
    "deepseek-v4-flash": "#D7D5E3",
    "GLM-5.2": "#E8D8D7",
    "GPT-5.6 Sol": "#F0E6D2",
    "Kimi K3": "#DDE2DE",
    "Agent (deepseek-v4-flash)": "#F2D3A0",
    "CellLineKG (deepseek-v4-flash)": "#88AED2",
}
CATEGORY_ORDER = [
    "Cell line metadata",
    "Copy number variation",
    "Drug response",
    "Gene dependency",
    "Variant",
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the accuracy figure.

    Arg:
        None.

    Returns:
        Parsed source-data workbook and figure output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_source_data(path: Path) -> pd.DataFrame:
    """Read and validate the category-accuracy source-data sheet.

    Arg:
        path: Fig2 source-data workbook path.

    Returns:
        Run-level category accuracy values for every method.
    """
    frame = pd.read_excel(path, sheet_name="Fig2a-e")
    required = ["run", "category", "method", "correct", "total", "accuracy"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    result = frame[required].copy()
    result["category"] = result["category"].astype(str).str.strip()
    result["method"] = result["method"].astype(str).str.strip()
    for column in ["correct", "total", "accuracy"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not result["method"].isin(METHOD_ORDER).all():
        raise ValueError(f"{path} contains an unknown method name.")
    if (result["total"] <= 0).any() or (result["correct"] > result["total"]).any():
        raise ValueError(f"{path} contains invalid correct or total counts.")
    return result


def summarize_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy for every category and method.

    Arg:
        results: Run-level category results from the Fig2a-e sheet.

    Returns:
        Category and method summaries with correct counts, totals, and accuracy.
    """
    summary = (
        results.groupby(["category", "method"], sort=False)
        .agg(
            correct=("correct", "sum"),
            total=("total", "sum"),
            runs=("run", "nunique"),
            accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        )
        .reset_index()
    )
    summary["accuracy_sd"] = summary["accuracy_sd"].fillna(0.0)
    return summary


def plot_accuracy(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create the five-panel category accuracy figure.

    Arg:
        summary: Accuracy summary for all categories and methods.
        output_dir: Directory receiving PNG and PDF outputs.

    Returns:
        None. Writes model_comparison_accuracy.png and PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharey=True)
    plot_axes = axes.flat

    for panel_index, category in enumerate(CATEGORY_ORDER):
        ax = plot_axes[panel_index]
        category_rows = (
            summary[summary["category"] == category]
            .set_index("method")
            .reindex(METHOD_ORDER)
        )
        if category_rows["accuracy"].isna().any():
            raise ValueError(f"Missing method scores for category: {category}")

        values = category_rows["accuracy"].to_numpy()
        errors = category_rows["accuracy_sd"].to_numpy()
        bars = ax.bar(
            range(len(METHOD_ORDER)),
            values,
            color=[METHOD_COLORS[method] for method in METHOD_ORDER],
            width=0.90,
            yerr=errors,
            error_kw={"elinewidth": 1.0, "ecolor": "#444444", "capsize": 3},
        )
        for bar, value, error in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + 0.025,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks([])
        ax.set_xlabel(category, fontsize=14, labelpad=7)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(
            axis="y",
            labelleft=True,
            labelsize=10,
            width=1.1,
            length=4,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.text(
            -0.08,
            1.05,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
        )
        ax.set_ylabel("Accuracy (%)", fontsize=12)

    legend_ax = plot_axes[5]
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=METHOD_COLORS[method], edgecolor="none", label=method)
        for method in METHOD_ORDER
    ]
    legend_ax.legend(
        handles=handles,
        title="Methods",
        loc="center left",
        frameon=False,
        fontsize=11,
        title_fontsize=12,
        handlelength=1.8,
        labelspacing=0.8,
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.08, wspace=0.22, hspace=0.30)
    fig.savefig(
        output_dir / "model_comparison_accuracy.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_dir / "model_comparison_accuracy.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    """Run the multi-model accuracy plotting workflow.

    Arg:
        None.

    Returns:
        None.
    """
    args = parse_args()
    results = read_source_data(args.source_data)
    summary = summarize_accuracy(results)
    plot_accuracy(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote model comparison accuracy figures to {args.output_dir}")


if __name__ == "__main__":
    main()
