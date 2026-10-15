"""Plot overall evidence completeness against median token consumption.

Input:
    The Fig4e sheet in Fig4_source_data.xlsx.

Output:
    PNG and PDF scatter plots comparing completeness and token consumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
import pandas as pd


DEFAULT_SOURCE_DATA = PROJECT_ROOT / "results" / "source_data" / "Fig4_source_data.xlsx"
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
    "deepseek-v4-flash": (10, -25),
    "GLM-5.2": (10, -45),
    "GPT-5.6 Sol": (10, 25),
    "Kimi K3": (10, 0),
    "Agent (deepseek-v4-flash)": (-10, -30),
    "CellLineKG (deepseek-v4-flash)": (10, -20),
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the evidence scatter plot.

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
    """Read and validate the evidence-completeness summary sheet.

    Arg:
        source_data: Fig4 source-data workbook path.

    Returns:
        One summary row per evaluated method.
    """
    summary = pd.read_excel(source_data, sheet_name="Fig4e")
    required = ["method", "questions", "numerator", "denominator", "completeness", "median_tokens"]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise ValueError(f"{source_data} is missing columns: {', '.join(missing)}")
    summary = summary[required].copy()
    summary["method"] = summary["method"].astype(str).str.strip()
    for column in ["questions", "numerator", "denominator", "completeness", "median_tokens"]:
        summary[column] = pd.to_numeric(summary[column], errors="raise")
    if summary["method"].duplicated().any() or set(summary["method"]) != set(METHOD_ORDER):
        raise ValueError(f"{source_data} does not contain exactly one row per method.")
    summary = summary.set_index("method").reindex(METHOD_ORDER).reset_index()
    if (summary["denominator"] <= 0).any() or (summary["numerator"] > summary["denominator"]).any():
        raise ValueError(f"{source_data} contains invalid completeness counts.")
    if (summary["median_tokens"] <= 0).any():
        raise ValueError(f"{source_data} contains non-positive median token values.")
    if summary["questions"].nunique() != 1 or summary["denominator"].nunique() != 1:
        raise ValueError("Evidence-path workbooks do not use the same benchmark scope.")
    return summary


def plot_evidence_vs_tokens(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create the evidence completeness and token scatter plot.

    Arg:
        summary: Overall evidence completeness and token values for all methods.
        output_dir: Directory receiving PNG and PDF outputs.

    Returns:
        None. Writes evidence_completeness_vs_token_consumption.png and PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for _, row in summary.iterrows():
        method = str(row["method"])
        x_value = float(row["median_tokens"])
        y_value = float(row["completeness"])
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
            f"{method}\n{y_value:.1%} "
            f"({int(row['numerator'])}/{int(row['denominator'])}), "
            f"{x_value:,.0f} tokens",
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=9,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(400, 180000)
    ax.set_ylim(0, 1.07)
    ax.set_xlabel(r"Median total tokens per question (log$_2$ scale)", fontsize=11)
    ax.set_ylabel("Overall evidence completeness", fontsize=11)
    ax.set_title(
        "Overall Evidence Completeness vs.\nToken Consumption",
        fontsize=14,
        pad=12,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
    ax.set_xticks([500, 2000, 8000, 32000, 128000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "evidence_completeness_vs_token_consumption.png",
        dpi=300,
        facecolor="white",
    )
    fig.savefig(
        output_dir / "evidence_completeness_vs_token_consumption.pdf",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    """Run the evidence completeness and token plotting workflow.

    Arg:
        None.

    Returns:
        None.
    """
    args = parse_args()
    summary = read_summary(args.source_data)
    plot_evidence_vs_tokens(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote evidence completeness and token figures to {args.output_dir}")


if __name__ == "__main__":
    main()
