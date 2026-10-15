"""Plot candidate completeness across all evaluated methods.

Input:
    The Fig3d sheet in Fig3_source_data.xlsx.

Output:
    PNG and PDF annotated heatmaps with questions as rows and methods as columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


DEFAULT_SOURCE_DATA = PROJECT_ROOT / "results" / "source_data" / "Fig3_source_data.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark" / "figures"
METHOD_ORDER = [
    "deepseek-v4-flash",
    "GLM-5.2",
    "GPT-5.6 Sol",
    "Kimi K3",
    "Agent (deepseek-v4-flash)",
    "CellLineKG (deepseek-v4-flash)",
]
METHOD_LABELS = [method.replace(" (", "\n(") for method in METHOD_ORDER]
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "model_candidate_completeness",
    ["#F7F2FA", "#E6D8EF", "#CFB5DF", "#B58DCA"],
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the candidate figure.

    Arg:
        None.

    Returns:
        Parsed source-data workbook and figure output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_results(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read and align the candidate-completeness source-data sheet.

    Arg:
        path: Fig3 source-data workbook path.

    Returns:
        Question labels, numeric fractions, and original score annotations.
    """
    frame = pd.read_excel(path, sheet_name="Fig3d")
    required = ["question", "method", "numerator", "denominator", "completeness"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frame = frame[required].copy()
    frame["question"] = frame["question"].astype(str).str.strip()
    frame["method"] = frame["method"].astype(str).str.strip()
    for column in ["numerator", "denominator", "completeness"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.duplicated(["question", "method"]).any():
        raise ValueError(f"{path} contains duplicate question and method rows.")
    if not frame["method"].isin(METHOD_ORDER).all():
        raise ValueError(f"{path} contains an unknown method name.")
    if (frame["denominator"] <= 0).any() or (frame["numerator"] > frame["denominator"]).any():
        raise ValueError(f"{path} contains invalid completeness counts.")

    question_order = frame["question"].drop_duplicates().tolist()
    numerators = frame.pivot(index="question", columns="method", values="numerator").reindex(
        index=question_order, columns=METHOD_ORDER
    )
    denominators = frame.pivot(index="question", columns="method", values="denominator").reindex(
        index=question_order, columns=METHOD_ORDER
    )
    completeness = frame.pivot(index="question", columns="method", values="completeness").reindex(
        index=question_order, columns=METHOD_ORDER
    )
    if numerators.isna().any().any() or denominators.isna().any().any():
        raise ValueError(f"{path} does not contain every question and method combination.")
    annotations = (
        numerators.astype(int).astype(str) + "/" + denominators.astype(int).astype(str)
    ).to_numpy()
    fractions = completeness.to_numpy(dtype=float)
    row_labels = [number if number.upper().startswith("Q") else f"Q{number}" for number in question_order]
    return row_labels, fractions, annotations


def plot_heatmap(
    row_labels: list[str],
    fractions: np.ndarray,
    annotations: np.ndarray,
    output_dir: Path,
) -> None:
    """Create the six-method candidate-completeness heatmap.

    Arg:
        row_labels: Question labels displayed on the y-axis.
        fractions: Numeric completeness values used for cell colors.
        annotations: Original numerator/denominator values displayed in cells.
        output_dir: Directory receiving PNG and PDF outputs.

    Returns:
        None. Writes model_comparison_candidate.png and PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization = Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots(figsize=(12.0, 6.7))

    for row_index in range(fractions.shape[0]):
        for column_index in range(fractions.shape[1]):
            ax.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    facecolor=PURPLE_CMAP(normalization(fractions[row_index, column_index])),
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            ax.text(
                column_index,
                row_index,
                annotations[row_index, column_index],
                ha="center",
                va="center",
                color="#111111",
                fontsize=10,
            )

    ax.set_xlim(-0.5, fractions.shape[1] - 0.5)
    ax.set_ylim(fractions.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(len(METHOD_ORDER)), labels=METHOD_LABELS)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", length=0, pad=9, labelsize=10)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    color_mappable = ScalarMappable(norm=normalization, cmap=PURPLE_CMAP)
    color_mappable.set_array([])
    colorbar = fig.colorbar(color_mappable, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_ticks([0, 0.25, 0.50, 0.75, 1.00])
    colorbar.ax.tick_params(labelsize=9, length=3)
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "model_comparison_candidate.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_dir / "model_comparison_candidate.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    """Run the multi-model candidate plotting workflow.

    Arg:
        None.

    Returns:
        None.
    """
    args = parse_args()
    row_labels, fractions, annotations = read_results(args.source_data)
    plot_heatmap(row_labels, fractions, annotations, args.output_dir)
    print(f"Wrote model comparison candidate figures to {args.output_dir}")


if __name__ == "__main__":
    main()
