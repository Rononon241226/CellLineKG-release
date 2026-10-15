"""Read the fixed benchmark cell-line scope without exposing other outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CellScopeError(ValueError):
    """Report a missing or invalid benchmark cell-line scope."""


class CellScopeReader:
    """Load the allowed DepMap ModelIDs for final benchmark answers."""

    def __init__(self, workspace_root: Path):
        """Initialize the fixed scope file location.

        Arg:
            workspace_root: Project root containing benchmark resources.

        Returns:
            None.
        """
        self.scope_path = (
            workspace_root.resolve()
            / "benchmark"
            / "baseline"
            / "Agent"
            / "resources"
            / "benchmark_scope"
            / "depmap_model_ids.csv"
        )

    def read_cell_scope(self) -> pd.Series:
        """Read the benchmark DepMap ModelIDs as a membership Series.

        Arg:
            None.

        Returns:
            Series containing unique allowed ModelIDs.
        """
        if not self.scope_path.is_file():
            raise CellScopeError(
                f"Benchmark cell scope not found: {self.scope_path}"
            )
        frame = pd.read_csv(self.scope_path, dtype=str)
        if list(frame.columns) != ["ModelID"]:
            raise CellScopeError(
                "Benchmark cell scope must contain only the ModelID column."
            )
        if frame["ModelID"].isna().any() or frame["ModelID"].str.strip().eq("").any():
            raise CellScopeError("Benchmark cell scope contains an empty ModelID.")
        normalized = frame["ModelID"].str.strip()
        if normalized.duplicated().any():
            raise CellScopeError("Benchmark cell scope contains duplicate ModelIDs.")
        return normalized.reset_index(drop=True).rename("ModelID")
