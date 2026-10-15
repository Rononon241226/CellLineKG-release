"""Read raw project data through a strict path boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_DATA_SUFFIXES = {".csv", ".json", ".txt", ".xls", ".xlsx"}


class DataAccessError(ValueError):
    """Report an invalid or disallowed raw-data access request."""


class DataAccess:
    """Provide read-only access to files below the project data directory."""

    def __init__(self, workspace_root: Path):
        """Initialize the data access boundary.

        Arg:
            workspace_root: Project root containing the data directory.

        Returns:
            None.
        """
        self.workspace_root = workspace_root.resolve()
        self.data_root = (self.workspace_root / "data" / "raw").resolve()
        if not self.data_root.is_dir():
            raise DataAccessError(f"Raw data directory not found: {self.data_root}")

    def list_data_files(self) -> list[str]:
        """List readable raw-data files without recommending a file.

        Arg:
            None.

        Returns:
            Project-relative paths below the data directory.
        """
        paths = [
            path.relative_to(self.workspace_root).as_posix()
            for path in self.data_root.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_DATA_SUFFIXES
        ]
        return sorted(paths, key=str.casefold)

    def read_csv(self, relative_path: str, **kwargs: Any) -> pd.DataFrame:
        """Read one CSV file below the data directory.

        Arg:
            relative_path: Project-relative path beginning with data/.
            **kwargs: Keyword arguments forwarded to pandas.read_csv.

        Returns:
            DataFrame loaded from the requested CSV file.
        """
        path = self._resolve_file(relative_path, {".csv"})
        return pd.read_csv(path, **kwargs)

    def read_excel(self, relative_path: str, **kwargs: Any) -> pd.DataFrame:
        """Read one Excel file below the data directory.

        Arg:
            relative_path: Project-relative path beginning with data/.
            **kwargs: Keyword arguments forwarded to pandas.read_excel.

        Returns:
            DataFrame loaded from the requested Excel file.
        """
        path = self._resolve_file(relative_path, {".xls", ".xlsx"})
        return pd.read_excel(path, **kwargs)

    def read_json(self, relative_path: str) -> Any:
        """Read one JSON file below the data directory.

        Arg:
            relative_path: Project-relative path beginning with data/.

        Returns:
            Parsed JSON-compatible Python value.
        """
        path = self._resolve_file(relative_path, {".json"})
        return json.loads(path.read_text(encoding="utf-8"))

    def read_text(
        self,
        relative_path: str,
        start_line: int = 1,
        max_lines: int = 200,
    ) -> str:
        """Read a bounded line range from one text file.

        Arg:
            relative_path: Project-relative path beginning with data/.
            start_line: One-based first line to return.
            max_lines: Maximum number of lines to return.

        Returns:
            Requested text lines joined by newline characters.
        """
        if start_line < 1:
            raise DataAccessError("start_line must be at least 1.")
        if not 1 <= max_lines <= 1000:
            raise DataAccessError("max_lines must be between 1 and 1000.")

        path = self._resolve_file(relative_path, {".txt"})
        selected: list[str] = []
        end_line = start_line + max_lines
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < start_line:
                    continue
                if line_number >= end_line:
                    break
                selected.append(line.rstrip("\r\n"))
        return "\n".join(selected)

    def search_text(
        self,
        relative_path: str,
        pattern: str,
        *,
        use_regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 100,
    ) -> list[dict[str, Any]]:
        """Search a chosen text file for matching lines.

        Arg:
            relative_path: Project-relative path beginning with data/.
            pattern: Literal text or regular expression selected by the caller.
            use_regex: Interpret pattern as a regular expression when true.
            case_sensitive: Preserve case during matching when true.
            max_matches: Maximum number of matching lines to return.

        Returns:
            Matching line numbers and text.
        """
        if not pattern:
            raise DataAccessError("pattern must not be empty.")
        if not 1 <= max_matches <= 1000:
            raise DataAccessError("max_matches must be between 1 and 1000.")

        path = self._resolve_file(relative_path, {".txt"})
        flags = 0 if case_sensitive else re.IGNORECASE
        expression = re.compile(pattern if use_regex else re.escape(pattern), flags)
        matches: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if expression.search(line):
                    matches.append(
                        {
                            "line_number": line_number,
                            "text": line.rstrip("\r\n"),
                        }
                    )
                    if len(matches) >= max_matches:
                        break
        return matches

    def _resolve_file(self, relative_path: str, suffixes: set[str]) -> Path:
        """Resolve and validate one requested raw-data path.

        Arg:
            relative_path: Project-relative path beginning with data/.
            suffixes: Allowed file suffixes for this reader.

        Returns:
            Resolved absolute file path inside the data directory.
        """
        requested = Path(str(relative_path))
        if requested.is_absolute():
            raise DataAccessError("Absolute paths are not allowed.")

        candidate = (self.workspace_root / requested).resolve()
        try:
            candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise DataAccessError("Only files below the data directory are allowed.") from exc

        if candidate.suffix.lower() not in suffixes:
            allowed = ", ".join(sorted(suffixes))
            raise DataAccessError(f"Expected one of these file types: {allowed}")
        if not candidate.is_file():
            raise DataAccessError(f"Data file not found: {relative_path}")
        return candidate
