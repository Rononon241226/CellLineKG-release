from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


CELL_LINES_HEADER = ["sanger_model_id", "gdsc_name", "cosmic_id"]
DRUG_SCREENS_HEADER = [
    "drug_screen_id",
    "dataset_name",
    "sanger_model_id",
    "cosmic_id",
    "gdsc_drug_id",
    "ln_ic50",
    "ic50",
    "auc",
    "z_score",
    "drug_sensitivity",
    "threshold_rule",
]
DRUGS_HEADER = ["gdsc_drug_id", "name", "synonyms", "target", "target_pathway"]
DATASETS_HEADER = ["dataset_name", "dataset_type", "source", "version"]
CELL_LINE_ID_MAPPING_HEADER = [
    "sanger_model_id",
    "id_source",
    "mapped_id",
    "mapping_type",
]

GDSC_VERSION = "GDSC Release 2023-10-27 (GDSC2)"
DRUG_SENSITIVITY_RULE = "sensitive: z_score<-2; resistant: z_score>2; otherwise: intermediate"


def export_source_tables(
    dose_response_input_path: Path,
    compounds_input_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    dose_response_df = pd.read_excel(dose_response_input_path)
    compounds_df = pd.read_csv(compounds_input_path)

    dose_response_df["SANGER_MODEL_ID"] = dose_response_df["SANGER_MODEL_ID"].map(_normalize_text)
    dose_response_df["CELL_LINE_NAME"] = dose_response_df["CELL_LINE_NAME"].map(_normalize_text)
    dose_response_df["COSMIC_ID"] = dose_response_df["COSMIC_ID"].map(_normalize_cosmic_id)
    dose_response_df["DRUG_ID"] = dose_response_df["DRUG_ID"].map(_normalize_text)
    dose_response_df["DATASET"] = dose_response_df["DATASET"].map(_normalize_text)

    compounds_df["DRUG_ID"] = compounds_df["DRUG_ID"].map(_normalize_text)
    compounds_df["DRUG_NAME"] = compounds_df["DRUG_NAME"].map(_normalize_text)
    compounds_df["SYNONYMS"] = compounds_df["SYNONYMS"].map(_normalize_text)
    compounds_df["TARGET"] = compounds_df["TARGET"].map(_normalize_text)
    compounds_df["TARGET_PATHWAY"] = compounds_df["TARGET_PATHWAY"].map(_normalize_text)

    used_drug_ids = set(dose_response_df["DRUG_ID"])
    used_drug_ids.discard("")

    with (
        _open_writer(output_dir / "cell_lines.csv", CELL_LINES_HEADER) as cell_line_writer,
        _open_writer(output_dir / "drug_screens.csv", DRUG_SCREENS_HEADER) as drug_screen_writer,
        _open_writer(output_dir / "drugs.csv", DRUGS_HEADER) as drug_writer,
        _open_writer(output_dir / "datasets.csv", DATASETS_HEADER) as dataset_writer,
        _open_writer(
            output_dir / "cell_line_id_mapping.csv",
            CELL_LINE_ID_MAPPING_HEADER,
        ) as mapping_writer,
    ):
        _write_cell_lines_and_mappings(dose_response_df, cell_line_writer, mapping_writer, counters)
        _write_drug_screens(dose_response_df, drug_screen_writer, counters)
        _write_datasets(dose_response_df, dataset_writer, counters)
        _write_drugs(compounds_df, used_drug_ids, drug_writer, counters)

    summary = {
        "dose_response_input_path": str(dose_response_input_path),
        "compounds_input_path": str(compounds_input_path),
        "output_dir": str(output_dir),
        "counts": dict(counters),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _write_cell_lines_and_mappings(
    dose_response_df: pd.DataFrame,
    cell_line_writer: csv.DictWriter,
    mapping_writer: csv.DictWriter,
    counters: Counter,
) -> None:
    seen_cell_lines: set[str] = set()
    written_mapping_rows: set[tuple[str, str, str, str]] = set()

    for row in dose_response_df[
        ["SANGER_MODEL_ID", "CELL_LINE_NAME", "COSMIC_ID"]
    ].drop_duplicates().itertuples(index=False):
        sanger_model_id = row.SANGER_MODEL_ID
        if not sanger_model_id or sanger_model_id in seen_cell_lines:
            continue

        gdsc_name = row.CELL_LINE_NAME
        cosmic_id = row.COSMIC_ID
        cell_line_writer.writerow(
            {
                "sanger_model_id": sanger_model_id,
                "gdsc_name": gdsc_name,
                "cosmic_id": cosmic_id,
            }
        )
        seen_cell_lines.add(sanger_model_id)
        counters["cell_lines"] += 1

        mapping_rows = [
            {
                "sanger_model_id": sanger_model_id,
                "id_source": "GDSC",
                "mapped_id": sanger_model_id,
                "mapping_type": "primary_model_id",
            }
        ]
        if gdsc_name:
            mapping_rows.append(
                {
                    "sanger_model_id": sanger_model_id,
                    "id_source": "GDSC",
                    "mapped_id": gdsc_name,
                    "mapping_type": "model_name",
                }
            )
        if cosmic_id:
            mapping_rows.append(
                {
                    "sanger_model_id": sanger_model_id,
                    "id_source": "COSMIC",
                    "mapped_id": cosmic_id,
                    "mapping_type": "cosmic_id",
                }
            )

        for mapping_row in mapping_rows:
            mapping_key = (
                mapping_row["sanger_model_id"],
                mapping_row["id_source"],
                mapping_row["mapped_id"],
                mapping_row["mapping_type"],
            )
            if mapping_key in written_mapping_rows:
                continue
            mapping_writer.writerow(mapping_row)
            written_mapping_rows.add(mapping_key)
            counters["cell_line_id_mapping_rows"] += 1


def _write_drug_screens(
    dose_response_df: pd.DataFrame,
    drug_screen_writer: csv.DictWriter,
    counters: Counter,
) -> None:
    for row in dose_response_df.itertuples(index=False):
        dataset_name = _normalize_text(row.DATASET)
        sanger_model_id = _normalize_text(row.SANGER_MODEL_ID)
        cosmic_id = _normalize_cosmic_id(row.COSMIC_ID)
        gdsc_drug_id = _normalize_text(row.DRUG_ID)
        drug_screen_writer.writerow(
            {
                "drug_screen_id": build_drug_screen_id(
                    dataset_name,
                    sanger_model_id,
                    cosmic_id,
                    gdsc_drug_id,
                ),
                "dataset_name": dataset_name,
                "sanger_model_id": sanger_model_id,
                "cosmic_id": cosmic_id,
                "gdsc_drug_id": gdsc_drug_id,
                "ln_ic50": _normalize_numeric(row.LN_IC50),
                "ic50": _exp_numeric(row.LN_IC50),
                "auc": _normalize_numeric(row.AUC),
                "z_score": _normalize_numeric(row.Z_SCORE),
                "drug_sensitivity": _classify_drug_sensitivity(row.Z_SCORE),
                "threshold_rule": DRUG_SENSITIVITY_RULE,
            }
        )
        counters["drug_screens"] += 1


def _write_datasets(
    dose_response_df: pd.DataFrame,
    dataset_writer: csv.DictWriter,
    counters: Counter,
) -> None:
    seen_datasets: set[str] = set()
    for dataset_name in dose_response_df["DATASET"].drop_duplicates():
        normalized_name = _normalize_text(dataset_name)
        if not normalized_name or normalized_name in seen_datasets:
            continue
        dataset_writer.writerow(
            {
                "dataset_name": normalized_name,
                "dataset_type": "drug_response",
                "source": "GDSC",
                "version": GDSC_VERSION,
            }
        )
        seen_datasets.add(normalized_name)
        counters["datasets"] += 1


def _write_drugs(
    compounds_df: pd.DataFrame,
    used_drug_ids: set[str],
    drug_writer: csv.DictWriter,
    counters: Counter,
) -> None:
    seen_drug_ids: set[str] = set()
    for row in compounds_df.itertuples(index=False):
        gdsc_drug_id = _normalize_text(row.DRUG_ID)
        if not gdsc_drug_id or gdsc_drug_id not in used_drug_ids or gdsc_drug_id in seen_drug_ids:
            continue
        drug_writer.writerow(
            {
                "gdsc_drug_id": gdsc_drug_id,
                "name": _normalize_text(row.DRUG_NAME),
                "synonyms": _normalize_text(row.SYNONYMS),
                "target": _normalize_text(row.TARGET),
                "target_pathway": _normalize_text(row.TARGET_PATHWAY),
            }
        )
        seen_drug_ids.add(gdsc_drug_id)
        counters["drugs"] += 1


def build_drug_screen_id(
    dataset_name: str,
    sanger_model_id: str,
    cosmic_id: str,
    gdsc_drug_id: str,
) -> str:
    record_key = "|".join(
        [
            dataset_name.strip().lower(),
            cosmic_id.strip(),
            gdsc_drug_id.strip(),
        ]
    )
    return f"drug_screen::{sanger_model_id.strip()}::{record_key}"


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _normalize_cosmic_id(value) -> str:
    normalized = _normalize_text(value)
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def _normalize_numeric(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _exp_numeric(value) -> str:
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return str(math.exp(numeric))


def _classify_drug_sensitivity(value) -> str:
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric < -2:
        return "sensitive"
    if numeric > 2:
        return "resistant"
    return "intermediate"


def _open_writer(path: Path, fieldnames: list[str]):
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return _CsvWriterContext(handle, writer)


class _CsvWriterContext:
    def __init__(self, handle, writer: csv.DictWriter):
        self.handle = handle
        self.writer = writer

    def __enter__(self) -> csv.DictWriter:
        return self.writer

    def __exit__(self, exc_type, exc, tb) -> None:
        self.handle.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse GDSC source files into source-level CSV tables."
    )
    parser.add_argument(
        "--dose-response-input",
        type=Path,
        default=Path("data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"),
        help="Path to GDSC fitted dose-response Excel file.",
    )
    parser.add_argument(
        "--compounds-input",
        type=Path,
        default=Path("data/raw/gdsc/screened_compounds_rel_8.5.csv"),
        help="Path to GDSC screened compounds CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/gdsc"),
        help="Directory where source-level GDSC tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = export_source_tables(
        args.dose_response_input,
        args.compounds_input,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "cell_lines": summary["counts"].get("cell_lines", 0),
                "drug_screens": summary["counts"].get("drug_screens", 0),
                "drugs": summary["counts"].get("drugs", 0),
                "datasets": summary["counts"].get("datasets", 0),
                "cell_line_id_mapping_rows": summary["counts"].get(
                    "cell_line_id_mapping_rows", 0
                ),
                "output_dir": summary["output_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
