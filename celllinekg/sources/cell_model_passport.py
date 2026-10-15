from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CELL_LINES_HEADER = ["cmp_model_id", "cmp_model_name"]
PROTEIN_ABUNDANCES_HEADER = [
    "cmp_model_id",
    "cmp_model_name",
    "uniprot_id",
    "protein_symbol",
    "protein_intensity",
    "value_type",
    "z_score",
    "level",
    "rank_within_cellline",
    "threshold_rule",
    "selection_rule",
]
CELL_LINE_ID_MAPPING_HEADER = [
    "cmp_model_id",
    "id_source",
    "mapped_id",
    "mapping_type",
]
DATASETS_HEADER = ["dataset_name", "dataset_type", "source", "version"]

ZSCORE_THRESHOLD = 2.0
TOP_N = 500
THRESHOLD_RULE = "abs(z_score)>2"
SELECTION_RULE = "abs(z_score) > 2 and rank_within_cellline <= 500"
VALUE_TYPE = "protein_intensity"
DATASET_NAME = "CellModelPassport_Proteomics"
CMP_PROTEOMICS_VERSION = "Cell Model Passport Proteomics Release 2025-02-11"


def export_source_tables(
    proteomics_input_path: Path,
    output_dir: Path,
    model_list_input_path: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    model_metadata = _load_model_metadata(model_list_input_path)
    seen_cell_lines: set[str] = set()
    written_mapping_rows: set[tuple[str, str, str, str]] = set()

    with (
        _open_writer(output_dir / "cell_lines.csv", CELL_LINES_HEADER) as cell_line_writer,
        _open_writer(
            output_dir / "protein_abundances_selected.csv",
            PROTEIN_ABUNDANCES_HEADER,
        ) as abundance_writer,
        _open_writer(
            output_dir / "cell_line_id_mapping.csv",
            CELL_LINE_ID_MAPPING_HEADER,
        ) as mapping_writer,
        _open_writer(output_dir / "datasets.csv", DATASETS_HEADER) as datasets_writer,
    ):
        datasets_writer.writerow(
            {
                "dataset_name": DATASET_NAME,
                "dataset_type": "protein_abundance",
                "source": "CellModelPassport",
                "version": CMP_PROTEOMICS_VERSION,
            }
        )
        counters["datasets"] += 1
        for cmp_model_id, selected_rows in _iter_selected_proteomics_rows(
            proteomics_input_path
        ):
            cmp_model_name = _resolve_model_name(
                cmp_model_id,
                selected_rows[0]["cmp_model_name"] if selected_rows else "",
                model_metadata,
            )

            if cmp_model_id not in seen_cell_lines:
                cell_line_writer.writerow(
                    {
                        "cmp_model_id": cmp_model_id,
                        "cmp_model_name": cmp_model_name,
                    }
                )
                seen_cell_lines.add(cmp_model_id)
                counters["cell_lines"] += 1

            for mapping_row in _build_mapping_rows(
                cmp_model_id,
                cmp_model_name,
                model_metadata.get(cmp_model_id, {}),
            ):
                mapping_key = (
                    mapping_row["cmp_model_id"],
                    mapping_row["id_source"],
                    mapping_row["mapped_id"],
                    mapping_row["mapping_type"],
                )
                if mapping_key in written_mapping_rows:
                    continue
                mapping_writer.writerow(mapping_row)
                written_mapping_rows.add(mapping_key)
                counters["id_mapping_rows"] += 1

            for rank, row in enumerate(selected_rows, start=1):
                abundance_writer.writerow(
                    {
                        "cmp_model_id": cmp_model_id,
                        "cmp_model_name": cmp_model_name,
                        "uniprot_id": row["uniprot_id"],
                        "protein_symbol": row["protein_symbol"],
                        "protein_intensity": row["protein_intensity"],
                        "value_type": VALUE_TYPE,
                        "z_score": row["z_score"],
                        "level": row["level"],
                        "rank_within_cellline": str(rank),
                        "threshold_rule": THRESHOLD_RULE,
                        "selection_rule": SELECTION_RULE,
                    }
                )
                counters["protein_abundances_selected"] += 1
                counters[f"protein_abundances_{row['level']}"] += 1

            if selected_rows:
                counters["cell_lines_with_proteomics"] += 1

    summary = {
        "proteomics_input_path": str(proteomics_input_path),
        "model_list_input_path": str(model_list_input_path) if model_list_input_path else "",
        "output_dir": str(output_dir),
        "counts": dict(counters),
        "threshold_rule": THRESHOLD_RULE,
        "selection_rule": SELECTION_RULE,
        "value_type": VALUE_TYPE,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _iter_selected_proteomics_rows(path: Path):
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            cmp_model_id = (raw_row.get("model_id") or "").strip()
            if not cmp_model_id:
                continue

            z_score_text = (raw_row.get("zscore") or "").strip()
            if not z_score_text:
                continue
            try:
                z_score_value = float(z_score_text)
            except ValueError:
                continue
            if abs(z_score_value) <= ZSCORE_THRESHOLD:
                continue

            if cmp_model_id not in grouped_rows:
                grouped_rows[cmp_model_id] = []
                order.append(cmp_model_id)

            grouped_rows[cmp_model_id].append(
                {
                    "cmp_model_name": (raw_row.get("model_name") or "").strip(),
                    "uniprot_id": (raw_row.get("uniprot_id") or "").strip(),
                    "protein_symbol": (raw_row.get("symbol") or "").strip(),
                    "protein_intensity": (raw_row.get("protein_intensity") or "").strip(),
                    "z_score": z_score_text,
                    "z_score_value": z_score_value,
                    "level": "high" if z_score_value > 0 else "low",
                }
            )

    for cmp_model_id in order:
        selected_rows = grouped_rows[cmp_model_id]
        selected_rows.sort(
            key=lambda row: (
                -abs(row["z_score_value"]),
                row["z_score_value"],
                row["uniprot_id"],
                row["protein_symbol"],
            )
        )
        yield cmp_model_id, selected_rows[:TOP_N]


def _load_model_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    metadata: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cmp_model_id = (row.get("model_id") or "").strip()
            if not cmp_model_id:
                continue
            metadata[cmp_model_id] = {
                "model_name": (row.get("model_name") or "").strip(),
                "broad_id": (row.get("BROAD_ID") or "").strip(),
                "ccle_id": (row.get("CCLE_ID") or "").strip(),
                "rrid": (row.get("RRID") or "").strip(),
                "cosmic_id": _normalize_cosmic_id(row.get("COSMIC_ID")),
                "sample_id": (row.get("sample_id") or "").strip(),
                "patient_id": (row.get("patient_id") or "").strip(),
            }
    return metadata


def _resolve_model_name(
    cmp_model_id: str,
    proteomics_model_name: str,
    model_metadata: dict[str, dict[str, str]],
) -> str:
    metadata = model_metadata.get(cmp_model_id, {})
    return metadata.get("model_name", "") or proteomics_model_name


def _build_mapping_rows(
    cmp_model_id: str,
    cmp_model_name: str,
    metadata: dict[str, str],
) -> list[dict[str, str]]:
    rows = [
        {
            "cmp_model_id": cmp_model_id,
            "id_source": "CellModelPassport",
            "mapped_id": cmp_model_id,
            "mapping_type": "primary_model_id",
        }
    ]

    if cmp_model_name:
        rows.append(
            {
                "cmp_model_id": cmp_model_id,
                "id_source": "CellModelPassport",
                "mapped_id": cmp_model_name,
                "mapping_type": "model_name",
            }
        )

    optional_mappings = [
        ("sample_id", metadata.get("sample_id", ""), "sample_id"),
        ("patient_id", metadata.get("patient_id", ""), "patient_id"),
        ("DepMap", metadata.get("broad_id", ""), "depmap_model_id"),
        ("CCLE", metadata.get("ccle_id", ""), "ccle_id"),
        ("Cellosaurus", metadata.get("rrid", ""), "rrid"),
        ("COSMIC", metadata.get("cosmic_id", ""), "cosmic_id"),
    ]
    for id_source, mapped_id, mapping_type in optional_mappings:
        if not mapped_id:
            continue
        rows.append(
            {
                "cmp_model_id": cmp_model_id,
                "id_source": id_source,
                "mapped_id": mapped_id,
                "mapping_type": mapping_type,
            }
        )
    return rows


def _normalize_cosmic_id(value: str | None) -> str:
    normalized = (value or "").strip()
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


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
        description="Parse Cell Model Passports proteomics data into source-level CSV tables."
    )
    parser.add_argument(
        "--proteomics-input",
        type=Path,
        default=Path(
            "data/raw/cell_model_passport/Proteomics_20250211/proteomics_all_20250211.csv"
        ),
        help="Path to Cell Model Passports proteomics long-table CSV.",
    )
    parser.add_argument(
        "--model-list-input",
        type=Path,
        default=Path("data/raw/cell_model_passport/model_list_20260303.csv"),
        help="Optional Cell Model Passports model list CSV used for cell line ID mappings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/cell_model_passport"),
        help="Directory where source-level Cell Model Passports tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    model_list_input = args.model_list_input if args.model_list_input.exists() else None
    summary = export_source_tables(
        args.proteomics_input,
        args.output_dir,
        model_list_input_path=model_list_input,
    )
    print(
        json.dumps(
            {
                "cell_lines": summary["counts"].get("cell_lines", 0),
                "datasets": summary["counts"].get("datasets", 0),
                "protein_abundances_selected": summary["counts"].get(
                    "protein_abundances_selected", 0
                ),
                "cell_lines_with_proteomics": summary["counts"].get(
                    "cell_lines_with_proteomics", 0
                ),
                "id_mapping_rows": summary["counts"].get("id_mapping_rows", 0),
                "output_dir": summary["output_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
