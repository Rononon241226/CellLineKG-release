from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


CELL_LINE_NODE_HEADER = ["name", "cmp_model_id"]
PROTEIN_NODE_HEADER = ["protein_id", "uniprot_id", "protein_symbol", "display_name"]
PROTEIN_ABUNDANCE_NODE_HEADER = [
    "protein_abundance_id",
    "value",
    "value_type",
    "z_score",
    "level",
    "rank_within_cellline",
    "threshold_rule",
    "selection_rule",
]
CELL_LINE_PROTEIN_ABUNDANCE_EDGE_HEADER = [
    "source_cmp_model_id",
    "target_protein_abundance_id",
    "relation",
]
PROTEIN_ABUNDANCE_PROTEIN_EDGE_HEADER = [
    "source_protein_abundance_id",
    "target_protein_id",
    "relation",
]
CELL_LINE_ID_MAPPING_HEADER = [
    "cmp_model_id",
    "id_source",
    "mapped_id",
    "mapping_type",
]
DATASET_NODE_HEADER = ["dataset_id", "name", "type", "source", "version"]
PROTEIN_ABUNDANCE_DATASET_EDGE_HEADER = ["source_protein_abundance_id", "target_dataset_id", "relation"]
CELL_LINE_DATASET_EDGE_HEADER = ["source_cmp_model_id", "target_dataset_id", "relation"]


@dataclass
class SourceTables:
    cell_lines: dict[str, dict[str, str]]
    protein_abundances: dict[str, list[dict[str, str]]]
    datasets: dict[str, dict[str, str]]
    id_mappings: dict[str, list[dict[str, str]]]


def run_mapping(source_dir: Path, schema_output_dir: Path) -> dict[str, object]:
    schema_output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_source_tables(source_dir)

    counters = Counter()
    seen_proteins: set[str] = set()
    seen_protein_abundances: set[str] = set()
    seen_abundance_protein: set[tuple[str, str]] = set()
    seen_abundance_dataset: set[tuple[str, str]] = set()
    seen_cellline_dataset: set[tuple[str, str]] = set()

    with ExitStack() as stack:
        cell_line_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_cellline.csv", CELL_LINE_NODE_HEADER)
        )
        protein_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_protein.csv", PROTEIN_NODE_HEADER)
        )
        protein_abundance_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "nodes_protein_abundance.csv",
                PROTEIN_ABUNDANCE_NODE_HEADER,
            )
        )
        dataset_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_dataset.csv", DATASET_NODE_HEADER)
        )
        cellline_abundance_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_cellline_has_protein_abundance.csv",
                CELL_LINE_PROTEIN_ABUNDANCE_EDGE_HEADER,
            )
        )
        abundance_protein_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_protein_abundance_measured_protein.csv",
                PROTEIN_ABUNDANCE_PROTEIN_EDGE_HEADER,
            )
        )
        abundance_dataset_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_protein_abundance_from_dataset.csv",
                PROTEIN_ABUNDANCE_DATASET_EDGE_HEADER,
            )
        )
        cellline_dataset_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_cellline_in_dataset.csv",
                CELL_LINE_DATASET_EDGE_HEADER,
            )
        )
        mapping_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "cell_line_id_mapping.csv",
                CELL_LINE_ID_MAPPING_HEADER,
            )
        )

        dataset_ids = {dataset_name: build_name_id(dataset_name) for dataset_name in tables.datasets}
        for dataset_name, dataset_row in tables.datasets.items():
            dataset_writer.writerow(
                {
                    "dataset_id": dataset_ids[dataset_name],
                    "name": dataset_row["dataset_name"],
                    "type": dataset_row["dataset_type"],
                    "source": dataset_row["source"],
                    "version": dataset_row["version"],
                }
            )
            counters["nodes_dataset"] += 1

        for cmp_model_id, base_row in tables.cell_lines.items():
            cell_line_writer.writerow(
                {
                    "name": base_row["cmp_model_name"],
                    "cmp_model_id": cmp_model_id,
                }
            )
            counters["nodes_cellline"] += 1

            dataset_id = next(iter(dataset_ids.values()), "")
            if dataset_id:
                pair = (cmp_model_id, dataset_id)
                if pair not in seen_cellline_dataset:
                    cellline_dataset_writer.writerow(
                        {
                            "source_cmp_model_id": cmp_model_id,
                            "target_dataset_id": dataset_id,
                            "relation": "in_dataset",
                        }
                    )
                    seen_cellline_dataset.add(pair)
                    counters["edges_cellline_in_dataset"] += 1

            for abundance in tables.protein_abundances.get(cmp_model_id, []):
                protein_id = build_protein_id(
                    abundance["uniprot_id"],
                    abundance["protein_symbol"],
                )
                if not protein_id:
                    continue

                if protein_id not in seen_proteins:
                    protein_writer.writerow(
                        {
                            "protein_id": protein_id,
                            "uniprot_id": abundance["uniprot_id"],
                            "protein_symbol": abundance["protein_symbol"],
                            "display_name": abundance["protein_symbol"] or abundance["uniprot_id"],
                        }
                    )
                    seen_proteins.add(protein_id)
                    counters["nodes_protein"] += 1

                protein_abundance_id = build_protein_abundance_id(
                    cmp_model_id,
                    protein_id,
                )
                if protein_abundance_id not in seen_protein_abundances:
                    protein_abundance_writer.writerow(
                        {
                            "protein_abundance_id": protein_abundance_id,
                            "value": abundance["protein_intensity"],
                            "value_type": abundance["value_type"],
                            "z_score": abundance["z_score"],
                            "level": abundance["level"],
                            "rank_within_cellline": abundance["rank_within_cellline"],
                            "threshold_rule": abundance["threshold_rule"],
                            "selection_rule": abundance["selection_rule"],
                        }
                    )
                    seen_protein_abundances.add(protein_abundance_id)
                    counters["nodes_protein_abundance"] += 1

                cellline_abundance_writer.writerow(
                    {
                        "source_cmp_model_id": cmp_model_id,
                        "target_protein_abundance_id": protein_abundance_id,
                        "relation": "has_protein_abundance",
                    }
                )
                counters["edges_cellline_has_protein_abundance"] += 1

                pair = (protein_abundance_id, protein_id)
                if pair not in seen_abundance_protein:
                    abundance_protein_writer.writerow(
                        {
                            "source_protein_abundance_id": protein_abundance_id,
                            "target_protein_id": protein_id,
                            "relation": "measured_protein",
                        }
                    )
                    seen_abundance_protein.add(pair)
                    counters["edges_protein_abundance_measured_protein"] += 1

                if dataset_id:
                    dataset_pair = (protein_abundance_id, dataset_id)
                    if dataset_pair not in seen_abundance_dataset:
                        abundance_dataset_writer.writerow(
                            {
                                "source_protein_abundance_id": protein_abundance_id,
                                "target_dataset_id": dataset_id,
                                "relation": "from_dataset",
                            }
                        )
                        seen_abundance_dataset.add(dataset_pair)
                        counters["edges_protein_abundance_from_dataset"] += 1

            for mapping_row in tables.id_mappings.get(cmp_model_id, []):
                mapping_writer.writerow(mapping_row)
                counters["cell_line_id_mapping_rows"] += 1

    summary = {
        "input_dir": str(source_dir),
        "output_dir": str(schema_output_dir),
        "counts": dict(counters),
        "relation_types": [
            "has_protein_abundance",
            "measured_protein",
            "from_dataset",
            "in_dataset",
        ],
    }
    (schema_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def load_source_tables(source_dir: Path) -> SourceTables:
    cell_lines = {
        row["cmp_model_id"]: row
        for row in _read_csv_rows(source_dir / "cell_lines.csv")
    }
    protein_abundances = _group_structured_rows(
        source_dir / "protein_abundances_selected.csv",
        [
            "uniprot_id",
            "protein_symbol",
            "protein_intensity",
            "value_type",
            "z_score",
            "level",
            "rank_within_cellline",
            "threshold_rule",
            "selection_rule",
        ],
    )
    datasets = {
        row["dataset_name"]: row
        for row in _read_csv_rows(source_dir / "datasets.csv")
    }
    id_mappings = _group_structured_rows(
        source_dir / "cell_line_id_mapping.csv",
        ["cmp_model_id", "id_source", "mapped_id", "mapping_type"],
    )

    return SourceTables(
        cell_lines=cell_lines,
        protein_abundances=protein_abundances,
        datasets=datasets,
        id_mappings=id_mappings,
    )


def build_protein_id(uniprot_id: str | None, protein_symbol: str | None) -> str:
    normalized_uniprot_id = (uniprot_id or "").strip()
    if normalized_uniprot_id:
        return normalized_uniprot_id
    normalized_symbol = (protein_symbol or "").strip()
    if normalized_symbol:
        return normalized_symbol.lower()
    return ""


def build_protein_abundance_id(
    cmp_model_id: str,
    protein_id: str,
) -> str:
    return f"protein_abundance::{cmp_model_id.strip()}::{protein_id.strip()}"


def build_name_id(name: str | None) -> str:
    return (name or "").strip().lower()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_structured_rows(path: Path, fields: list[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        grouped[row["cmp_model_id"]].append({field: row[field] for field in fields})
    return dict(grouped)


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
        description="Map Cell Model Passports source-level CSV tables into schema-level CSV tables."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/intermediate/normalized/cell_model_passport"),
        help="Directory containing source-level Cell Model Passports tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/schema/cell_model_passport"),
        help="Directory where schema-level tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_mapping(args.source_dir, args.output_dir)
    print(
        json.dumps(
            {
                "source_dir": summary["input_dir"],
                "output_dir": summary["output_dir"],
                "nodes_cellline": summary["counts"].get("nodes_cellline", 0),
                "nodes_dataset": summary["counts"].get("nodes_dataset", 0),
                "nodes_protein": summary["counts"].get("nodes_protein", 0),
                "nodes_protein_abundance": summary["counts"].get(
                    "nodes_protein_abundance", 0
                ),
                "edges_cellline_has_protein_abundance": summary["counts"].get(
                    "edges_cellline_has_protein_abundance", 0
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
