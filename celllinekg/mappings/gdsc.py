from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


CELL_LINE_NODE_HEADER = ["name", "sanger_model_id", "cosmic_id", "gdsc_name"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRUG_EXTERNAL_IDS_PATH = PROJECT_ROOT / "data" / "curation" / "gdsc_drug_external_ids.csv"
DRUG_NODE_HEADER = [
    "gdsc_drug_id",
    "name",
    "synonyms",
    "mychem_id",
    "external_aliases",
    "target",
    "target_pathway",
]
DRUG_SCREEN_NODE_HEADER = [
    "drug_screen_id",
    "ln_ic50",
    "ic50",
    "auc",
    "z_score",
    "drug_sensitivity",
    "threshold_rule",
]
DATASET_NODE_HEADER = ["dataset_id", "name", "type", "source", "version"]
CELL_LINE_DRUG_SCREEN_EDGE_HEADER = [
    "source_sanger_model_id",
    "target_drug_screen_id",
    "relation",
]
DRUG_SCREEN_DRUG_EDGE_HEADER = ["source_drug_screen_id", "target_drug_id", "relation"]
DRUG_SCREEN_DATASET_EDGE_HEADER = [
    "source_drug_screen_id",
    "target_dataset_id",
    "relation",
]
CELL_LINE_DATASET_EDGE_HEADER = ["source_sanger_model_id", "target_dataset_id", "relation"]
CELL_LINE_ID_MAPPING_HEADER = [
    "sanger_model_id",
    "id_source",
    "mapped_id",
    "mapping_type",
]


@dataclass
class SourceTables:
    cell_lines: dict[str, dict[str, str]]
    drug_screens: dict[str, list[dict[str, str]]]
    drugs: dict[str, dict[str, str]]
    datasets: dict[str, dict[str, str]]
    id_mappings: dict[str, list[dict[str, str]]]


def run_mapping(source_dir: Path, schema_output_dir: Path) -> dict[str, object]:
    schema_output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_source_tables(source_dir)
    drug_external_ids = _load_drug_external_ids(DRUG_EXTERNAL_IDS_PATH)
    retained_drug_ids = {
        drug_id
        for drug_id, drug_row in tables.drugs.items()
        if drug_row.get("target", "").strip()
    }

    counters = Counter()
    written_dataset_edges: set[tuple[str, str]] = set()

    with ExitStack() as stack:
        cell_line_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_cellline.csv", CELL_LINE_NODE_HEADER)
        )
        drug_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_drug.csv", DRUG_NODE_HEADER)
        )
        drug_screen_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_drug_screen.csv", DRUG_SCREEN_NODE_HEADER)
        )
        dataset_writer = stack.enter_context(
            _open_writer(schema_output_dir / "nodes_dataset.csv", DATASET_NODE_HEADER)
        )
        cell_line_drug_screen_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_cellline_has_drug_screen.csv",
                CELL_LINE_DRUG_SCREEN_EDGE_HEADER,
            )
        )
        drug_screen_drug_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_drug_screen_tested_drug.csv",
                DRUG_SCREEN_DRUG_EDGE_HEADER,
            )
        )
        drug_screen_dataset_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_drug_screen_from_dataset.csv",
                DRUG_SCREEN_DATASET_EDGE_HEADER,
            )
        )
        cell_line_dataset_writer = stack.enter_context(
            _open_writer(
                schema_output_dir / "edges_cellline_in_dataset.csv",
                CELL_LINE_DATASET_EDGE_HEADER,
            )
        )
        mapping_writer = stack.enter_context(
            _open_writer(schema_output_dir / "cell_line_id_mapping.csv", CELL_LINE_ID_MAPPING_HEADER)
        )

        for dataset_name, dataset_row in tables.datasets.items():
            dataset_id = build_dataset_id(dataset_name)
            dataset_writer.writerow(
                {
                    "dataset_id": dataset_id,
                    "name": dataset_row["dataset_name"],
                    "type": dataset_row["dataset_type"],
                    "source": dataset_row["source"],
                    "version": dataset_row["version"],
                }
            )
            counters["nodes_dataset"] += 1

        for gdsc_drug_id, drug_row in tables.drugs.items():
            if gdsc_drug_id not in retained_drug_ids:
                counters["filtered_drugs_missing_target"] += 1
                continue
            drug_writer.writerow(
                _build_drug_node_row(
                    drug_row,
                    drug_external_ids.get(gdsc_drug_id, {}),
                )
            )
            counters["nodes_drug"] += 1

        for sanger_model_id, cell_line_row in tables.cell_lines.items():
            cell_line_writer.writerow(
                {
                    "name": cell_line_row["gdsc_name"],
                    "sanger_model_id": sanger_model_id,
                    "cosmic_id": cell_line_row["cosmic_id"],
                    "gdsc_name": cell_line_row["gdsc_name"],
                }
            )
            counters["nodes_cellline"] += 1

            for screen_row in tables.drug_screens.get(sanger_model_id, []):
                drug_screen_id = screen_row["drug_screen_id"]
                dataset_id = build_dataset_id(screen_row["dataset_name"])
                gdsc_drug_id = screen_row["gdsc_drug_id"]
                if gdsc_drug_id not in retained_drug_ids:
                    counters["filtered_drug_screens_missing_drug_target"] += 1
                    continue

                drug_screen_writer.writerow(
                    {
                        "drug_screen_id": drug_screen_id,
                        "ln_ic50": screen_row["ln_ic50"],
                        "ic50": screen_row["ic50"],
                        "auc": screen_row["auc"],
                        "z_score": screen_row["z_score"],
                        "drug_sensitivity": screen_row["drug_sensitivity"],
                        "threshold_rule": screen_row["threshold_rule"],
                    }
                )
                counters["nodes_drug_screen"] += 1

                cell_line_drug_screen_writer.writerow(
                    {
                        "source_sanger_model_id": sanger_model_id,
                        "target_drug_screen_id": drug_screen_id,
                        "relation": "has_drug_screen",
                    }
                )
                counters["edges_cellline_has_drug_screen"] += 1

                drug_screen_drug_writer.writerow(
                    {
                        "source_drug_screen_id": drug_screen_id,
                        "target_drug_id": gdsc_drug_id,
                        "relation": "tested_drug",
                    }
                )
                counters["edges_drug_screen_tested_drug"] += 1

                drug_screen_dataset_writer.writerow(
                    {
                        "source_drug_screen_id": drug_screen_id,
                        "target_dataset_id": dataset_id,
                        "relation": "from_dataset",
                    }
                )
                counters["edges_drug_screen_from_dataset"] += 1

                dataset_pair = (sanger_model_id, dataset_id)
                if dataset_pair not in written_dataset_edges:
                    cell_line_dataset_writer.writerow(
                        {
                            "source_sanger_model_id": sanger_model_id,
                            "target_dataset_id": dataset_id,
                            "relation": "in_dataset",
                        }
                    )
                    written_dataset_edges.add(dataset_pair)
                    counters["edges_cellline_in_dataset"] += 1

            for mapping_row in tables.id_mappings.get(sanger_model_id, []):
                mapping_writer.writerow(mapping_row)
                counters["cell_line_id_mapping_rows"] += 1

    summary = {
        "input_dir": str(source_dir),
        "output_dir": str(schema_output_dir),
        "counts": dict(counters),
        "relation_types": [
            "has_drug_screen",
            "tested_drug",
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
        row["sanger_model_id"]: row
        for row in _read_csv_rows(source_dir / "cell_lines.csv")
    }
    drug_screens = _group_structured_rows(
        source_dir / "drug_screens.csv",
        [
            "drug_screen_id",
            "dataset_name",
            "cosmic_id",
            "gdsc_drug_id",
            "ln_ic50",
            "ic50",
            "auc",
            "z_score",
            "drug_sensitivity",
            "threshold_rule",
        ],
    )
    drugs = {
        row["gdsc_drug_id"]: row
        for row in _read_csv_rows(source_dir / "drugs.csv")
    }
    datasets = {
        row["dataset_name"]: row
        for row in _read_csv_rows(source_dir / "datasets.csv")
    }
    id_mappings = _group_structured_rows(
        source_dir / "cell_line_id_mapping.csv",
        ["sanger_model_id", "id_source", "mapped_id", "mapping_type"],
    )

    return SourceTables(
        cell_lines=cell_lines,
        drug_screens=drug_screens,
        drugs=drugs,
        datasets=datasets,
        id_mappings=id_mappings,
    )


def _load_drug_external_ids(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {
        row["gdsc_drug_id"]: row
        for row in _read_csv_rows(path)
    }


def _build_drug_node_row(
    drug_row: dict[str, str],
    external_ids: dict[str, str],
) -> dict[str, str]:
    mychem_id = external_ids.get("mychem_id", "").strip()
    pubchem_cid = external_ids.get("pubchem_cid", "").strip()
    chembl_id = external_ids.get("chembl_id", "").strip()
    external_aliases = _join_pipe_list(
        [
            f"PUBCHEM:{pubchem_cid}" if pubchem_cid else "",
            f"CHEMBL:{chembl_id}" if chembl_id else "",
        ]
    )
    if not (mychem_id or pubchem_cid or chembl_id):
        external_aliases = _join_pipe_list(
            [external_aliases, _build_target_alias(drug_row.get("target", ""))]
        )

    return {
        "gdsc_drug_id": drug_row["gdsc_drug_id"],
        "name": drug_row["name"],
        "synonyms": drug_row["synonyms"],
        "mychem_id": mychem_id,
        "external_aliases": external_aliases,
        "target": drug_row["target"],
        "target_pathway": drug_row["target_pathway"],
    }


def _build_target_alias(target: str) -> str:
    target = target.strip()
    if not target:
        return ""
    if _looks_like_non_protein_mechanism(target):
        return target
    parts = [part.strip() for part in re.split(r"\s*,\s*", target) if part.strip()]
    if not parts:
        return ""
    return f"{'/'.join(parts)} inhibitor"


def _looks_like_non_protein_mechanism(target: str) -> bool:
    normalized = target.strip().lower()
    mechanism_terms = (
        "antimetabolite",
        "crosslinker",
        "destabiliser",
        "stabiliser",
        "retinoic acid",
        "break induction",
        "g-quadruplex",
        "metabolism",
        "anti-oxidant",
        "inflammatory",
    )
    return any(term in normalized for term in mechanism_terms)


def _join_pipe_list(values: list[str]) -> str:
    seen: set[str] = set()
    aliases: list[str] = []
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        aliases.append(value)
    return "||".join(aliases)


def build_dataset_id(name: str) -> str:
    return name.strip().lower()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_structured_rows(path: Path, fields: list[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        grouped[row["sanger_model_id"]].append({field: row[field] for field in fields})
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
        description="Map GDSC source-level CSV tables into schema-level CSV tables."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/intermediate/normalized/gdsc"),
        help="Directory containing source-level GDSC tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/schema/gdsc"),
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
                "nodes_drug": summary["counts"].get("nodes_drug", 0),
                "nodes_drug_screen": summary["counts"].get("nodes_drug_screen", 0),
                "nodes_dataset": summary["counts"].get("nodes_dataset", 0),
                "edges_cellline_has_drug_screen": summary["counts"].get(
                    "edges_cellline_has_drug_screen", 0
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
