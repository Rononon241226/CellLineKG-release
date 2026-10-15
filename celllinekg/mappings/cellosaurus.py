from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CELLLINE_NODE_HEADER = [
    "name",
    "cellosaurus_id",
    "synonyms",
    "cross_reference",
    "reference_pmid",
    "species",
    "hierarchy",
    "Same_individual",
    "category",
]

DISEASE_NODE_HEADER = ["name", "disease_source", "disease_source_id"]

CELLLINE_DISEASE_EDGE_HEADER = [
    "source_cellosaurus_id",
    "target_disease_id",
    "relation",
]

CELLLINE_EDGE_HEADER = [
    "source_cellosaurus_id",
    "target_cellosaurus_id",
    "relation",
]

CELLLINE_ID_MAPPING_HEADER = [
    "name",
    "cellosaurus_id",
    "mapped_id_source",
    "mapped_id",
    "mapping_type",
]


@dataclass
class SchemaRows:
    cellline_node: dict[str, str]
    disease_nodes: list[dict[str, str]]
    disease_edges: list[dict[str, str]]
    parent_edges: list[dict[str, str]]
    same_donor_edges: list[dict[str, str]]
    id_mapping_rows: list[dict[str, str]]
    disease_name_conflicts: int = 0


@dataclass
class SourceTables:
    cell_lines: dict[str, dict[str, str]]
    synonyms: dict[str, list[str]]
    secondary_accessions: dict[str, list[str]]
    cross_references: dict[str, list[dict[str, str]]]
    references: dict[str, list[dict[str, str]]]
    diseases: dict[str, list[dict[str, str]]]
    species: dict[str, list[dict[str, str]]]
    hierarchy: dict[str, list[dict[str, str]]]
    same_individual: dict[str, list[dict[str, str]]]


def run_mapping(source_dir: Path, schema_output_dir: Path) -> dict[str, object]:
    schema_output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_source_tables(source_dir)

    schema_counters = Counter()
    disease_nodes_by_id: dict[str, dict[str, str]] = {}
    disease_name_conflicts = 0

    with _open_writer(schema_output_dir / "nodes_cellline.csv", CELLLINE_NODE_HEADER) as schema_cellline_writer, \
        _open_writer(schema_output_dir / "nodes_disease.csv", DISEASE_NODE_HEADER) as schema_disease_writer, \
        _open_writer(schema_output_dir / "edges_cellline_has_disease.csv", CELLLINE_DISEASE_EDGE_HEADER) as schema_disease_edge_writer, \
        _open_writer(schema_output_dir / "edges_has_parent.csv", CELLLINE_EDGE_HEADER) as schema_parent_edge_writer, \
        _open_writer(schema_output_dir / "edges_same_donor_as.csv", CELLLINE_EDGE_HEADER) as schema_same_donor_edge_writer, \
        _open_writer(schema_output_dir / "cell_line_id_mapping.csv", CELLLINE_ID_MAPPING_HEADER) as schema_id_mapping_writer:

        for cellosaurus_id, base_row in tables.cell_lines.items():
            schema_rows = map_source_row_to_schema_rows(
                cellosaurus_id=cellosaurus_id,
                base_row=base_row,
                synonyms=tables.synonyms.get(cellosaurus_id, []),
                secondary_accessions=tables.secondary_accessions.get(cellosaurus_id, []),
                cross_references=tables.cross_references.get(cellosaurus_id, []),
                references=tables.references.get(cellosaurus_id, []),
                diseases=tables.diseases.get(cellosaurus_id, []),
                species=tables.species.get(cellosaurus_id, []),
                hierarchy=tables.hierarchy.get(cellosaurus_id, []),
                same_individual=tables.same_individual.get(cellosaurus_id, []),
                disease_nodes_by_id=disease_nodes_by_id,
            )
            disease_name_conflicts += schema_rows.disease_name_conflicts

            schema_cellline_writer.writerow(schema_rows.cellline_node)
            schema_counters["nodes_cellline"] += 1

            for disease_node in schema_rows.disease_nodes:
                schema_disease_writer.writerow(disease_node)
                schema_counters["nodes_disease"] += 1

            for disease_edge in schema_rows.disease_edges:
                schema_disease_edge_writer.writerow(disease_edge)
                schema_counters["edges_cellline_has_disease"] += 1

            for parent_edge in schema_rows.parent_edges:
                schema_parent_edge_writer.writerow(parent_edge)
                schema_counters["edges_has_parent"] += 1

            for same_donor_edge in schema_rows.same_donor_edges:
                schema_same_donor_edge_writer.writerow(same_donor_edge)
                schema_counters["edges_same_donor_as"] += 1

            for mapping_row in schema_rows.id_mapping_rows:
                schema_id_mapping_writer.writerow(mapping_row)
                schema_counters["cell_line_id_mapping_rows"] += 1

    summary = {
        "input_dir": str(source_dir),
        "output_dir": str(schema_output_dir),
        "counts": dict(schema_counters),
        "multi_value_format": "Delimited strings in CSV cells (values separated by ' || ', structured pairs by '::')",
        "disease_name_conflicts": disease_name_conflicts,
        "relation_types": ["has_disease", "has_parent", "same_donor_as"],
    }
    (schema_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def load_source_tables(source_dir: Path) -> SourceTables:
    cell_lines = {
        row["cellosaurus_id"]: row
        for row in _read_csv_rows(source_dir / "cell_lines.csv")
    }
    synonyms = _group_simple_rows(source_dir / "cell_line_synonyms.csv", "synonym")
    secondary_accessions = _group_simple_rows(
        source_dir / "cell_line_secondary_accessions.csv", "secondary_accession"
    )
    cross_references = _group_structured_rows(
        source_dir / "cell_line_cross_references.csv", ["resource", "identifier"]
    )
    references = _group_structured_rows(
        source_dir / "cell_line_references.csv", ["source", "identifier"]
    )
    diseases = _group_structured_rows(
        source_dir / "cell_line_diseases.csv", ["resource", "identifier", "name"]
    )
    species = _group_structured_rows(
        source_dir / "cell_line_species.csv", ["taxonomy_id", "name"]
    )
    hierarchy = _group_structured_rows(
        source_dir / "cell_line_hierarchy.csv", ["parent_cellosaurus_id", "parent_name"]
    )
    same_individual = _group_structured_rows(
        source_dir / "cell_line_same_individual.csv", ["related_cellosaurus_id", "related_name"]
    )

    return SourceTables(
        cell_lines=cell_lines,
        synonyms=synonyms,
        secondary_accessions=secondary_accessions,
        cross_references=cross_references,
        references=references,
        diseases=diseases,
        species=species,
        hierarchy=hierarchy,
        same_individual=same_individual,
    )


def map_source_row_to_schema_rows(
    *,
    cellosaurus_id: str,
    base_row: dict[str, str],
    synonyms: list[str],
    secondary_accessions: list[str],
    cross_references: list[dict[str, str]],
    references: list[dict[str, str]],
    diseases: list[dict[str, str]],
    species: list[dict[str, str]],
    hierarchy: list[dict[str, str]],
    same_individual: list[dict[str, str]],
    disease_nodes_by_id: dict[str, dict[str, str]],
) -> SchemaRows:
    disease_rows: list[dict[str, str]] = []
    disease_edges: list[dict[str, str]] = []
    parent_edges: list[dict[str, str]] = []
    same_donor_edges: list[dict[str, str]] = []
    disease_name_conflicts = 0

    cellline_node = {
        "name": base_row["primary_name"],
        "cellosaurus_id": cellosaurus_id,
        "synonyms": _join_strings(synonyms),
        "cross_reference": _join_cross_references(cross_references),
        "reference_pmid": _join_strings(_extract_pubmed_ids(references)),
        "species": _join_species(species),
        "hierarchy": _join_hierarchy(hierarchy),
        "Same_individual": _join_same_individual(same_individual),
        "category": base_row["category"],
    }

    id_mapping_rows = [
        {
            "name": base_row["primary_name"],
            "cellosaurus_id": cellosaurus_id,
            "mapped_id_source": "Cellosaurus",
            "mapped_id": cellosaurus_id,
            "mapping_type": "primary_accession",
        }
    ]

    for secondary_accession in secondary_accessions:
        id_mapping_rows.append(
            {
                "name": base_row["primary_name"],
                "cellosaurus_id": cellosaurus_id,
                "mapped_id_source": "Cellosaurus",
                "mapped_id": secondary_accession,
                "mapping_type": "secondary_accession",
            }
        )

    for cross_reference in cross_references:
        id_mapping_rows.append(
            {
                "name": base_row["primary_name"],
                "cellosaurus_id": cellosaurus_id,
                "mapped_id_source": cross_reference["resource"],
                "mapped_id": cross_reference["identifier"],
                "mapping_type": "cross_reference",
            }
        )

    for disease in diseases:
        disease_id = build_disease_id(disease["resource"], disease["identifier"])
        disease_node = {
            "name": disease["name"],
            "disease_source": disease["resource"],
            "disease_source_id": disease["identifier"],
        }
        existing_disease = disease_nodes_by_id.get(disease_id)
        if existing_disease is None:
            disease_nodes_by_id[disease_id] = disease_node
            disease_rows.append(disease_node)
        elif existing_disease["name"] != disease["name"]:
            disease_name_conflicts += 1

        disease_edges.append(
            {
                "source_cellosaurus_id": cellosaurus_id,
                "target_disease_id": disease_id,
                "relation": "has_disease",
            }
        )

    for relation in hierarchy:
        parent_edges.append(
            {
                "source_cellosaurus_id": cellosaurus_id,
                "target_cellosaurus_id": relation["parent_cellosaurus_id"],
                "relation": "has_parent",
            }
        )

    for relation in same_individual:
        same_donor_edges.append(
            {
                "source_cellosaurus_id": cellosaurus_id,
                "target_cellosaurus_id": relation["related_cellosaurus_id"],
                "relation": "same_donor_as",
            }
        )

    return SchemaRows(
        cellline_node=cellline_node,
        disease_nodes=disease_rows,
        disease_edges=disease_edges,
        parent_edges=parent_edges,
        same_donor_edges=same_donor_edges,
        id_mapping_rows=id_mapping_rows,
        disease_name_conflicts=disease_name_conflicts,
    )


def build_disease_id(resource: str, identifier: str) -> str:
    return f"{resource}:{identifier}"


def _extract_pubmed_ids(references: list[dict[str, str]]) -> list[str]:
    return [
        reference["identifier"]
        for reference in references
        if reference.get("source") == "PubMed" and reference.get("identifier")
    ]


def _join_strings(values: list[str]) -> str:
    return " || ".join(value for value in values if value)


def _join_cross_references(cross_references: list[dict[str, str]]) -> str:
    values = [
        f"{item['resource']}::{item['identifier']}"
        for item in cross_references
        if item.get("resource") and item.get("identifier")
    ]
    return _join_strings(values)


def _join_species(species_items: list[dict[str, str]]) -> str:
    values = [
        f"{item['taxonomy_id']}::{item['name']}"
        for item in species_items
        if item.get("taxonomy_id") and item.get("name")
    ]
    return _join_strings(values)


def _join_hierarchy(relations: list[dict[str, str]]) -> str:
    values = [
        f"{item['parent_cellosaurus_id']}::{item['parent_name']}"
        for item in relations
        if item.get("parent_cellosaurus_id") and item.get("parent_name")
    ]
    return _join_strings(values)


def _join_same_individual(relations: list[dict[str, str]]) -> str:
    values = [
        f"{item['related_cellosaurus_id']}::{item['related_name']}"
        for item in relations
        if item.get("related_cellosaurus_id") and item.get("related_name")
    ]
    return _join_strings(values)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_simple_rows(path: Path, value_key: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in _read_csv_rows(path):
        grouped[row["cellosaurus_id"]].append(row[value_key])
    return dict(grouped)


def _group_structured_rows(path: Path, fields: list[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        grouped[row["cellosaurus_id"]].append({field: row[field] for field in fields})
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

    def __exit__(self, exc_type, exc, tb) -> None:   #报错类型 报错内容 报错位置
        self.handle.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map Cellosaurus source-level CSV tables into schema-level CSV tables."
    )   #创建空的命令接收器
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/intermediate/normalized/cellosaurus"),
        help="Directory containing source-level Cellosaurus tables.",
    )    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/schema/cellosaurus"),
        help="Directory where schema-level tables will be written.",
    )     #添加命令
    return parser    #args = parser.parse_args()可以用来读输入的信息


def main() -> int:    #打印结果
    args = build_arg_parser().parse_args()
    summary = run_mapping(args.source_dir, args.output_dir)
    print(
        json.dumps(
            {
                "source_dir": summary["input_dir"],
                "output_dir": summary["output_dir"],
                "nodes_cellline": summary["counts"].get("nodes_cellline", 0),
                "nodes_disease": summary["counts"].get("nodes_disease", 0),
                "edges_cellline_has_disease": summary["counts"].get(
                    "edges_cellline_has_disease", 0
                ),
            },
            indent=2,  #缩进两格
            ensure_ascii=False,  #支持中文
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
