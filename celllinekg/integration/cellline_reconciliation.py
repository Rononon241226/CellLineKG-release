from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


RECONCILIATION_HEADER = [
    "canonical_cellline_id",
    "source_database",
    "source_cellline_id",
    "source_name",
    "match_confidence",
    "matched_by",
    "matched_identifiers",
    "cluster_size",
]

SOURCE_PRIORITY = {
    "Cellosaurus": 0,
    "DepMap": 1,
    "GDSC": 2,
    "CellModelPassport": 3,
}


@dataclass(frozen=True)
class EntityKey:
    source_database: str
    source_cellline_id: str


@dataclass
class EntityRecord:
    key: EntityKey
    source_name: str
    identifiers: dict[str, set[str]]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[EntityKey, EntityKey] = {}

    def add(self, item: EntityKey) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: EntityKey) -> EntityKey:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: EntityKey, right: EntityKey) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def run_reconciliation(
    cellosaurus_mapping_path: Path,
    depmap_mapping_path: Path,
    gdsc_mapping_path: Path,
    cmp_mapping_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    entities = _load_entities(
        cellosaurus_mapping_path,
        depmap_mapping_path,
        gdsc_mapping_path,
        cmp_mapping_path,
    )

    union_find = UnionFind()
    for entity_key in entities:
        union_find.add(entity_key)

    identifier_index: dict[tuple[str, str], list[EntityKey]] = defaultdict(list)
    for entity_key, entity in entities.items():
        for identifier_type, values in entity.identifiers.items():
            for value in values:
                identifier_index[(identifier_type, value)].append(entity_key)

    matched_identifier_counter = Counter()
    union_identifier_types = {"cellosaurus_accession", "depmap_model_id", "sidm_model_id"}
    for key, entity_keys in identifier_index.items():
        if len(entity_keys) < 2:
            continue
        matched_identifier_counter[key[0]] += 1
        if key[0] not in union_identifier_types:
            continue
        anchor = entity_keys[0]
        for other in entity_keys[1:]:
            union_find.union(anchor, other)

    clusters: dict[EntityKey, list[EntityKey]] = defaultdict(list)
    for entity_key in entities:
        clusters[union_find.find(entity_key)].append(entity_key)

    counters = Counter()
    clean_clusters: list[tuple[str, list[EntityKey], dict[str, list[str]]]] = []
    with _open_writer(output_dir / "cell_line_reconciliation.csv", RECONCILIATION_HEADER) as writer:
        for cluster_entities in sorted(
            clusters.values(),
            key=lambda items: (_cluster_priority(items), _cluster_anchor_value(items)),
        ):
            cluster_entities.sort(key=_entity_sort_key)
            canonical_cellline_id = _build_canonical_id(cluster_entities)
            evidence_by_type = _collect_cluster_evidence(cluster_entities, entities)
            matched_by = " || ".join(sorted(evidence_by_type)) if evidence_by_type else "unmatched"
            matched_identifiers = _format_matched_identifiers(evidence_by_type)
            match_confidence = "high" if evidence_by_type else "unmatched"
            cluster_size = len(cluster_entities)

            counters["clusters"] += 1
            if cluster_size == 1:
                counters["singleton_clusters"] += 1
            else:
                counters["matched_clusters"] += 1
                counters[f"cluster_size_{cluster_size}"] += 1

            if _is_clean_cluster(cluster_entities):
                clean_clusters.append((canonical_cellline_id, cluster_entities, evidence_by_type))
                counters["clean_clusters"] += 1
                counters[f"clean_cluster_size_{cluster_size}"] += 1

            for entity_key in cluster_entities:
                entity = entities[entity_key]
                writer.writerow(
                    {
                        "canonical_cellline_id": canonical_cellline_id,
                        "source_database": entity_key.source_database,
                        "source_cellline_id": entity_key.source_cellline_id,
                        "source_name": entity.source_name,
                        "match_confidence": match_confidence,
                        "matched_by": matched_by,
                        "matched_identifiers": matched_identifiers,
                        "cluster_size": str(cluster_size),
                    }
                )
                counters["rows"] += 1
                counters[f"rows_{entity_key.source_database.lower()}"] += 1

    with _open_writer(output_dir / "cell_line_reconciliation_clean.csv", RECONCILIATION_HEADER) as writer:
        for canonical_cellline_id, cluster_entities, evidence_by_type in clean_clusters:
            matched_by = " || ".join(sorted(evidence_by_type)) if evidence_by_type else "unmatched"
            matched_identifiers = _format_matched_identifiers(evidence_by_type)
            cluster_size = len(cluster_entities)
            for entity_key in cluster_entities:
                entity = entities[entity_key]
                writer.writerow(
                    {
                        "canonical_cellline_id": canonical_cellline_id,
                        "source_database": entity_key.source_database,
                        "source_cellline_id": entity_key.source_cellline_id,
                        "source_name": entity.source_name,
                        "match_confidence": "high" if evidence_by_type else "unmatched",
                        "matched_by": matched_by,
                        "matched_identifiers": matched_identifiers,
                        "cluster_size": str(cluster_size),
                    }
                )
                counters["clean_rows"] += 1
                counters[f"clean_rows_{entity_key.source_database.lower()}"] += 1

    summary = {
        "input_files": {
            "cellosaurus": str(cellosaurus_mapping_path),
            "depmap": str(depmap_mapping_path),
            "gdsc": str(gdsc_mapping_path),
            "cell_model_passport": str(cmp_mapping_path),
        },
        "output_dir": str(output_dir),
        "counts": dict(counters),
        "matched_identifier_types": dict(matched_identifier_counter),
        "matching_rules": [
            "Cellosaurus accession exact match (union)",
            "DepMap model ID exact match (union)",
            "SIDM model ID exact match between GDSC and Cell Model Passports (union)",
            "COSMIC ID exact match (evidence only)",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _load_entities(
    cellosaurus_mapping_path: Path,
    depmap_mapping_path: Path,
    gdsc_mapping_path: Path,
    cmp_mapping_path: Path,
) -> dict[EntityKey, EntityRecord]:
    entities: dict[EntityKey, EntityRecord] = {}
    _load_cellosaurus_entities(cellosaurus_mapping_path, entities)
    _load_depmap_entities(depmap_mapping_path, entities)
    _load_gdsc_entities(gdsc_mapping_path, entities)
    _load_cmp_entities(cmp_mapping_path, entities)
    return entities


def _load_cellosaurus_entities(path: Path, entities: dict[EntityKey, EntityRecord]) -> None:
    for row in _read_csv_rows(path):
        key = EntityKey("Cellosaurus", row["cellosaurus_id"])
        entity = entities.setdefault(
            key,
            EntityRecord(key=key, source_name=row.get("name", ""), identifiers=defaultdict(set)),
        )
        if row.get("name"):
            entity.source_name = row["name"]
        mapped_id_source = (row.get("mapped_id_source") or "").strip()
        mapped_id = (row.get("mapped_id") or "").strip()
        if mapped_id_source == "Cellosaurus" and mapped_id:
            entity.identifiers["cellosaurus_accession"].add(mapped_id.upper())
        if mapped_id_source == "COSMIC" and mapped_id:
            entity.identifiers["cosmic_id"].add(mapped_id)


def _load_depmap_entities(path: Path, entities: dict[EntityKey, EntityRecord]) -> None:
    for row in _read_csv_rows(path):
        key = EntityKey("DepMap", row["depmap_model_id"])
        entity = entities.setdefault(
            key,
            EntityRecord(key=key, source_name="", identifiers=defaultdict(set)),
        )
        mapped_id = (row.get("mapped_id") or "").strip()
        id_source = (row.get("id_source") or "").strip()
        mapping_type = (row.get("mapping_type") or "").strip()
        if mapping_type == "primary_model_id" and mapped_id:
            entity.identifiers["depmap_model_id"].add(mapped_id.upper())
        elif mapping_type == "rrid" and mapped_id:
            entity.identifiers["cellosaurus_accession"].add(mapped_id.upper())
        elif mapping_type == "cosmic_id" and mapped_id:
            entity.identifiers["cosmic_id"].add(mapped_id)
        elif mapping_type == "model_name" and mapped_id:
            entity.source_name = mapped_id
        elif id_source == "DepMap" and mapped_id:
            entity.identifiers["depmap_model_id"].add(mapped_id.upper())


def _load_gdsc_entities(path: Path, entities: dict[EntityKey, EntityRecord]) -> None:
    for row in _read_csv_rows(path):
        key = EntityKey("GDSC", row["sanger_model_id"])
        entity = entities.setdefault(
            key,
            EntityRecord(key=key, source_name="", identifiers=defaultdict(set)),
        )
        mapped_id = (row.get("mapped_id") or "").strip()
        mapping_type = (row.get("mapping_type") or "").strip()
        if mapping_type == "primary_model_id" and mapped_id:
            entity.identifiers["sidm_model_id"].add(mapped_id.upper())
        elif mapping_type == "model_name" and mapped_id:
            entity.source_name = mapped_id
        elif mapping_type == "cosmic_id" and mapped_id:
            entity.identifiers["cosmic_id"].add(mapped_id)


def _load_cmp_entities(path: Path, entities: dict[EntityKey, EntityRecord]) -> None:
    for row in _read_csv_rows(path):
        key = EntityKey("CellModelPassport", row["cmp_model_id"])
        entity = entities.setdefault(
            key,
            EntityRecord(key=key, source_name="", identifiers=defaultdict(set)),
        )
        mapped_id = (row.get("mapped_id") or "").strip()
        id_source = (row.get("id_source") or "").strip()
        mapping_type = (row.get("mapping_type") or "").strip()
        if mapping_type == "primary_model_id" and mapped_id:
            entity.identifiers["sidm_model_id"].add(mapped_id.upper())
        elif mapping_type == "model_name" and mapped_id:
            entity.source_name = mapped_id
        elif mapping_type == "depmap_model_id" and mapped_id:
            entity.identifiers["depmap_model_id"].add(mapped_id.upper())
        elif mapping_type == "rrid" and mapped_id:
            entity.identifiers["cellosaurus_accession"].add(mapped_id.upper())
        elif mapping_type == "cosmic_id" and mapped_id:
            entity.identifiers["cosmic_id"].add(mapped_id)
        elif id_source == "CellModelPassport" and mapped_id and mapped_id.upper().startswith("SIDM"):
            entity.identifiers["sidm_model_id"].add(mapped_id.upper())


def _collect_cluster_evidence(
    cluster_entities: list[EntityKey],
    entities: dict[EntityKey, EntityRecord],
) -> dict[str, list[str]]:
    values_by_type: dict[str, list[str]] = defaultdict(list)
    for entity_key in cluster_entities:
        for identifier_type, values in entities[entity_key].identifiers.items():
            values_by_type[identifier_type].extend(values)

    evidence: dict[str, list[str]] = {}
    for identifier_type, values in values_by_type.items():
        counts = Counter(values)
        matched_values = sorted(value for value, count in counts.items() if count > 1)
        if matched_values:
            evidence[identifier_type] = matched_values
    return evidence


def _format_matched_identifiers(evidence_by_type: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for identifier_type in sorted(evidence_by_type):
        for value in evidence_by_type[identifier_type]:
            parts.append(f"{identifier_type}:{value}")
    return " || ".join(parts)


def _build_canonical_id(cluster_entities: list[EntityKey]) -> str:
    anchor = min(cluster_entities, key=_entity_sort_key)
    return f"{anchor.source_database}:{anchor.source_cellline_id}"


def _is_clean_cluster(cluster_entities: list[EntityKey]) -> bool:
    source_counts = Counter(entity_key.source_database for entity_key in cluster_entities)
    return all(count == 1 for count in source_counts.values())


def _cluster_priority(cluster_entities: list[EntityKey]) -> tuple[int, str]:
    anchor = min(cluster_entities, key=_entity_sort_key)
    return SOURCE_PRIORITY.get(anchor.source_database, 99), anchor.source_cellline_id


def _cluster_anchor_value(cluster_entities: list[EntityKey]) -> str:
    return min(cluster_entities, key=_entity_sort_key).source_cellline_id


def _entity_sort_key(entity_key: EntityKey) -> tuple[int, str, str]:
    return (
        SOURCE_PRIORITY.get(entity_key.source_database, 99),
        entity_key.source_database,
        entity_key.source_cellline_id,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        description="Reconcile cell line identifiers across Cellosaurus, DepMap, GDSC, and Cell Model Passports."
    )
    parser.add_argument(
        "--cellosaurus-mapping",
        type=Path,
        default=Path("data/intermediate/schema/cellosaurus/cell_line_id_mapping.csv"),
        help="Path to Cellosaurus cell_line_id_mapping.csv.",
    )
    parser.add_argument(
        "--depmap-mapping",
        type=Path,
        default=Path("data/intermediate/schema/depmap/cell_line_id_mapping.csv"),
        help="Path to DepMap cell_line_id_mapping.csv.",
    )
    parser.add_argument(
        "--gdsc-mapping",
        type=Path,
        default=Path("data/intermediate/schema/gdsc/cell_line_id_mapping.csv"),
        help="Path to GDSC cell_line_id_mapping.csv.",
    )
    parser.add_argument(
        "--cmp-mapping",
        type=Path,
        default=Path("data/intermediate/schema/cell_model_passport/cell_line_id_mapping.csv"),
        help="Path to Cell Model Passports cell_line_id_mapping.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/integration/cellline_reconciliation"),
        help="Directory where reconciliation outputs will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_reconciliation(
        args.cellosaurus_mapping,
        args.depmap_mapping,
        args.gdsc_mapping,
        args.cmp_mapping,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "clusters": summary["counts"].get("clusters", 0),
                "matched_clusters": summary["counts"].get("matched_clusters", 0),
                "rows": summary["counts"].get("rows", 0),
                "clean_clusters": summary["counts"].get("clean_clusters", 0),
                "clean_rows": summary["counts"].get("clean_rows", 0),
                "rows_cellosaurus": summary["counts"].get("rows_cellosaurus", 0),
                "rows_depmap": summary["counts"].get("rows_depmap", 0),
                "rows_gdsc": summary["counts"].get("rows_gdsc", 0),
                "rows_cellmodelpassport": summary["counts"].get("rows_cellmodelpassport", 0),
                "output_dir": summary["output_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
