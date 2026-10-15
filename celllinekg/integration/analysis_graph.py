from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NODE_HEADERS = {
    "nodes_cellline.csv": [
        "cellosaurus_id",
        "name",
        "synonyms",
        "tumor_origin",
        "growth_pattern",
    ],
    "nodes_disease.csv": ["disease_id", "name", "oncotree_disease_code", "ncit_disease_id"],
    "nodes_subtype.csv": ["subtype_id", "name", "oncotree_subtype_code", "ncit_subtype_id"],
    "nodes_lineage.csv": ["lineage_id", "name"],
    "nodes_media.csv": ["media_id", "formulation"],
    "nodes_gene.csv": ["gene_id", "entrez_id", "symbol", "display_name"],
    "nodes_variant.csv": ["variant_id", "variant_type", "protein_change", "rs_id"],
    "nodes_copy_number.csv": ["copy_number_id", "status", "value"],
    "nodes_dependency.csv": ["dependency_id", "score", "rank_within_cellline"],
    "nodes_gene_expression.csv": ["expression_id", "z_score", "level", "rank_within_cellline"],
    "nodes_drug.csv": [
        "drug_id",
        "name",
        "synonyms",
        "mychem_id",
        "external_aliases",
        "target",
        "target_pathway",
    ],
    "nodes_drug_screen.csv": ["drug_screen_id", "ic50", "auc", "z_score", "drug_sensitivity"],
    "nodes_protein.csv": ["protein_id", "uniprot_id", "symbol", "display_name"],
    "nodes_protein_abundance.csv": ["protein_abundance_id", "z_score", "level", "rank_within_cellline"],
}

EDGE_HEADERS = {
    "edges_cellline_has_disease.csv": ["source_cellosaurus_id", "target_disease_id", "relation", "source"],
    "edges_cellline_has_subtype.csv": ["source_cellosaurus_id", "target_subtype_id", "relation", "source"],
    "edges_cellline_has_lineage.csv": ["source_cellosaurus_id", "target_lineage_id", "relation", "source"],
    "edges_cellline_cultures_in.csv": ["source_cellosaurus_id", "target_media_id", "relation", "source"],
    "edges_subtype_of.csv": ["source_subtype_id", "target_disease_id", "relation", "source"],
    "edges_disease_part_of.csv": ["source_disease_id", "target_lineage_id", "relation", "source"],
    "edges_cellline_has_variant.csv": ["source_cellosaurus_id", "target_variant_id", "relation", "source"],
    "edges_variant_in_gene.csv": ["source_variant_id", "target_gene_id", "relation", "source"],
    "edges_cellline_has_copy_number.csv": ["source_cellosaurus_id", "target_copy_number_id", "relation", "source"],
    "edges_copy_number_target_gene.csv": ["source_copy_number_id", "target_gene_id", "relation", "source"],
    "edges_cellline_has_dependency.csv": ["source_cellosaurus_id", "target_dependency_id", "relation", "source"],
    "edges_dependency_target_gene.csv": ["source_dependency_id", "target_gene_id", "relation", "source"],
    "edges_cellline_has_expression.csv": ["source_cellosaurus_id", "target_expression_id", "relation", "source"],
    "edges_expression_measured_gene.csv": ["source_expression_id", "target_gene_id", "relation", "source"],
    "edges_cellline_has_drug_screen.csv": ["source_cellosaurus_id", "target_drug_screen_id", "relation", "source"],
    "edges_drug_screen_tested_drug.csv": ["source_drug_screen_id", "target_drug_id", "relation", "source"],
    "edges_cellline_has_protein_abundance.csv": ["source_cellosaurus_id", "target_protein_abundance_id", "relation", "source"],
    "edges_protein_abundance_measured_protein.csv": ["source_protein_abundance_id", "target_protein_id", "relation", "source"],
    "edges_protein_encoded_by_gene.csv": ["source_protein_id", "target_gene_id", "relation", "source"],
    "edges_drug_targets_protein.csv": ["source_drug_id", "target_protein_id", "relation", "source"],
}

DOWNSTREAM_SOURCES = {"DepMap", "GDSC", "CellModelPassport"}
EXCLUDED_DATASET_IDS = {"depmap_model"}


def build_analysis_subset(
    reconciliation_clean_path: Path,
    cellosaurus_nodes_path: Path,
    cellosaurus_source_path: Path,
    cellosaurus_schema_dir: Path,
    depmap_schema_dir: Path,
    gdsc_schema_dir: Path,
    cmp_schema_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    nodes_dir = output_dir / "nodes"
    edges_dir = output_dir / "edges"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)
    for output_path in list(nodes_dir.glob("*.csv")) + list(edges_dir.glob("*.csv")):
        output_path.unlink()

    human_cellosaurus = _load_human_cellosaurus_ids(cellosaurus_nodes_path)
    cancer_cellosaurus = _load_cellosaurus_ids_by_category(cellosaurus_source_path, "Cancer cell line")
    selected_clusters = _select_analysis_clusters(
        _load_clusters(reconciliation_clean_path),
        human_cellosaurus,
        cancer_cellosaurus,
    )
    selected_records = _build_selected_records(selected_clusters)
    selected_records = _filter_records_with_dataset_coverage(
        selected_records,
        depmap_schema_dir,
        gdsc_schema_dir,
        cmp_schema_dir,
    )
    selected_records = _filter_depmap_non_cancerous_records(
        selected_records,
        depmap_schema_dir,
    )
    graph = _build_unified_graph(
        cellosaurus_schema_dir,
        depmap_schema_dir,
        gdsc_schema_dir,
        cmp_schema_dir,
        selected_records,
    )

    for filename, headers in NODE_HEADERS.items():
        _write_csv(nodes_dir / filename, headers, graph.nodes[filename])
    for filename, headers in EDGE_HEADERS.items():
        _write_csv(edges_dir / filename, headers, graph.edges[filename])

    summary = {
        "inputs": {
            "reconciliation_clean": str(reconciliation_clean_path),
            "cellosaurus_nodes": str(cellosaurus_nodes_path),
            "cellosaurus_source": str(cellosaurus_source_path),
            "cellosaurus_schema_dir": str(cellosaurus_schema_dir),
            "depmap_schema_dir": str(depmap_schema_dir),
            "gdsc_schema_dir": str(gdsc_schema_dir),
            "cell_model_passport_schema_dir": str(cmp_schema_dir),
        },
        "output_dir": str(output_dir),
        "selection_rule": "human Cellosaurus Cancer cell lines present in at least one real downstream dataset; clean reconciliation only; exclude DepMap Non-Cancerous disease labels",
        "selected_cellline_count": len(selected_records),
        "source_coverage": _build_source_coverage(selected_records),
        "counts": {
            "nodes": {filename.replace('.csv', ''): len(rows) for filename, rows in graph.nodes.items()},
            "edges": {filename.replace('.csv', ''): len(rows) for filename, rows in graph.edges.items()},
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_unified_graph(
    cellosaurus_schema_dir: Path,
    depmap_schema_dir: Path,
    gdsc_schema_dir: Path,
    cmp_schema_dir: Path,
    selected_records: dict[str, dict[str, str]],
) -> "_GraphAccumulator":
    graph = _GraphAccumulator()
    selected_cellosaurus_ids = set(selected_records)
    depmap_to_cellosaurus = {v["depmap_model_id"]: k for k, v in selected_records.items() if v["depmap_model_id"]}
    gdsc_to_cellosaurus = {v["sanger_model_id"]: k for k, v in selected_records.items() if v["sanger_model_id"]}
    cmp_to_cellosaurus = {v["cmp_model_id"]: k for k, v in selected_records.items() if v["cmp_model_id"]}

    subtype_ids: set[str] = set()
    lineage_ids: set[str] = set()
    media_ids: set[str] = set()
    gene_ids: set[str] = set()
    variant_ids: set[str] = set()
    copy_number_ids: set[str] = set()
    dependency_ids: set[str] = set()
    expression_ids: set[str] = set()
    drug_ids: set[str] = set()
    drug_screen_ids: set[str] = set()
    protein_ids: set[str] = set()
    protein_abundance_ids: set[str] = set()

    cellosaurus_nodes = _row_map(cellosaurus_schema_dir / "nodes_cellline.csv", "cellosaurus_id")
    depmap_cellline_rows = _row_map(depmap_schema_dir / "nodes_cellline.csv", "depmap_model_id")
    disease_book = _DiseaseBook(graph)

    for cellosaurus_id, record in sorted(selected_records.items()):
        row = cellosaurus_nodes[cellosaurus_id]
        depmap_row = depmap_cellline_rows.get(record["depmap_model_id"], {})
        graph.add_node("nodes_cellline.csv", cellosaurus_id, {
            "cellosaurus_id": cellosaurus_id,
            "name": row["name"],
            "synonyms": row["synonyms"],
            "tumor_origin": depmap_row.get("tumor_origin", ""),
            "growth_pattern": depmap_row.get("growth_pattern", ""),
        })

    depmap_disease_rows = {row["name"].strip().lower(): row for row in _read_rows(depmap_schema_dir / "nodes_disease.csv")}
    depmap_subtype_rows = {_depmap_subtype_id(row): row for row in _read_rows(depmap_schema_dir / "nodes_subtype.csv")}
    depmap_lineage_rows = {row["name"].strip().lower(): row for row in _read_rows(depmap_schema_dir / "nodes_lineage.csv")}
    depmap_media_rows = _row_map(depmap_schema_dir / "nodes_media.csv", "media_id")
    depmap_gene_rows = _row_map(depmap_schema_dir / "nodes_gene.csv", "gene_id")
    depmap_variant_rows = _row_map(depmap_schema_dir / "nodes_variant.csv", "variant_id")
    depmap_copy_number_rows = _row_map(depmap_schema_dir / "nodes_copy_number.csv", "copy_number_id")
    depmap_dependency_rows = _row_map(depmap_schema_dir / "nodes_dependency.csv", "dependency_id")
    depmap_expression_rows = _row_map(depmap_schema_dir / "nodes_gene_expression.csv", "expression_id")
    for row in _read_rows(depmap_schema_dir / "edges_cellline_has_disease.csv"):
        source_id = row["source_depmap_model_id"]
        if source_id not in depmap_to_cellosaurus:
            continue
        disease_row = depmap_disease_rows[row["target_disease_id"]]
        disease_book.add(
            cellosaurus_id=depmap_to_cellosaurus[source_id],
            disease_name=disease_row["name"],
            oncotree_disease_code=disease_row.get("oncotree_disease_code", ""),
            ncit_disease_id=disease_row.get("ncit_disease_id", ""),
        )

    edge_specs = [
        ("edges_cellline_has_subtype.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_subtype_id", subtype_ids),
        ("edges_cellline_has_lineage.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_lineage_id", lineage_ids),
        ("edges_cellline_cultures_in.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_media_id", media_ids),
        ("edges_cellline_has_variant.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_variant_id", variant_ids),
        ("edges_cellline_has_copy_number.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_copy_number_id", copy_number_ids),
        ("edges_cellline_has_dependency.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_dependency_id", dependency_ids),
        ("edges_cellline_has_expression.csv", "source_depmap_model_id", depmap_to_cellosaurus, "target_expression_id", expression_ids),
    ]
    for filename, source_field, mapping, target_field, id_set in edge_specs:
        _rewrite_cellline_edge_file(graph, depmap_schema_dir / filename, source_field, mapping, target_field, id_set)

    for row in _read_rows(depmap_schema_dir / "edges_subtype_of.csv"):
        if row["source_subtype_id"] in subtype_ids and _build_disease_id_from_name(depmap_disease_rows[row["target_disease_id"]]["name"]) in disease_book.ids:
            graph.add_edge("edges_subtype_of.csv", (row["source_subtype_id"], _build_disease_id_from_name(depmap_disease_rows[row["target_disease_id"]]["name"])), {
                "source_subtype_id": row["source_subtype_id"],
                "target_disease_id": _build_disease_id_from_name(depmap_disease_rows[row["target_disease_id"]]["name"]),
                "relation": row["relation"],
                "source": "DepMap",
            })
    for row in _read_rows(depmap_schema_dir / "edges_disease_part_of.csv"):
        unified_disease_id = _build_disease_id_from_name(depmap_disease_rows[row["source_disease_id"]]["name"])
        if unified_disease_id in disease_book.ids and row["target_lineage_id"] in lineage_ids:
            graph.add_edge("edges_disease_part_of.csv", (unified_disease_id, row["target_lineage_id"]), {
                "source_disease_id": unified_disease_id,
                "target_lineage_id": row["target_lineage_id"],
                "relation": row["relation"],
                "source": "DepMap",
            })
    for row in _read_rows(depmap_schema_dir / "edges_variant_in_gene.csv"):
        if row["source_variant_id"] in variant_ids:
            gene_ids.add(row["target_gene_id"])
            graph.add_edge("edges_variant_in_gene.csv", (row["source_variant_id"], row["target_gene_id"]), _with_source(row, "DepMap"))
    for row in _read_rows(depmap_schema_dir / "edges_copy_number_target_gene.csv"):
        if row["source_copy_number_id"] in copy_number_ids:
            gene_ids.add(row["target_gene_id"])
            graph.add_edge("edges_copy_number_target_gene.csv", (row["source_copy_number_id"], row["target_gene_id"]), _with_source(row, "DepMap"))
    for row in _read_rows(depmap_schema_dir / "edges_dependency_target_gene.csv"):
        if row["source_dependency_id"] in dependency_ids:
            gene_ids.add(row["target_gene_id"])
            graph.add_edge("edges_dependency_target_gene.csv", (row["source_dependency_id"], row["target_gene_id"]), _with_source(row, "DepMap"))
    for row in _read_rows(depmap_schema_dir / "edges_expression_measured_gene.csv"):
        if row["source_expression_id"] in expression_ids:
            gene_ids.add(row["target_gene_id"])
            graph.add_edge("edges_expression_measured_gene.csv", (row["source_expression_id"], row["target_gene_id"]), _with_source(row, "DepMap"))

    for subtype_id in sorted(subtype_ids):
        row = depmap_subtype_rows[subtype_id]
        graph.add_node(
            "nodes_subtype.csv",
            subtype_id,
            {
                "subtype_id": subtype_id,
                "name": row["name"],
                "oncotree_subtype_code": _get_oncotree_subtype_code(row),
                "ncit_subtype_id": row.get("ncit_subtype_id", ""),
            },
        )
    for lineage_id in sorted(lineage_ids):
        row = depmap_lineage_rows[lineage_id]
        graph.add_node("nodes_lineage.csv", lineage_id, {"lineage_id": lineage_id, "name": row["name"]})
    for media_id in sorted(media_ids):
        row = depmap_media_rows[media_id]
        graph.add_node("nodes_media.csv", media_id, {"media_id": media_id, "formulation": row["formulation"]})
    for gene_id in sorted(gene_ids):
        row = depmap_gene_rows[gene_id]
        graph.add_node("nodes_gene.csv", gene_id, {
            "gene_id": gene_id,
            "entrez_id": row["entrez_id"],
            "symbol": row["gene_symbol"],
            "display_name": row.get("display_name", "") or row["gene_symbol"],
        })
    for variant_id in sorted(variant_ids):
        row = depmap_variant_rows[variant_id]
        graph.add_node("nodes_variant.csv", variant_id, {
            "variant_id": variant_id,
            "variant_type": row["variant_type"],
            "protein_change": row["protein_change"],
            "rs_id": row.get("rs_id", ""),
        })
    for copy_number_id in sorted(copy_number_ids):
        row = depmap_copy_number_rows[copy_number_id]
        graph.add_node("nodes_copy_number.csv", copy_number_id, {
            "copy_number_id": copy_number_id,
            "status": row["status"],
            "value": row["value"],
        })
    for dependency_id in sorted(dependency_ids):
        row = depmap_dependency_rows[dependency_id]
        graph.add_node("nodes_dependency.csv", dependency_id, {
            "dependency_id": dependency_id,
            "score": row["score"],
            "rank_within_cellline": row["rank_within_cellline"],
        })
    for expression_id in sorted(expression_ids):
        row = depmap_expression_rows[expression_id]
        graph.add_node("nodes_gene_expression.csv", expression_id, {
            "expression_id": expression_id,
            "z_score": row["z_score"],
            "level": row["level"],
            "rank_within_cellline": row["rank_within_cellline"],
        })
    gdsc_drug_rows = _row_map(gdsc_schema_dir / "nodes_drug.csv", "gdsc_drug_id")
    gdsc_screen_rows = _row_map(gdsc_schema_dir / "nodes_drug_screen.csv", "drug_screen_id")
    gdsc_retained_drug_ids = {
        drug_id
        for drug_id, row in gdsc_drug_rows.items()
        if row.get("target", "").strip()
    }
    gdsc_screen_to_drug = {
        row["source_drug_screen_id"]: row["target_drug_id"]
        for row in _read_rows(gdsc_schema_dir / "edges_drug_screen_tested_drug.csv")
        if row["target_drug_id"] in gdsc_retained_drug_ids
    }
    for row in _read_rows(gdsc_schema_dir / "edges_cellline_has_drug_screen.csv"):
        source_id = row["source_sanger_model_id"]
        drug_screen_id = row["target_drug_screen_id"]
        if source_id not in gdsc_to_cellosaurus:
            continue
        if drug_screen_id not in gdsc_screen_to_drug:
            continue
        drug_screen_ids.add(drug_screen_id)
        drug_ids.add(gdsc_screen_to_drug[drug_screen_id])
        graph.add_edge("edges_cellline_has_drug_screen.csv", (gdsc_to_cellosaurus[source_id], drug_screen_id), {
            "source_cellosaurus_id": gdsc_to_cellosaurus[source_id],
            "target_drug_screen_id": drug_screen_id,
            "relation": row["relation"],
            "source": "GDSC",
        })
    for row in _read_rows(gdsc_schema_dir / "edges_drug_screen_tested_drug.csv"):
        if row["source_drug_screen_id"] in drug_screen_ids:
            graph.add_edge("edges_drug_screen_tested_drug.csv", (row["source_drug_screen_id"], row["target_drug_id"]), _with_source(row, "GDSC"))
    for drug_id in sorted(drug_ids):
        row = gdsc_drug_rows[drug_id]
        graph.add_node("nodes_drug.csv", drug_id, {
            "drug_id": drug_id,
            "name": row["name"],
            "synonyms": row["synonyms"],
            "mychem_id": row.get("mychem_id", ""),
            "external_aliases": row.get("external_aliases", ""),
            "target": row["target"],
            "target_pathway": row["target_pathway"],
        })
    for drug_screen_id in sorted(drug_screen_ids):
        row = gdsc_screen_rows[drug_screen_id]
        graph.add_node("nodes_drug_screen.csv", drug_screen_id, {
            "drug_screen_id": drug_screen_id,
            "ic50": row["ic50"],
            "auc": row["auc"],
            "z_score": row["z_score"],
            "drug_sensitivity": row["drug_sensitivity"],
        })
    cmp_protein_rows = _row_map(cmp_schema_dir / "nodes_protein.csv", "protein_id")
    cmp_abundance_rows = _row_map(cmp_schema_dir / "nodes_protein_abundance.csv", "protein_abundance_id")
    _rewrite_cellline_edge_file(graph, cmp_schema_dir / "edges_cellline_has_protein_abundance.csv", "source_cmp_model_id", cmp_to_cellosaurus, "target_protein_abundance_id", protein_abundance_ids, source="CellModelPassport")
    for row in _read_rows(cmp_schema_dir / "edges_protein_abundance_measured_protein.csv"):
        if row["source_protein_abundance_id"] in protein_abundance_ids:
            protein_ids.add(row["target_protein_id"])
            graph.add_edge("edges_protein_abundance_measured_protein.csv", (row["source_protein_abundance_id"], row["target_protein_id"]), _with_source(row, "CellModelPassport"))
    for protein_id in sorted(protein_ids):
        row = cmp_protein_rows[protein_id]
        graph.add_node("nodes_protein.csv", protein_id, {
            "protein_id": protein_id,
            "uniprot_id": row["uniprot_id"],
            "symbol": row["protein_symbol"],
            "display_name": row.get("display_name", "") or row["protein_symbol"] or row["uniprot_id"],
        })

    gene_by_symbol = {
        row.get("gene_symbol", "").strip().upper(): row["gene_id"]
        for row in depmap_gene_rows.values()
        if row.get("gene_symbol", "").strip()
    }
    protein_by_symbol = _build_protein_symbol_index(cmp_protein_rows, protein_ids)
    for protein_id in sorted(protein_ids):
        for protein_symbol in _split_protein_symbols(cmp_protein_rows[protein_id].get("protein_symbol", "")):
            gene_id = gene_by_symbol.get(protein_symbol)
            if not gene_id:
                continue
            gene_ids.add(gene_id)
            graph.add_edge(
                "edges_protein_encoded_by_gene.csv",
                (protein_id, gene_id),
                    {
                        "source_protein_id": protein_id,
                        "target_gene_id": gene_id,
                        "relation": "encoded_by",
                        "source": "CellModelPassport",
                    },
                )
    for drug_id in sorted(drug_ids):
        target = gdsc_drug_rows[drug_id].get("target", "")
        for target_symbol in _split_drug_target_symbols(target):
            for protein_id in protein_by_symbol.get(target_symbol, []):
                graph.add_edge(
                    "edges_drug_targets_protein.csv",
                    (drug_id, protein_id),
                    {
                        "source_drug_id": drug_id,
                        "target_protein_id": protein_id,
                        "relation": "targets",
                        "source": "GDSC",
                    },
                )

    for abundance_id in sorted(protein_abundance_ids):
        row = cmp_abundance_rows[abundance_id]
        graph.add_node("nodes_protein_abundance.csv", abundance_id, {
            "protein_abundance_id": abundance_id,
            "z_score": row["z_score"],
            "level": row["level"],
            "rank_within_cellline": row["rank_within_cellline"],
        })
    disease_book.flush_nodes()
    return graph


class _GraphAccumulator:
    def __init__(self) -> None:
        self.nodes = {name: [] for name in NODE_HEADERS}
        self.edges = {name: [] for name in EDGE_HEADERS}
        self._node_seen = {name: set() for name in NODE_HEADERS}
        self._edge_seen = {name: set() for name in EDGE_HEADERS}

    def add_node(self, filename: str, key: str, row: dict[str, str]) -> None:
        if key in self._node_seen[filename]:
            return
        self._node_seen[filename].add(key)
        self.nodes[filename].append({field: str(row.get(field, "")) for field in NODE_HEADERS[filename]})

    def add_edge(self, filename: str, key: tuple[str, str], row: dict[str, str]) -> None:
        cache_key = "::".join(key)
        if cache_key in self._edge_seen[filename]:
            return
        self._edge_seen[filename].add(cache_key)
        self.edges[filename].append({field: str(row.get(field, "")) for field in EDGE_HEADERS[filename]})


class _DiseaseBook:
    def __init__(self, graph: _GraphAccumulator) -> None:
        self.graph = graph
        self.records: dict[str, dict[str, object]] = {}
        self.ids: set[str] = set()

    def add(
        self,
        cellosaurus_id: str,
        disease_name: str,
        oncotree_disease_code: str,
        ncit_disease_id: str,
    ) -> None:
        if not disease_name.strip():
            return
        disease_id = _build_disease_id_from_name(disease_name)
        self.ids.add(disease_id)
        record = self.records.setdefault(
            disease_id,
            {
                "name": disease_name,
                "oncotree_disease_code": "",
                "ncit_disease_id": "",
            },
        )
        if oncotree_disease_code and not record["oncotree_disease_code"]:
            record["oncotree_disease_code"] = oncotree_disease_code
        if ncit_disease_id and not record["ncit_disease_id"]:
            record["ncit_disease_id"] = ncit_disease_id
        self.graph.add_edge("edges_cellline_has_disease.csv", (cellosaurus_id, disease_id), {
            "source_cellosaurus_id": cellosaurus_id,
            "target_disease_id": disease_id,
            "relation": "has_disease",
            "source": "DepMap",
        })

    def flush_nodes(self) -> None:
        for disease_id, record in sorted(self.records.items()):
            self.graph.add_node("nodes_disease.csv", disease_id, {
                "disease_id": disease_id,
                "name": record["name"],
                "oncotree_disease_code": record["oncotree_disease_code"],
                "ncit_disease_id": record["ncit_disease_id"],
            })

def _rewrite_cellline_edge_file(
    graph: _GraphAccumulator,
    path: Path,
    source_field: str,
    source_mapping: dict[str, str],
    target_field: str,
    id_set: set[str],
    *,
    source: str = "DepMap",
) -> None:
    for row in _read_rows(path):
        source_id = row[source_field]
        if source_id not in source_mapping:
            continue
        cellosaurus_id = source_mapping[source_id]
        target_id = row[target_field]
        if target_id in EXCLUDED_DATASET_IDS:
            continue
        id_set.add(target_id)
        graph.add_edge(path.name, (cellosaurus_id, target_id), {
            "source_cellosaurus_id": cellosaurus_id,
            target_field: target_id,
            "relation": row["relation"],
            "source": source,
        })


def _with_source(row: dict[str, str], source: str) -> dict[str, str]:
    enriched = dict(row)
    enriched["source"] = source
    return enriched


def _build_selected_records(selected_clusters: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    selected_records = {}
    for rows in selected_clusters.values():
        row_by_source = {row["source_database"]: row for row in rows}
        cellosaurus_row = row_by_source["Cellosaurus"]
        cellosaurus_id = cellosaurus_row["source_cellline_id"]
        selected_records[cellosaurus_id] = {
            "cellosaurus_id": cellosaurus_id,
            "cellosaurus_name": cellosaurus_row["source_name"],
            "depmap_model_id": row_by_source.get("DepMap", {}).get("source_cellline_id", ""),
            "sanger_model_id": row_by_source.get("GDSC", {}).get("source_cellline_id", ""),
            "cmp_model_id": row_by_source.get("CellModelPassport", {}).get("source_cellline_id", ""),
        }
    return selected_records


def _filter_records_with_dataset_coverage(
    selected_records: dict[str, dict[str, str]],
    depmap_schema_dir: Path,
    gdsc_schema_dir: Path,
    cmp_schema_dir: Path,
) -> dict[str, dict[str, str]]:
    depmap_ids = _collect_dataset_backed_source_ids(depmap_schema_dir / "edges_cellline_in_dataset.csv", "source_depmap_model_id")
    gdsc_ids = _collect_dataset_backed_source_ids(gdsc_schema_dir / "edges_cellline_in_dataset.csv", "source_sanger_model_id")
    cmp_ids = _collect_dataset_backed_source_ids(cmp_schema_dir / "edges_cellline_in_dataset.csv", "source_cmp_model_id")

    filtered = {}
    for cellosaurus_id, record in selected_records.items():
        has_dataset = (
            (record["depmap_model_id"] and record["depmap_model_id"] in depmap_ids)
            or (record["sanger_model_id"] and record["sanger_model_id"] in gdsc_ids)
            or (record["cmp_model_id"] and record["cmp_model_id"] in cmp_ids)
        )
        if has_dataset:
            filtered[cellosaurus_id] = record
    return filtered


def _filter_depmap_non_cancerous_records(
    selected_records: dict[str, dict[str, str]],
    depmap_schema_dir: Path,
) -> dict[str, dict[str, str]]:
    non_cancerous_depmap_ids = _collect_depmap_ids_by_disease_name(
        depmap_schema_dir,
        disease_name="Non-Cancerous",
    )
    return {
        cellosaurus_id: record
        for cellosaurus_id, record in selected_records.items()
        if not (
            record["depmap_model_id"]
            and record["depmap_model_id"] in non_cancerous_depmap_ids
        )
    }


def _collect_depmap_ids_by_disease_name(
    depmap_schema_dir: Path,
    *,
    disease_name: str,
) -> set[str]:
    disease_rows = _read_rows(depmap_schema_dir / "nodes_disease.csv")
    matched_disease_ids = {
        row.get("name", "").strip().lower()
        for row in disease_rows
        if row.get("name", "").strip().lower() == disease_name.strip().lower()
    }
    if not matched_disease_ids:
        return set()

    depmap_ids: set[str] = set()
    for row in _read_rows(depmap_schema_dir / "edges_cellline_has_disease.csv"):
        if row.get("target_disease_id") in matched_disease_ids:
            depmap_id = row.get("source_depmap_model_id", "")
            if depmap_id:
                depmap_ids.add(depmap_id)
    return depmap_ids


def _collect_dataset_backed_source_ids(path: Path, source_field: str) -> set[str]:
    source_ids: set[str] = set()
    for row in _read_rows(path):
        if row.get("target_dataset_id") in EXCLUDED_DATASET_IDS:
            continue
        source_id = row.get(source_field, "")
        if source_id:
            source_ids.add(source_id)
    return source_ids


def _build_source_coverage(selected_records: dict[str, dict[str, str]]) -> dict[str, object]:
    counters = Counter()
    combinations = Counter()
    for record in selected_records.values():
        labels = []
        if record["depmap_model_id"]:
            counters["depmap"] += 1
            labels.append("DepMap")
        if record["sanger_model_id"]:
            counters["gdsc"] += 1
            labels.append("GDSC")
        if record["cmp_model_id"]:
            counters["cell_model_passport"] += 1
            labels.append("CellModelPassport")
        combinations[" + ".join(labels)] += 1
    return {"by_source": dict(counters), "by_source_combination": dict(sorted(combinations.items()))}


def _build_disease_id_from_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized or "unknown_disease"


def _load_human_cellosaurus_ids(path: Path) -> set[str]:
    human_ids: set[str] = set()
    for row in _read_rows(path):
        species = row.get("species", "")
        if "9606" in species or "Homo sapiens" in species:
            human_ids.add(row["cellosaurus_id"])
    return human_ids


def _load_cellosaurus_ids_by_category(path: Path, category: str) -> set[str]:
    matched_ids: set[str] = set()
    for row in _read_rows(path):
        if row.get("category", "").strip() == category:
            matched_ids.add(row["cellosaurus_id"])
    return matched_ids


def _load_clusters(path: Path) -> dict[str, list[dict[str, str]]]:
    clusters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_rows(path):
        clusters[row["canonical_cellline_id"]].append(row)
    return dict(clusters)


def _select_analysis_clusters(
    clusters: dict[str, list[dict[str, str]]],
    human_cellosaurus: set[str],
    cancer_cellosaurus: set[str],
) -> dict[str, list[dict[str, str]]]:
    selected = {}
    for cluster_id, rows in clusters.items():
        source_names = {row["source_database"] for row in rows}
        if not (source_names & DOWNSTREAM_SOURCES):
            continue
        if not any(
            row["source_database"] == "Cellosaurus"
            and row["source_cellline_id"] in human_cellosaurus
            and row["source_cellline_id"] in cancer_cellosaurus
            for row in rows
        ):
            continue
        selected[cluster_id] = rows
    return selected


def _depmap_subtype_id(row: dict[str, str]) -> str:
    return (_get_oncotree_subtype_code(row).lower() or row["name"].strip().lower())


def _get_oncotree_subtype_code(row: dict[str, str]) -> str:
    return (
        row.get("oncotree_subtype_code")
        or row.get("oncotree_code")
        or ""
    ).strip()


DRUG_TARGET_SYMBOL_ALIASES = {
    "BCL-XL": ["BCL2L1"],
    "BCLXL": ["BCL2L1"],
    "BCL-W": ["BCL2L2"],
    "BCLW": ["BCL2L2"],
    "BCL-B": ["BCL2L10"],
    "BCLB": ["BCL2L10"],
    "BFL1": ["BCL2A1"],
    "PI3KALPHA": ["PIK3CA"],
    "PI3K ALPHA": ["PIK3CA"],
    "PI3K-BETA": ["PIK3CB"],
    "PI3KBETA": ["PIK3CB"],
    "PI3K BETA": ["PIK3CB"],
    "PI3KDELTA": ["PIK3CD"],
    "PI3K DELTA": ["PIK3CD"],
    "PI3KGAMMA": ["PIK3CG"],
    "PI3K GAMMA": ["PIK3CG"],
    "PI3K (BETA SPARING)": ["PIK3CA", "PIK3CD", "PIK3CG"],
    "PI3K (CLASS 1)": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    "MUTANT RAS": ["KRAS", "NRAS", "HRAS"],
    "RAS": ["KRAS", "NRAS", "HRAS"],
    "MTORC1": ["MTOR"],
    "MTORC2": ["MTOR"],
    "S6K1": ["RPS6KB1"],
    "RSK1": ["RPS6KA1"],
    "RSK2": ["RPS6KA3"],
    "RSK3": ["RPS6KA2"],
}


def _build_protein_symbol_index(
    protein_rows: dict[str, dict[str, str]],
    protein_ids: set[str],
) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for protein_id in protein_ids:
        for symbol in _split_protein_symbols(protein_rows[protein_id].get("protein_symbol", "")):
            index[symbol].append(protein_id)
    return dict(index)


def _split_protein_symbols(symbols: str) -> list[str]:
    return [
        symbol
        for symbol in (part.strip().upper() for part in re.split(r"[;|]", symbols or ""))
        if re.fullmatch(r"[A-Z0-9]+", symbol)
    ]


def _split_drug_target_symbols(target: str) -> list[str]:
    symbols: list[str] = []
    for part in re.split(r"[,;/]", target or ""):
        normalized = re.sub(r"\s+", " ", part.strip().upper())
        if not normalized:
            continue
        if normalized in DRUG_TARGET_SYMBOL_ALIASES:
            symbols.extend(DRUG_TARGET_SYMBOL_ALIASES[normalized])
            continue
        compact = normalized.replace("-", "").replace(" ", "")
        if compact in DRUG_TARGET_SYMBOL_ALIASES:
            symbols.extend(DRUG_TARGET_SYMBOL_ALIASES[compact])
            continue
        if re.fullmatch(r"[A-Z0-9]+", normalized):
            symbols.append(normalized)
    return list(dict.fromkeys(symbols))


def _row_map(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    return {row[key_field]: row for row in _read_rows(path)}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build unified analysis graph outputs for human cell lines supported by at least one downstream source.")
    parser.add_argument("--reconciliation-clean-input", type=Path, default=Path("data/intermediate/integration/cellline_reconciliation/cell_line_reconciliation_clean.csv"))
    parser.add_argument("--cellosaurus-nodes-input", type=Path, default=Path("data/intermediate/schema/cellosaurus/nodes_cellline.csv"))
    parser.add_argument("--cellosaurus-source-input", type=Path, default=Path("data/intermediate/normalized/cellosaurus/cell_lines.csv"))
    parser.add_argument("--cellosaurus-schema-dir", type=Path, default=Path("data/intermediate/schema/cellosaurus"))
    parser.add_argument("--depmap-schema-dir", type=Path, default=Path("data/intermediate/schema/depmap"))
    parser.add_argument("--gdsc-schema-dir", type=Path, default=Path("data/intermediate/schema/gdsc"))
    parser.add_argument("--cell-model-passport-schema-dir", type=Path, default=Path("data/intermediate/schema/cell_model_passport"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/neo4j_import"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_analysis_subset(
        args.reconciliation_clean_input,
        args.cellosaurus_nodes_input,
        args.cellosaurus_source_input,
        args.cellosaurus_schema_dir,
        args.depmap_schema_dir,
        args.gdsc_schema_dir,
        args.cell_model_passport_schema_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
