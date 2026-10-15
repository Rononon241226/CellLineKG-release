from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


CELL_LINE_NODE_HEADER = [
    "name",
    "depmap_model_id",
    "rrid",
    "tumor_origin",
    "growth_pattern",
    "cosmic_id",
]
DISEASE_NODE_HEADER = ["name", "oncotree_disease_code", "ncit_disease_id"]
SUBTYPE_NODE_HEADER = ["name", "oncotree_subtype_code", "ncit_subtype_id"]
LINEAGE_NODE_HEADER = ["name"]
MEDIA_NODE_HEADER = ["media_id", "formulation"]
GENE_NODE_HEADER = ["gene_id", "entrez_id", "gene_symbol", "display_name"]
VARIANT_NODE_HEADER = [
    "variant_id",
    "chrom",
    "position",
    "ref",
    "alt",
    "variant_type",
    "dna_change",
    "protein_change",
    "rs_id",
]
COPY_NUMBER_NODE_HEADER = [
    "copy_number_id",
    "value_type",
    "threshold_rule",
    "selection_rule",
    "status",
    "value",
    "model_condition_id",
]
DEPENDENCY_NODE_HEADER = [
    "dependency_id",
    "score",
    "value_type",
    "rank_within_cellline",
    "threshold_rule",
    "selection_rule",
]
GENE_EXPRESSION_NODE_HEADER = [
    "expression_id",
    "value",
    "value_type",
    "z_score",
    "level",
    "rank_within_cellline",
    "threshold_rule",
    "selection_rule",
]
DATASET_NODE_HEADER = ["dataset_id", "name", "type", "source", "version"]

CELL_LINE_DISEASE_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_disease_id",
    "relation",
]
CELL_LINE_SUBTYPE_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_subtype_id",
    "relation",
]
CELL_LINE_LINEAGE_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_lineage_id",
    "relation",
]
CELL_LINE_MEDIA_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_media_id",
    "relation",
]
CELL_LINE_VARIANT_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_variant_id",
    "relation",
]
VARIANT_GENE_EDGE_HEADER = ["source_variant_id", "target_gene_id", "relation"]
CELL_LINE_COPY_NUMBER_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_copy_number_id",
    "relation",
]
COPY_NUMBER_GENE_EDGE_HEADER = ["source_copy_number_id", "target_gene_id", "relation"]
CELL_LINE_DEPENDENCY_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_dependency_id",
    "relation",
]
DEPENDENCY_GENE_EDGE_HEADER = ["source_dependency_id", "target_gene_id", "relation"]
CELL_LINE_EXPRESSION_EDGE_HEADER = [
    "source_depmap_model_id",
    "target_expression_id",
    "relation",
]
EXPRESSION_GENE_EDGE_HEADER = ["source_expression_id", "target_gene_id", "relation"]
SUBTYPE_DISEASE_EDGE_HEADER = ["source_subtype_id", "target_disease_id", "relation"]
DISEASE_LINEAGE_EDGE_HEADER = ["source_disease_id", "target_lineage_id", "relation"]
VARIANT_DATASET_EDGE_HEADER = ["source_variant_id", "target_dataset_id", "relation"]
COPY_NUMBER_DATASET_EDGE_HEADER = ["source_copy_number_id", "target_dataset_id", "relation"]
DEPENDENCY_DATASET_EDGE_HEADER = ["source_dependency_id", "target_dataset_id", "relation"]
EXPRESSION_DATASET_EDGE_HEADER = ["source_expression_id", "target_dataset_id", "relation"]
CELL_LINE_DATASET_EDGE_HEADER = ["source_depmap_model_id", "target_dataset_id", "relation"]
CELL_LINE_ID_MAPPING_HEADER = [
    "depmap_model_id",
    "id_source",
    "mapped_id",
    "mapping_type",
]

MODEL_DATASET_NAME = "DepMap_Model"
MUTATION_DATASET_NAME = "DepMap_OmicsSomaticMutations"
COPY_NUMBER_DATASET_NAME = "DepMap_OmicsCNGeneWGS"
DEPENDENCY_DATASET_NAME = "DepMap_CRISPRGeneEffect"
EXPRESSION_DATASET_NAME = "DepMap_OmicsExpressionTPMLogp1HumanProteinCodingGenes"
ONCOTREE_INPUT_PATH = Path("data/raw/oncotree/oncotree_2025_10_03.json")


@dataclass
class SourceTables:
    cell_lines: dict[str, dict[str, str]]
    diseases: dict[str, list[dict[str, str]]]
    subtypes: dict[str, list[dict[str, str]]]
    lineages: dict[str, list[dict[str, str]]]
    media: dict[str, list[dict[str, str]]]
    mutations: dict[str, list[dict[str, str]]]
    copy_number_events: dict[str, list[dict[str, str]]]
    dependencies: dict[str, list[dict[str, str]]]
    expressions: dict[str, list[dict[str, str]]]
    datasets: dict[str, dict[str, str]]
    id_mappings: dict[str, list[dict[str, str]]]


def run_mapping(source_dir: Path, schema_output_dir: Path) -> dict[str, object]:
    schema_output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_source_tables(source_dir)
    oncotree = OncoTreeIndex.from_json(ONCOTREE_INPUT_PATH)
    disease_code_by_id = _build_disease_code_by_id(tables, oncotree)

    counters = Counter()
    seen_diseases: set[str] = set()
    seen_subtypes: set[str] = set()
    seen_lineages: set[str] = set()
    seen_media: set[str] = set()
    seen_genes: set[str] = set()
    seen_variants: set[str] = set()
    seen_copy_numbers: set[str] = set()
    seen_dependencies: set[str] = set()
    seen_expressions: set[str] = set()
    seen_subtype_disease: set[tuple[str, str]] = set()
    seen_disease_lineage: set[tuple[str, str]] = set()
    seen_variant_gene: set[tuple[str, str]] = set()
    seen_copy_number_gene: set[tuple[str, str]] = set()
    seen_dependency_gene: set[tuple[str, str]] = set()
    seen_expression_gene: set[tuple[str, str]] = set()
    seen_variant_dataset: set[tuple[str, str]] = set()
    seen_copy_number_dataset: set[tuple[str, str]] = set()
    seen_dependency_dataset: set[tuple[str, str]] = set()
    seen_expression_dataset: set[tuple[str, str]] = set()
    seen_cellline_dataset: set[tuple[str, str]] = set()

    with ExitStack() as stack:
        cell_line_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_cellline.csv", CELL_LINE_NODE_HEADER))
        disease_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_disease.csv", DISEASE_NODE_HEADER))
        subtype_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_subtype.csv", SUBTYPE_NODE_HEADER))
        lineage_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_lineage.csv", LINEAGE_NODE_HEADER))
        media_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_media.csv", MEDIA_NODE_HEADER))
        gene_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_gene.csv", GENE_NODE_HEADER))
        variant_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_variant.csv", VARIANT_NODE_HEADER))
        copy_number_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_copy_number.csv", COPY_NUMBER_NODE_HEADER))
        dependency_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_dependency.csv", DEPENDENCY_NODE_HEADER))
        expression_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_gene_expression.csv", GENE_EXPRESSION_NODE_HEADER))
        dataset_writer = stack.enter_context(_open_writer(schema_output_dir / "nodes_dataset.csv", DATASET_NODE_HEADER))
        cellline_disease_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_disease.csv", CELL_LINE_DISEASE_EDGE_HEADER))
        cellline_subtype_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_subtype.csv", CELL_LINE_SUBTYPE_EDGE_HEADER))
        cellline_lineage_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_lineage.csv", CELL_LINE_LINEAGE_EDGE_HEADER))
        cellline_media_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_cultures_in.csv", CELL_LINE_MEDIA_EDGE_HEADER))
        cellline_variant_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_variant.csv", CELL_LINE_VARIANT_EDGE_HEADER))
        variant_gene_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_variant_in_gene.csv", VARIANT_GENE_EDGE_HEADER))
        cellline_copy_number_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_copy_number.csv", CELL_LINE_COPY_NUMBER_EDGE_HEADER))
        copy_number_gene_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_copy_number_target_gene.csv", COPY_NUMBER_GENE_EDGE_HEADER))
        cellline_dependency_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_dependency.csv", CELL_LINE_DEPENDENCY_EDGE_HEADER))
        dependency_gene_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_dependency_target_gene.csv", DEPENDENCY_GENE_EDGE_HEADER))
        cellline_expression_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_has_expression.csv", CELL_LINE_EXPRESSION_EDGE_HEADER))
        expression_gene_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_expression_measured_gene.csv", EXPRESSION_GENE_EDGE_HEADER))
        subtype_disease_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_subtype_of.csv", SUBTYPE_DISEASE_EDGE_HEADER))
        disease_lineage_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_disease_part_of.csv", DISEASE_LINEAGE_EDGE_HEADER))
        variant_dataset_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_variant_from_dataset.csv", VARIANT_DATASET_EDGE_HEADER))
        copy_number_dataset_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_copy_number_from_dataset.csv", COPY_NUMBER_DATASET_EDGE_HEADER))
        dependency_dataset_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_dependency_from_dataset.csv", DEPENDENCY_DATASET_EDGE_HEADER))
        expression_dataset_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_expression_from_dataset.csv", EXPRESSION_DATASET_EDGE_HEADER))
        cellline_dataset_writer = stack.enter_context(_open_writer(schema_output_dir / "edges_cellline_in_dataset.csv", CELL_LINE_DATASET_EDGE_HEADER))
        mapping_writer = stack.enter_context(_open_writer(schema_output_dir / "cell_line_id_mapping.csv", CELL_LINE_ID_MAPPING_HEADER))
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
        for depmap_model_id, base_row in tables.cell_lines.items():
            cell_line_writer.writerow(
                {
                    "name": base_row["depmap_name"],
                    "depmap_model_id": depmap_model_id,
                    "rrid": base_row["rrid"],
                    "tumor_origin": base_row["tumor_origin"],
                    "growth_pattern": base_row["growth_pattern"],
                    "cosmic_id": base_row["cosmic_id"],
                }
            )
            counters["nodes_cellline"] += 1

            disease_rows = tables.diseases.get(depmap_model_id, [])
            subtype_rows = tables.subtypes.get(depmap_model_id, [])
            lineage_rows = tables.lineages.get(depmap_model_id, [])
            media_rows = tables.media.get(depmap_model_id, [])
            mutation_rows = tables.mutations.get(depmap_model_id, [])
            copy_number_rows = tables.copy_number_events.get(depmap_model_id, [])
            dependency_rows = tables.dependencies.get(depmap_model_id, [])
            expression_rows = tables.expressions.get(depmap_model_id, [])

            _write_cellline_dataset_edge(depmap_model_id, dataset_ids.get(MODEL_DATASET_NAME, ""), seen_cellline_dataset, cellline_dataset_writer, counters)
            if mutation_rows:
                _write_cellline_dataset_edge(depmap_model_id, dataset_ids.get(MUTATION_DATASET_NAME, ""), seen_cellline_dataset, cellline_dataset_writer, counters)
            if copy_number_rows:
                _write_cellline_dataset_edge(depmap_model_id, dataset_ids.get(COPY_NUMBER_DATASET_NAME, ""), seen_cellline_dataset, cellline_dataset_writer, counters)
            if dependency_rows:
                _write_cellline_dataset_edge(depmap_model_id, dataset_ids.get(DEPENDENCY_DATASET_NAME, ""), seen_cellline_dataset, cellline_dataset_writer, counters)
            if expression_rows:
                _write_cellline_dataset_edge(depmap_model_id, dataset_ids.get(EXPRESSION_DATASET_NAME, ""), seen_cellline_dataset, cellline_dataset_writer, counters)

            for disease in disease_rows:
                disease_id = build_name_id(disease["disease_name"])
                if disease_id not in seen_diseases:
                    disease_writer.writerow(
                        {
                            "name": disease["disease_name"],
                            "oncotree_disease_code": disease_code_by_id.get(disease_id, ""),
                            "ncit_disease_id": oncotree.get_ncit_ids(
                                disease_code_by_id.get(disease_id, "")
                            ),
                        }
                    )
                    seen_diseases.add(disease_id)
                    counters["nodes_disease"] += 1

                cellline_disease_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_disease_id": disease_id,
                        "relation": "has_disease",
                    }
                )
                counters["edges_cellline_has_disease"] += 1

            for subtype in subtype_rows:
                subtype_id = build_subtype_id(
                    subtype["subtype_name"], _get_subtype_code(subtype)
                )
                if subtype_id not in seen_subtypes:
                    subtype_writer.writerow(
                        {
                            "name": subtype["subtype_name"],
                            "oncotree_subtype_code": _get_subtype_code(subtype),
                            "ncit_subtype_id": oncotree.get_ncit_ids(
                                _get_subtype_code(subtype)
                            ),
                        }
                    )
                    seen_subtypes.add(subtype_id)
                    counters["nodes_subtype"] += 1

                cellline_subtype_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_subtype_id": subtype_id,
                        "relation": "has_subtype",
                    }
                )
                counters["edges_cellline_has_subtype"] += 1

                for disease in disease_rows:
                    disease_id = build_name_id(disease["disease_name"])
                    pair = (subtype_id, disease_id)
                    if pair in seen_subtype_disease:
                        continue
                    subtype_disease_writer.writerow(
                        {
                            "source_subtype_id": subtype_id,
                            "target_disease_id": disease_id,
                            "relation": "subtype_of",
                        }
                    )
                    seen_subtype_disease.add(pair)
                    counters["edges_subtype_of"] += 1

            for lineage in lineage_rows:
                lineage_id = build_name_id(lineage["lineage_name"])
                if lineage_id not in seen_lineages:
                    lineage_writer.writerow({"name": lineage["lineage_name"]})
                    seen_lineages.add(lineage_id)
                    counters["nodes_lineage"] += 1

                cellline_lineage_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_lineage_id": lineage_id,
                        "relation": "has_lineage",
                    }
                )
                counters["edges_cellline_has_lineage"] += 1

                for disease in disease_rows:
                    disease_id = build_name_id(disease["disease_name"])
                    pair = (disease_id, lineage_id)
                    if pair in seen_disease_lineage:
                        continue
                    disease_lineage_writer.writerow(
                        {
                            "source_disease_id": disease_id,
                            "target_lineage_id": lineage_id,
                            "relation": "part_of",
                        }
                    )
                    seen_disease_lineage.add(pair)
                    counters["edges_disease_part_of"] += 1

            for media in media_rows:
                media_id = build_media_id(media["formulation_id"])
                if media_id not in seen_media:
                    media_writer.writerow(
                        {
                            "media_id": media_id,
                            "formulation": media["formulation_id"],
                        }
                    )
                    seen_media.add(media_id)
                    counters["nodes_media"] += 1

                cellline_media_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_media_id": media_id,
                        "relation": "cultures_in",
                    }
                )
                counters["edges_cellline_cultures_in"] += 1

            for mutation in mutation_rows:
                gene_id = build_gene_id(mutation["entrez_id"], mutation["gene_symbol"])
                if gene_id and gene_id not in seen_genes:
                    gene_writer.writerow(
                        {
                            "gene_id": gene_id,
                            "entrez_id": mutation["entrez_id"],
                            "gene_symbol": mutation["gene_symbol"],
                            "display_name": mutation["gene_symbol"],
                        }
                    )
                    seen_genes.add(gene_id)
                    counters["nodes_gene"] += 1

                variant_id = build_variant_id(depmap_model_id, gene_id, mutation)
                if variant_id not in seen_variants:
                    variant_writer.writerow(
                        {
                            "variant_id": variant_id,
                            "chrom": mutation["chrom"],
                            "position": mutation["position"],
                            "ref": mutation["ref"],
                            "alt": mutation["alt"],
                            "variant_type": mutation["variant_type"],
                            "dna_change": mutation["dna_change"],
                            "protein_change": mutation["protein_change"],
                            "rs_id": mutation.get("rs_id", ""),
                        }
                    )
                    seen_variants.add(variant_id)
                    counters["nodes_variant"] += 1

                cellline_variant_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_variant_id": variant_id,
                        "relation": "has_variant",
                    }
                )
                counters["edges_cellline_has_variant"] += 1

                if gene_id:
                    pair = (variant_id, gene_id)
                    if pair not in seen_variant_gene:
                        variant_gene_writer.writerow(
                            {
                                "source_variant_id": variant_id,
                                "target_gene_id": gene_id,
                                "relation": "in_gene",
                            }
                        )
                        seen_variant_gene.add(pair)
                        counters["edges_variant_in_gene"] += 1

                mutation_dataset_id = dataset_ids.get(MUTATION_DATASET_NAME, "")
                if mutation_dataset_id:
                    pair = (variant_id, mutation_dataset_id)
                    if pair not in seen_variant_dataset:
                        variant_dataset_writer.writerow(
                            {
                                "source_variant_id": variant_id,
                                "target_dataset_id": mutation_dataset_id,
                                "relation": "from_dataset",
                            }
                        )
                        seen_variant_dataset.add(pair)
                        counters["edges_variant_from_dataset"] += 1

            for copy_number in copy_number_rows:
                gene_id = build_gene_id(copy_number["entrez_id"], copy_number["gene_symbol"])
                if gene_id and gene_id not in seen_genes:
                    gene_writer.writerow(
                        {
                            "gene_id": gene_id,
                            "entrez_id": copy_number["entrez_id"],
                            "gene_symbol": copy_number["gene_symbol"],
                            "display_name": copy_number["gene_symbol"],
                        }
                    )
                    seen_genes.add(gene_id)
                    counters["nodes_gene"] += 1

                if not gene_id:
                    continue

                copy_number_id = build_copy_number_id(depmap_model_id, gene_id)
                if copy_number_id not in seen_copy_numbers:
                    copy_number_writer.writerow(
                        {
                            "copy_number_id": copy_number_id,
                            "value_type": copy_number["value_type"],
                            "threshold_rule": copy_number["threshold_rule"],
                            "selection_rule": copy_number["selection_rule"],
                            "status": copy_number["status"],
                            "value": copy_number["copy_number_value"],
                            "model_condition_id": copy_number["model_condition_id"],
                        }
                    )
                    seen_copy_numbers.add(copy_number_id)
                    counters["nodes_copy_number"] += 1

                cellline_copy_number_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_copy_number_id": copy_number_id,
                        "relation": "has_copy_number",
                    }
                )
                counters["edges_cellline_has_copy_number"] += 1

                pair = (copy_number_id, gene_id)
                if pair not in seen_copy_number_gene:
                    copy_number_gene_writer.writerow(
                        {
                            "source_copy_number_id": copy_number_id,
                            "target_gene_id": gene_id,
                            "relation": "target_gene",
                        }
                    )
                    seen_copy_number_gene.add(pair)
                    counters["edges_copy_number_target_gene"] += 1

                copy_number_dataset_id = dataset_ids.get(COPY_NUMBER_DATASET_NAME, "")
                if copy_number_dataset_id:
                    pair = (copy_number_id, copy_number_dataset_id)
                    if pair not in seen_copy_number_dataset:
                        copy_number_dataset_writer.writerow(
                            {
                                "source_copy_number_id": copy_number_id,
                                "target_dataset_id": copy_number_dataset_id,
                                "relation": "from_dataset",
                            }
                        )
                        seen_copy_number_dataset.add(pair)
                        counters["edges_copy_number_from_dataset"] += 1

            for dependency in dependency_rows:
                gene_id = build_gene_id(dependency["entrez_id"], dependency["gene_symbol"])
                if gene_id and gene_id not in seen_genes:
                    gene_writer.writerow(
                        {
                            "gene_id": gene_id,
                            "entrez_id": dependency["entrez_id"],
                            "gene_symbol": dependency["gene_symbol"],
                            "display_name": dependency["gene_symbol"],
                        }
                    )
                    seen_genes.add(gene_id)
                    counters["nodes_gene"] += 1

                if not gene_id:
                    continue

                dependency_id = build_dependency_id(
                    depmap_model_id,
                    gene_id,
                )
                if dependency_id not in seen_dependencies:
                    dependency_writer.writerow(
                        {
                            "dependency_id": dependency_id,
                            "score": dependency["dependency_score"],
                            "value_type": dependency["value_type"],
                            "rank_within_cellline": dependency["rank_within_cellline"],
                            "threshold_rule": dependency["threshold_rule"],
                            "selection_rule": dependency["selection_rule"],
                        }
                    )
                    seen_dependencies.add(dependency_id)
                    counters["nodes_dependency"] += 1

                cellline_dependency_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_dependency_id": dependency_id,
                        "relation": "has_dependency",
                    }
                )
                counters["edges_cellline_has_dependency"] += 1

                pair = (dependency_id, gene_id)
                if pair not in seen_dependency_gene:
                    dependency_gene_writer.writerow(
                        {
                            "source_dependency_id": dependency_id,
                            "target_gene_id": gene_id,
                            "relation": "target_gene",
                        }
                    )
                    seen_dependency_gene.add(pair)
                    counters["edges_dependency_target_gene"] += 1

                dependency_dataset_id = dataset_ids.get(DEPENDENCY_DATASET_NAME, "")
                if dependency_dataset_id:
                    pair = (dependency_id, dependency_dataset_id)
                    if pair not in seen_dependency_dataset:
                        dependency_dataset_writer.writerow(
                            {
                                "source_dependency_id": dependency_id,
                                "target_dataset_id": dependency_dataset_id,
                                "relation": "from_dataset",
                            }
                        )
                        seen_dependency_dataset.add(pair)
                        counters["edges_dependency_from_dataset"] += 1

            for expression in expression_rows:
                gene_id = build_gene_id(expression["entrez_id"], expression["gene_symbol"])
                if gene_id and gene_id not in seen_genes:
                    gene_writer.writerow(
                        {
                            "gene_id": gene_id,
                            "entrez_id": expression["entrez_id"],
                            "gene_symbol": expression["gene_symbol"],
                            "display_name": expression["gene_symbol"],
                        }
                    )
                    seen_genes.add(gene_id)
                    counters["nodes_gene"] += 1

                if not gene_id:
                    continue

                expression_id = build_expression_id(
                    depmap_model_id,
                    gene_id,
                )
                if expression_id not in seen_expressions:
                    expression_writer.writerow(
                        {
                            "expression_id": expression_id,
                            "value": expression["expression_value"],
                            "value_type": expression["value_type"],
                            "z_score": expression["z_score"],
                            "level": expression["level"],
                            "rank_within_cellline": expression["rank_within_cellline"],
                            "threshold_rule": expression["threshold_rule"],
                            "selection_rule": expression["selection_rule"],
                        }
                    )
                    seen_expressions.add(expression_id)
                    counters["nodes_gene_expression"] += 1

                cellline_expression_writer.writerow(
                    {
                        "source_depmap_model_id": depmap_model_id,
                        "target_expression_id": expression_id,
                        "relation": "has_expression",
                    }
                )
                counters["edges_cellline_has_expression"] += 1

                pair = (expression_id, gene_id)
                if pair not in seen_expression_gene:
                    expression_gene_writer.writerow(
                        {
                            "source_expression_id": expression_id,
                            "target_gene_id": gene_id,
                            "relation": "measured_gene",
                        }
                    )
                    seen_expression_gene.add(pair)
                    counters["edges_expression_measured_gene"] += 1

                expression_dataset_id = dataset_ids.get(EXPRESSION_DATASET_NAME, "")
                if expression_dataset_id:
                    pair = (expression_id, expression_dataset_id)
                    if pair not in seen_expression_dataset:
                        expression_dataset_writer.writerow(
                            {
                                "source_expression_id": expression_id,
                                "target_dataset_id": expression_dataset_id,
                                "relation": "from_dataset",
                            }
                        )
                        seen_expression_dataset.add(pair)
                        counters["edges_expression_from_dataset"] += 1

            for mapping_row in tables.id_mappings.get(depmap_model_id, []):
                mapping_writer.writerow(mapping_row)
                counters["cell_line_id_mapping_rows"] += 1

    summary = {
        "input_dir": str(source_dir),
        "output_dir": str(schema_output_dir),
        "counts": dict(counters),
        "relation_types": [
            "has_disease",
            "has_subtype",
            "has_lineage",
            "cultures_in",
            "has_variant",
            "in_gene",
            "has_copy_number",
            "target_gene",
            "has_dependency",
            "has_expression",
            "measured_gene",
            "subtype_of",
            "part_of",
            "from_dataset",
            "in_dataset",
        ],
    }
    (schema_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


@dataclass
class OncoTreeNode:
    code: str
    name: str
    parent: str
    aliases: tuple[str, ...]
    nci_ids: tuple[str, ...]


class OncoTreeIndex:
    def __init__(self, nodes: dict[str, OncoTreeNode]):
        self.nodes = nodes
        self.alias_to_code: dict[str, str] = {}
        self.name_to_codes: dict[str, set[str]] = defaultdict(set)
        for code, node in nodes.items():
            self.name_to_codes[node.name.strip().lower()].add(code)
            for alias in node.aliases:
                if alias:
                    self.alias_to_code[alias] = code

    @classmethod
    def from_json(cls, path: Path) -> "OncoTreeIndex":
        if not path.exists():
            return cls({})

        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes: dict[str, OncoTreeNode] = {}

        def walk(raw_node: dict[str, object]) -> None:
            code = str(raw_node.get("code") or "").strip()
            if code:
                aliases: list[str] = []
                for field in ("history", "precursors", "revocations"):
                    values = raw_node.get(field) or []
                    if isinstance(values, list):
                        aliases.extend(str(value).strip() for value in values if str(value).strip())
                nodes[code] = OncoTreeNode(
                    code=code,
                    name=str(raw_node.get("name") or "").strip(),
                    parent=str(raw_node.get("parent") or "").strip(),
                    aliases=tuple(aliases),
                    nci_ids=_extract_oncotree_nci_ids(raw_node),
                )
            children = raw_node.get("children") or {}
            if isinstance(children, dict):
                for child in children.values():
                    if isinstance(child, dict):
                        walk(child)

        if isinstance(payload, dict):
            for root in payload.values():
                if isinstance(root, dict):
                    walk(root)
        return cls(nodes)

    def resolve_code(self, code: str | None) -> str:
        normalized = (code or "").strip()
        if not normalized:
            return ""
        if normalized in self.nodes:
            return normalized
        return self.alias_to_code.get(normalized, normalized)

    def resolve_disease_code(self, disease_name: str, subtype_codes: list[str]) -> str:
        name = disease_name.strip().lower()
        direct_codes = self.name_to_codes.get(name, set())
        if len(direct_codes) == 1:
            return next(iter(direct_codes))

        evidence: set[str] = set()
        for subtype_code in subtype_codes:
            for node in self.iter_ancestors(subtype_code):
                if node.name.strip().lower() == name:
                    evidence.add(node.code)
        if len(evidence) == 1:
            return next(iter(evidence))
        return ""

    def get_ncit_ids(self, code: str | None) -> str:
        resolved_code = self.resolve_code(code)
        node = self.nodes.get(resolved_code)
        if node is None:
            return ""
        return " || ".join(node.nci_ids)

    def iter_ancestors(self, code: str | None) -> list[OncoTreeNode]:
        current_code = self.resolve_code(code)
        ancestors: list[OncoTreeNode] = []
        seen: set[str] = set()
        while current_code and current_code not in seen:
            node = self.nodes.get(current_code)
            if node is None:
                break
            seen.add(current_code)
            ancestors.append(node)
            current_code = node.parent
        return ancestors


def _extract_oncotree_nci_ids(raw_node: dict[str, object]) -> tuple[str, ...]:
    external_references = raw_node.get("externalReferences") or {}
    if not isinstance(external_references, dict):
        return ()
    values = external_references.get("NCI") or []
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if _is_ncit_id(str(value).strip())
        )
    )


def _is_ncit_id(value: str) -> bool:
    return bool(re.fullmatch(r"C\d+", value.strip(), flags=re.IGNORECASE))


def _build_disease_code_by_id(
    tables: SourceTables,
    oncotree: OncoTreeIndex,
) -> dict[str, str]:
    disease_subtype_codes: dict[str, set[str]] = defaultdict(set)
    disease_names: dict[str, str] = {}
    for depmap_model_id, disease_rows in tables.diseases.items():
        subtype_codes = {
            _get_subtype_code(subtype)
            for subtype in tables.subtypes.get(depmap_model_id, [])
            if _get_subtype_code(subtype)
        }
        for disease in disease_rows:
            disease_name = disease["disease_name"]
            disease_id = build_name_id(disease_name)
            disease_names[disease_id] = disease_name
            disease_subtype_codes[disease_id].update(subtype_codes)

    return {
        disease_id: oncotree.resolve_disease_code(
            disease_name,
            sorted(disease_subtype_codes.get(disease_id, set())),
        )
        for disease_id, disease_name in disease_names.items()
    }


def _get_subtype_code(subtype: dict[str, str | None]) -> str:
    return (
        subtype.get("oncotree_subtype_code")
        or subtype.get("oncotree_code")
        or ""
    ).strip()


def load_source_tables(source_dir: Path) -> SourceTables:
    cell_lines = {
        row["depmap_model_id"]: row
        for row in _read_csv_rows(source_dir / "cell_lines.csv")
    }
    diseases = _group_structured_rows(
        source_dir / "cell_line_diseases.csv", ["disease_name"]
    )
    subtypes = _group_structured_rows(
        source_dir / "cell_line_subtypes.csv", ["subtype_name", "oncotree_subtype_code", "oncotree_code"]
    )
    lineages = _group_structured_rows(
        source_dir / "cell_line_lineages.csv", ["lineage_name"]
    )
    media = _group_structured_rows(
        source_dir / "cell_line_media.csv", ["formulation_id"]
    )
    mutations = _group_structured_rows(
        source_dir / "somatic_mutations.csv",
        [
            "chrom",
            "position",
            "ref",
            "alt",
            "variant_type",
            "dna_change",
            "protein_change",
            "rs_id",
            "gene_symbol",
            "entrez_id",
        ],
    )
    copy_number_events = _group_structured_rows(
        source_dir / "gene_copy_numbers_selected.csv",
        [
            "model_condition_id",
            "gene_symbol",
            "entrez_id",
            "copy_number_value",
            "value_type",
            "status",
            "threshold_rule",
            "selection_rule",
        ],
    )
    dependencies = _group_structured_rows(
        source_dir / "gene_dependencies_selected.csv",
        [
            "gene_symbol",
            "entrez_id",
            "dependency_score",
            "value_type",
            "rank_within_cellline",
            "threshold_rule",
            "selection_rule",
        ],
    )
    expressions = _group_structured_rows(
        source_dir / "gene_expressions_selected.csv",
        [
            "gene_symbol",
            "entrez_id",
            "expression_value",
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
        ["depmap_model_id", "id_source", "mapped_id", "mapping_type"],
    )

    return SourceTables(
        cell_lines=cell_lines,
        diseases=diseases,
        subtypes=subtypes,
        lineages=lineages,
        media=media,
        mutations=mutations,
        copy_number_events=copy_number_events,
        dependencies=dependencies,
        expressions=expressions,
        datasets=datasets,
        id_mappings=id_mappings,
    )


def build_name_id(name: str | None) -> str:
    return (name or "").strip().lower()


def build_subtype_id(name: str | None, oncotree_code: str | None) -> str:
    code = (oncotree_code or "").strip()
    if code:
        return code.lower()
    return build_name_id(name)


def build_media_id(formulation_id: str | None) -> str:
    return (formulation_id or "").strip().lower()


def build_gene_id(entrez_id: str | None, gene_symbol: str | None) -> str:
    # Some trailing missing columns are read as None by DictReader.
    normalized_entrez_id = (entrez_id or "").strip()
    if normalized_entrez_id:
        return normalized_entrez_id
    normalized_gene_symbol = (gene_symbol or "").strip()
    if normalized_gene_symbol:
        return normalized_gene_symbol.lower()
    return ""


def build_variant_id(
    depmap_model_id: str,
    gene_id: str,
    mutation: dict[str, str | None],
) -> str:
    # Keep the ID compact while using genomic coordinates to avoid protein-change collisions.
    variant_key = "|".join(
        [
            gene_id.strip(),
            (mutation.get("chrom") or "").strip(),
            (mutation.get("position") or "").strip(),
            (mutation.get("ref") or "").strip(),
            (mutation.get("alt") or "").strip(),
        ]
    )
    return f"variant::{depmap_model_id.strip()}::{variant_key}"


def build_copy_number_id(depmap_model_id: str, gene_id: str) -> str:
    return f"copy_number::{depmap_model_id.strip()}::{gene_id.strip()}"


def build_dependency_id(depmap_model_id: str, gene_id: str) -> str:
    return f"dependency::{depmap_model_id.strip()}::{gene_id.strip()}"


def build_expression_id(depmap_model_id: str, gene_id: str) -> str:
    return f"gene_expression::{depmap_model_id.strip()}::{gene_id.strip()}"


def _write_cellline_dataset_edge(
    depmap_model_id: str,
    dataset_id: str,
    seen_pairs: set[tuple[str, str]],
    writer: csv.DictWriter,
    counters: Counter,
) -> None:
    if not dataset_id:
        return
    pair = (depmap_model_id, dataset_id)
    if pair in seen_pairs:
        return
    writer.writerow(
        {
            "source_depmap_model_id": depmap_model_id,
            "target_dataset_id": dataset_id,
            "relation": "in_dataset",
        }
    )
    seen_pairs.add(pair)
    counters["edges_cellline_in_dataset"] += 1


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_structured_rows(path: Path, fields: list[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        grouped[row["depmap_model_id"]].append({field: row.get(field, "") or "" for field in fields})
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
        description="Map DepMap source-level CSV tables into schema-level CSV tables."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/intermediate/normalized/depmap"),
        help="Directory containing source-level DepMap tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/schema/depmap"),
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
                "nodes_disease": summary["counts"].get("nodes_disease", 0),
                "nodes_variant": summary["counts"].get("nodes_variant", 0),
                "nodes_copy_number": summary["counts"].get("nodes_copy_number", 0),
                "nodes_dependency": summary["counts"].get("nodes_dependency", 0),
                "nodes_gene_expression": summary["counts"].get("nodes_gene_expression", 0),
                "edges_cellline_has_expression": summary["counts"].get(
                    "edges_cellline_has_expression", 0
                ),
                "edges_cellline_in_dataset": summary["counts"].get(
                    "edges_cellline_in_dataset", 0
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
