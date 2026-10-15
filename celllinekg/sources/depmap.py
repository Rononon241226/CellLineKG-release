from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


CELL_LINES_HEADER = [
    "depmap_model_id",
    "depmap_name",
    "rrid",
    "oncotree_primary_disease",
    "oncotree_subtype",
    "oncotree_code",
    "oncotree_lineage",
    "tumor_origin",
    "growth_pattern",
    "formulation_id",
    "cosmic_id",
]
CELL_LINE_DISEASES_HEADER = ["depmap_model_id", "disease_name"]
CELL_LINE_SUBTYPES_HEADER = ["depmap_model_id", "subtype_name", "oncotree_subtype_code"]
CELL_LINE_LINEAGES_HEADER = ["depmap_model_id", "lineage_name"]
CELL_LINE_MEDIA_HEADER = ["depmap_model_id", "formulation_id"]
DATASETS_HEADER = ["dataset_name", "dataset_type", "source", "version"]
CELL_LINE_ID_MAPPING_HEADER = [
    "depmap_model_id",
    "id_source",
    "mapped_id",
    "mapping_type",
]
SOMATIC_MUTATIONS_HEADER = [
    "depmap_model_id",
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
]
COPY_NUMBER_EVENTS_HEADER = [
    "depmap_model_id",
    "model_condition_id",
    "gene_symbol",
    "entrez_id",
    "copy_number_value",
    "value_type",
    "status",
    "threshold_rule",
    "selection_rule",
]
GENE_DEPENDENCIES_HEADER = [
    "depmap_model_id",
    "gene_symbol",
    "entrez_id",
    "dependency_score",
    "value_type",
    "rank_within_cellline",
    "threshold_rule",
    "selection_rule",
]
GENE_EXPRESSIONS_HEADER = [
    "depmap_model_id",
    "gene_symbol",
    "entrez_id",
    "expression_value",
    "value_type",
    "z_score",
    "level",
    "rank_within_cellline",
    "threshold_rule",
    "selection_rule",
]

DEPENDENCY_SCORE_THRESHOLD = -1.0
DEPENDENCY_TOP_N = 500
DEPENDENCY_SELECTION_RULE = "score <= -1 and rank_within_cellline <= 500"
DEPENDENCY_THRESHOLD_RULE = "score<=-1"
DEPENDENCY_VALUE_TYPE = "gene_dependency_score"
EXPRESSION_ZSCORE_THRESHOLD = 2.0
EXPRESSION_TOP_N = 500
EXPRESSION_THRESHOLD_RULE = "abs(z_score)>2"
EXPRESSION_SELECTION_RULE = "abs(z_score) > 2 and rank_within_cellline <= 500"
EXPRESSION_VALUE_TYPE = "tpm_logp1_expression"
COPY_NUMBER_VALUE_TYPE = "linear_relative_copy_number"
DEP_MAP_VERSION = "DepMap Release 25Q3"
MODEL_DATASET_NAME = "DepMap_Model"
MUTATION_DATASET_NAME = "DepMap_OmicsSomaticMutations"
COPY_NUMBER_DATASET_NAME = "DepMap_OmicsCNGeneWGS"
DEPENDENCY_DATASET_NAME = "DepMap_CRISPRGeneEffect"
EXPRESSION_DATASET_NAME = "DepMap_OmicsExpressionTPMLogp1HumanProteinCodingGenes"


def export_source_tables(
    model_input_path: Path,
    output_dir: Path,
    mutation_input_path: Path | None = None,
    cn_input_path: Path | None = None,
    dependency_input_path: Path | None = None,
    expression_input_path: Path | None = None,
    cn_amplification_threshold: float = 2.0,
    cn_deletion_threshold: float = 0.5,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()

    with (
        _open_writer(output_dir / "cell_lines.csv", CELL_LINES_HEADER) as cell_lines_writer,
        _open_writer(output_dir / "cell_line_diseases.csv", CELL_LINE_DISEASES_HEADER) as diseases_writer,
        _open_writer(output_dir / "cell_line_subtypes.csv", CELL_LINE_SUBTYPES_HEADER) as subtypes_writer,
        _open_writer(output_dir / "cell_line_lineages.csv", CELL_LINE_LINEAGES_HEADER) as lineages_writer,
        _open_writer(output_dir / "cell_line_media.csv", CELL_LINE_MEDIA_HEADER) as media_writer,
        _open_writer(output_dir / "datasets.csv", DATASETS_HEADER) as datasets_writer,
        _open_writer(output_dir / "cell_line_id_mapping.csv", CELL_LINE_ID_MAPPING_HEADER) as mapping_writer,
    ):
        _write_dataset_row(datasets_writer, MODEL_DATASET_NAME, "metadata", counters)
        for row in _read_csv_rows(model_input_path):
            model_id = row["ModelID"].strip()
            if not model_id:
                continue

            base_row = {
                "depmap_model_id": model_id,
                "depmap_name": row["CellLineName"].strip(),
                "rrid": row["RRID"].strip(),
                "oncotree_primary_disease": row["OncotreePrimaryDisease"].strip(),
                "oncotree_subtype": row["OncotreeSubtype"].strip(),
                "oncotree_code": row["OncotreeCode"].strip(),
                "oncotree_lineage": row["OncotreeLineage"].strip(),
                "tumor_origin": row["PrimaryOrMetastasis"].strip(),
                "growth_pattern": row["GrowthPattern"].strip(),
                "formulation_id": row["FormulationID"].strip(),
                "cosmic_id": _normalize_cosmic_id(row["COSMICID"]),
            }
            cell_lines_writer.writerow(base_row)
            counters["cell_lines"] += 1

            if base_row["oncotree_primary_disease"]:
                diseases_writer.writerow(
                    {
                        "depmap_model_id": model_id,
                        "disease_name": base_row["oncotree_primary_disease"],
                    }
                )
                counters["disease_links"] += 1

            if base_row["oncotree_subtype"]:
                subtypes_writer.writerow(
                    {
                        "depmap_model_id": model_id,
                        "subtype_name": base_row["oncotree_subtype"],
                        "oncotree_subtype_code": base_row["oncotree_code"],
                    }
                )
                counters["subtype_links"] += 1

            if base_row["oncotree_lineage"]:
                lineages_writer.writerow(
                    {
                        "depmap_model_id": model_id,
                        "lineage_name": base_row["oncotree_lineage"],
                    }
                )
                counters["lineage_links"] += 1

            if base_row["formulation_id"]:
                media_writer.writerow(
                    {
                        "depmap_model_id": model_id,
                        "formulation_id": base_row["formulation_id"],
                    }
                )
                counters["media_links"] += 1

            _write_mapping_rows(base_row, mapping_writer, counters)

    if mutation_input_path is not None:
        with _open_writer(output_dir / "somatic_mutations.csv", SOMATIC_MUTATIONS_HEADER) as mutation_writer:
            _append_dataset_row(output_dir / "datasets.csv", MUTATION_DATASET_NAME, "mutation")
            counters["datasets"] += 1
            for row in _read_csv_rows(mutation_input_path):
                if not _is_default_row(row):
                    continue

                model_id = row["ModelID"].strip()
                if not model_id:
                    continue

                mutation_writer.writerow(
                    {
                        "depmap_model_id": model_id,
                        "chrom": row["Chrom"].strip(),
                        "position": row["Pos"].strip(),
                        "ref": row["Ref"].strip(),
                        "alt": row["Alt"].strip(),
                        "variant_type": row["VariantType"].strip(),
                        "dna_change": row["DNAChange"].strip(),
                        "protein_change": row["ProteinChange"].strip(),
                        "rs_id": _normalize_rs_id(row.get("DbsnpRsID", "")),
                        "gene_symbol": row["HugoSymbol"].strip(),
                        "entrez_id": row["EntrezGeneID"].strip(),
                    }
                )
                counters["somatic_mutations"] += 1

    if cn_input_path is not None:
        with cn_input_path.open(encoding="utf-8", newline="") as handle, _open_writer(
            output_dir / "gene_copy_numbers_selected.csv",
            COPY_NUMBER_EVENTS_HEADER,
        ) as cn_writer:
            _append_dataset_row(
                output_dir / "datasets.csv",
                COPY_NUMBER_DATASET_NAME,
                "copy_number",
            )
            counters["datasets"] += 1
            reader = csv.reader(handle)
            header = next(reader)
            parsed_genes = [_parse_gene_column_name(name) for name in header[6:]]
            threshold_rule = (
                f"amplification: value>={cn_amplification_threshold}; "
                f"deletion: value<={cn_deletion_threshold}"
            )
            selection_rule = (
                f"value>={cn_amplification_threshold} or value<={cn_deletion_threshold}"
            )
            for row in reader:
                if not _is_default_row_by_index(row):
                    continue
                model_id = row[2].strip()
                model_condition_id = row[4].strip()
                if not model_id:
                    continue
                for index, value in enumerate(row[6:]):
                    if value == "":
                        continue
                    try:
                        copy_number_value = float(value)
                    except ValueError:
                        continue
                    if copy_number_value >= cn_amplification_threshold:
                        status = "amplification"
                    elif copy_number_value <= cn_deletion_threshold:
                        status = "deletion"
                    else:
                        continue
                    gene_symbol, entrez_id = parsed_genes[index]
                    cn_writer.writerow(
                        {
                            "depmap_model_id": model_id,
                            "model_condition_id": model_condition_id,
                            "gene_symbol": gene_symbol,
                            "entrez_id": entrez_id,
                            "copy_number_value": value.strip(),
                            "value_type": COPY_NUMBER_VALUE_TYPE,
                            "status": status,
                            "threshold_rule": threshold_rule,
                            "selection_rule": selection_rule,
                        }
                    )
                    counters["copy_number_events"] += 1
                    counters[f"copy_number_{status}_events"] += 1

    if dependency_input_path is not None:
        with dependency_input_path.open(encoding="utf-8", newline="") as handle, _open_writer(
            output_dir / "gene_dependencies_selected.csv",
            GENE_DEPENDENCIES_HEADER,
        ) as dependency_writer:
            _append_dataset_row(
                output_dir / "datasets.csv",
                DEPENDENCY_DATASET_NAME,
                "dependency",
            )
            counters["datasets"] += 1
            reader = csv.reader(handle)
            header = next(reader)
            parsed_genes = [_parse_gene_column_name(name) for name in header[1:]]
            for row in reader:
                if not row:
                    continue

                model_id = row[0].strip()
                if not model_id:
                    continue

                selected_dependencies: list[tuple[float, str, str]] = []
                for index, value in enumerate(row[1:]):
                    item = value.strip()
                    if item == "":
                        continue
                    try:
                        dependency_score = float(item)
                    except ValueError:
                        continue

                    if dependency_score > DEPENDENCY_SCORE_THRESHOLD:
                        continue

                    gene_symbol, entrez_id = parsed_genes[index]
                    if not gene_symbol and not entrez_id:
                        continue
                    selected_dependencies.append((dependency_score, gene_symbol, entrez_id))

                selected_dependencies.sort(key=lambda entry: entry[0])
                for rank, (dependency_score, gene_symbol, entrez_id) in enumerate(
                    selected_dependencies[:DEPENDENCY_TOP_N],
                    start=1,
                ):
                    dependency_writer.writerow(
                        {
                            "depmap_model_id": model_id,
                            "gene_symbol": gene_symbol,
                            "entrez_id": entrez_id,
                            "dependency_score": str(dependency_score),
                            "value_type": DEPENDENCY_VALUE_TYPE,
                            "rank_within_cellline": str(rank),
                            "threshold_rule": DEPENDENCY_THRESHOLD_RULE,
                            "selection_rule": DEPENDENCY_SELECTION_RULE,
                        }
                    )
                    counters["gene_dependencies_selected"] += 1

                if selected_dependencies:
                    counters["cell_lines_with_dependencies"] += 1

    if expression_input_path is not None:
        gene_stats = _compute_expression_gene_stats(expression_input_path)
        with expression_input_path.open(encoding="utf-8", newline="") as handle, _open_writer(
            output_dir / "gene_expressions_selected.csv",
            GENE_EXPRESSIONS_HEADER,
        ) as expression_writer:
            _append_dataset_row(
                output_dir / "datasets.csv",
                EXPRESSION_DATASET_NAME,
                "gene_expression",
            )
            counters["datasets"] += 1
            reader = csv.reader(handle)
            header = next(reader)
            parsed_genes = [_parse_gene_column_name(name) for name in header[6:]]
            for row in reader:
                if not _is_default_row_by_index(row):
                    continue

                model_id = row[2].strip()
                if not model_id:
                    continue

                selected_expressions: list[tuple[float, float, float, str, str, str]] = []
                for index, value in enumerate(row[6:]):
                    item = value.strip()
                    if item == "":
                        continue
                    try:
                        expression_value = float(item)
                    except ValueError:
                        continue

                    stats = gene_stats[index]
                    if stats[1] == 0.0:
                        continue
                    z_score = (expression_value - stats[0]) / stats[1]
                    abs_z_score = abs(z_score)
                    if abs_z_score <= EXPRESSION_ZSCORE_THRESHOLD:
                        continue

                    gene_symbol, entrez_id = parsed_genes[index]
                    if not gene_symbol and not entrez_id:
                        continue
                    level = "high" if z_score > 0 else "low"
                    selected_expressions.append(
                        (abs_z_score, z_score, expression_value, gene_symbol, entrez_id, level)
                    )

                selected_expressions.sort(key=lambda entry: (-entry[0], entry[1]))
                for rank, (_, z_score, expression_value, gene_symbol, entrez_id, level) in enumerate(
                    selected_expressions[:EXPRESSION_TOP_N],
                    start=1,
                ):
                    expression_writer.writerow(
                        {
                            "depmap_model_id": model_id,
                            "gene_symbol": gene_symbol,
                            "entrez_id": entrez_id,
                            "expression_value": str(expression_value),
                            "value_type": EXPRESSION_VALUE_TYPE,
                            "z_score": str(z_score),
                            "level": level,
                            "rank_within_cellline": str(rank),
                            "threshold_rule": EXPRESSION_THRESHOLD_RULE,
                            "selection_rule": EXPRESSION_SELECTION_RULE,
                        }
                    )
                    counters["gene_expressions_selected"] += 1
                    counters[f"gene_expressions_{level}"] += 1

                if selected_expressions:
                    counters["cell_lines_with_expression_profiles"] += 1

    summary = {
        "model_input_path": str(model_input_path),
        "mutation_input_path": str(mutation_input_path) if mutation_input_path else "",
        "cn_input_path": str(cn_input_path) if cn_input_path else "",
        "dependency_input_path": str(dependency_input_path) if dependency_input_path else "",
        "expression_input_path": str(expression_input_path) if expression_input_path else "",
        "output_dir": str(output_dir),
        "counts": dict(counters),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _compute_expression_gene_stats(path: Path) -> list[tuple[float, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        gene_count = max(len(header) - 6, 0)
        counts = [0] * gene_count
        means = [0.0] * gene_count
        m2_values = [0.0] * gene_count

        for row in reader:
            if not _is_default_row_by_index(row):
                continue
            for index, value in enumerate(row[6:]):
                item = value.strip()
                if item == "":
                    continue
                try:
                    expression_value = float(item)
                except ValueError:
                    continue
                counts[index] += 1
                delta = expression_value - means[index]
                means[index] += delta / counts[index]
                delta2 = expression_value - means[index]
                m2_values[index] += delta * delta2

    stats: list[tuple[float, float]] = []
    for index in range(gene_count):
        if counts[index] < 2:
            stats.append((means[index], 0.0))
            continue
        variance = m2_values[index] / counts[index]
        stats.append((means[index], math.sqrt(variance)))
    return stats


def _write_mapping_rows(
    row: dict[str, str],
    mapping_writer: csv.DictWriter,
    counters: Counter,
) -> None:
    mapping_rows = [
        {
            "depmap_model_id": row["depmap_model_id"],
            "id_source": "DepMap",
            "mapped_id": row["depmap_model_id"],
            "mapping_type": "primary_model_id",
        }
    ]

    if row["rrid"]:
        mapping_rows.append(
            {
                "depmap_model_id": row["depmap_model_id"],
                "id_source": "Cellosaurus",
                "mapped_id": row["rrid"],
                "mapping_type": "rrid",
            }
        )

    if row["cosmic_id"]:
        mapping_rows.append(
            {
                "depmap_model_id": row["depmap_model_id"],
                "id_source": "COSMIC",
                "mapped_id": row["cosmic_id"],
                "mapping_type": "cosmic_id",
            }
        )

    for mapping_row in mapping_rows:
        mapping_writer.writerow(mapping_row)
        counters["id_mapping_rows"] += 1


def _write_dataset_row(
    datasets_writer: csv.DictWriter,
    dataset_name: str,
    dataset_type: str,
    counters: Counter,
) -> None:
    datasets_writer.writerow(
        {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "source": "DepMap",
            "version": DEP_MAP_VERSION,
        }
    )
    counters["datasets"] += 1


def _append_dataset_row(path: Path, dataset_name: str, dataset_type: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASETS_HEADER)
        writer.writerow(
            {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "source": "DepMap",
                "version": DEP_MAP_VERSION,
            }
        )


def _normalize_cosmic_id(value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def _normalize_rs_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    if normalized.lower().startswith("rs"):
        return normalized
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    if normalized.isdigit():
        return f"rs{normalized}"
    return normalized


def _parse_gene_column_name(value: str) -> tuple[str, str]:
    item = value.strip()
    if not item:
        return "", ""
    if item.endswith(")") and "(" in item:
        gene_symbol, entrez_id = item.rsplit("(", 1)
        return gene_symbol.strip(), entrez_id[:-1].strip()
    return item, ""


def _is_default_row(row: dict[str, str]) -> bool:
    return (
        row.get("IsDefaultEntryForModel", "").strip() == "Yes"
        and row.get("IsDefaultEntryForMC", "").strip() == "Yes"
    )


def _is_default_row_by_index(row: list[str]) -> bool:
    return len(row) > 5 and row[3].strip() == "Yes" and row[5].strip() == "Yes"


def _read_csv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


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
        description="Parse DepMap source files into source-level CSV tables."
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=Path("data/raw/depmap/Model.csv"),
        help="Path to DepMap Model.csv.",
    )
    parser.add_argument(
        "--mutation-input",
        type=Path,
        default=Path("data/raw/depmap/OmicsSomaticMutations.csv"),
        help="Path to DepMap OmicsSomaticMutations.csv.",
    )
    parser.add_argument(
        "--cn-input",
        type=Path,
        default=Path("data/raw/depmap/OmicsCNGeneWGS.csv"),
        help="Path to DepMap OmicsCNGeneWGS.csv.",
    )
    parser.add_argument(
        "--dependency-input",
        type=Path,
        default=Path("data/raw/depmap/CRISPRGeneEffect.csv"),
        help="Path to DepMap CRISPRGeneEffect.csv.",
    )
    parser.add_argument(
        "--expression-input",
        type=Path,
        default=Path("data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"),
        help="Path to DepMap OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv.",
    )
    parser.add_argument(
        "--cn-amplification-threshold",
        type=float,
        default=2.0,
        help="Copy-number value threshold used to keep amplification events.",
    )
    parser.add_argument(
        "--cn-deletion-threshold",
        type=float,
        default=0.5,
        help="Copy-number value threshold used to keep deletion events.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/depmap"),
        help="Directory where source-level DepMap tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = export_source_tables(
        args.model_input,
        args.output_dir,
        mutation_input_path=args.mutation_input,
        cn_input_path=args.cn_input,
        dependency_input_path=args.dependency_input,
        expression_input_path=args.expression_input,
        cn_amplification_threshold=args.cn_amplification_threshold,
        cn_deletion_threshold=args.cn_deletion_threshold,
    )
    print(
        json.dumps(
            {
                "cell_lines": summary["counts"].get("cell_lines", 0),
                "datasets": summary["counts"].get("datasets", 0),
                "disease_links": summary["counts"].get("disease_links", 0),
                "subtype_links": summary["counts"].get("subtype_links", 0),
                "somatic_mutations": summary["counts"].get("somatic_mutations", 0),
                "copy_number_events": summary["counts"].get("copy_number_events", 0),
                "gene_dependencies_selected": summary["counts"].get(
                    "gene_dependencies_selected", 0
                ),
                "gene_expressions_selected": summary["counts"].get(
                    "gene_expressions_selected", 0
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
