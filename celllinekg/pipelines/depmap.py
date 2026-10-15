from __future__ import annotations

import argparse
import json
from pathlib import Path

from celllinekg.mappings.depmap import run_mapping
from celllinekg.sources.depmap import export_source_tables


def run_pipeline(
    model_input_path: Path,
    mutation_input_path: Path,
    cn_input_path: Path,
    dependency_input_path: Path,
    expression_input_path: Path,
    source_output_dir: Path,
    schema_output_dir: Path,
    cn_amplification_threshold: float,
    cn_deletion_threshold: float,
) -> dict[str, object]:
    source_summary = export_source_tables(
        model_input_path,
        source_output_dir,
        mutation_input_path=mutation_input_path,
        cn_input_path=cn_input_path,
        dependency_input_path=dependency_input_path,
        expression_input_path=expression_input_path,
        cn_amplification_threshold=cn_amplification_threshold,
        cn_deletion_threshold=cn_deletion_threshold,
    )
    schema_summary = run_mapping(source_output_dir, schema_output_dir)
    return {"source": source_summary, "schema": schema_summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DepMap pipeline: source files -> source tables -> schema tables."
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
        "--source-output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/depmap"),
        help="Directory where source-level parsed tables will be written.",
    )
    parser.add_argument(
        "--schema-output-dir",
        type=Path,
        default=Path("data/intermediate/schema/depmap"),
        help="Directory where schema-level tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_pipeline(
        args.model_input,
        args.mutation_input,
        args.cn_input,
        args.dependency_input,
        args.expression_input,
        args.source_output_dir,
        args.schema_output_dir,
        args.cn_amplification_threshold,
        args.cn_deletion_threshold,
    )
    print(
        json.dumps(
            {
                "cell_lines": summary["source"]["counts"].get("cell_lines", 0),
                "somatic_mutations": summary["source"]["counts"].get("somatic_mutations", 0),
                "copy_number_events": summary["source"]["counts"].get("copy_number_events", 0),
                "gene_dependencies_selected": summary["source"]["counts"].get(
                    "gene_dependencies_selected", 0
                ),
                "gene_expressions_selected": summary["source"]["counts"].get(
                    "gene_expressions_selected", 0
                ),
                "source_output_dir": summary["source"]["output_dir"],
                "schema_output_dir": summary["schema"]["output_dir"],
                "nodes_gene_expression": summary["schema"]["counts"].get(
                    "nodes_gene_expression", 0
                ),
                "edges_cellline_has_expression": summary["schema"]["counts"].get(
                    "edges_cellline_has_expression", 0
                ),
                "cell_line_id_mapping_rows": summary["schema"]["counts"].get(
                    "cell_line_id_mapping_rows", 0
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
