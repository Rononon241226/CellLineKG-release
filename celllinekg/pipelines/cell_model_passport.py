from __future__ import annotations

import argparse
import json
from pathlib import Path

from celllinekg.mappings.cell_model_passport import run_mapping
from celllinekg.sources.cell_model_passport import export_source_tables


def run_pipeline(
    proteomics_input_path: Path,
    model_list_input_path: Path | None,
    source_output_dir: Path,
    schema_output_dir: Path,
) -> dict[str, object]:
    source_summary = export_source_tables(
        proteomics_input_path,
        source_output_dir,
        model_list_input_path=model_list_input_path,
    )
    schema_summary = run_mapping(source_output_dir, schema_output_dir)
    return {"source": source_summary, "schema": schema_summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Cell Model Passports pipeline: source files -> source tables -> schema tables."
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
        "--source-output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/cell_model_passport"),
        help="Directory where source-level parsed tables will be written.",
    )
    parser.add_argument(
        "--schema-output-dir",
        type=Path,
        default=Path("data/intermediate/schema/cell_model_passport"),
        help="Directory where schema-level tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    model_list_input = args.model_list_input if args.model_list_input.exists() else None
    summary = run_pipeline(
        args.proteomics_input,
        model_list_input,
        args.source_output_dir,
        args.schema_output_dir,
    )
    print(
        json.dumps(
            {
                "cell_lines": summary["source"]["counts"].get("cell_lines", 0),
                "protein_abundances_selected": summary["source"]["counts"].get(
                    "protein_abundances_selected", 0
                ),
                "source_output_dir": summary["source"]["output_dir"],
                "schema_output_dir": summary["schema"]["output_dir"],
                "nodes_protein": summary["schema"]["counts"].get("nodes_protein", 0),
                "nodes_protein_abundance": summary["schema"]["counts"].get(
                    "nodes_protein_abundance", 0
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
