from __future__ import annotations

import argparse
import json
from pathlib import Path

from celllinekg.mappings.gdsc import run_mapping
from celllinekg.sources.gdsc import export_source_tables


def run_pipeline(
    dose_response_input_path: Path,
    compounds_input_path: Path,
    source_output_dir: Path,
    schema_output_dir: Path,
) -> dict[str, object]:
    source_summary = export_source_tables(
        dose_response_input_path,
        compounds_input_path,
        source_output_dir,
    )
    schema_summary = run_mapping(source_output_dir, schema_output_dir)
    return {"source": source_summary, "schema": schema_summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the GDSC pipeline: source files -> source tables -> schema tables."
    )
    parser.add_argument(
        "--dose-response-input",
        type=Path,
        default=Path("data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"),
        help="Path to GDSC fitted dose-response Excel file.",
    )
    parser.add_argument(
        "--compounds-input",
        type=Path,
        default=Path("data/raw/gdsc/screened_compounds_rel_8.5.csv"),
        help="Path to GDSC screened compounds CSV file.",
    )
    parser.add_argument(
        "--source-output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/gdsc"),
        help="Directory where source-level parsed tables will be written.",
    )
    parser.add_argument(
        "--schema-output-dir",
        type=Path,
        default=Path("data/intermediate/schema/gdsc"),
        help="Directory where schema-level tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_pipeline(
        args.dose_response_input,
        args.compounds_input,
        args.source_output_dir,
        args.schema_output_dir,
    )
    print(
        json.dumps(
            {
                "cell_lines": summary["source"]["counts"].get("cell_lines", 0),
                "drug_screens": summary["source"]["counts"].get("drug_screens", 0),
                "drugs": summary["source"]["counts"].get("drugs", 0),
                "datasets": summary["source"]["counts"].get("datasets", 0),
                "source_output_dir": summary["source"]["output_dir"],
                "schema_output_dir": summary["schema"]["output_dir"],
                "nodes_drug_screen": summary["schema"]["counts"].get("nodes_drug_screen", 0),
                "edges_drug_screen_tested_drug": summary["schema"]["counts"].get(
                    "edges_drug_screen_tested_drug", 0
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
