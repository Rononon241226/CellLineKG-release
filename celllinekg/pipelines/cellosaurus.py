from __future__ import annotations

import argparse
import json
from pathlib import Path

from celllinekg.mappings.cellosaurus import run_mapping
from celllinekg.sources.cellosaurus import export_source_tables


def run_pipeline(
    input_path: Path,
    source_output_dir: Path,
    schema_output_dir: Path,
) -> dict[str, object]:
    source_summary = export_source_tables(input_path, source_output_dir)
    schema_summary = run_mapping(source_output_dir, schema_output_dir)
    return {"source": source_summary, "schema": schema_summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Cellosaurus pipeline: raw file -> source tables -> schema tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/cellosaurus/cellosaurus.txt"),
        help="Path to the Cellosaurus flat file.",
    )
    parser.add_argument(
        "--source-output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/cellosaurus"),
        help="Directory where source-level parsed tables will be written.",
    )
    parser.add_argument(
        "--schema-output-dir",
        type=Path,
        default=Path("data/intermediate/schema/cellosaurus"),
        help="Directory where schema-level tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_pipeline(args.input, args.source_output_dir, args.schema_output_dir)
    print(
        json.dumps(
            {
                "cell_lines": summary["source"]["counts"].get("cell_lines", 0),
                "source_output_dir": summary["source"]["output_dir"],
                "schema_output_dir": summary["schema"]["output_dir"],
                "nodes_disease": summary["schema"]["counts"].get("nodes_disease", 0),
                "edges_cellline_has_disease": summary["schema"]["counts"].get(
                    "edges_cellline_has_disease", 0
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
