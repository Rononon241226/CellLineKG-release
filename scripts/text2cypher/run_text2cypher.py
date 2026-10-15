"""Run the CellLineKG Text2Cypher workflow from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celllinekg.text2cypher.text2cypher_service import Text2CypherService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CellLineKG Text2Cypher workflow against Neo4j."
    )
    parser.add_argument("question", help="Natural-language question to ask the graph.")
    parser.add_argument(
        "--schema-mode",
        choices=["full", "filtered"],
        default="full",
        help="Schema context mode. Use filtered to omit the full database schema.",
    )
    args = parser.parse_args()

    service = Text2CypherService(schema_mode=args.schema_mode)
    payload = service.ask(args.question)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
