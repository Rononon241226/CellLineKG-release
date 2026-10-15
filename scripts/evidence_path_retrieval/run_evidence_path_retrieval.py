"""Run CellLineKG evidence-path retrieval from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celllinekg.evidence_path_retrieval import run_evidence_path_retrieval
from celllinekg.llm import get_chat_llm


def main() -> None:
    """
    Run CellLineKG evidence-path retrieval for one question.

    Arg:
        None. The question is read from command-line arguments.

    Returns:
        None. The evidence-path retrieval result is printed.
    """
    parser = argparse.ArgumentParser(description="Run CellLineKG evidence-path retrieval.")
    parser.add_argument("question", help="User question for evidence-path retrieval.")
    args = parser.parse_args()

    llm = get_chat_llm()
    result = run_evidence_path_retrieval(llm=llm, question=args.question)
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
