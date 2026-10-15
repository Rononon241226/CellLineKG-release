"""Run the CellLineKG framework from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celllinekg.framework import CellLineKGFramework


def main() -> None:
    """
    Run the CellLineKG framework for one user question.

    Arg:
        None. The question is read from command-line arguments.

    Returns:
        None. The final answer is printed.
    """
    parser = argparse.ArgumentParser(description="Run the CellLineKG framework.")
    parser.add_argument("question", help="User question.")
    args = parser.parse_args()

    framework = CellLineKGFramework()
    answer = framework.ask(args.question)
    print(answer)


if __name__ == "__main__":
    main()
