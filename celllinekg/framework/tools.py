"""Tool wrappers for CellLineKG workflows."""

from __future__ import annotations

from typing import Any

from celllinekg.evidence_path_retrieval import run_evidence_path_retrieval
from celllinekg.text2cypher import Text2CypherService


def run_text2cypher_tool(question: str, llm: Any | None = None) -> str:
    """
    Run the Text2Cypher QA tool.

    Arg:
        question: User question for the Text2Cypher QA system.
        llm: Optional injected chat model, including a tracked model proxy.

    Returns:
        The final answer string.
    """
    result = Text2CypherService().ask(question, llm=llm)
    return str(result.get("answer", ""))


def run_evidence_path_retrieval_tool(llm: Any, question: str) -> str:
    """
    Run the CellLineKG evidence-path retrieval tool.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path retrieval.

    Returns:
        The final answer string.
    """
    result = run_evidence_path_retrieval(llm=llm, question=question)
    if isinstance(result, str):
        return result
    return str(result.get("answer") or result.get("message") or "")
