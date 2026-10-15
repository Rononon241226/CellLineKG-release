"""Select a CellLineKG workflow for each user question."""

from __future__ import annotations

import json
from typing import Any

from celllinekg.llm.usage_tracker import invoke_with_tracking


WORKFLOW_SELECTOR_SYSTEM_PROMPT = """You select the appropriate CellLineKG workflow for each user question.

Choose exactly one workflow for the user question.

Available workflows:
- text2cypher: Use for specific factual questions that ask for specific entities, lists, counts, properties, values, or relationships such as gene copy number, gene dependency scores from CRISPR knockout screens, drug sensitivity and resistance, gene expression levels, protein abundance measurements and gene mutation.
- evidence_path_retrieval: Use evidence-path retrieval when the question asks for graph evidence paths linking a specific gene to a specific cell line, especially for molecular profile, landscape, integrated evidence, biological evidence, experimental evidence, or pharmacological evidence.

Return JSON only:
{"workflow": "text2cypher"}
or
{"workflow": "evidence_path_retrieval"}
"""


def select_workflow(llm: Any, question: str) -> str:
    """
    Choose the workflow for one CellLineKG user question.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question to route.

    Returns:
        The selected workflow name: text2cypher or evidence_path_retrieval.
    """
    response = invoke_with_tracking(
        llm,
        "workflow_selection",
        [
            ("system", WORKFLOW_SELECTOR_SYSTEM_PROMPT),
            ("user", question),
        ]
    )
    content = str(getattr(response, "content", response)).strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return "text2cypher"

    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return "text2cypher"

    workflow = str(payload.get("workflow", "")).strip().lower()
    if workflow not in {"text2cypher", "evidence_path_retrieval"}:
        return "text2cypher"
    return workflow
