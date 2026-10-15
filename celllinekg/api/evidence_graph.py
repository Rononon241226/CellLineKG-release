"""QA evidence graph endpoint for CellLineKG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from celllinekg.evidence_graph import (
    build_evidence_graph,
    empty_evidence_graph,
    execute_text2cypher_evidence_query,
    merge_evidence_graphs,
)
from celllinekg.evidence_path_retrieval import run_evidence_path_retrieval_with_evidence
from celllinekg.framework.workflow_selector import select_workflow
from celllinekg.llm import get_chat_llm
from celllinekg.text2cypher import Text2CypherService


class QAEvidenceRequest(BaseModel):
    """Natural-language question request for an evidence graph."""

    question: str = Field(..., min_length=1)
    mode: Literal["auto", "text2cypher", "evidence_path_retrieval"] = "auto"
    include_debug: bool = False


class EvidenceNode(BaseModel):
    """One node in a QA evidence graph."""

    id: str
    label: str
    type: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    """One directed edge in a QA evidence graph."""

    id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class EvidenceGraph(BaseModel):
    """Nodes and edges used as QA evidence."""

    nodes: list[EvidenceNode] = Field(default_factory=list)
    edges: list[EvidenceEdge] = Field(default_factory=list)


class QAEvidenceResponse(BaseModel):
    """Natural-language answer with its evidence graph."""

    answer: str
    workflow: Literal["text2cypher", "evidence_path_retrieval"]
    graph: EvidenceGraph
    cypher: str | None = None
    rows: list[dict[str, Any]] | None = None
    debug: dict[str, Any] | None = None
    graph_warning: str | None = None


router = APIRouter(tags=["QA evidence graph"])


@router.get("/qa/evidence-view", include_in_schema=False)
def view_qa_evidence_graph() -> FileResponse:
    """
    Return the interactive D3 evidence graph page.

    Arg:
        None.

    Returns:
        The HTML page used to query and visualize QA evidence.
    """
    html_path = Path(__file__).resolve().parent / "static" / "qa_evidence_graph.html"
    return FileResponse(html_path, media_type="text/html")


@router.post("/qa/evidence-graph", response_model=QAEvidenceResponse)
def ask_question_with_evidence(request: QAEvidenceRequest) -> QAEvidenceResponse:
    """
    Answer a question and return the graph evidence used by the selected workflow.

    Arg:
        request: Question, workflow mode, and debug preference.

    Returns:
        The answer, selected workflow, query details, and evidence graph.
    """
    try:
        if request.mode == "auto":
            llm = get_chat_llm()
            workflow = select_workflow(llm=llm, question=request.question)
            if workflow == "evidence_path_retrieval":
                return _run_evidence_path_retrieval_with_evidence(
                    llm=llm,
                    question=request.question,
                )
            return _run_text2cypher_with_evidence(
                question=request.question,
                include_debug=request.include_debug,
            )

        if request.mode == "evidence_path_retrieval":
            llm = get_chat_llm()
            return _run_evidence_path_retrieval_with_evidence(
                llm=llm,
                question=request.question,
            )

        return _run_text2cypher_with_evidence(
            question=request.question,
            include_debug=request.include_debug,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - external services vary
        raise HTTPException(
            status_code=500,
            detail=f"CellLineKG evidence query failed: {exc}",
        ) from exc


def _run_text2cypher_with_evidence(
    question: str,
    include_debug: bool,
) -> QAEvidenceResponse:
    """
    Run the unchanged Text2Cypher answer flow and collect graph evidence separately.

    Arg:
        question: User question for Text2Cypher.
        include_debug: Whether to include existing Text2Cypher debug details.

    Returns:
        A Text2Cypher answer with its evidence graph.
    """
    service = Text2CypherService()
    payload = service.ask(question)
    cypher, rows = _extract_text2cypher_steps(payload)
    graph = build_evidence_graph(rows or [])
    graph_warning = None

    if cypher:
        try:
            companion_graph = execute_text2cypher_evidence_query(
                cypher,
                uri=service.settings.neo4j_uri,
                user=service.settings.neo4j_user,
                password=service.settings.neo4j_password,
            )
            graph = merge_evidence_graphs(graph, companion_graph)
        except Exception as exc:  # pragma: no cover - depends on generated Cypher and Neo4j
            # Evidence extraction must not replace a valid QA answer with an API failure.
            graph_warning = f"The answer was generated, but graph extraction failed: {exc}"

    debug = payload.get("debug")
    if not include_debug or not isinstance(debug, dict):
        debug = None

    return QAEvidenceResponse(
        answer=str(payload.get("answer", "")),
        workflow="text2cypher",
        graph=EvidenceGraph(**graph),
        cypher=cypher,
        rows=rows,
        debug=debug,
        graph_warning=graph_warning,
    )


def _run_evidence_path_retrieval_with_evidence(
    llm: Any,
    question: str,
) -> QAEvidenceResponse:
    """
    Run the evidence-path retrieval workflow and return its evidence graph.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path retrieval.

    Returns:
        An answer with the relevant Neo4j paths as an evidence graph.
    """
    payload = run_evidence_path_retrieval_with_evidence(llm=llm, question=question)
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        graph = empty_evidence_graph()

    return QAEvidenceResponse(
        answer=str(payload.get("answer", "")),
        workflow="evidence_path_retrieval",
        graph=EvidenceGraph(**graph),
        cypher=str(payload["cypher"]) if payload.get("cypher") else None,
    )


def _extract_text2cypher_steps(
    payload: dict[str, object],
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """
    Extract the final query and result preview from an existing Text2Cypher response.

    Arg:
        payload: Response returned by Text2CypherService.ask.

    Returns:
        The final Cypher query and optional result rows.
    """
    cypher = None
    rows = None
    intermediate_steps = payload.get("intermediate_steps", [])
    if not isinstance(intermediate_steps, list):
        return cypher, rows

    for step in intermediate_steps:
        if not isinstance(step, dict):
            continue
        if "query" in step:
            cypher = str(step["query"])
        if "context" in step and isinstance(step["context"], list):
            rows = step["context"]
    return cypher, rows
