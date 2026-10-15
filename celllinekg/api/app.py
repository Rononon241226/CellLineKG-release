#python -m uvicorn celllinekg.api.app:app --host 127.0.0.1 --port 8000 --reload

"""HTTP API for CellLineKG."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from celllinekg.api.evidence_graph import router as evidence_graph_router
from celllinekg.config import load_dotenv
from celllinekg.evidence_path_retrieval import run_evidence_path_retrieval
from celllinekg.framework.workflow_selector import select_workflow
from celllinekg.llm import get_chat_llm
from celllinekg.text2cypher import Text2CypherService


SUPPORTED_ENTITY_LABELS = {
    "CellLine": {
        "id": "cellosaurus_id",
        "name": "name",
        "search": ["cellosaurus_id", "name", "synonyms"],
    },
    "Gene": {
        "id": "gene_id",
        "name": "symbol",
        "search": ["gene_id", "entrez_id", "symbol", "name"],
    },
    "Protein": {
        "id": "protein_id",
        "name": "symbol",
        "search": ["protein_id", "uniprot_id", "symbol"],
    },
    "Drug": {
        "id": "drug_id",
        "name": "name",
        "search": ["drug_id", "name", "synonyms", "mychem_id", "external_aliases", "target"],
    },
    "Disease": {
        "id": "disease_id",
        "name": "name",
        "search": ["disease_id", "name", "oncotree_disease_code", "ncit_disease_id"],
    },
    "Subtype": {
        "id": "subtype_id",
        "name": "name",
        "search": ["subtype_id", "name", "oncotree_subtype_code", "ncit_subtype_id"],
    },
    "Lineage": {
        "id": "lineage_id",
        "name": "name",
        "search": ["lineage_id", "name"],
    },
}


class QARequest(BaseModel):
    """Natural-language question request."""

    question: str = Field(..., min_length=1)
    mode: Literal["auto", "text2cypher", "evidence_path_retrieval"] = "auto"
    include_debug: bool = False


class QAResponse(BaseModel):
    """Natural-language question response."""

    answer: str
    workflow: Literal["text2cypher", "evidence_path_retrieval"]
    cypher: str | None = None
    rows: list[dict[str, Any]] | None = None
    debug: dict[str, Any] | None = None


app = FastAPI(
    title="CellLineKG API",
    version="0.1.0",
    description="HTTP API for querying CellLineKG over Neo4j.",
)
app.include_router(evidence_graph_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service and Neo4j connectivity status."""
    try:
        _run_read_query("RETURN 1 AS ok")
    except Exception as exc:  # pragma: no cover - depends on local Neo4j
        return {"status": "degraded", "neo4j": "error", "detail": str(exc)}
    return {"status": "ok", "neo4j": "ok"}


@app.post("/qa", response_model=QAResponse)
def ask_question(request: QARequest) -> QAResponse:
    """Answer a natural-language question with a CellLineKG workflow."""
    try:
        if request.mode == "auto":
            llm = get_chat_llm()
            workflow = select_workflow(llm=llm, question=request.question)
            if workflow == "evidence_path_retrieval":
                return _run_evidence_path_retrieval(llm=llm, question=request.question)
            return _run_text2cypher(request.question, include_debug=request.include_debug)

        if request.mode == "evidence_path_retrieval":
            llm = get_chat_llm()
            return _run_evidence_path_retrieval(llm=llm, question=request.question)

        return _run_text2cypher(request.question, include_debug=request.include_debug)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - external services vary
        raise HTTPException(status_code=500, detail=f"CellLineKG query failed: {exc}") from exc


@app.get("/cell-lines/{cellosaurus_id}")
def get_cell_line(cellosaurus_id: str) -> dict[str, Any]:
    """Return one CellLine and its main graph neighborhoods."""
    query = """
    MATCH (c:CellLine {cellosaurus_id: $cellosaurus_id})
    OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease)
    OPTIONAL MATCH (c)-[:HAS_SUBTYPE]->(s:Subtype)
    OPTIONAL MATCH (c)-[:HAS_LINEAGE]->(l:Lineage)
    OPTIONAL MATCH (c)-[:CULTURES_IN]->(m:Media)
    RETURN
      properties(c) AS cell_line,
      collect(DISTINCT properties(d)) AS diseases,
      collect(DISTINCT properties(s)) AS subtypes,
      collect(DISTINCT properties(l)) AS lineages,
      collect(DISTINCT properties(m)) AS media
    """
    try:
        rows = _run_read_query(query, {"cellosaurus_id": cellosaurus_id})
    except Exception as exc:  # pragma: no cover - depends on local Neo4j
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not rows:
        raise HTTPException(status_code=404, detail="CellLine not found.")
    return _drop_empty_collections(rows[0])


@app.get("/search/entities")
def search_entities(
    q: str = Query(..., min_length=1),
    label: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Search supported graph entities by identifier, name, or synonym fields."""
    labels = _select_labels(label)
    per_label_limit = max(1, min(limit, 100))
    results: list[dict[str, Any]] = []

    try:
        for entity_label in labels:
            results.extend(_search_label(entity_label, q, per_label_limit))
    except Exception as exc:  # pragma: no cover - depends on local Neo4j
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"query": q, "count": len(results[:limit]), "results": results[:limit]}


def _run_text2cypher(question: str, include_debug: bool) -> QAResponse:
    payload = Text2CypherService().ask(question)
    debug = payload.get("debug") if include_debug and isinstance(payload.get("debug"), dict) else None
    intermediate_steps = payload.get("intermediate_steps", [])
    cypher = None
    rows = None
    if isinstance(intermediate_steps, list):
        for step in intermediate_steps:
            if not isinstance(step, dict):
                continue
            if "query" in step:
                cypher = str(step["query"])
            if "context" in step and isinstance(step["context"], list):
                rows = step["context"]

    return QAResponse(
        answer=str(payload.get("answer", "")),
        workflow="text2cypher",
        cypher=cypher,
        rows=rows,
        debug=debug,
    )


def _run_evidence_path_retrieval(llm: Any, question: str) -> QAResponse:
    result = run_evidence_path_retrieval(llm=llm, question=question)
    if isinstance(result, dict):
        answer = str(result.get("answer") or result.get("message") or "")
    else:
        answer = str(result)
    return QAResponse(answer=answer, workflow="evidence_path_retrieval")


def _search_label(label: str, text: str, limit: int) -> list[dict[str, Any]]:
    metadata = SUPPORTED_ENTITY_LABELS[label]
    id_field = metadata["id"]
    name_field = metadata["name"]
    search_fields = metadata["search"]
    where_clause = " OR ".join(
        f"toLower(coalesce(toString(n.{field}), '')) CONTAINS $q" for field in search_fields
    )
    query = f"""
    MATCH (n:{label})
    WHERE {where_clause}
    RETURN
      $label AS label,
      toString(n.{id_field}) AS id,
      toString(n.{name_field}) AS name,
      properties(n) AS properties
    LIMIT $limit
    """
    return _run_read_query(
        query,
        {
            "q": text.lower(),
            "label": label,
            "limit": limit,
        },
    )


def _select_labels(label: str | None) -> list[str]:
    if label is None:
        return list(SUPPORTED_ENTITY_LABELS)
    normalized = label.strip()
    for supported_label in SUPPORTED_ENTITY_LABELS:
        if supported_label.lower() == normalized.lower():
            return [supported_label]
    supported = ", ".join(SUPPORTED_ENTITY_LABELS)
    raise HTTPException(status_code=400, detail=f"Unsupported label. Use one of: {supported}.")


def _run_read_query(
    query: str,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    load_dotenv()
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Neo4j dependency is missing. Install it with: `python -m pip install neo4j`.") from exc

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            records = session.run(query, parameters or {})
            return [record.data() for record in records]
    finally:
        driver.close()


def _drop_empty_collections(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    for key, value in list(cleaned.items()):
        if isinstance(value, list):
            cleaned[key] = [item for item in value if item]
    return cleaned

