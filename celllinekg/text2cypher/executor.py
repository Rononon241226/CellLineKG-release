"""Cypher execution helpers."""

from __future__ import annotations

from typing import Any

from .models import QueryExecutionResult


class CypherExecutor:
    """Execute generated Cypher against Neo4jGraph."""

    def execute(self, graph: Any, cypher: str | None) -> QueryExecutionResult:
        if not cypher:
            return QueryExecutionResult(cypher=None, rows=[], error="No Cypher was generated.")

        try:
            rows = graph.query(cypher)
            if not isinstance(rows, list):
                rows = list(rows)
            return QueryExecutionResult(cypher=cypher, rows=rows, error=None)
        except Exception as exc:
            return QueryExecutionResult(cypher=cypher, rows=[], error=str(exc))

