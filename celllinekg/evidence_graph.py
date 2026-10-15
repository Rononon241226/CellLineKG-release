"""Build API-ready evidence graphs from Neo4j query values."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


PREFERRED_LABEL_FIELDS = (
    "name",
    "symbol",
    "cellosaurus_id",
    "gene_id",
    "protein_id",
    "drug_id",
    "variant_id",
    "expression_id",
    "dependency_id",
    "copy_number_id",
    "protein_abundance_id",
    "drug_screen_id",
)


def empty_evidence_graph() -> dict[str, list[dict[str, Any]]]:
    """
    Create an empty evidence graph.

    Arg:
        None.

    Returns:
        A dictionary with empty nodes and edges lists.
    """
    return {"nodes": [], "edges": []}


def build_evidence_graph(values: Any) -> dict[str, list[dict[str, Any]]]:
    """
    Convert nested Neo4j values into nodes and edges for the API.

    Arg:
        values: A Neo4j node, relationship, path, record value, or nested collection.

    Returns:
        A dictionary containing deduplicated nodes and edges.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def add_node(node: Any) -> str | None:
        if not _is_neo4j_node(node):
            return None

        properties = _json_safe(dict(node))
        labels = sorted(str(label) for label in getattr(node, "labels", []))
        node_type = labels[0] if labels else "Node"
        node_id = _neo4j_identity(node, node_type, properties)
        nodes[node_id] = {
            "id": node_id,
            "label": _display_label(properties, node_type),
            "type": node_type,
            "labels": labels,
            "properties": properties,
        }
        return node_id

    def add_relationship(relationship: Any) -> None:
        if not _is_neo4j_relationship(relationship):
            return

        source_id = add_node(getattr(relationship, "start_node", None))
        target_id = add_node(getattr(relationship, "end_node", None))
        if not source_id or not target_id:
            return

        relationship_type = str(
            getattr(relationship, "type", relationship.__class__.__name__)
        )
        properties = _json_safe(dict(relationship))
        edge_id = _neo4j_identity(
            relationship,
            relationship_type,
            {
                "source": source_id,
                "target": target_id,
                "properties": properties,
            },
        )
        edges[edge_id] = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "type": relationship_type,
            "properties": properties,
        }

    def visit(value: Any) -> None:
        if value is None:
            return
        if _is_neo4j_path(value):
            for node in getattr(value, "nodes", []):
                add_node(node)
            for relationship in getattr(value, "relationships", []):
                add_relationship(relationship)
            return
        if _is_neo4j_relationship(value):
            add_relationship(value)
            return
        if _is_neo4j_node(value):
            add_node(value)
            return
        if isinstance(value, Mapping):
            for nested_value in value.values():
                visit(nested_value)
            return
        if isinstance(value, (list, tuple, set)):
            for nested_value in value:
                visit(nested_value)

    visit(values)
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def merge_evidence_graphs(
    *graphs: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Merge evidence graphs without duplicating nodes or edges.

    Arg:
        graphs: Evidence graph dictionaries to merge.

    Returns:
        One evidence graph containing unique nodes and edges.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for graph in graphs:
        for node in graph.get("nodes", []):
            node_id = str(node.get("id", ""))
            if node_id:
                nodes[node_id] = node
        for edge in graph.get("edges", []):
            edge_id = str(edge.get("id", ""))
            if edge_id:
                edges[edge_id] = edge
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def build_text2cypher_evidence_query(
    cypher: str,
    max_nodes: int = 100,
) -> str | None:
    """
    Build a read-only companion query from a Text2Cypher query.

    Arg:
        cypher: The verified Cypher query used to generate the QA answer.
        max_nodes: Maximum number of evidence nodes included in the graph.

    Returns:
        A companion Cypher query, or None when the source query cannot be adapted safely.
    """
    # Never replay a query prefix that can mutate the graph or invoke a procedure.
    forbidden_clause = re.search(
        r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|FOREACH|UNION)\b",
        cypher,
        flags=re.IGNORECASE,
    )
    if forbidden_clause:
        return None

    return_matches = list(re.finditer(r"\bRETURN\b", cypher, flags=re.IGNORECASE))
    if not return_matches:
        return None

    # Use the last RETURN so subqueries before it remain intact.
    query_prefix = cypher[: return_matches[-1].start()].rstrip()
    if not query_prefix:
        return None

    node_variables = list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(
                r"\(\s*([A-Za-z_]\w*)\s*(?=[:{)])",
                query_prefix,
            )
        )
    )
    if not node_variables:
        return None

    relationship_types = list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(
                r"\[\s*(?:[A-Za-z_]\w*\s*)?:\s*([A-Za-z_]\w*)",
                query_prefix,
            )
        )
    )

    node_list = ", ".join(node_variables)
    query_lines = [
        query_prefix,
        f"WITH [candidate IN [{node_list}] WHERE candidate IS NOT NULL] AS row_nodes",
        "LIMIT 500",
        "UNWIND row_nodes AS evidence_node",
        f"WITH collect(DISTINCT evidence_node)[..{max(1, int(max_nodes))}] AS evidence_nodes",
    ]

    if relationship_types:
        escaped_types = ", ".join(
            f'"{relationship_type}"' for relationship_type in relationship_types
        )
        query_lines.extend(
            [
                "UNWIND evidence_nodes AS evidence_source",
                "UNWIND evidence_nodes AS evidence_target",
                "OPTIONAL MATCH (evidence_source)-[evidence_relationship]->(evidence_target)",
                f"WHERE type(evidence_relationship) IN [{escaped_types}]",
                "RETURN evidence_nodes AS nodes,",
                "       collect(DISTINCT evidence_relationship) AS relationships",
            ]
        )
    else:
        query_lines.append("RETURN evidence_nodes AS nodes, [] AS relationships")

    return "\n".join(query_lines)


def execute_text2cypher_evidence_query(
    cypher: str,
    *,
    uri: str,
    user: str,
    password: str,
    max_nodes: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """
    Execute a companion query and return its evidence graph.

    Arg:
        cypher: The verified Cypher query used for the QA answer.
        uri: Neo4j connection URI.
        user: Neo4j user name.
        password: Neo4j password.
        max_nodes: Maximum number of evidence nodes included in the graph.

    Returns:
        A dictionary containing evidence nodes and edges.
    """
    evidence_query = build_text2cypher_evidence_query(cypher, max_nodes=max_nodes)
    if not evidence_query:
        return empty_evidence_graph()

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Neo4j dependency is missing. Install it with: `python -m pip install neo4j`."
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        records, _, _ = driver.execute_query(evidence_query)
        # Keep raw Neo4j values because Record.data() converts graph entities to plain dictionaries.
        raw_values = [value for record in records for value in record.values()]
        return build_evidence_graph(raw_values)
    finally:
        driver.close()


def _is_neo4j_node(value: Any) -> bool:
    return isinstance(value, Mapping) and hasattr(value, "labels")


def _is_neo4j_relationship(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and hasattr(value, "start_node")
        and hasattr(value, "end_node")
    )


def _is_neo4j_path(value: Any) -> bool:
    return hasattr(value, "nodes") and hasattr(value, "relationships")


def _neo4j_identity(value: Any, value_type: str, fallback: Any) -> str:
    identity = getattr(value, "element_id", getattr(value, "id", None))
    if identity is not None:
        return str(identity)
    serialized = json.dumps(_json_safe(fallback), sort_keys=True, ensure_ascii=False)
    return f"{value_type}:{serialized}"


def _display_label(properties: dict[str, Any], fallback: str) -> str:
    for field in PREFERRED_LABEL_FIELDS:
        value = properties.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
