"""Evidence-path retrieval for CellLineKG."""

from __future__ import annotations

import json
import os
from typing import Any

from celllinekg.llm.usage_tracker import invoke_with_tracking
from celllinekg.config import load_dotenv
from celllinekg.text2cypher.entity_resolver import EntityResolver
from celllinekg.text2cypher.models import EntityMention


ALLOWED_SEED_LABELS = {
    "CellLine",
    "Gene",
    "Protein",
    "Drug",
    "Subtype",
    "Disease",
    "Lineage",
}


MODEL_FILTER_RELATIONSHIPS = {
    "Lineage": "HAS_LINEAGE",
    "Disease": "HAS_DISEASE",
    "Subtype": "HAS_SUBTYPE",
}


TARGET_NAME_PROPERTIES = {
    "Gene": "symbol",
    "Protein": "symbol",
    "Drug": "name",
}


PATH_TEXT_FIELD = {
    "CellLine": "name",
    "Lineage": "name",
    "Disease": "name",
    "Subtype": "name",
    "Gene": "symbol",
    "Protein": "symbol",
    "Drug": "name",
    "Variant": "protein_change",
    "GeneExpression": "level",
    "Dependency": "score",
    "CopyNumber": "status",
    "ProteinAbundance": "level",
    "DrugScreen": "drug_sensitivity",
}


SIGNATURE_FIELD = {
    "CellLine": "name",
    "Lineage": "name",
    "Disease": "name",
    "Subtype": "name",
    "Gene": "symbol",
    "Protein": "symbol",
    "Drug": "name",
    "Variant": "variant_id",
    "GeneExpression": "expression_id",
    "Dependency": "dependency_id",
    "CopyNumber": "copy_number_id",
    "ProteinAbundance": "protein_abundance_id",
    "DrugScreen": "drug_screen_id",
}


NODE_PROPERTY_KEEP = {
    "CellLine": [
        "name",
    ],
    "Lineage": [
        "name",
    ],
    "Disease": [
        "name",
    ],
    "Subtype": [
        "name",
    ],
    "Gene": [
        "symbol",
    ],
    "Protein": [
        "symbol",
    ],
    "Drug": [
        "name",
        "target",
        "target_pathway",
    ],
    "Variant": [
        "protein_change",
        "variant_type",
        "rs_id",
    ],
    "GeneExpression": [
        "level",
    ],
    "Dependency": [
        "score",
    ],
    "CopyNumber": [
        "status",
    ],
    "ProteinAbundance": [
        "level",
    ],
    "DrugScreen": [
        "ic50",
        "drug_sensitivity",
    ],
}


SEED_EXTRACTION_SYSTEM_PROMPT = """You are a professional biomedical expert tasked with extracting and inferring seed entities for CellLineKG graph search.

Candidate labels:
CellLine, Gene, Protein, Drug, Subtype, Disease, Lineage

Rules:
- Extract explicit and implicit graph-search seed entities from the question.
- Use only candidate labels supported by CellLineKG.
- The same text may appear with multiple labels when it can map to multiple graph node types, e.g. EGFR as both Gene and Protein.
- For abstract biomedical concepts, output the closest graph-search seed only when it can guide retrieval.
- If the phrase is a generic organ/tissue + cancer phrase, classify it as Lineage and set text to the canonical organ/tissue noun only. Do not use adjective forms, e.g. "Bone cancer" -> {"text": "Bone", "label": "Lineage"}.
- Return a JSON array only. Do not include markdown or explanation.

Example:
Question: What biological evidence supports EGFR-targeted therapy in lung cancer models?
Output:
[
  {"text": "EGFR", "label": "Gene"},
  {"text": "lung", "label": "Lineage"}
]
"""


def extract_seeds(llm: Any, question: str) -> list[dict[str, str]]:
    """
    Extract seed entities for CellLineKG evidence-path retrieval.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path seed extraction.

    Returns:
        A list of seed dictionaries. Each seed has text and label fields.
    """
    response = invoke_with_tracking(
        llm,
        "seed_extraction",
        [
            ("system", SEED_EXTRACTION_SYSTEM_PROMPT),
            ("user", question),
        ]
    )
    content = str(getattr(response, "content", response)).strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()

    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in LLM seed extraction response.")

    payload = json.loads(content[start : end + 1])

    if not isinstance(payload, list):
        raise ValueError("Seed extraction response must be a JSON array.")

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).strip()
        if not text or label not in ALLOWED_SEED_LABELS:
            continue

        key = (text.lower(), label)
        if key in seen:
            continue

        normalized.append({"text": text, "label": label})
        seen.add(key)

    return normalized


def resolve_seeds(seeds: list[dict[str, str]]) -> list[dict[str, str | None]]:
    """
    Resolve evidence-path seeds to graph node names.

    Arg:
        seeds: Seed list from extract_seeds. Each seed has text and label fields.

    Returns:
        A seed list with an added name field. name is None when the seed cannot be resolved.
    """
    mentions: list[EntityMention] = []
    for seed in seeds:
        text = str(seed.get("text", "")).strip()
        label = str(seed.get("label", "")).strip()
        if not text or label not in ALLOWED_SEED_LABELS:
            continue
        mentions.append(
            EntityMention(
                text=text,
                entity_type=label,
                semantic_group="direct",
                modifiers=[],
                anchors=[],
            )
        )

    resolved_entities = EntityResolver().resolve(mentions)
    resolved_by_seed: dict[tuple[str, str], str | None] = {}
    for entity in resolved_entities:
        key = (entity.text.lower(), entity.entity_type)
        resolved_by_seed[key] = entity.resolved_name if entity.status == "resolved" else None

    resolved_seeds: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for seed in seeds:
        text = str(seed.get("text", "")).strip()
        label = str(seed.get("label", "")).strip()
        if not text or label not in ALLOWED_SEED_LABELS:
            continue

        key = (text.lower(), label)
        if key in seen:
            continue

        resolved_seeds.append(
            {
                "text": text,
                "label": label,
                "name": resolved_by_seed.get(key),
            }
        )
        seen.add(key)

    return resolved_seeds


def build_cellline_path_query(
    resolved_seeds: list[dict[str, str | None]],
    max_hops: int = 4,
    limit: int = 600,
) -> str:
    """
    Build a Cypher query for retrieving paths from CellLine nodes to resolved seed nodes.

    Arg:
        resolved_seeds: Seed list from resolve_seeds. Each seed has text, label, and name fields.
        max_hops: Maximum path length from CellLine nodes to target seed nodes.
        limit: Maximum number of paths returned by the query.

    Returns:
        A Cypher query string that can be executed against Neo4j.
    """
    match_clauses: list[str] = []
    target_conditions: list[str] = []

    model_index = 0
    target_index = 0
    for seed in resolved_seeds:
        label = str(seed.get("label", "")).strip()
        name = seed.get("name")
        name = str(name).strip() if name is not None else None

        if label == "CellLine" and name:
            escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
            match_clauses.append(f'MATCH (cl:CellLine {{name: "{escaped_name}"}})')
            model_index += 1
            continue

        if label in MODEL_FILTER_RELATIONSHIPS and name:
            escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
            relationship = MODEL_FILTER_RELATIONSHIPS[label]
            match_clauses.append(
                f'MATCH (cl:CellLine)-[:{relationship}]->(:{label} {{name: "{escaped_name}"}})'
            )
            model_index += 1
            continue

        if label in TARGET_NAME_PROPERTIES:
            property_name = TARGET_NAME_PROPERTIES[label]
            if name:
                escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
                target_conditions.append(
                    f'(target:{label} AND target.{property_name} = "{escaped_name}")'
                )
            else:
                target_conditions.append(f"(target:{label})")
            target_index += 1

    if not match_clauses:
        match_clauses.append("MATCH (cl:CellLine)")

    query_lines = [*match_clauses]
    query_lines.append(f"MATCH p = (cl)-[*1..{int(max_hops)}]->(target)")
    if target_conditions:
        query_lines.append("WHERE")
        query_lines.append("  " + "\n  OR ".join(target_conditions))
    query_lines.append("RETURN p")
    query_lines.append(f"LIMIT {int(limit)}")

    return "\n".join(query_lines)


def parse_neo4j_path(path: Any) -> dict[str, object]:
    """
    Parse a Neo4j path into graph evidence for evidence-path retrieval.

    Arg:
        path: Neo4j Path object returned as p from a Cypher query.

    Returns:
        A path dictionary with path_text, nodes, and relationships.
    """
    nodes = list(getattr(path, "nodes", []))
    relationships = list(getattr(path, "relationships", []))

    node_texts: list[str] = []
    parsed_nodes: list[dict[str, object]] = []

    for node in nodes:
        labels = list(getattr(node, "labels", []))
        label = next((item for item in PATH_TEXT_FIELD if item in labels), labels[0] if labels else "")
        properties = dict(node)

        path_text_field = PATH_TEXT_FIELD.get(label, "")
        node_text = str(properties.get(path_text_field, "") or label or "Node")
        node_texts.append(node_text)
        node_properties = {}
        if path_text_field:
            value = properties.get(path_text_field)
            if value is not None and value != "":
                node_properties[path_text_field] = value
        parsed_nodes.append(
            {
                "label": label,
                "properties": node_properties,
            }
        )

    path_parts: list[str] = []
    parsed_relationships: list[dict[str, str]] = []
    if node_texts:
        path_parts.append(node_texts[0])

    for index, relationship in enumerate(relationships):
        relationship_type = getattr(relationship, "type", relationship.__class__.__name__)
        parsed_relationships.append({"type": relationship_type})
        direction = "->"
        current_node = nodes[index] if index < len(nodes) else None
        next_node = nodes[index + 1] if index + 1 < len(nodes) else None
        start_node = getattr(relationship, "start_node", None)
        end_node = getattr(relationship, "end_node", None)
        if start_node is not None and end_node is not None and current_node is not None:
            current_id = getattr(current_node, "element_id", getattr(current_node, "id", None))
            next_id = getattr(next_node, "element_id", getattr(next_node, "id", None))
            start_id = getattr(start_node, "element_id", getattr(start_node, "id", None))
            end_id = getattr(end_node, "element_id", getattr(end_node, "id", None))
            if current_id == end_id and next_id == start_id:
                direction = "<-"

        if direction == "<-":
            path_parts.append(f"<-[{relationship_type}]-")
        else:
            path_parts.append(f"-[{relationship_type}]->")
        if index + 1 < len(node_texts):
            path_parts.append(node_texts[index + 1])

    return {
        "path_text": " ".join(path_parts),
        "nodes": parsed_nodes,
        "relationships": parsed_relationships,
    }


def build_evidence_table(paths: list[dict[str, object]]) -> dict[str, list[dict[str, str | None]]]:
    """
    Build an evidence table from retrieved graph paths.

    Arg:
        paths: Parsed path dictionaries from parse_neo4j_path.

    Returns:
        A table with cell-line rows and molecular evidence columns.
    """
    rows_by_key: dict[tuple[str, str], dict[str, str | None]] = {}

    for path in paths:
        nodes = path.get("nodes", [])
        if not isinstance(nodes, list):
            continue

        cell_line = None
        target_gene = None
        drug_name = None
        drug_sensitivity = None
        evidence_updates: dict[str, str] = {}

        for node in nodes:
            if not isinstance(node, dict):
                continue
            label = str(node.get("label", ""))
            properties = node.get("properties", {})
            if not isinstance(properties, dict):
                continue

            if label == "CellLine":
                value = str(properties.get("name", "") or "")
                if value:
                    cell_line = value
            elif label == "Gene":
                value = str(properties.get("symbol", "") or "")
                if value:
                    target_gene = value
            elif label == "Variant":
                value = str(properties.get("protein_change", "") or "")
                if value:
                    evidence_updates["mutation"] = value
            elif label == "GeneExpression":
                value = str(properties.get("level", "") or "")
                if value:
                    evidence_updates["gene_expression"] = value
            elif label == "CopyNumber":
                value = str(properties.get("status", "") or "")
                if value:
                    evidence_updates["copy_number_variation"] = value
            elif label == "Dependency":
                evidence_updates["dependency"] = "dependent"
            elif label == "ProteinAbundance":
                value = str(properties.get("level", "") or "")
                if value:
                    evidence_updates["protein_abundance"] = value
            elif label == "Drug":
                value = str(properties.get("name", "") or "")
                if value:
                    drug_name = value
            elif label == "DrugScreen":
                value = str(properties.get("drug_sensitivity", "") or "")
                if value:
                    drug_sensitivity = value

        if drug_sensitivity:
            if drug_name:
                evidence_updates["drug_response"] = f"{drug_name}: {drug_sensitivity}"
            else:
                evidence_updates["drug_response"] = drug_sensitivity

        if not cell_line or not target_gene:
            continue

        key = (cell_line, target_gene)
        if key not in rows_by_key:
            rows_by_key[key] = {
                "cell_line": cell_line,
                "target_gene": target_gene,
                "mutation": None,
                "gene_expression": None,
                "copy_number_variation": None,
                "dependency": None,
                "protein_abundance": None,
                "drug_response": None,
            }

        for column, value in evidence_updates.items():
            current_value = rows_by_key[key].get(column)
            if current_value:
                values = str(current_value).split("; ")
                if value not in values:
                    rows_by_key[key][column] = f"{current_value}; {value}"
            else:
                rows_by_key[key][column] = value

    return {"rows": list(rows_by_key.values())}


def format_evidence_table_markdown(table: dict[str, list[dict[str, str | None]]]) -> str:
    """
    Format an evidence table as a Markdown table.

    Arg:
        table: Evidence table from build_evidence_table.

    Returns:
        A Markdown table string.
    """
    base_columns = [
        ("cell_line", "Cell line"),
        ("target_gene", "Target gene"),
    ]
    evidence_columns = [
        ("mutation", "Mutation"),
        ("gene_expression", "Gene expression"),
        ("copy_number_variation", "Copy number variation"),
        ("dependency", "Dependency"),
        ("protein_abundance", "Protein abundance"),
        ("drug_response", "Drug response"),
    ]
    rows = table.get("rows", [])
    if not rows:
        return ""

    active_evidence_columns = [
        (key, title)
        for key, title in evidence_columns
        if any(row.get(key) not in {None, ""} for row in rows)
    ]
    columns = base_columns + active_evidence_columns

    header = "| " + " | ".join(title for _, title in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append(str(value) if value not in {None, ""} else "")
        body_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body_lines])


def generate_evidence_path_summary(llm: Any, question: str, table_markdown: str) -> str:
    """
    Generate a concise summary from the evidence-path table.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path retrieval.
        table_markdown: Markdown evidence table.

    Returns:
        A concise answer summary grounded only in the table.
    """
    system_prompt = """You are a biomedical assistant summarizing CellLineKG evidence.

rules:
Summarize only what is directly supported by the evidence table. Do not claim causality, mechanism, gain-of-function effects, or therapeutic relevance.
Do not add facts that are not shown in the table.
Blank cells mean that no evidence was retrieved for that column in the current table. Do not describe blank cells as absent, negative, normal, unchanged, or not reported.
Mention only non-empty evidence fields.
Use concise variant names (e.g. Q61R, R248Q) instead of HGVS prefixes like p.Q61R.
Write one concise paragraph.
"""
    user_prompt = f"""Question:
{question}

Evidence table:
{table_markdown}
"""
    response = invoke_with_tracking(
        llm,
        "evidence_path_summary",
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )
    return str(getattr(response, "content", response)).strip()


def filter_relevant_paths(
    llm: Any,
    question: str,
    paths: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Filter unrelated evidence paths using the user question.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path retrieval.
        paths: Parsed path dictionaries with path_text.

    Returns:
        The path dictionaries selected as relevant by the LLM.
    """
    path_lines = "\n".join(
        f"{index}. {path.get('path_text', '')}"
        for index, path in enumerate(paths)
    )
    system_prompt = """You are filtering evidence paths for CellLineKG.

Keep paths that are absolutely relevant to the user question.
Remove paths that merely contain the seed entities but do not match the requested evidence type.

Return a JSON array of selected path indexes only, such as:
[0, 2, 5]
"""
    user_prompt = f"""Question:
{question}

Paths:
{path_lines}
"""
    response = invoke_with_tracking(
        llm,
        "path_filtering",
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )
    content = str(getattr(response, "content", response)).strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()

    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        return paths

    try:
        selected_indexes = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return paths

    if not isinstance(selected_indexes, list):
        return paths

    selected_paths: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in selected_indexes:
        if not isinstance(item, int):
            continue
        if item in seen or item < 0 or item >= len(paths):
            continue
        selected_paths.append(paths[item])
        seen.add(item)

    return selected_paths


def run_evidence_path_retrieval(
    llm: Any,
    question: str,
    path_limit: int = 100,
    max_hops: int = 4,
) -> str | dict[str, str]:
    """
    Run the CellLineKG evidence-path retrieval workflow.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path retrieval.
        path_limit: Maximum number of paths allowed before returning an overflow answer.
        max_hops: Maximum path length from CellLine nodes to the Gene node.

    Returns:
        A Markdown table followed by a summary, or a simple unsupported/overflow message.
    """
    seeds = extract_seeds(llm=llm, question=question)
    resolved_seeds = resolve_seeds(seeds)

    model_clauses: list[str] = []
    gene_names: list[str] = []

    for seed in resolved_seeds:
        label = str(seed.get("label", "")).strip()
        name = seed.get("name")
        name = str(name).strip() if name is not None else None
        if not name:
            continue

        escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
        if label == "CellLine":
            model_clauses.append(f'MATCH (cl:CellLine {{name: "{escaped_name}"}})')
        elif label in MODEL_FILTER_RELATIONSHIPS:
            relationship = MODEL_FILTER_RELATIONSHIPS[label]
            model_clauses.append(
                f'MATCH (cl:CellLine)-[:{relationship}]->(:{label} {{name: "{escaped_name}"}})'
            )
        elif label == "Gene":
            gene_names.append(escaped_name)

    if not model_clauses or not gene_names:
        return {"message": "the question is unsupport"}

    gene_condition = " OR ".join(
        f'target.symbol = "{gene_name}"' for gene_name in dict.fromkeys(gene_names)
    )
    query_lines = [*model_clauses]
    query_lines.append(f"MATCH p = (cl)-[*1..{int(max_hops)}]->(target:Gene)")
    query_lines.append(f"WHERE {gene_condition}")
    query_lines.append("RETURN p")
    query_lines.append(f"LIMIT {int(path_limit) + 1}")
    query = "\n".join(query_lines)

    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Neo4j dependency is missing. Install it with: `python -m pip install neo4j`.") from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        records, _, _ = driver.execute_query(query)
        if len(records) > path_limit:
            return {"answer": "the query result is out of limitation"}

        parsed_paths: list[dict[str, object]] = []
        for record in records:
            path = record["p"]
            if path is None:
                continue
            parsed_path = parse_neo4j_path(path)
            if parsed_path["path_text"]:
                parsed_paths.append(parsed_path)
        if not parsed_paths:
            return {"message": "no relevant information found"}
        relevant_paths = filter_relevant_paths(llm=llm, question=question, paths=parsed_paths)
        table = build_evidence_table(relevant_paths)
        rows = table.get("rows", [])
        if not rows:
            return {"message": "no relevant information found"}

        table_markdown = format_evidence_table_markdown(table)
        if not table_markdown:
            return {"message": "no relevant information found"}

        if len(rows) == 1:
            return generate_evidence_path_summary(
                llm=llm,
                question=question,
                table_markdown=table_markdown,
            )
        return table_markdown
    finally:
        driver.close()


def run_evidence_path_retrieval_with_evidence(
    llm: Any,
    question: str,
    path_limit: int = 100,
    max_hops: int = 4,
) -> dict[str, object]:
    """
    Run evidence-path retrieval and keep the paths used to generate the answer.

    Arg:
        llm: A chat LLM client with an invoke method.
        question: User question for evidence-path retrieval.
        path_limit: Maximum number of paths allowed before returning an overflow answer.
        max_hops: Maximum path length from CellLine nodes to the Gene node.

    Returns:
        A dictionary containing the answer, Cypher query, and evidence graph.
    """
    from celllinekg.evidence_graph import build_evidence_graph, empty_evidence_graph

    seeds = extract_seeds(llm=llm, question=question)
    resolved_seeds = resolve_seeds(seeds)

    model_clauses: list[str] = []
    gene_names: list[str] = []

    for seed in resolved_seeds:
        label = str(seed.get("label", "")).strip()
        name = seed.get("name")
        name = str(name).strip() if name is not None else None
        if not name:
            continue

        escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
        if label == "CellLine":
            model_clauses.append(f'MATCH (cl:CellLine {{name: "{escaped_name}"}})')
        elif label in MODEL_FILTER_RELATIONSHIPS:
            relationship = MODEL_FILTER_RELATIONSHIPS[label]
            model_clauses.append(
                f'MATCH (cl:CellLine)-[:{relationship}]->(:{label} {{name: "{escaped_name}"}})'
            )
        elif label == "Gene":
            gene_names.append(escaped_name)

    if not model_clauses or not gene_names:
        return {
            "answer": "the question is unsupport",
            "cypher": None,
            "graph": empty_evidence_graph(),
        }

    gene_condition = " OR ".join(
        f'target.symbol = "{gene_name}"' for gene_name in dict.fromkeys(gene_names)
    )
    query_lines = [*model_clauses]
    query_lines.append(f"MATCH p = (cl)-[*1..{int(max_hops)}]->(target:Gene)")
    query_lines.append(f"WHERE {gene_condition}")
    query_lines.append("RETURN p")
    query_lines.append(f"LIMIT {int(path_limit) + 1}")
    query = "\n".join(query_lines)

    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Neo4j dependency is missing. Install it with: `python -m pip install neo4j`."
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        records, _, _ = driver.execute_query(query)
        if len(records) > path_limit:
            return {
                "answer": "the query result is out of limitation",
                "cypher": query,
                "graph": empty_evidence_graph(),
            }

        # Keep raw Neo4j paths beside parsed paths so graph identities and directions remain exact.
        path_pairs: list[tuple[Any, dict[str, object]]] = []
        for record in records:
            path = record["p"]
            if path is None:
                continue
            parsed_path = parse_neo4j_path(path)
            if parsed_path["path_text"]:
                path_pairs.append((path, parsed_path))

        if not path_pairs:
            return {
                "answer": "no relevant information found",
                "cypher": query,
                "graph": empty_evidence_graph(),
            }

        parsed_paths = [parsed_path for _, parsed_path in path_pairs]
        relevant_paths = filter_relevant_paths(
            llm=llm,
            question=question,
            paths=parsed_paths,
        )
        relevant_path_ids = {id(path) for path in relevant_paths}
        raw_relevant_paths = [
            raw_path
            for raw_path, parsed_path in path_pairs
            if id(parsed_path) in relevant_path_ids
        ]

        table = build_evidence_table(relevant_paths)
        rows = table.get("rows", [])
        if not rows:
            return {
                "answer": "no relevant information found",
                "cypher": query,
                "graph": empty_evidence_graph(),
            }

        table_markdown = format_evidence_table_markdown(table)
        if not table_markdown:
            return {
                "answer": "no relevant information found",
                "cypher": query,
                "graph": empty_evidence_graph(),
            }

        if len(rows) == 1:
            answer = generate_evidence_path_summary(
                llm=llm,
                question=question,
                table_markdown=table_markdown,
            )
        else:
            answer = table_markdown

        return {
            "answer": answer,
            "cypher": query,
            "graph": build_evidence_graph(raw_relevant_paths),
        }
    finally:
        driver.close()
