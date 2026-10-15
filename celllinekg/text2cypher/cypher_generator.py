"""LLM-based Cypher generation."""

from __future__ import annotations

import re
from typing import Any

from celllinekg.llm.usage_tracker import invoke_with_tracking

from .models import CypherGenerationResult, CypherVerificationResult, ResolvedEntity


GENERAL_CYPHER_PROMPT = """
Generate a Cypher query for CellLineKG.

Rules:
- Most questions are about CellLine-centered queries.
Always prioritize paths starting from CellLine and follow the schema centered on CellLine.
- Use only labels, relationships, and properties that explicitly exist in the provided schema.
- Relationship directions must strictly match the schema.
Never reverse a relationship direction.
For example, if the schema is:
(CellLine)-[:HAS_EXPRESSION]->(GeneExpression)
you must not generate:
(GeneExpression)-[:HAS_EXPRESSION]->(CellLine)
- When generating multi-hop queries, strictly follow the exact path order defined in the schema.
For example:
(CellLine)-[:HAS_EXPRESSION]->(GeneExpression)-[:MEASURED_GENE]->(Gene)
must not be reordered or reversed.
- Do not add Disease or Subtype nodes unless the user explicitly asks about diseases/subtypes or the resolved entities include them.
- Output only valid Cypher query.
Do not provide explanations, comments, markdown, or natural language.
- Entity listing queries must use DISTINCT.
""".strip()


class CypherGenerator:
    """Generate Cypher from question, schema context, and resolved entities."""

    def generate(
        self,
        llm: Any,
        question: str,
        schema_context: str,
        resolved_entities: list[ResolvedEntity],
        graph_schema: str,
        system_prompt: str = GENERAL_CYPHER_PROMPT,
    ) -> CypherGenerationResult:
        resolved_block = self._format_resolved_entities(resolved_entities)
        user_prompt = f"""Use the following graph context to generate a Cypher query.
Resolved entities:
{resolved_block}

Entity-specific schema:
{schema_context}

Database schema:
{graph_schema}

Question:
{question}
"""
        response = invoke_with_tracking(
            llm,
            "cypher_generation",
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        raw = str(getattr(response, "content", response)).strip()
        cypher = self._extract_cypher(raw)
        prompt = self._format_debug_prompt(system_prompt, user_prompt)
        return CypherGenerationResult(cypher=cypher, prompt=prompt, raw_response=raw)

    def verify_and_rewrite(
        self,
        llm: Any,
        question: str,
        cypher: str | None,
        schema_context: str,
        graph_schema: str,
    ) -> CypherVerificationResult:
        """Verify generated Cypher against selected schema and rewrite if needed."""
        if not cypher:
            return CypherVerificationResult(
                cypher=None,
                changed=False,
                prompt="",
                raw_response="",
            )

        system_prompt = """You verify Cypher queries for CellLineKG.

Rules:
- Compare the Cypher with the provided entity-specific schema.
- Every label, relationship, relationship direction, multi-hop path, and property must exist in the schema.
- If a path is not in the schema, rewrite the Cypher using the closest valid schema path.
- Do not invent labels, relationships, or properties.
- Preserve DISTINCT for entity listing queries.
- Output only the final executable Cypher query.
- If the original Cypher is already valid, output it unchanged.
"""
        user_prompt = f"""Question:
{question}

Generated Cypher:
{cypher}

Entity-specific schema:
{schema_context}

Database schema:
{graph_schema}
"""
        response = invoke_with_tracking(
            llm,
            "cypher_verification",
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        raw = str(getattr(response, "content", response)).strip()
        verified_cypher = self._extract_cypher(raw)
        prompt = self._format_debug_prompt(system_prompt, user_prompt)
        final_cypher = verified_cypher or cypher
        changed = bool(verified_cypher and verified_cypher.strip() != cypher.strip())
        return CypherVerificationResult(
            cypher=final_cypher,
            changed=changed,
            prompt=prompt,
            raw_response=raw,
        )

    def _format_debug_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"""System:
{system_prompt}

User:
{user_prompt}
"""

    def _format_resolved_entities(self, resolved_entities: list[ResolvedEntity]) -> str:
        if not resolved_entities:
            return "- none"
        lines: list[str] = []
        for entity in resolved_entities:
            lines.append(
                f"- mention={entity.text}; type={entity.entity_type}; status={entity.status}; "
                f"resolved_id={entity.resolved_id}; resolved_name={entity.resolved_name}; "
                f"match_field={entity.match_field}"
            )
        return "\n".join(lines)

    def _extract_cypher(self, raw: str) -> str | None:
        """Extract executable Cypher from an LLM response.

        Arg:
            raw: Raw LLM response that may contain markdown or explanatory text.

        Returns:
            Clean Cypher query text, or None when no query-like text is found.
        """
        cleaned = raw.strip()
        if not cleaned:
            return None

        fenced_match = re.search(
            r"```(?:cypher)?\s*(.*?)```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        else:
            start_match = re.search(
                r"\b(MATCH|OPTIONAL\s+MATCH|WITH|CALL|UNWIND|RETURN)\b",
                cleaned,
                flags=re.IGNORECASE,
            )
            if start_match:
                cleaned = cleaned[start_match.start() :].strip()
            else:
                return None

        cleaned = cleaned.strip().strip("`").strip()
        if cleaned.lower().startswith("cypher"):
            cleaned = cleaned[6:].strip()

        lines: list[str] = []
        for line in cleaned.splitlines():
            if line.strip().startswith("```"):
                break
            lines.append(line.rstrip())

        cypher = "\n".join(lines).strip()
        if cypher.endswith(";"):
            cypher = cypher[:-1].strip()
        return cypher or None
