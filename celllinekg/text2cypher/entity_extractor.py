"""LLM-based entity mention extraction."""

from __future__ import annotations

import json
from typing import Any

from celllinekg.llm.usage_tracker import invoke_with_tracking

from .models import EntityAnchor, EntityMention


ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a professional biomedical assistant extracting entity mentions from CellLineKG user questions.

Return only valid JSON with this schema:
{
  "mentions": [
    {
      "text": "...",
      "entity_type": "...",
      "semantic_group": "...",
      "modifiers": ["..."],
      "anchors": [
        {"text": "...", "entity_type": "..."}
      ]
    }
  ]
}

Field meanings:
- `text`: the mention text used for graph entity resolution. 
- `entity_type`: CellLine|Dataset|Drug|Gene|Media|Protein|Variant|Lineage|Disease|Subtype|CopyNumber|Dependency|DrugScreen|GeneExpression|ProteinAbundance, written exactly as one of the allowed entity names.
- `semantic_group`: use `direct` for directly mentioned graph entities; use `indirect` for analysis semantics that depend on other entities.
- `modifiers`: normalized semantic qualifiers for indirect entities when present, such as `amplification`, `deletion`, `sensitive`, `resistant`, `high`, or `low`.
- `anchors`: the directly mentioned entities that an indirect entity is attached to. Each anchor must include both `text` and `entity_type`.

Rules:
- Direct entities are: CellLine, Drug, Gene, Media, Protein, Lineage, Disease, Subtype.
- Indirect entities are: CopyNumber, Dependency, DrugScreen, GeneExpression, ProteinAbundance, Variant. CellLineKG contains corresponding analysis data types: gene copy number, gene dependency scores from CRISPR knockout screens, drug sensitivity and resistance, gene expression levels, protein abundance measurements and gene mutation. When the user's question semantically implies any of these data types, list the corresponding indirect entity.
- CellLine entitity is only extracted when specific cell line names are mentioned.
- For direct mentions, return empty `modifiers` and empty `anchors`.
- For indirect mentions, include `modifiers` when present and anchor them to direct entities when possible.
- For indirect mentions, `anchors` should point only to direct entities and must contain the anchored entity text plus entity type and only appear when the relationship is clearly indicated in the sentence.
- For any direct Drug mention, `text` MUST be the English generic drug name, e.g. "奥希替尼" -> "Osimertinib".
- For any direct Lineage, Disease, Subtype mention, `text` MUST be the corresponding English biomedical term. 
- If the phrase is a organ name + cancer, classify it as Lineage and set `text` to the canonical organ/tissue noun only. Do not use adjective forms, e.g. "Bone cancer"->"text": "Bone", "entity_type": "Lineage"
- When an organ/tissue term is immediately followed by a cancer histology term, treat the full phrase as one Disease/Subtype mention; do not separately extract the organ/tissue as Lineage.
- If no entity is found, return {{"mentions": []}}.
- Entity extraction should be context-dependent. If multiple entities appear in one sentence, ensure that entities are anchored to the correct semantic relationships.

Example:
Question: KRAS amplification in skin cell lines
Output:
{
  "mentions": [
    {
      "text": "KRAS",
      "entity_type": "Gene",
      "semantic_group": "direct",
      "modifiers": [],
      "anchors": []
    },
    {
      "text": "skin",
      "entity_type": "Lineage",
      "semantic_group": "direct",
      "modifiers": [],
      "anchors": []
    },
    {
      "text": "amplification",
      "entity_type": "CopyNumber",
      "semantic_group": "indirect",
      "modifiers": ["amplification"],
      "anchors": [
        {"text": "KRAS", "entity_type": "Gene"}
      ]
    }
  ]
}

"""


class LLMEntityExtractor:
    """Extract entity mentions from a natural-language question."""

    def extract(self, llm: Any, question: str) -> list[EntityMention]:
        try:
            response = invoke_with_tracking(
                llm,
                "entity_extraction",
                [
                    ("system", ENTITY_EXTRACTION_SYSTEM_PROMPT),
                    ("user", question),
                ]
            )
            if isinstance(response, dict):
                content = response.get("content", response)
            else:
                content = getattr(response, "content", response)
            payload = self._parse_json_object(content)
        except Exception:
            payload = {"mentions": []}

        mentions: list[EntityMention] = []
        for item in payload.get("mentions", []):
            text = str(item.get("text", "")).strip()
            entity_type = str(item.get("entity_type", "")).strip()
            semantic_group = str(item.get("semantic_group", "")).strip().lower()
            modifiers = self._parse_modifiers(item.get("modifiers", []))
            anchors = self._parse_anchors(item.get("anchors", []))
            if text and entity_type and semantic_group in {"direct", "indirect"}:
                if semantic_group == "direct":
                    modifiers = []
                    anchors = []
                mentions.append(
                    EntityMention(
                        text=text,
                        entity_type=entity_type,
                        semantic_group=semantic_group,
                        modifiers=modifiers,
                        anchors=anchors,
                    )
                )
        return mentions

    def _parse_json_object(self, text: Any) -> dict[str, Any]:
        if isinstance(text, dict):
            payload = text
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            return payload
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM response.")
        payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _parse_modifiers(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        modifiers: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                modifiers.append(text)
        return modifiers

    def _parse_anchors(self, value: Any) -> list[EntityAnchor]:
        if not isinstance(value, list):
            return []
        anchors: list[EntityAnchor] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            entity_type = str(item.get("entity_type", "")).strip()
            if text and entity_type:
                anchors.append(EntityAnchor(text=text, entity_type=entity_type))
        return anchors
