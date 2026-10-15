"""Structured models for the query pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityAnchor:
    text: str
    entity_type: str


@dataclass
class EntityMention:
    text: str
    entity_type: str
    semantic_group: str
    modifiers: list[str] = field(default_factory=list)
    anchors: list[EntityAnchor] = field(default_factory=list)


@dataclass
class ResolvedEntity:
    text: str
    entity_type: str
    status: str
    resolved_id: str | None = None
    resolved_name: str | None = None
    match_field: str | None = None
    candidates: list[dict[str, str | None]] = field(default_factory=list)


@dataclass
class ResolutionDecision:
    action: str
    resolved_entities: list[ResolvedEntity] = field(default_factory=list)
    unresolved_entities: list[ResolvedEntity] = field(default_factory=list)
    ambiguous_entities: list[ResolvedEntity] = field(default_factory=list)
    message: str | None = None


@dataclass
class BridgeDecision:
    action: str
    bridge_targets: list[dict[str, str]] = field(default_factory=list)
    bridged_entities: list[ResolvedEntity] = field(default_factory=list)
    still_unresolved: list[ResolvedEntity] = field(default_factory=list)
    message: str | None = None


@dataclass
class CypherGenerationResult:
    cypher: str | None
    prompt: str
    raw_response: str


@dataclass
class CypherVerificationResult:
    cypher: str | None
    changed: bool
    prompt: str
    raw_response: str


@dataclass
class QueryExecutionResult:
    cypher: str | None
    rows: list[dict[str, Any]]
    error: str | None = None
