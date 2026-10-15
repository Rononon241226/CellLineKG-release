"""Route resolved entities into query, bridge, or clarify actions."""

from __future__ import annotations

from .models import ResolvedEntity, ResolutionDecision


class ResolutionRouter:
    """Apply the minimal resolution branching strategy."""

    def _format_entities(self, entities: list[ResolvedEntity]) -> str:
        """Format entities with enough detail for pipeline diagnostics."""
        return "; ".join(
            f"{entity.text} as {entity.entity_type}" for entity in entities
        )

    def decide(self, resolved_entities: list[ResolvedEntity]) -> ResolutionDecision:
        resolved = [entity for entity in resolved_entities if entity.status == "resolved"]
        unresolved = [entity for entity in resolved_entities if entity.status == "unresolved"]
        ambiguous = [entity for entity in resolved_entities if entity.status == "ambiguous"]
        not_in_graph = [entity for entity in resolved_entities if entity.status == "not_in_graph"]

        if ambiguous:
            texts = self._format_entities(ambiguous)
            return ResolutionDecision(
                action="clarify",
                resolved_entities=resolved,
                unresolved_entities=unresolved,
                ambiguous_entities=ambiguous,
                message=f"Multiple candidates found for: {texts}. Please confirm the intended entity.",
            )

        if not_in_graph:
            texts = ", ".join(
                f"{entity.text} as {entity.entity_type} -> "
                f"{entity.resolved_name or entity.resolved_id or 'unknown'}"
                for entity in not_in_graph
            )
            return ResolutionDecision(
                action="fail",
                resolved_entities=resolved,
                unresolved_entities=unresolved,
                ambiguous_entities=ambiguous,
                message=f"The graph does not contain: {texts}.",
            )

        if unresolved:
            texts = self._format_entities(unresolved)
            return ResolutionDecision(
                action="bridge",
                resolved_entities=resolved,
                unresolved_entities=unresolved,
                ambiguous_entities=ambiguous,
                message=f"Local resolution failed for: {texts}. Try external bridge lookup next.",
            )

        return ResolutionDecision(
            action="query",
            resolved_entities=resolved,
            unresolved_entities=unresolved,
            ambiguous_entities=ambiguous,
            message="All direct entities resolved. Continue to graph query.",
        )
