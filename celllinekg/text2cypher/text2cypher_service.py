"""Query service orchestrating extraction, resolution, schema selection, and Cypher execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celllinekg.config import load_dotenv
from celllinekg.text2cypher.answer_synthesizer import AnswerSynthesizer
from celllinekg.text2cypher.cypher_generator import CypherGenerator
from celllinekg.text2cypher.entity_bridge import EntityBridge
from celllinekg.text2cypher.entity_extractor import LLMEntityExtractor
from celllinekg.text2cypher.entity_resolver import EntityResolver
from celllinekg.text2cypher.executor import CypherExecutor
from celllinekg.text2cypher.resolution_router import ResolutionRouter
from celllinekg.text2cypher.schema_registry import EntitySchemaRegistry


class FallbackNeo4jGraph:
    """Minimal Neo4j wrapper that does not depend on APOC schema procedures."""

    def __init__(self, *, url: str, username: str, password: str):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(url, auth=(username, password))
        self.schema = ""

    def refresh_schema(self) -> None:
        """Best-effort no-op fallback when APOC-backed schema refresh is unavailable."""
        self.schema = ""

    def query(self, cypher: str) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(cypher)
        return [record.data() for record in records]

    def close(self) -> None:
        self._driver.close()


@dataclass(frozen=True)
class Text2CypherSettings:
    """Settings for the query pipeline."""

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str | None = None

    @classmethod
    def from_env(cls) -> "Text2CypherSettings":
        load_dotenv()
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4j"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_base_url=os.getenv("LLM_BASE_URL"),
        )


class Text2CypherService:
    """Coordinate the query pipeline for CellLineKG."""

    def __init__(
        self,
        settings: Text2CypherSettings | None = None,
        schema_mode: str = "full",
    ):
        """
        Initialize the CellLineKG query service.

        Arg:
            settings: Optional runtime settings. When omitted, settings are loaded from .env.
            schema_mode: Schema context mode. Use "full" to pass filtered schema plus
                database schema, or "filtered" to pass only the filtered entity schema.

        Returns:
            None.
        """
        if schema_mode not in {"full", "filtered"}:
            raise ValueError("schema_mode must be either 'full' or 'filtered'.")
        self.settings = settings or Text2CypherSettings.from_env()
        self.schema_mode = schema_mode
        self.entity_extractor = LLMEntityExtractor()
        self.entity_resolver = EntityResolver()
        self.resolution_router = ResolutionRouter()
        self.entity_bridge = EntityBridge(self.entity_resolver)
        self.schema_registry = EntitySchemaRegistry()
        self.cypher_generator = CypherGenerator()
        self.executor = CypherExecutor()
        self.answer_synthesizer = AnswerSynthesizer()

    def ask(
        self,
        question: str,
        llm: Any | None = None,
    ) -> dict[str, object]:
        """Run the end-to-end query pipeline and return structured diagnostics.

        Arg:
            question: Natural-language CellLineKG question.
            llm: Optional injected chat model, including a tracked model proxy.

        Returns:
            Answer, query steps, and debug diagnostics.
        """
        if not self.settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")
        if not self.settings.llm_model:
            raise RuntimeError("LLM_MODEL is not configured.")

        if llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError(
                    "LangChain dependencies are missing or incompatible. Install in the same Python environment: "
                    "`python -m pip install -U langchain langchain-community langchain-openai langchain-neo4j neo4j openai`."
                ) from exc

            llm_kwargs: dict[str, object] = {
                "api_key": self.settings.llm_api_key,
                "model": self.settings.llm_model,
                "temperature": 0,
                "max_tokens": 4096,
                "timeout": 300,
            }
            if self.settings.llm_base_url:
                llm_kwargs["base_url"] = self.settings.llm_base_url

            llm = ChatOpenAI(**llm_kwargs)

        mentions = self.entity_extractor.extract(llm=llm, question=question)
        indirect_types = list(
            dict.fromkeys(
                mention.entity_type.strip().lower()
                for mention in mentions
                if mention.semantic_group == "indirect" and mention.entity_type
            )
        )
        resolved_entities = self.entity_resolver.resolve(mentions)
        resolution = self.resolution_router.decide(resolved_entities)

        if resolution.action == "bridge":
            bridge = self.entity_bridge.bridge(resolution.unresolved_entities)
            if bridge.action == "resolved":
                resolved_entities = resolution.resolved_entities + bridge.bridged_entities
                resolved_entities.extend(bridge.still_unresolved)
                resolution = self.resolution_router.decide(resolved_entities)
            else:
                return {
                    "question": question,
                    "answer": "KG 中没有找到与该问题匹配的相关数据。",
                    "intermediate_steps": [],
                    "debug": {
                        "mentions": [mention.__dict__ for mention in mentions],
                        "resolved_entities": [entity.__dict__ for entity in resolved_entities],
                        "resolution_action": resolution.action,
                        "resolution_message": resolution.message,
                        "bridge_action": bridge.action,
                        "bridge_message": bridge.message,
                        "bridge_targets": bridge.bridge_targets,
                        "bridged_entities": [entity.__dict__ for entity in bridge.bridged_entities],
                        "still_unresolved": [
                            entity.__dict__ for entity in bridge.still_unresolved
                        ],
                        "unresolved_entities": [
                            entity.__dict__ for entity in resolution.unresolved_entities
                        ],
                        "ambiguous_entities": [
                            entity.__dict__ for entity in resolution.ambiguous_entities
                        ],
                    },
                }

        if resolution.action != "query":
            return {
                "question": question,
                "answer": resolution.message,
                "intermediate_steps": [],
                "debug": {
                    "mentions": [mention.__dict__ for mention in mentions],
                    "resolved_entities": [entity.__dict__ for entity in resolved_entities],
                    "resolution_action": resolution.action,
                    "resolution_message": resolution.message,
                    "bridge_action": "resolved",
                    "unresolved_entities": [
                        entity.__dict__ for entity in resolution.unresolved_entities
                    ],
                    "ambiguous_entities": [
                        entity.__dict__ for entity in resolution.ambiguous_entities
                    ],
                },
            }

        graph, graph_schema_error = self._build_graph()
        entity_schema = self.schema_registry.build_context(resolved_entities, mentions=mentions)
        graph_schema = self._select_graph_schema(graph)

        generation = self.cypher_generator.generate(
            llm=llm,
            question=question,
            schema_context=entity_schema,
            resolved_entities=resolved_entities,
            graph_schema=graph_schema,
        )
        verification = self.cypher_generator.verify_and_rewrite(
            llm=llm,
            question=question,
            cypher=generation.cypher,
            schema_context=entity_schema,
            graph_schema=graph_schema,
        )
        execution = self.executor.execute(graph=graph, cypher=verification.cypher)
        answer = self.answer_synthesizer.synthesize(
            llm=llm,
            question=question,
            execution=execution,
            schema_context=entity_schema,
            final_cypher=verification.cypher,
        )

        return {
            "question": question,
            "answer": answer,
            "intermediate_steps": [
                {"query": verification.cypher},
                {"context": execution.rows[:10]},
            ],
            "debug": {
                "mentions": [mention.__dict__ for mention in mentions],
                "resolved_entities": [entity.__dict__ for entity in resolved_entities],
                "indirect_entity_types": indirect_types,
                "resolution_action": resolution.action,
                "resolution_message": resolution.message,
                "entity_schema": entity_schema,
                "graph_schema": graph_schema,
                "schema_mode": self.schema_mode,
                "graph_schema_error": graph_schema_error,
                "generation_prompt": generation.prompt,
                "generation_raw_response": generation.raw_response,
                "generated_cypher": generation.cypher,
                "verification_prompt": verification.prompt,
                "verification_raw_response": verification.raw_response,
                "verification_changed": verification.changed,
                "final_cypher": verification.cypher,
                "result_row_count": len(execution.rows),
                "result_preview": execution.rows[:10],
                "error": execution.error,
            },
        }

    def _build_graph(self) -> tuple[Any, str | None]:
        graph_schema_error: str | None = None

        try:
            try:
                from langchain_neo4j import Neo4jGraph
            except ImportError:
                from langchain_community.graphs import Neo4jGraph

            graph = Neo4jGraph(
                url=self.settings.neo4j_uri,
                username=self.settings.neo4j_user,
                password=self.settings.neo4j_password,
            )
            graph.refresh_schema()
            return graph, None
        except Exception as exc:
            graph_schema_error = str(exc)
            fallback_graph = FallbackNeo4jGraph(
                url=self.settings.neo4j_uri,
                username=self.settings.neo4j_user,
                password=self.settings.neo4j_password,
            )
            fallback_graph.refresh_schema()
            return fallback_graph, graph_schema_error

    def _select_graph_schema(self, graph: Any) -> str:
        """
        Select the graph schema text passed to the LLM.

        Arg:
            graph: Graph wrapper with an optional schema attribute.

        Returns:
            Full database schema text in full mode, or an empty string in filtered mode.
        """
        if self.schema_mode == "filtered":
            return ""
        return (getattr(graph, "schema", "") or "").strip()

    def query_only(self, question: str) -> dict[str, object]:
        """Run the query pipeline through Cypher execution without answer synthesis."""
        if not self.settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")
        if not self.settings.llm_model:
            raise RuntimeError("LLM_MODEL is not configured.")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "LangChain dependencies are missing or incompatible. Install in the same Python environment: "
                "`python -m pip install -U langchain langchain-community langchain-openai langchain-neo4j neo4j openai`."
            ) from exc

        llm_kwargs: dict[str, object] = {
            "api_key": self.settings.llm_api_key,
            "model": self.settings.llm_model,
            "temperature": 0,
            "max_tokens": 4096,
            "timeout": 300,
        }
        if self.settings.llm_base_url:
            llm_kwargs["base_url"] = self.settings.llm_base_url

        llm = ChatOpenAI(**llm_kwargs)

        mentions = self.entity_extractor.extract(llm=llm, question=question)
        resolved_entities = self.entity_resolver.resolve(mentions)
        resolution = self.resolution_router.decide(resolved_entities)

        if resolution.action == "bridge":
            bridge = self.entity_bridge.bridge(resolution.unresolved_entities)
            if bridge.action == "resolved":
                resolved_entities = resolution.resolved_entities + bridge.bridged_entities
                resolved_entities.extend(bridge.still_unresolved)
                resolution = self.resolution_router.decide(resolved_entities)
            else:
                return {
                    "question": question,
                    "rows": [],
                    "resolution_message": bridge.message,
                    "cypher": None,
                }

        if resolution.action != "query":
            return {
                "question": question,
                "rows": [],
                "resolution_message": resolution.message,
                "cypher": None,
            }

        graph, _ = self._build_graph()
        entity_schema = self.schema_registry.build_context(resolved_entities, mentions=mentions)
        graph_schema = self._select_graph_schema(graph)
        generation = self.cypher_generator.generate(
            llm=llm,
            question=question,
            schema_context=entity_schema,
            resolved_entities=resolved_entities,
            graph_schema=graph_schema,
        )
        execution = self.executor.execute(graph=graph, cypher=generation.cypher)
        resolution_message = resolution.message
        if execution.error:
            resolution_message = f"Cypher execution failed: {execution.error}"

        return {
            "question": question,
            "rows": execution.rows,
            "resolution_message": resolution_message,
            "cypher": execution.cypher,
        }


def main() -> None:
    """CLI entrypoint for running one question through the query pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the CellLineKG query pipeline against Neo4j."
    )
    parser.add_argument("question", help="Natural-language question to ask the graph.")
    parser.add_argument(
        "--schema-mode",
        choices=["full", "filtered"],
        default="full",
        help="Schema context mode. Use filtered to omit the full database schema.",
    )
    args = parser.parse_args()
    service = Text2CypherService(schema_mode=args.schema_mode)
    payload = service.ask(args.question)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
