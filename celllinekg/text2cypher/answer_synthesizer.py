"""Turn query rows into a natural-language answer."""

from __future__ import annotations

import json
from typing import Any

from celllinekg.llm.usage_tracker import invoke_with_tracking

from .models import QueryExecutionResult


ANSWER_SYSTEM_PROMPT = """You are a professional biomedical assistant answering questions about CellLineKG from the provided query results.

Rules:
- Must understand the question with biological domain knowledge.
- Understand the meaning of rows with Entity-specific schema.
- Answer concisely in one sentence, directly state the result first, and do not add generic explanations beyond the evidence provided.
- Organize the answer of the question from rows that exactly relate to the question, which means not all the properties are needed to mention.
- Do not expose Cypher, SQL, field aliases, or query expressions such as COUNT(...) > 0 in the final answer; express the result in natural language only.
"""

ANSWER_USER_PROMPT_TEMPLATE = """Question:
{question}

Final Cypher:
{final_cypher}

Rows:
{rows}

Entity-specific schema:
{schema_context}
"""


class AnswerSynthesizer:
    """Generate a concise natural-language answer from query rows."""

    def synthesize(
        self,
        llm: Any,
        question: str,
        execution: QueryExecutionResult,
        schema_context: str,
        final_cypher: str | None = None,
    ) -> str:
        """
        Generate a concise answer from executed query rows.

        Arg:
            llm: Chat LLM used to synthesize the final answer.
            question: User question.
            execution: Query execution result containing rows or an error.
            schema_context: Entity-specific schema context used for the query.
            final_cypher: Final Cypher query executed against the KG.

        Returns:
            Natural-language answer grounded in rows and final Cypher.
        """
        if execution.error:
            return f"Query execution failed: {execution.error}"
        if not execution.rows:
            return "No matching data was found in the KG."

        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            final_cypher=final_cypher or "",
            rows=json.dumps(execution.rows[:20], ensure_ascii=False, default=str),
            schema_context=schema_context,
        )
        response = invoke_with_tracking(
            llm,
            "answer_synthesis",
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                ("user", user_prompt),
            ]
        )
        return str(getattr(response, "content", response)).strip()
