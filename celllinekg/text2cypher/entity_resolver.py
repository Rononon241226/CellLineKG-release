"""Resolve extracted mentions to graph entities using local node tables."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback for environments without rapidfuzz
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(left: str, right: str) -> float:
            left_tokens = set(left.split())
            right_tokens = set(right.split())
            common = " ".join(sorted(left_tokens & right_tokens))
            left_only = " ".join(sorted(left_tokens - right_tokens))
            right_only = " ".join(sorted(right_tokens - left_tokens))
            return max(
                _FallbackFuzz._ratio(common, f"{common} {left_only}".strip()),
                _FallbackFuzz._ratio(common, f"{common} {right_only}".strip()),
                _FallbackFuzz._ratio(
                    f"{common} {left_only}".strip(),
                    f"{common} {right_only}".strip(),
                ),
            )

        @staticmethod
        def partial_ratio(left: str, right: str) -> float:
            if len(left) > len(right):
                left, right = right, left
            if not left:
                return 0.0
            best = 0.0
            for start in range(0, max(len(right) - len(left) + 1, 1)):
                best = max(best, _FallbackFuzz._ratio(left, right[start : start + len(left)]))
            return best

        @staticmethod
        def _ratio(left: str, right: str) -> float:
            if not left and not right:
                return 100.0
            return 100.0 * SequenceMatcher(None, left, right).ratio()

    fuzz = _FallbackFuzz()

from .models import EntityMention, ResolvedEntity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODES_DIR = PROJECT_ROOT / "data" / "processed" / "neo4j_import" / "nodes"

CELLLINE_CSV = NODES_DIR / "nodes_cellline.csv"
GENE_CSV = NODES_DIR / "nodes_gene.csv"
PROTEIN_CSV = NODES_DIR / "nodes_protein.csv"
DRUG_CSV = NODES_DIR / "nodes_drug.csv"
DISEASE_CSV = NODES_DIR / "nodes_disease.csv"
SUBTYPE_CSV = NODES_DIR / "nodes_subtype.csv"
LINEAGE_CSV = NODES_DIR / "nodes_lineage.csv"



SUPPORTED_DIRECT_TYPES = {
    "cellline",
    "gene",
    "protein",
    "drug",
    "disease",
    "subtype",
    "lineage",
}


def _normalize(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[\s\-_./,()]+", "", text)


def _normalize_fuzzy_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_fuzzy_text(text: str) -> set[str]:
    return {token for token in _normalize_fuzzy_text(text).split() if token}


def _split_pipe_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split("||") if part.strip())


def _split_comma_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _split_ncit_list(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for part in _split_pipe_list(text):
        normalized = _normalize_ncit_id(part)
        if normalized:
            values.append(normalized)
            values.append(f"NCIT:{normalized}")
            values.append(f"NCI:{normalized}")
    return tuple(dict.fromkeys(values))


def _normalize_ncit_id(text: str) -> str:
    value = text.strip().upper()
    value = re.sub(r"^(?:NCIT|NCI|NCI THESAURUS)[:_\\s-]*", "", value)
    return value if re.fullmatch(r"C\d+", value) else ""


def _make_candidate(
    resolved_id: str,
    resolved_name: str,
    match_field: str,
) -> dict[str, str]:
    return {
        "resolved_id": resolved_id,
        "resolved_name": resolved_name,
        "match_field": match_field,
    }


def _match_priority(match_field: str) -> int:
    priorities = {
        "name": 0,
        "symbol": 0, 

        "cellosaurus_id": 1,
        "depmap_model_id":1,
        "sanger_model_id":1,
        "cmp_model_id":1,
        "entrez_id": 1,
        "mychem_id": 1,

        "synonyms": 2,
        "external_aliases": 2,

        "oncotree_disease_code": 1,
        "oncotree_subtype_code": 1,
        "oncotree_code": 1,
        "ncit_disease_id": 1,
        "ncit_subtype_id": 1,

        "cross_reference": 3,
    }
    return priorities.get(match_field, 99)


class EntityResolver:
    """Resolve direct entity mentions against local graph node tables."""

    def __init__(self) -> None:
        self._resolvers = {
            "cellline": self._resolve_cellline,
            "gene": self._resolve_gene,
            "protein": self._resolve_protein,
            "drug": self._resolve_drug,
            "disease": self._resolve_disease,
            "subtype": self._resolve_subtype,
            "lineage": self._resolve_lineage,
        }

    def resolve(self, mentions: list[EntityMention]) -> list[ResolvedEntity]:
        resolved: list[ResolvedEntity] = []
        for mention in mentions:
            if mention.semantic_group.strip().lower() != "direct":
                continue

            normalized_type = mention.entity_type.strip().lower()
            resolver = self._resolvers.get(normalized_type)
            if resolver is None:
                resolved.append(
                    ResolvedEntity(
                        text=mention.text,
                        entity_type=mention.entity_type,
                        status="unsupported",
                    )
                )
                continue
            resolved.append(resolver(mention.text, mention.entity_type))
        return resolved

    def resolve_gene_entrez(
        self,
        entrez_id: str,
        *,
        original_text: str,
        resolved_name: str | None = None,
    ) -> ResolvedEntity:
        local_result = self._resolve_from_index(
            text=entrez_id,
            entity_type="Gene",
            index=_gene_index(),
        )
        if local_result.status == "resolved":
            local_result.text = original_text
            return local_result

        return ResolvedEntity(
            text=original_text,
            entity_type="Gene",
            status="not_in_graph",
            resolved_id=entrez_id,
            resolved_name=resolved_name,
            match_field="entrez_id",
        )

    def resolve_disease_ncit(
        self,
        ncit_id: str,
        *,
        original_text: str,
        resolved_name: str | None = None,
    ) -> ResolvedEntity:
        return self._resolve_ncit_against_indexes(
            ncit_id,
            original_text=original_text,
            resolved_name=resolved_name,
            indexes=(
                ("Disease", _disease_index()),
                ("Subtype", _subtype_index()),
            ),
        )

    def resolve_subtype_ncit(
        self,
        ncit_id: str,
        *,
        original_text: str,
        resolved_name: str | None = None,
    ) -> ResolvedEntity:
        return self._resolve_ncit_against_indexes(
            ncit_id,
            original_text=original_text,
            resolved_name=resolved_name,
            indexes=(
                ("Subtype", _subtype_index()),
                ("Disease", _disease_index()),
            ),
        )

    def _resolve_cellline(self, text: str, entity_type: str) -> ResolvedEntity:
        return self._resolve_from_index(
            text=text,
            entity_type=entity_type,
            index=_cellline_index(),
        )

    def _resolve_gene(self, text: str, entity_type: str) -> ResolvedEntity:
        return self._resolve_from_index(
            text=text,
            entity_type=entity_type,
            index=_gene_index(),
        )

    def _resolve_protein(self, text: str, entity_type: str) -> ResolvedEntity:
        return self._resolve_from_index(
            text=text,
            entity_type=entity_type,
            index=_protein_index(),
        )

    def _resolve_drug(self, text: str, entity_type: str) -> ResolvedEntity:
        return self._resolve_from_index(
            text=text,
            entity_type=entity_type,
            index=_drug_index(),
        )

    def _resolve_disease(self, text: str, entity_type: str) -> ResolvedEntity:
        return self._resolve_disease_or_subtype(text)

    def _resolve_subtype(self, text: str, entity_type: str) -> ResolvedEntity:
        return self._resolve_disease_or_subtype(text)

    def _resolve_disease_or_subtype(self, text: str) -> ResolvedEntity:
        """Resolve a cancer type mention against Disease and Subtype tables.

        Arg:
            text: disease or subtype mention text extracted from the question.

        Returns:
            Resolved entity from the graph, preferring Disease when both labels match.
        """
        result = self._resolve_from_index(
            text=text,
            entity_type="Disease",
            index=_disease_index(),
        )
        if result.status != "unresolved":
            return result

        result = self._resolve_from_index(
            text=text,
            entity_type="Subtype",
            index=_subtype_index(),
        )
        if result.status != "unresolved":
            return result

        result = self._resolve_fuzzy(
            text=text,
            entity_type="Disease",
            records=_disease_fuzzy_records(),
            minimum_score=90.0,
            minimum_margin=8.0,
            use_partial=False,
        )
        if result.status != "unresolved":
            return result

        return self._resolve_fuzzy(
            text=text,
            entity_type="Subtype",
            records=_subtype_fuzzy_records(),
            minimum_score=90.0,
            minimum_margin=8.0,
            use_partial=False,
        )

    def _resolve_lineage(self, text: str, entity_type: str) -> ResolvedEntity:
        result = self._resolve_from_index(
            text=text,
            entity_type=entity_type,
            index=_lineage_index(),
        )
        if result.status != "unresolved":
            return result
        return self._resolve_fuzzy(
            text=text,
            entity_type=entity_type,
            records=_lineage_fuzzy_records(),
            minimum_score=80.0,
            minimum_margin=5.0,
            use_partial=True,
        )

    def _resolve_from_index(
        self,
        text: str,
        entity_type: str,
        index: dict[str, tuple[dict[str, str], ...]],
    ) -> ResolvedEntity:
        key = _normalize(text)
        if not key:
            return ResolvedEntity(
                text=text,
                entity_type=entity_type,
                status="unresolved",
            )

        matches = index.get(key, ())
        if not matches:
            return ResolvedEntity(
                text=text,
                entity_type=entity_type,
                status="unresolved",
            )

        grouped = self._group_matches(matches)
        if len(grouped) == 1:
            match = next(iter(grouped.values()))
            return ResolvedEntity(
                text=text,
                entity_type=entity_type,
                status="resolved",
                resolved_id=match["resolved_id"],
                resolved_name=match["resolved_name"],
                match_field=match["match_field"],
            )

        return ResolvedEntity(
            text=text,
            entity_type=entity_type,
            status="ambiguous",
            candidates=[
                _make_candidate(
                    resolved_id=match["resolved_id"],
                    resolved_name=match["resolved_name"],
                    match_field=match["match_field"],
                )
                for match in grouped.values()
            ],
        )

    def _resolve_fuzzy(
        self,
        *,
        text: str,
        entity_type: str,
        records: tuple[dict[str, object], ...],
        minimum_score: float,
        minimum_margin: float,
        use_partial: bool,
    ) -> ResolvedEntity:
        query = _normalize_fuzzy_text(text)
        query_tokens = _tokenize_fuzzy_text(text)
        if not query:
            return ResolvedEntity(
                text=text,
                entity_type=entity_type,
                status="unresolved",
            )

        scored: list[tuple[float, dict[str, object]]] = []
        for record in records:
            candidate = str(record["fuzzy_name"])
            candidate_tokens = record["tokens"]
            score = float(fuzz.token_set_ratio(query, candidate))
            if query_tokens and isinstance(candidate_tokens, set) and query_tokens <= candidate_tokens:
                score = max(score, 95.0)
            if use_partial:
                score = max(score, float(fuzz.partial_ratio(query, candidate)))
                if isinstance(candidate_tokens, set):
                    for query_token in query_tokens:
                        for candidate_token in candidate_tokens:
                            score = max(
                                score,
                                float(fuzz.partial_ratio(query_token, candidate_token)),
                            )
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < minimum_score:
            return ResolvedEntity(
                text=text,
                entity_type=entity_type,
                status="unresolved",
            )

        best_score = scored[0][0]
        close_matches = [
            (score, record)
            for score, record in scored
            if best_score - score < minimum_margin and score >= minimum_score
        ]
        grouped: dict[str, tuple[float, dict[str, object]]] = {}
        for score, record in close_matches:
            resolved_id = str(record["resolved_id"])
            if resolved_id not in grouped or score > grouped[resolved_id][0]:
                grouped[resolved_id] = (score, record)

        if len(grouped) == 1:
            _, match = next(iter(grouped.values()))
            return ResolvedEntity(
                text=text,
                entity_type=entity_type,
                status="resolved",
                resolved_id=str(match["resolved_id"]),
                resolved_name=str(match["resolved_name"]),
                match_field="fuzzy_name",
            )

        return ResolvedEntity(
            text=text,
            entity_type=entity_type,
            status="ambiguous",
            candidates=[
                _make_candidate(
                    resolved_id=str(record["resolved_id"]),
                    resolved_name=str(record["resolved_name"]),
                    match_field="fuzzy_name",
                )
                for _, record in grouped.values()
            ],
        )

    def _resolve_ncit_against_indexes(
        self,
        ncit_id: str,
        *,
        original_text: str,
        resolved_name: str | None,
        indexes: tuple[tuple[str, dict[str, tuple[dict[str, str], ...]]], ...],
    ) -> ResolvedEntity:
        normalized_ncit_id = _normalize_ncit_id(ncit_id)
        if not normalized_ncit_id:
            return ResolvedEntity(
                text=original_text,
                entity_type=indexes[0][0],
                status="unresolved",
            )

        matches_by_type: list[ResolvedEntity] = []
        for entity_type, index in indexes:
            result = self._resolve_from_index(
                text=normalized_ncit_id,
                entity_type=entity_type,
                index=index,
            )
            if result.status == "resolved":
                result.text = original_text
                matches_by_type.append(result)

        if len(matches_by_type) == 1:
            return matches_by_type[0]
        if len(matches_by_type) > 1:
            return ResolvedEntity(
                text=original_text,
                entity_type=indexes[0][0],
                status="ambiguous",
                candidates=[
                    _make_candidate(
                        resolved_id=result.resolved_id or "",
                        resolved_name=result.resolved_name or "",
                        match_field=result.match_field or "",
                    )
                    for result in matches_by_type
                ],
            )

        return ResolvedEntity(
            text=original_text,
            entity_type=indexes[0][0],
            status="not_in_graph",
            resolved_id=normalized_ncit_id,
            resolved_name=resolved_name,
            match_field="ncit_id",
        )

    def _group_matches(
        self,
        matches: tuple[dict[str, str], ...],
    ) -> dict[str, dict[str, str]]:
        grouped: dict[str, dict[str, str]] = {}
        for match in matches:
            resolved_id = match["resolved_id"]
            existing = grouped.get(resolved_id)
            if existing is None or _match_priority(match["match_field"]) < _match_priority(
                existing["match_field"]
            ):
                grouped[resolved_id] = match
        return grouped


def _build_index(
    records: Iterable[dict[str, str]],
    alias_fields: tuple[str, ...],
) -> dict[str, tuple[dict[str, str], ...]]:
    bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()

    for record in records:
        for field in alias_fields:
            value = record.get(field, "")
            key = _normalize(value)
            if not key:
                continue
            signature = (key, record["resolved_id"], field)
            if signature in seen:
                continue
            seen.add(signature)
            bucket[key].append(record | {"match_field": field})

    return {key: tuple(values) for key, values in bucket.items()}


def _make_fuzzy_record(resolved_id: str, resolved_name: str) -> dict[str, object]:
    fuzzy_name = _normalize_fuzzy_text(resolved_name)
    return {
        "resolved_id": resolved_id,
        "resolved_name": resolved_name,
        "fuzzy_name": fuzzy_name,
        "tokens": _tokenize_fuzzy_text(resolved_name),
    }


@lru_cache(maxsize=1)
def _cellline_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with CELLLINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aliases = {
                "name": row.get("name", "") or "",
                "cellosaurus_id": row.get("cellosaurus_id", "") or "",
                "depmap_model_id": row.get("depmap_model_id", "") or "",
                "sanger_model_id": row.get("sanger_model_id", "") or "",
                "cmp_model_id": row.get("cmp_model_id", "") or "",
            }
            record = {
                "resolved_id": row.get("cellosaurus_id", "") or "",
                "resolved_name": row.get("name", "") or "",
                **aliases,
            }
            records.append(record)
            for synonym in _split_pipe_list(row.get("synonyms", "") or ""):
                records.append(
                    {
                        "resolved_id": record["resolved_id"],
                        "resolved_name": record["resolved_name"],
                        "synonyms": synonym,
                    }
                )
            for ref in _split_pipe_list(row.get("cross_reference", "") or ""):
                filtered_ref_prefixes={
                    "atcc", "dsmz", "ecacc","jcrb", "kclb", "cctcc", "bcrc"  
                    # "clo", "efo", "cldb", 
                    # "ncit", "mesh", "wikidata","pubchem_cell_line"
                }
                if "::" in ref:
                    ref_prefix, ref_value = ref.split("::", 1)
                    if ref_prefix.lower() in filtered_ref_prefixes:
                        records.append(
                            {
                                "resolved_id": record["resolved_id"],
                                "resolved_name": record["resolved_name"],
                                "cross_reference": ref_value,
                            }
                        )
    return _build_index(
        records=records,
        alias_fields=(
            "name",
            "synonyms",
            "cellosaurus_id",
            "depmap_model_id",
            "sanger_model_id",
            "cmp_model_id",
            "cross_reference",
        ),
    )


@lru_cache(maxsize=1)
def _gene_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with GENE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "resolved_id": (row.get("entrez_id", "") or row.get("gene_id", "") or ""),
                    "resolved_name": row.get("symbol", "") or "",
                    "symbol": row.get("symbol", "") or "",
                    "entrez_id": row.get("entrez_id", "") or "",
                    "gene_id": row.get("gene_id", "") or "",
                }
            )
    return _build_index(records=records, alias_fields=("symbol", "entrez_id", "gene_id"))


@lru_cache(maxsize=1)
def _protein_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with PROTEIN_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "resolved_id": row.get("protein_id", "") or "",
                    "resolved_name": row.get("display_name", "") or row.get("symbol", "") or "",
                    "protein_id": row.get("protein_id", "") or "",
                    "uniprot_id": row.get("uniprot_id", "") or "",
                    "symbol": row.get("symbol", "") or "",
                    "display_name": row.get("display_name", "") or "",
                }
            )
    return _build_index(
        records=records,
        alias_fields=("symbol", "display_name", "protein_id", "uniprot_id"),
    )


@lru_cache(maxsize=1)
def _drug_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with DRUG_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = {
                "resolved_id": row.get("drug_id", "") or "",
                "resolved_name": row.get("name", "") or "",
                "name": row.get("name", "") or "",
                "mychem_id": row.get("mychem_id", "") or "",
            }
            records.append(record)
            for synonym in _split_comma_list(row.get("synonyms", "") or ""):
                records.append(
                    {
                        "resolved_id": record["resolved_id"],
                        "resolved_name": record["resolved_name"],
                        "synonyms": synonym,
                    }
                )
            for alias in _split_pipe_list(row.get("external_aliases", "") or ""):
                records.append(
                    {
                        "resolved_id": record["resolved_id"],
                        "resolved_name": record["resolved_name"],
                        "external_aliases": alias,
                    }
                )
    return _build_index(
        records=records,
        alias_fields=("name", "synonyms", "mychem_id", "external_aliases"),
    )


@lru_cache(maxsize=1)
def _disease_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with DISEASE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = {
                "resolved_id": row.get("disease_id", "") or "",
                "resolved_name": row.get("name", "") or "",
                "name": row.get("name", "") or "",
                "disease_id": row.get("disease_id", "") or "",
                "oncotree_disease_code": row.get("oncotree_disease_code", "") or "",
                "ncit_disease_id": row.get("ncit_disease_id", "") or "",
            }
            records.append(record)
            for ncit_id in _split_ncit_list(row.get("ncit_disease_id", "") or ""):
                records.append(
                    {
                        "resolved_id": record["resolved_id"],
                        "resolved_name": record["resolved_name"],
                        "ncit_disease_id": ncit_id,
                    }
                )
    return _build_index(
        records=records,
        alias_fields=("name", "disease_id", "oncotree_disease_code", "ncit_disease_id"),
    )


@lru_cache(maxsize=1)
def _subtype_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with SUBTYPE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "resolved_id": row.get("subtype_id", "") or "",
                    "resolved_name": row.get("name", "") or "",
                    "name": row.get("name", "") or "",
                    "subtype_id": row.get("subtype_id", "") or "",
                    "oncotree_subtype_code": (
                        row.get("oncotree_subtype_code")
                        or row.get("oncotree_code")
                        or ""
                    ),
                    "ncit_subtype_id": row.get("ncit_subtype_id", "") or "",
                }
            )
            for ncit_id in _split_ncit_list(row.get("ncit_subtype_id", "") or ""):
                records.append(
                    {
                        "resolved_id": row.get("subtype_id", "") or "",
                        "resolved_name": row.get("name", "") or "",
                        "ncit_subtype_id": ncit_id,
                    }
                )
    return _build_index(
        records=records,
        alias_fields=("name", "subtype_id", "oncotree_subtype_code", "ncit_subtype_id"),
    )


@lru_cache(maxsize=1)
def _lineage_index() -> dict[str, tuple[dict[str, str], ...]]:
    records: list[dict[str, str]] = []
    with LINEAGE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "resolved_id": row.get("lineage_id", "") or "",
                    "resolved_name": row.get("name", "") or "",
                    "name": row.get("name", "") or "",
                    "lineage_id": row.get("lineage_id", "") or "",
                }
            )
    return _build_index(records=records, alias_fields=("name", "lineage_id"))


@lru_cache(maxsize=1)
def _disease_fuzzy_records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with DISEASE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            resolved_id = row.get("disease_id", "") or ""
            resolved_name = row.get("name", "") or ""
            if resolved_id and resolved_name:
                records.append(_make_fuzzy_record(resolved_id, resolved_name))
    return tuple(records)


@lru_cache(maxsize=1)
def _subtype_fuzzy_records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with SUBTYPE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            resolved_id = row.get("subtype_id", "") or ""
            resolved_name = row.get("name", "") or ""
            if resolved_id and resolved_name:
                records.append(_make_fuzzy_record(resolved_id, resolved_name))
    return tuple(records)


@lru_cache(maxsize=1)
def _lineage_fuzzy_records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with LINEAGE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            resolved_id = row.get("lineage_id", "") or ""
            resolved_name = row.get("name", "") or ""
            if resolved_id and resolved_name:
                records.append(_make_fuzzy_record(resolved_id, resolved_name))
    return tuple(records)
