"""External bridge support for unresolved entities."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from .entity_resolver import EntityResolver
from .models import BridgeDecision, EntityMention, ResolvedEntity


MYGENE_BASE_URL = "https://mygene.info/v3"
MYDISEASE_BASE_URL = "https://mydisease.info/v1"
MYCHEM_BASE_URL = "https://mychem.info/v1"
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


class BaseBridgeProvider(ABC):
    """Bridge one entity family to an external identifier source."""

    bridge_name: str

    def __init__(self, resolver: EntityResolver):
        self.resolver = resolver

    @abstractmethod
    def bridge(self, entity: ResolvedEntity) -> tuple[ResolvedEntity | None, str | None]:
        """Try to bridge one unresolved entity."""


class GeneBridgeProvider(BaseBridgeProvider):
    """Resolve unresolved genes through MyGene.info."""

    bridge_name = "MyGene.info"

    def bridge(self, entity: ResolvedEntity) -> tuple[ResolvedEntity | None, str | None]:
        candidate, error = self._lookup_gene_entrez(entity.text)
        if candidate is None:
            return None, error
        return (
            self.resolver.resolve_gene_entrez(
                candidate["entrez_id"],
                original_text=entity.text,
                resolved_name=candidate.get("symbol"),
            ),
            None,
        )

    def _lookup_gene_entrez(self, text: str) -> tuple[dict[str, str] | None, str | None]:
        query = text.strip()
        if not query:
            return None, "empty gene query"

        errors: list[str] = []

        if re.fullmatch(r"ENSG\d+(?:\.\d+)?", query, flags=re.IGNORECASE):
            ensembl_id = query.split(".", 1)[0]
            for candidate_id in dict.fromkeys([query, ensembl_id]):
                candidate, error = self._lookup_gene_by_gene_id(candidate_id)
                if candidate is not None:
                    return candidate, None
                if error:
                    errors.append(error)

            candidate, error = self._lookup_gene_by_query(
                query=ensembl_id,
                scopes=("ensembl.gene", "ensembl"),
            )
            if candidate is not None:
                return candidate, None
            if error:
                errors.append(error)
            return None, self._collapse_lookup_errors(
                errors, default="no MyGene match for Ensembl ID"
            )

        if query.isdigit():
            candidate, error = self._lookup_gene_by_gene_id(query)
            if candidate is not None:
                return candidate, None
            if error:
                errors.append(error)
            candidate, error = self._lookup_gene_by_query(
                query=query,
                scopes=("entrezgene",),
            )
            if candidate is not None:
                return candidate, None
            if error:
                errors.append(error)
            return None, self._collapse_lookup_errors(
                errors, default="no MyGene match for Entrez ID"
            )

        candidate, error = self._lookup_gene_by_query(query)
        if candidate is not None:
            return candidate, None
        if error:
            errors.append(error)
        return None, self._collapse_lookup_errors(errors, default="no unique MyGene hit")

    def _lookup_gene_by_gene_id(self, gene_id: str) -> tuple[dict[str, str] | None, str | None]:
        url = (
            f"{MYGENE_BASE_URL}/gene/{quote(gene_id)}?"
            "fields=entrezgene,symbol,name,taxid"
        )
        try:
            payload = self._get_json(url)
        except HTTPError as exc:
            return None, f"MyGene /gene lookup failed for {gene_id}: HTTP {exc.code}"
        except URLError as exc:
            return None, f"MyGene /gene lookup failed for {gene_id}: {exc.reason}"

        if not isinstance(payload, dict):
            return None, f"MyGene /gene lookup returned non-object for {gene_id}"

        hit = self._normalize_gene_hit(payload)
        if hit is None:
            return None, f"MyGene /gene lookup returned no usable human hit for {gene_id}"
        return hit, None

    def _lookup_gene_by_query(
        self,
        query: str,
        *,
        scopes: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, str] | None, str | None]:
        query_values = [
            f"symbol:{query}",
            f"alias:{query}",
            query,
        ]
        if scopes:
            query_values = list(scopes)

        errors: list[str] = []
        for query_value in query_values:
            params_dict = {
                "species": "human",
                "size": 5,
                "fields": "entrezgene,symbol,name,taxid,alias",
                "entrezonly": "true",
            }
            if scopes:
                params_dict["q"] = query
                params_dict["scopes"] = query_value
            else:
                params_dict["q"] = query_value
            params = urlencode(params_dict)
            try:
                payload = self._get_json(f"{MYGENE_BASE_URL}/query?{params}")
            except HTTPError as exc:
                errors.append(f"MyGene /query failed for {query}: HTTP {exc.code}")
                continue
            except URLError as exc:
                errors.append(f"MyGene /query failed for {query}: {exc.reason}")
                continue

            hit = self._select_gene_hit(payload, original_text=query)
            if hit is not None:
                return hit, None

        if errors:
            return None, self._collapse_lookup_errors(errors, default="MyGene query failed")
        return None, None

    def _get_json(self, url: str) -> dict[str, object] | list[object] | None:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _select_gene_hit(
        self,
        payload: dict[str, object] | list[object] | None,
        *,
        original_text: str,
    ) -> dict[str, str] | None:
        if not isinstance(payload, dict):
            return None

        hits = payload.get("hits", [])
        if not isinstance(hits, list):
            return None

        normalized_query = original_text.strip().lower()
        normalized_hits: list[dict[str, str]] = []
        for raw_hit in hits:
            if not isinstance(raw_hit, dict):
                continue
            hit = self._normalize_gene_hit(raw_hit)
            if hit is None:
                continue
            normalized_hits.append(hit)

        if not normalized_hits:
            return None

        exact_symbol_hits = [
            hit for hit in normalized_hits if hit["symbol"].strip().lower() == normalized_query
        ]
        unique_exact_symbol_hits = self._dedupe_gene_hits(exact_symbol_hits)
        if len(unique_exact_symbol_hits) == 1:
            return unique_exact_symbol_hits[0]

        exact_alias_hits = [
            hit for hit in normalized_hits if normalized_query in hit.get("aliases", ())
        ]
        unique_exact_alias_hits = self._dedupe_gene_hits(exact_alias_hits)
        if len(unique_exact_alias_hits) == 1:
            return unique_exact_alias_hits[0]

        unique_hits = self._dedupe_gene_hits(normalized_hits)
        if len(unique_hits) == 1:
            return unique_hits[0]
        return None

    def _normalize_gene_hit(self, hit: dict[str, object]) -> dict[str, str] | None:
        taxid = str(hit.get("taxid", "")).strip()
        entrez = str(hit.get("entrezgene", "")).strip()
        symbol = str(hit.get("symbol", "")).strip()
        if taxid and taxid != "9606":
            return None
        if not entrez:
            return None

        aliases = hit.get("alias", [])
        alias_values: tuple[str, ...]
        if isinstance(aliases, list):
            alias_values = tuple(
                str(item).strip().lower() for item in aliases if str(item).strip()
            )
        elif isinstance(aliases, str) and aliases.strip():
            alias_values = (aliases.strip().lower(),)
        else:
            alias_values = ()

        return {
            "entrez_id": entrez,
            "symbol": symbol or entrez,
            "aliases": alias_values,
        }

    def _dedupe_gene_hits(self, hits: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: dict[str, dict[str, str]] = {}
        for hit in hits:
            deduped[hit["entrez_id"]] = hit
        return list(deduped.values())

    def _collapse_lookup_errors(self, errors: list[str], *, default: str) -> str:
        unique_errors = list(dict.fromkeys(error for error in errors if error))
        if not unique_errors:
            return default
        return "; ".join(unique_errors)


class DiseaseBridgeProvider(BaseBridgeProvider):
    """Resolve unresolved diseases or subtypes through MyDisease.info NCIt xrefs."""

    bridge_name = "MyDisease.info"

    def bridge(self, entity: ResolvedEntity) -> tuple[ResolvedEntity | None, str | None]:
        candidates, error = self._lookup_ncit_ids(entity.text)
        if not candidates:
            return None, error

        bridged_results: list[ResolvedEntity] = []
        for candidate in candidates:
            if entity.entity_type.strip().lower() == "subtype":
                bridged = self.resolver.resolve_subtype_ncit(
                    candidate["ncit_id"],
                    original_text=entity.text,
                    resolved_name=candidate.get("name"),
                )
            else:
                bridged = self.resolver.resolve_disease_ncit(
                    candidate["ncit_id"],
                    original_text=entity.text,
                    resolved_name=candidate.get("name"),
                )
            if bridged.status == "resolved":
                bridged_results.append(bridged)

        unique_results = self._dedupe_resolved_entities(bridged_results)
        if len(unique_results) == 1:
            return unique_results[0], None
        if len(unique_results) > 1:
            return (
                ResolvedEntity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    status="ambiguous",
                    candidates=[
                        {
                            "resolved_id": result.resolved_id,
                            "resolved_name": result.resolved_name,
                            "match_field": result.match_field,
                        }
                        for result in unique_results
                    ],
                ),
                None,
            )

        return None, "MyDisease NCIt xrefs were not found in graph"

    def _lookup_ncit_ids(self, text: str) -> tuple[list[dict[str, str]], str | None]:
        query = text.strip()
        if not query:
            return [], "empty disease query"

        params = urlencode(
            {
                "q": query,
                "size": 5,
                "fields": "mondo.label,mondo.xrefs,disease_ontology.name,disease_ontology.xrefs",
            }
        )
        try:
            payload = self._get_json(f"{MYDISEASE_BASE_URL}/query?{params}")
        except HTTPError as exc:
            return [], f"MyDisease /query failed for {query}: HTTP {exc.code}"
        except URLError as exc:
            return [], f"MyDisease /query failed for {query}: {exc.reason}"

        if not isinstance(payload, dict):
            return [], f"MyDisease /query returned non-object for {query}"

        hits = payload.get("hits", [])
        if not isinstance(hits, list):
            return [], f"MyDisease /query returned invalid hits for {query}"

        candidates: list[dict[str, str]] = []
        for raw_hit in hits:
            if not isinstance(raw_hit, dict):
                continue
            ncit_ids = self._extract_ncit_ids(raw_hit)
            if not ncit_ids:
                continue
            name = self._extract_hit_name(raw_hit)
            for ncit_id in ncit_ids:
                candidates.append({"ncit_id": ncit_id, "name": name})
        return self._dedupe_ncit_candidates(candidates), None

    def _get_json(self, url: str) -> dict[str, object] | list[object] | None:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_hit_name(self, hit: dict[str, object]) -> str:
        mondo = hit.get("mondo", {})
        if isinstance(mondo, dict):
            label = str(mondo.get("label") or "").strip()
            if label:
                return label
        disease_ontology = hit.get("disease_ontology", {})
        if isinstance(disease_ontology, dict):
            name = str(disease_ontology.get("name") or "").strip()
            if name:
                return name
        return str(hit.get("_id") or "").strip()

    def _extract_ncit_ids(self, value: object) -> list[str]:
        found: list[str] = []

        def visit(item: object, *, in_nci_context: bool = False) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    key_text = str(key).strip().lower()
                    visit(
                        nested,
                        in_nci_context=(
                            in_nci_context
                            or key_text in {"ncit", "nci", "nci_thesaurus", "nci thesaurus"}
                        ),
                    )
                return
            if isinstance(item, list):
                for nested in item:
                    visit(nested, in_nci_context=in_nci_context)
                return
            if isinstance(item, str):
                for ncit_id in _extract_ncit_ids_from_text(
                    item,
                    allow_bare=in_nci_context,
                ):
                    found.append(ncit_id)

        visit(value)
        return list(dict.fromkeys(found))

    def _dedupe_ncit_candidates(
        self,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        deduped: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            deduped[candidate["ncit_id"]] = candidate
        return list(deduped.values())

    def _dedupe_resolved_entities(
        self,
        entities: list[ResolvedEntity],
    ) -> list[ResolvedEntity]:
        deduped: dict[tuple[str, str], ResolvedEntity] = {}
        for entity in entities:
            deduped[(entity.entity_type, entity.resolved_id or "")] = entity
        return list(deduped.values())


def _extract_ncit_ids_from_text(text: str, *, allow_bare: bool = False) -> list[str]:
    values: list[str] = []
    patterns = [r"\b(?:NCIT|NCI|NCI\s+Thesaurus)[:_\s-]*(C\d+)\b"]
    if allow_bare:
        patterns.append(r"\b(C\d{3,})\b")
    for pattern in patterns:
        values.extend(
            match.group(1).upper()
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
        return list(dict.fromkeys(values))


class DrugBridgeProvider(BaseBridgeProvider):
    """Resolve unresolved drugs through MyChem, PubChem, and ChEMBL identifiers."""

    bridge_name = "MyChem/PubChem/ChEMBL"

    def bridge(self, entity: ResolvedEntity) -> tuple[ResolvedEntity | None, str | None]:
        candidates, error = self._lookup_drug_identifiers(entity.text)
        if not candidates:
            return None, error

        bridged_results: list[ResolvedEntity] = []
        for candidate in candidates:
            resolved = self._resolve_candidate(candidate)
            if resolved.status == "resolved":
                resolved.text = entity.text
                bridged_results.append(resolved)

        unique_results = self._dedupe_resolved_entities(bridged_results)
        if len(unique_results) == 1:
            return unique_results[0], None
        if len(unique_results) > 1:
            return (
                ResolvedEntity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    status="ambiguous",
                    candidates=[
                        {
                            "resolved_id": result.resolved_id,
                            "resolved_name": result.resolved_name,
                            "match_field": result.match_field,
                        }
                        for result in unique_results
                    ],
                ),
                None,
            )

        return None, "external drug identifiers were not found in graph"

    def _lookup_drug_identifiers(self, text: str) -> tuple[list[dict[str, str]], str | None]:
        query = text.strip()
        if not query:
            return [], "empty drug query"

        candidates: list[dict[str, str]] = []
        errors: list[str] = []

        mychem_id, error = self._lookup_mychem_id(query)
        if mychem_id:
            candidates.append({"value": mychem_id})
        elif error:
            errors.append(error)

        pubchem_cid, error = self._lookup_pubchem_cid(query)
        if pubchem_cid:
            candidates.append({"value": f"PUBCHEM:{pubchem_cid}"})
        elif error:
            errors.append(error)

        chembl_id, error = self._lookup_chembl_id(query)
        if chembl_id:
            candidates.append({"value": f"CHEMBL:{chembl_id}"})
        elif error:
            errors.append(error)

        deduped = list({candidate["value"]: candidate for candidate in candidates}.values())
        if deduped:
            return deduped, None
        return [], "; ".join(errors) if errors else "no external drug identifier match"

    def _lookup_mychem_id(self, query: str) -> tuple[str | None, str | None]:
        params = urlencode({"q": query, "size": 1, "fields": "_id"})
        try:
            payload = self._get_json(f"{MYCHEM_BASE_URL}/query?{params}")
        except HTTPError as exc:
            return None, f"MyChem /query failed for {query}: HTTP {exc.code}"
        except URLError as exc:
            return None, f"MyChem /query failed for {query}: {exc.reason}"
        if not isinstance(payload, dict):
            return None, f"MyChem /query returned non-object for {query}"
        hits = payload.get("hits", [])
        if not isinstance(hits, list) or not hits:
            return None, None
        hit = hits[0]
        if not isinstance(hit, dict):
            return None, None
        mychem_id = str(hit.get("_id") or "").strip()
        return (mychem_id or None), None

    def _lookup_pubchem_cid(self, query: str) -> tuple[str | None, str | None]:
        url = f"{PUBCHEM_BASE_URL}/compound/name/{quote(query, safe='')}/cids/JSON"
        try:
            payload = self._get_json(url)
        except HTTPError as exc:
            if exc.code in {400, 404}:
                return None, None
            return None, f"PubChem lookup failed for {query}: HTTP {exc.code}"
        except URLError as exc:
            return None, f"PubChem lookup failed for {query}: {exc.reason}"
        if not isinstance(payload, dict):
            return None, f"PubChem lookup returned non-object for {query}"
        identifier_list = payload.get("IdentifierList", {})
        if not isinstance(identifier_list, dict):
            return None, None
        cids = identifier_list.get("CID", [])
        if not isinstance(cids, list) or not cids:
            return None, None
        return str(cids[0]).strip(), None

    def _lookup_chembl_id(self, query: str) -> tuple[str | None, str | None]:
        params = urlencode({"q": query, "limit": 1})
        try:
            payload = self._get_json(f"{CHEMBL_BASE_URL}/molecule/search.json?{params}")
        except HTTPError as exc:
            return None, f"ChEMBL molecule search failed for {query}: HTTP {exc.code}"
        except URLError as exc:
            return None, f"ChEMBL molecule search failed for {query}: {exc.reason}"
        if not isinstance(payload, dict):
            return None, f"ChEMBL search returned non-object for {query}"
        molecules = payload.get("molecules", [])
        if not isinstance(molecules, list) or not molecules:
            return None, None
        molecule = molecules[0]
        if not isinstance(molecule, dict):
            return None, None
        chembl_id = str(molecule.get("molecule_chembl_id") or "").strip()
        return (chembl_id or None), None

    def _resolve_candidate(self, candidate: dict[str, str]) -> ResolvedEntity:
        return self.resolver.resolve(
            [
                EntityMention(
                    text=candidate["value"],
                    entity_type="Drug",
                    semantic_group="direct",
                )
            ]
        )[0]

    def _get_json(self, url: str) -> dict[str, object] | list[object] | None:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _dedupe_resolved_entities(
        self,
        entities: list[ResolvedEntity],
    ) -> list[ResolvedEntity]:
        deduped: dict[tuple[str, str], ResolvedEntity] = {}
        for entity in entities:
            deduped[(entity.entity_type, entity.resolved_id or "")] = entity
        return list(deduped.values())


class EntityBridge:
    """Bridge unresolved entities to external identifier systems."""

    def __init__(self, resolver: EntityResolver):
        self.resolver = resolver
        self.last_lookup_errors: dict[str, str] = {}
        self.providers: dict[str, BaseBridgeProvider] = {
            "gene": GeneBridgeProvider(resolver),
            "disease": DiseaseBridgeProvider(resolver),
            "subtype": DiseaseBridgeProvider(resolver),
            "drug": DrugBridgeProvider(resolver),
        }

    def bridge(self, unresolved_entities: list[ResolvedEntity]) -> BridgeDecision:
        self.last_lookup_errors = {}
        bridge_targets: list[dict[str, str]] = []
        bridged_entities: list[ResolvedEntity] = []
        still_unresolved: list[ResolvedEntity] = []

        for entity in unresolved_entities:
            normalized_type = entity.entity_type.strip().lower()
            provider = self.providers.get(normalized_type)
            if provider is None:
                still_unresolved.append(entity)
                continue

            bridge_target = {
                "text": entity.text,
                "entity_type": entity.entity_type,
                "bridge": provider.bridge_name,
            }
            bridge_targets.append(bridge_target)

            bridged, error = provider.bridge(entity)
            if bridged is None:
                if error:
                    bridge_target["error"] = error
                    self.last_lookup_errors[entity.text] = error
                still_unresolved.append(entity)
                continue

            bridged_entities.append(bridged)

        if bridged_entities:
            labels = ", ".join(
                f"{entity.text} -> {entity.resolved_name or entity.resolved_id or 'unknown'}"
                for entity in bridged_entities
            )
            return BridgeDecision(
                action="resolved",
                bridge_targets=bridge_targets,
                bridged_entities=bridged_entities,
                still_unresolved=still_unresolved,
                message=f"External bridge resolved: {labels}.",
            )

        lookup_errors = [
            f"{item['text']}: {item['error']}" for item in bridge_targets if item.get("error")
        ]
        if lookup_errors:
            return BridgeDecision(
                action="fail",
                bridge_targets=bridge_targets,
                bridged_entities=bridged_entities,
                still_unresolved=still_unresolved,
                message="External bridge lookup failed for: " + "; ".join(lookup_errors) + ".",
            )

        if bridge_targets:
            target_labels = ", ".join(
                f"{item['text']} -> {item['bridge']}" for item in bridge_targets
            )
            return BridgeDecision(
                action="bridge",
                bridge_targets=bridge_targets,
                bridged_entities=bridged_entities,
                still_unresolved=still_unresolved,
                message=f"External bridge lookup is needed for: {target_labels}.",
            )

        unresolved_texts = ", ".join(entity.text for entity in still_unresolved) or "none"
        return BridgeDecision(
            action="fail",
            bridge_targets=bridge_targets,
            bridged_entities=bridged_entities,
            still_unresolved=still_unresolved,
            message=f"No external bridge is configured for: {unresolved_texts}.",
        )
