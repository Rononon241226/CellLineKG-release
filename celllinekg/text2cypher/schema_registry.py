"""Entity-focused schema context for Cypher generation."""

from __future__ import annotations

from .models import EntityMention, ResolvedEntity


VARIANT_SCHEMA = """
Entity schema for Variant:
- Node label: Variant

Fields:
- protein_change: protein_change uses HGVS-like protein notation with a p. prefix, e.g. p.V600E.
- rs_id: dbSNP rs identifier when present, such as rs121913529.
- variant_type: variant category.

Paths:
- (:CellLine)-[:HAS_VARIANT]->(:Variant)-[:IN_GENE]->(:Gene)

Rule:

"""


GENE_EXPRESSION_SCHEMA = """
Entity schema for GeneExpression:
- Node label: GeneExpression

Path:
- (:CellLine)-[:HAS_EXPRESSION]->(:GeneExpression)-[:MEASURED_GENE]->(:Gene)

Fields:
- z_score: Z-score standardized expression value.
- level: categorical gene-expression label (high/low).
- rank_within_cellline: rank within one cell line by descending abs(z_score); smaller rank means more extreme expression change.

Rule:
- For questions of comparing expression among cell lines, sort by z_score.
- Use level = "high" for high-expression queries and level = "low" for low-expression queries.
"""


DEPENDENCY_SCHEMA = """
Entity schema for Dependency:
- Node label: Dependency

Path:
- (:CellLine)-[:HAS_DEPENDENCY]->(:Dependency)-[:TARGET_GENE]->(:Gene)

Fields:
- score: dependency score (More negative dependency scores indicate stronger gene dependency / essentiality for cell survival.).
- rank_within_cellline: rank of the gene dependency score within one cell line; smaller rank means stronger dependency.

Rule:
- For dependency questions, sort by score.
"""


COPY_NUMBER_SCHEMA = """
Entity schema for CopyNumber:
- Node label: CopyNumber

Path:
- (:CellLine)-[:HAS_COPY_NUMBER]->(:CopyNumber)-[:TARGET_GENE]->(:Gene)

Fields:
- value: value of copy-number/2.
- status: called status (amplification/deletion).
"""


DRUG_SCREEN_SCHEMA = """
Entity schema for DrugScreen:
- Node label: DrugScreen

Path:
- (:CellLine)-[:HAS_DRUG_SCREEN]->(:DrugScreen)-[:TESTED_DRUG]->(:Drug)

Fields:
- auc: area under the dose-response curve.
- ic50: half-maximal inhibitory concentration.
- z_score: Z-score standardized drug response score.
- drug_sensitivity: sensitivity classification (sensitive/intermediate/resistant).

Rule:
"""


DRUG_SCHEMA = """
Entity schema for Drug:
- Node label: Drug

Fields:
- name: drug name.
- target: drug target.
- target_pathway: target pathway.

Paths:
- (:DrugScreen)-[:TESTED_DRUG]->(:Drug)
- (:Drug)-[:TARGETS]->(:Protein)
- (:Drug)-[:TARGETS]->(:Protein)-[:ENCODED_BY]->(:Gene)

Rule:
- Match target_pathway values exactly as graph values. Do not append words such as "pathway" to the target_pathway value.
"""


PROTEIN_SCHEMA = """
Entity schema for Protein:
- Node label: Protein

Fields:
- protein_id: canonical graph identifier for protein matching.
- uniprot_id: UniProt accession.
- symbol: protein/gene symbol from Cell Model Passport.

Paths:
- (:ProteinAbundance)-[:MEASURED_PROTEIN]->(:Protein)
- (:Drug)-[:TARGETS]->(:Protein)
- (:Protein)-[:ENCODED_BY]->(:Gene)
"""


PROTEIN_ABUNDANCE_SCHEMA = """
Entity schema for ProteinAbundance:
- Node label: ProteinAbundance

Path:
- (:CellLine)-[:HAS_PROTEIN_ABUNDANCE]->(:ProteinAbundance)-[:MEASURED_PROTEIN]->(:Protein)
- (:Protein)-[:ENCODED_BY]->(:Gene)

Fields:
- z_score: Z-score standardized abundance score.
- level: categorical abundance label (high/low).
- rank_within_cellline: rank within one cell line by descending abs(z_score); smaller rank means more extreme protein abundance change.
"""


CELLLINE_SCHEMA = """
Entity schema for CellLine:
- Node label: CellLine

Fields:
- cellosaurus_id: canonical graph identifier for cell lines.
- name: primary cell-line name.
- synonyms: alternate cell-line names.
- growth_pattern: tumor growth type (Adherent/Suspension/Mixed)
- tumor_origin: tumor tissue origin (Primary/Metastatic)

Paths:
- (:CellLine)-[:CULTURES_IN]->(:Media)
- (:CellLine)-[:HAS_DISEASE]->(:Disease)
- (:CellLine)-[:HAS_SUBTYPE]->(:Subtype)
- (:CellLine)-[:HAS_LINEAGE]->(:Lineage)
- (:CellLine)-[:HAS_VARIANT]->(:Variant)
- (:CellLine)-[:HAS_COPY_NUMBER]->(:CopyNumber)
- (:CellLine)-[:HAS_DEPENDENCY]->(:Dependency)
- (:CellLine)-[:HAS_EXPRESSION]->(:GeneExpression)
- (:CellLine)-[:HAS_DRUG_SCREEN]->(:DrugScreen)
- (:CellLine)-[:HAS_PROTEIN_ABUNDANCE]->(:ProteinAbundance)

Rule:

"""

GENE_SCHEMA = """
Entity schema for Gene:
- Node label: Gene

Fields:
- entrez_id: canonical graph identifier for gene matching.
- symbol: gene symbol.
- name: gene name.

Paths:
- (:Variant)-[:IN_GENE]->(:Gene)
- (:CopyNumber)-[:TARGET_GENE]->(:Gene)
- (:Dependency)-[:TARGET_GENE]->(:Gene)
- (:GeneExpression)-[:MEASURED_GENE]->(:Gene)
- (:Protein)-[:ENCODED_BY]->(:Gene)

Rule:
"""

MEDIA_SCHEMA="""
Entity schema for Media:
- Node label: Media

Fields:
- formulation: complete media formulation including supplements.

"""

DISEASE_SCHEMA = """
Entity schema for Disease:
- Node label: Disease

Fields:
- disease_id: canonical graph identifier for disease matching.
- name: disease name.

Paths:
- (:CellLine)-[:HAS_DISEASE]->(:Disease)
- (:Subtype)-[:SUBTYPE_OF]->(:Disease)
- (:Disease)-[:PART_OF]->(:Lineage)
"""


SUBTYPE_SCHEMA = """
Entity schema for Subtype:
- Node label: Subtype

Fields:
- subtype_id: canonical graph identifier for subtype matching.
- name: subtype name.

Paths:
- (:CellLine)-[:HAS_SUBTYPE]->(:Subtype)
- (:Subtype)-[:SUBTYPE_OF]->(:Disease)
"""


LINEAGE_SCHEMA = """
Entity schema for Lineage:
- Node label: Lineage

Fields:
- lineage_id: canonical graph identifier for tissue or organ lineage matching.
- name: lineage name.

Paths:
- (:CellLine)-[:HAS_LINEAGE]->(:Lineage)
"""


SCHEMAS_BY_TYPE: dict[str, str] = {
    "variant": VARIANT_SCHEMA,
    "geneexpression": GENE_EXPRESSION_SCHEMA,
    "dependency": DEPENDENCY_SCHEMA,
    "copynumber": COPY_NUMBER_SCHEMA,
    "drugscreen": DRUG_SCREEN_SCHEMA,
    "drug": DRUG_SCHEMA,
    "protein": PROTEIN_SCHEMA,
    "proteinabundance": PROTEIN_ABUNDANCE_SCHEMA,
    "cellline": CELLLINE_SCHEMA,
    "gene": GENE_SCHEMA,
    "media": MEDIA_SCHEMA,
    "disease": DISEASE_SCHEMA,
    "subtype": SUBTYPE_SCHEMA,
    "lineage": LINEAGE_SCHEMA,
}


class EntitySchemaRegistry:
    """Return relevant schema snippets based on resolved entities."""

    def build_context(
        self,
        resolved_entities: list[ResolvedEntity],
        mentions: list[EntityMention] | None = None,
    ) -> str:
        entity_types: list[str] = []
        for entity in resolved_entities:
            normalized_type = entity.entity_type.strip().lower()
            if normalized_type in SCHEMAS_BY_TYPE and normalized_type not in entity_types:
                entity_types.append(normalized_type)

        for mention in mentions or []:
            if mention.semantic_group != "indirect":
                continue
            normalized_type = mention.entity_type.strip().lower()
            if normalized_type in SCHEMAS_BY_TYPE and normalized_type not in entity_types:
                entity_types.append(normalized_type)

        if not entity_types:
            return "No entity-specific schema was selected."

        parts = [SCHEMAS_BY_TYPE[entity_type].strip() for entity_type in entity_types]
        return "\n\n".join(parts)
