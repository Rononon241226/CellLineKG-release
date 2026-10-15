# Neo4j Import Guide

This document describes how to import the processed CellLineKG graph tables into Neo4j database.

## Environment

- Neo4j: 2026.02.3
- Edition: Community Edition

## Required files

The import requires the complete `neo4j_import` output produced by the
CellLineKG data-processing pipeline. It contains 14 node tables and 20 edge tables.

Files are under the Neo4j import directory with this structure:

```text
<neo4j-import-directory>/
└── neo4j_import/
    ├── nodes/
    │   ├── nodes_cellline.csv
    │   ├── nodes_copy_number.csv
    │   ├── nodes_dependency.csv
    │   ├── nodes_disease.csv
    │   ├── nodes_drug.csv
    │   ├── nodes_drug_screen.csv
    │   ├── nodes_gene.csv
    │   ├── nodes_gene_expression.csv
    │   ├── nodes_lineage.csv
    │   ├── nodes_media.csv
    │   ├── nodes_protein.csv
    │   ├── nodes_protein_abundance.csv
    │   ├── nodes_subtype.csv
    │   └── nodes_variant.csv
    └── edges/
        ├── edges_cellline_cultures_in.csv
        ├── edges_cellline_has_copy_number.csv
        ├── edges_cellline_has_dependency.csv
        ├── edges_cellline_has_disease.csv
        ├── edges_cellline_has_drug_screen.csv
        ├── edges_cellline_has_expression.csv
        ├── edges_cellline_has_lineage.csv
        ├── edges_cellline_has_protein_abundance.csv
        ├── edges_cellline_has_subtype.csv
        ├── edges_cellline_has_variant.csv
        ├── edges_copy_number_target_gene.csv
        ├── edges_dependency_target_gene.csv
        ├── edges_disease_part_of.csv
        ├── edges_drug_screen_tested_drug.csv
        ├── edges_drug_targets_protein.csv
        ├── edges_expression_measured_gene.csv
        ├── edges_protein_abundance_measured_protein.csv
        ├── edges_protein_encoded_by_gene.csv
        ├── edges_subtype_of.csv
        └── edges_variant_in_gene.csv
```

## Import order

Run the following scripts in exactly this order:

1. `scripts/neo4j/analysis_01_import_cellosaurus_neo4j.cypher`
2. `scripts/neo4j/analysis_02_import_depmap_neo4j.cypher`
3. `scripts/neo4j/analysis_03_import_gdsc_neo4j.cypher`
4. `scripts/neo4j/analysis_04_import_cell_model_passport_neo4j.cypher`


## Verify the imported graph

Count all nodes:

```cypher
MATCH (n)
RETURN count(n) AS node_count;
```

Expected result:

```text
2946803
```

Count all relationships:

```cypher
MATCH ()-[r]->()
RETURN count(r) AS relationship_count;
```

Expected result:

```text
5844778
```

The complete expected counts and source-coverage statistics are stored
in [`../check/summary.json`].

