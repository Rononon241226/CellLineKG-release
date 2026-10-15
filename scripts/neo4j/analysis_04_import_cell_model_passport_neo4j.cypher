CREATE CONSTRAINT analysis_protein_id IF NOT EXISTS
FOR (p:Protein) REQUIRE p.protein_id IS UNIQUE;

CREATE CONSTRAINT analysis_protein_abundance_id IF NOT EXISTS
FOR (p:ProteinAbundance) REQUIRE p.protein_abundance_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_protein.csv' AS row
CALL {
    WITH row
    MERGE (p:Protein {protein_id: row.protein_id})
    SET p.uniprot_id = row.uniprot_id,
        p.symbol = row.symbol
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_protein_abundance.csv' AS row
CALL {
    WITH row
    MERGE (p:ProteinAbundance {protein_abundance_id: row.protein_abundance_id})
    REMOVE p.name, p.value, p.value_type, p.threshold_rule, p.selection_rule
    SET p.z_score = toFloat(row.z_score),
        p.level = row.level,
        p.rank_within_cellline = toInteger(row.rank_within_cellline)
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_protein_abundance.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (p:ProteinAbundance {protein_abundance_id: row.target_protein_abundance_id})
    MERGE (c)-[r:HAS_PROTEIN_ABUNDANCE]->(p)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_protein_abundance_measured_protein.csv' AS row
CALL {
    WITH row
    MATCH (p1:ProteinAbundance {protein_abundance_id: row.source_protein_abundance_id})
    MATCH (p2:Protein {protein_id: row.target_protein_id})
    MERGE (p1)-[r:MEASURED_PROTEIN]->(p2)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_protein_encoded_by_gene.csv' AS row
CALL {
    WITH row
    MATCH (p:Protein {protein_id: row.source_protein_id})
    MATCH (g:Gene {gene_id: row.target_gene_id})
    MERGE (p)-[r:ENCODED_BY]->(g)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_drug_targets_protein.csv' AS row
CALL {
    WITH row
    MATCH (d:Drug {drug_id: row.source_drug_id})
    MATCH (p:Protein {protein_id: row.target_protein_id})
    MERGE (d)-[r:TARGETS]->(p)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;
