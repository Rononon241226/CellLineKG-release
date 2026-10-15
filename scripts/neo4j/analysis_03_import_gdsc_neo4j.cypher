CREATE CONSTRAINT analysis_drug_id IF NOT EXISTS
FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE;

CREATE CONSTRAINT analysis_drug_screen_id IF NOT EXISTS
FOR (d:DrugScreen) REQUIRE d.drug_screen_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_drug.csv' AS row
CALL {
    WITH row
    MERGE (d:Drug {drug_id: row.drug_id})
    SET d.name = row.name,
        d.synonyms = row.synonyms,
        d.mychem_id = row.mychem_id,
        d.external_aliases = row.external_aliases,
        d.target = row.target,
        d.target_pathway = row.target_pathway
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_drug_screen.csv' AS row
CALL {
    WITH row
    MERGE (d:DrugScreen {drug_screen_id: row.drug_screen_id})
    REMOVE d.name, d.ln_ic50, d.threshold_rule
    SET d.ic50 = toFloat(row.ic50),
        d.auc = toFloat(row.auc),
        d.z_score = toFloat(row.z_score),
        d.drug_sensitivity = row.drug_sensitivity
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_drug_screen.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (d:DrugScreen {drug_screen_id: row.target_drug_screen_id})
    MERGE (c)-[r:HAS_DRUG_SCREEN]->(d)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_drug_screen_tested_drug.csv' AS row
CALL {
    WITH row
    MATCH (d1:DrugScreen {drug_screen_id: row.source_drug_screen_id})
    MATCH (d2:Drug {drug_id: row.target_drug_id})
    MERGE (d1)-[r:TESTED_DRUG]->(d2)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;
