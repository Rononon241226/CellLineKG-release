MATCH ()-[r:FROM_DATASET|IN_DATASET]->()
DELETE r;

MATCH (d:Dataset)
DETACH DELETE d;

CREATE CONSTRAINT analysis_cellline_cellosaurus_id IF NOT EXISTS
FOR (c:CellLine) REQUIRE c.cellosaurus_id IS UNIQUE;

CREATE CONSTRAINT analysis_disease_id IF NOT EXISTS
FOR (d:Disease) REQUIRE d.disease_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_cellline.csv' AS row
CALL {
    WITH row
    MERGE (c:CellLine {cellosaurus_id: row.cellosaurus_id})
    REMOVE c.cross_reference,
           c.reference_pmid,
           c.depmap_model_id,
           c.sanger_model_id,
           c.cmp_model_id,
           c.has_depmap,
           c.has_gdsc,
           c.has_cell_model_passport
    SET c.name = row.name,
        c.synonyms = row.synonyms,
        c.tumor_origin = row.tumor_origin,
        c.growth_pattern = row.growth_pattern
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_disease.csv' AS row
CALL {
    WITH row
    MERGE (d:Disease {disease_id: row.disease_id})
    SET d.name = row.name,
        d.oncotree_disease_code = row.oncotree_disease_code,
        d.ncit_disease_id = row.ncit_disease_id
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_disease.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (d:Disease {disease_id: row.target_disease_id})
    MERGE (c)-[r:HAS_DISEASE]->(d)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;
