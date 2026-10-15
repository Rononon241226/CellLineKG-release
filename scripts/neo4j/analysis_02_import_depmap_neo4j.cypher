CREATE CONSTRAINT analysis_subtype_id IF NOT EXISTS
FOR (s:Subtype) REQUIRE s.subtype_id IS UNIQUE;

CREATE CONSTRAINT analysis_lineage_id IF NOT EXISTS
FOR (l:Lineage) REQUIRE l.lineage_id IS UNIQUE;

CREATE CONSTRAINT analysis_media_id IF NOT EXISTS
FOR (m:Media) REQUIRE m.media_id IS UNIQUE;

CREATE CONSTRAINT analysis_gene_id IF NOT EXISTS
FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE;

CREATE CONSTRAINT analysis_variant_id IF NOT EXISTS
FOR (v:Variant) REQUIRE v.variant_id IS UNIQUE;

CREATE CONSTRAINT analysis_copy_number_id IF NOT EXISTS
FOR (c:CopyNumber) REQUIRE c.copy_number_id IS UNIQUE;

CREATE CONSTRAINT analysis_dependency_id IF NOT EXISTS
FOR (d:Dependency) REQUIRE d.dependency_id IS UNIQUE;

CREATE CONSTRAINT analysis_expression_id IF NOT EXISTS
FOR (e:GeneExpression) REQUIRE e.expression_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_subtype.csv' AS row
CALL {
    WITH row
    MERGE (s:Subtype {subtype_id: row.subtype_id})
    SET s.name = row.name,
        s.oncotree_subtype_code = row.oncotree_subtype_code,
        s.ncit_subtype_id = row.ncit_subtype_id
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_lineage.csv' AS row
CALL {
    WITH row
    MERGE (l:Lineage {lineage_id: row.lineage_id})
    SET l.name = row.name
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_media.csv' AS row
CALL {
    WITH row
    MERGE (m:Media {media_id: row.media_id})
    SET m.formulation = row.formulation
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_gene.csv' AS row
CALL {
    WITH row
    MERGE (g:Gene {gene_id: row.gene_id})
    SET g.name = row.symbol,
        g.entrez_id = row.entrez_id,
        g.symbol = row.symbol
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_variant.csv' AS row
CALL {
    WITH row
    MERGE (v:Variant {variant_id: row.variant_id})
    REMOVE v.name, v.chrom, v.position, v.ref, v.alt, v.dna_change
    SET v.variant_type = row.variant_type,
        v.protein_change = row.protein_change,
        v.rs_id = row.rs_id
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_copy_number.csv' AS row
CALL {
    WITH row
    MERGE (c:CopyNumber {copy_number_id: row.copy_number_id})
    REMOVE c.value_type, c.threshold_rule, c.selection_rule, c.model_condition_id
    SET c.status = row.status,
        c.value = toFloat(row.value)
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_dependency.csv' AS row
CALL {
    WITH row
    MERGE (d:Dependency {dependency_id: row.dependency_id})
    REMOVE d.value_type, d.threshold_rule, d.selection_rule, d.model_condition_id
    SET d.score = toFloat(row.score),
        d.rank_within_cellline = toInteger(row.rank_within_cellline)
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/nodes/nodes_gene_expression.csv' AS row
CALL {
    WITH row
    MERGE (e:GeneExpression {expression_id: row.expression_id})
    REMOVE e.value, e.value_type, e.threshold_rule, e.selection_rule, e.model_condition_id
    SET e.z_score = toFloat(row.z_score),
        e.level = row.level,
        e.rank_within_cellline = toInteger(row.rank_within_cellline)
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_subtype.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (s:Subtype {subtype_id: row.target_subtype_id})
    MERGE (c)-[r:HAS_SUBTYPE]->(s)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_lineage.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (l:Lineage {lineage_id: row.target_lineage_id})
    MERGE (c)-[r:HAS_LINEAGE]->(l)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_cultures_in.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (m:Media {media_id: row.target_media_id})
    MERGE (c)-[r:CULTURES_IN]->(m)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_subtype_of.csv' AS row
CALL {
    WITH row
    MATCH (s:Subtype {subtype_id: row.source_subtype_id})
    MATCH (d:Disease {disease_id: row.target_disease_id})
    MERGE (s)-[r:SUBTYPE_OF]->(d)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_disease_part_of.csv' AS row
CALL {
    WITH row
    MATCH (d:Disease {disease_id: row.source_disease_id})
    MATCH (l:Lineage {lineage_id: row.target_lineage_id})
    MERGE (d)-[r:PART_OF]->(l)
    SET r.source = row.source
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_variant.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (v:Variant {variant_id: row.target_variant_id})
    MERGE (c)-[r:HAS_VARIANT]->(v)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_variant_in_gene.csv' AS row
CALL {
    WITH row
    MATCH (v:Variant {variant_id: row.source_variant_id})
    MATCH (g:Gene {gene_id: row.target_gene_id})
    MERGE (v)-[r:IN_GENE]->(g)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_copy_number.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (n:CopyNumber {copy_number_id: row.target_copy_number_id})
    MERGE (c)-[r:HAS_COPY_NUMBER]->(n)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_copy_number_target_gene.csv' AS row
CALL {
    WITH row
    MATCH (n:CopyNumber {copy_number_id: row.source_copy_number_id})
    MATCH (g:Gene {gene_id: row.target_gene_id})
    MERGE (n)-[r:TARGET_GENE]->(g)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_dependency.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (d:Dependency {dependency_id: row.target_dependency_id})
    MERGE (c)-[r:HAS_DEPENDENCY]->(d)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_dependency_target_gene.csv' AS row
CALL {
    WITH row
    MATCH (d:Dependency {dependency_id: row.source_dependency_id})
    MATCH (g:Gene {gene_id: row.target_gene_id})
    MERGE (d)-[r:TARGET_GENE]->(g)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_cellline_has_expression.csv' AS row
CALL {
    WITH row
    MATCH (c:CellLine {cellosaurus_id: row.source_cellosaurus_id})
    MATCH (e:GeneExpression {expression_id: row.target_expression_id})
    MERGE (c)-[r:HAS_EXPRESSION]->(e)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/edges/edges_expression_measured_gene.csv' AS row
CALL {
    WITH row
    MATCH (e:GeneExpression {expression_id: row.source_expression_id})
    MATCH (g:Gene {gene_id: row.target_gene_id})
    MERGE (e)-[r:MEASURED_GENE]->(g)
    SET r.source = row.source
} IN TRANSACTIONS OF 200 ROWS;
