# CellLineKG

CellLineKG is an LLM-based framework for answering biomedical cell-line questions over a Neo4j knowledge graph. It selects between Text2Cypher and evidence-path retrieval according to the question.

## Repository structure

```text
celllinekg/          Core data processing and question-answering code
scripts/             Command-line entry points
benchmark/           Questions, baselines, runners, and plotting code
data/                Raw, intermediate, and processed data
results/             Benchmark, KEGG, Source Data, and figure outputs
docs/                Data-source and Neo4j import documentation
```

Run all commands below from the repository root.

## Installation

Use Python 3.10 or later and install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

R dependencies for the KEGG analysis are listed in [`docs/R_REQUIREMENTS.md`](docs/R_REQUIREMENTS.md).

## 1. Prepare source data

Download the required files and place them under `data/raw/` as described in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## 2. Build graph tables

```bash
python -m celllinekg.pipelines.cellosaurus
python -m celllinekg.pipelines.depmap
python -m celllinekg.pipelines.gdsc
python -m celllinekg.pipelines.cell_model_passport
python -m celllinekg.integration.cellline_reconciliation
python -m celllinekg.integration.analysis_graph
```

The final Neo4j import tables are written to:

```text
data/processed/neo4j_import/nodes/
data/processed/neo4j_import/edges/
```

## 3. Import Neo4j

Use a new empty Neo4j Community Edition 2026.02.3 database and run the four Cypher scripts in `scripts/neo4j/` in numerical order. See [`docs/NEO4J_IMPORT.md`](docs/NEO4J_IMPORT.md).

## 4. Configure and run CellLineKG

Copy `.env.example` to `.env` and provide the required credentials.

Run the complete framework:

```bash
python scripts/framework/run_celllinekg.py "your question"
```

Start the API:

```bash
python -m uvicorn celllinekg.api.app:app --host 127.0.0.1 --port 8000
```

## 5. Run benchmarks

Benchmark questions and reference answers are under `benchmark/data/`. The three runners evaluate an LLM-only baseline, the tool-using Agent baseline, and CellLineKG:

```bash
python benchmark/runner/run_llm_only_batch.py --help
python benchmark/runner/run_batch.py --help
python benchmark/runner/run_celllinekg_batch.py --help
```

Raw runs are written to `results/benchmark/runs/`. Manually checked workbooks are under `results/benchmark/results_checked/`.

## 6. Run KEGG enrichment analysis

```bash
Rscript scripts/kegg/kegg_gene_batch.r
Rscript scripts/kegg/kegg_protein_batch.r
```

KEGG results are written to `results/kegg/`.

## 7. Prepare Source Data

```bash
python scripts/figures/prepare_source_data.py
```

The Source Data workbooks are generated in `results/source_data/`.

## 8. Generate figures

Run the six scripts in `benchmark/plotting/`. Figures are written to `results/benchmark/figures/`.

## Main outputs

| Output | Location |
| --- | --- |
| Neo4j import tables | `data/processed/neo4j_import/` |
| Benchmark runs | `results/benchmark/runs/` |
| Checked benchmark results | `results/benchmark/results_checked/` |
| KEGG results | `results/kegg/` |
| Figure Source Data | `results/source_data/` |
| Benchmark figures | `results/benchmark/figures/` |
