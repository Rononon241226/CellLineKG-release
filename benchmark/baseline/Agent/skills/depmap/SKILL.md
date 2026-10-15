---
name: depmap
description: Interpret and process the raw DepMap files for cell-line metadata, somatic mutations, gene copy number, CRISPR gene effect, and gene expression. Use when a question requires evidence from data/raw/depmap.
---

# DepMap Raw Data

Use the raw DepMap files directly. Inspect the data and perform the required filtering and calculations yourself. Do not treat this skill as a query plan, and do not assume that a particular file is relevant until the question and data support that choice.

## Files and references

- `Model.csv`: cell-line names and metadata. Read [model-metadata.md](references/model-metadata.md).
- `OmicsSomaticMutations.csv`: one somatic mutation record per row. Read [somatic-mutations.md](references/somatic-mutations.md).
- `OmicsCNGeneWGS.csv`: a wide gene copy-number matrix. Read [copy-number.md](references/copy-number.md).
- `CRISPRGeneEffect.csv`: a wide CRISPR gene-effect matrix. Read [gene-dependency.md](references/gene-dependency.md).
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`: a wide gene-expression matrix. Read [gene-expression.md](references/gene-expression.md).

Read only the references needed for the current task.

## Shared conventions

- `ModelID` is the DepMap cell-line identifier used across these files.
- Gene-matrix columns use the form `gene symbol (Entrez Gene ID)`, for example `TP53 (7157)`.

## Benchmark cell-line scope

- `read_cell_scope()` returns the 1,780 benchmark DepMap `ModelID` values as a pandas Series. When an answer returns, counts, or compares cell lines, restrict the final results with `frame["ModelID"].isin(read_cell_scope())`.
