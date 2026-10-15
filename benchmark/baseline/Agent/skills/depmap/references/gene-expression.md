# Gene expression

## Raw file

`data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`

The file is a wide matrix. The first six columns contain record metadata, including `ModelID` and the two default-entry flags. The remaining columns are genes in the form `gene symbol (Entrez Gene ID)`.

The gene values are log-transformed expression values, `log2(TPM + 1)`.

## Processing rules

- When this file is used to answer a question, restrict the relevant records to rows where both `IsDefaultEntryForModel` and `IsDefaultEntryForMC` equal `Yes`
- For each gene, calculate the mean and population standard deviation across all retained rows.
- A gene-wise z-score is calculated as `(value - mean) / population standard deviation`.
- For each cell line, up to 500 genes with the largest `abs(z-score)` among those with `abs(z-score) > 2` are retained; positive z-scores indicate high expression and negative z-scores indicate low expression.
