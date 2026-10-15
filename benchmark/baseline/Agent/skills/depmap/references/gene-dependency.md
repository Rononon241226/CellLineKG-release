# CRISPR gene dependency

## Raw file

`data/raw/depmap/CRISPRGeneEffect.csv`

The file is a wide matrix. The first column, named `Unnamed: 0` when read directly with pandas, contains the DepMap `ModelID`. The remaining columns are genes in the form `gene symbol (Entrez Gene ID)`.

More negative gene-effect scores indicate stronger dependency.

## Processing rules

- A gene is considered a dependency for a cell line when its gene-effect score is less than or equal to `-1.0` and it is among the 500 lowest qualifying scores for that cell line.
