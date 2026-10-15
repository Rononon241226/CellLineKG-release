# Gene copy number

## Raw file

`data/raw/depmap/OmicsCNGeneWGS.csv`

The file is a wide matrix. The first six columns contain record metadata, including `ModelID` and the two default-entry flags. The remaining columns are genes. Gene columns use the form `gene symbol (Entrez Gene ID)`.

The file contains continuous, linear-scale gene copy-number values.

## Processing rules

- When this file is used to answer a question, restrict the relevant records to rows where both `IsDefaultEntryForModel` and `IsDefaultEntryForMC` equal `Yes`
- Ignore retained rows with an empty `ModelID`.
- Values greater than or equal to `2.0` are classified as `amplification`, values less than or equal to `0.5` are classified as `deletion`.


