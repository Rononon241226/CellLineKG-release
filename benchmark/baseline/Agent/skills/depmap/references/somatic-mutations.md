# Somatic mutations

## Raw file

`data/raw/depmap/OmicsSomaticMutations.csv`

The file is in long format: each row is one mutation record. `ModelID` identifies the cell line. The main fields are:

- `ModelID`: DepMap cell-line identifier.
- `Chrom`: chromosome containing the variant.
- `Pos`: genomic position of the variant.
- `Ref`: reference allele.
- `Alt`: alternate allele.
- `VariantType`: variant category, such as SNV, insertion, or deletion.
- `DNAChange`: nucleotide-level change reported by DepMap.
- `ProteinChange`: protein-level change, such as `p.V600E`.
- `DbsnpRsID`: dbSNP rs identifier when available.
- `HugoSymbol`: gene symbol.
- `EntrezGeneID`: Entrez Gene identifier.

## Processing rules

- When this file is used to answer a question, restrict the relevant records to rows where both `IsDefaultEntryForModel` and `IsDefaultEntryForMC` equal `Yes`.
- Ignore retained rows with an empty `ModelID`.
- Do not apply an additional mutation threshold.

