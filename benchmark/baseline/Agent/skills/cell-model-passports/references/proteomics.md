# Proteomics

## Raw file

`data/raw/cell_model_passport/Proteomics_20250211/proteomics_all_20250211.csv`

The file contains protein abundance records in long format. The relevant fields are:

- `model_id`: Cell Model Passports model identifier.
- `model_name`: model name.
- `uniprot_id`: UniProt protein identifier.
- `symbol`: protein symbol.
- `protein_intensity`: recorded protein intensity.
- `zscore`: file-provided protein abundance z-score.

## Processing rules

- For each cell line, up to 500 proteins with the largest `abs(zscore)` among those with `abs(zscore) > 2` are retained; A `zscore` greater than `2` indicates high protein abundance, while a `zscore` less than `-2` indicates low protein abundance.
