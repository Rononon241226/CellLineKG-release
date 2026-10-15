# Model metadata

## Raw file

`data/raw/depmap/Model.csv`

The file has one model record per row. `ModelID` is the DepMap cell-line identifier, and `CellLineName` is the displayed DepMap name. The relevant metadata fields are:

- `ModelID`: DepMap cell-line identifier.
- `CellLineName`: cell-line name used by DepMap.
- `RRID`: Cellosaurus cell-line identifier.
- `OncotreeLineage`: broad tissue or lineage classification.
- `OncotreePrimaryDisease`: primary disease classification.
- `OncotreeSubtype`: more specific disease subtype.
- `OncotreeCode`: OncoTree classification code.
- `PrimaryOrMetastasis`: whether the model originated from a primary or metastatic sample.
- `GrowthPattern`: adherent, suspension, or another recorded growth pattern.
- `FormulationID`: recorded culture formulation or medium.
- `COSMICID`: COSMIC cell-line identifier.

## Processing rules

- Ignore a row when `ModelID` is empty.
- Strip surrounding whitespace from retained text values.
- Remove a trailing `.0` from `COSMICID` when present.
- `ModelID` can be used to associate this metadata with other DepMap files.
- Match the classification level requested by the question. Do not broaden a specific disease or subtype to its wider lineage unless the question asks for that wider category.
