# Dose response

## Raw file

`data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx`

The file contains fitted cell-line drug-response records. The relevant fields are:

- `SANGER_MODEL_ID`: Sanger model identifier.
- `CELL_LINE_NAME`: cell-line name recorded by GDSC.
- `COSMIC_ID`: COSMIC cell-line identifier.
- `DRUG_ID`: GDSC drug identifier.
- `LN_IC50`: fitted log IC50 value.
- `AUC`: area under the fitted dose-response curve.
- `Z_SCORE`: standardized drug-response score.

## Processing rules

- A `Z_SCORE` less than `-2` indicates drug sensitivity.
- A `Z_SCORE` greater than `2` indicates drug resistance.
