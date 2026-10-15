# Data sources

Download the files below and keep the listed filenames.

| Source | Version used | Files | Place in |
| --- | --- | --- | --- |
| [Cellosaurus](https://ftp.expasy.org/databases/cellosaurus/) | 54.0, 27 November 2025 | `cellosaurus.txt` | `data/raw/cellosaurus/` |
| [DepMap](https://depmap.org/portal/data_page/) | Public 25Q3 | `Model.csv`, `OmicsSomaticMutations.csv`, `OmicsCNGeneWGS.csv`, `CRISPRGeneEffect.csv`, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | `data/raw/depmap/` |
| [GDSC](https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/) | Release 8.5, 27 October 2023 | `GDSC2_fitted_dose_response_27Oct23.xlsx`, `screened_compounds_rel_8.5.csv` | `data/raw/gdsc/` |
| [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/downloads) | Proteomics export 11 February 2025; model-list export 3 March 2026 | `Proteomics_20250211/proteomics_all_20250211.csv`, `model_list_20260303.csv` | `data/raw/cell_model_passport/` |

`data/curation/gdsc_drug_external_ids.csv` is the retained GDSC drug identifier mapping used during graph construction. It was prepared with the [BioThings](https://mychem.info/) identifier-mapping service and is included with the code.


