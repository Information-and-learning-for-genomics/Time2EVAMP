# Data preprocessing

## Phenotype preprocessing

`phen_preprocess.py` - Preprocessing of phenotype data for time-to-event analysis. The script uses participant indices and a participant table containing the outcome, age, birth date, and assessment date to generate training and test phenotype data.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `--train-ind` | `None` | File containing indices of samples used for training. |
| `--test-ind` | `None` | File containing indices of samples used for testing. |
| `--participant-table` | `None` | Comma-separated participant table containing the outcome and age-related information. The table must contain the columns `eid`, `y` (date of event), `birth` (year), and `assesment` (date of assesment). |
| `--out` | — | Output prefix. |

## Covariate adjustment

`data_cov_adj.py` Adjustment of the design matrix for covariate effects using a linear or exponential model. The script matches samples to a covariate table using participant IDs.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `--X` | `None` | Design matrix to be adjusted for covariates. |
| `--indlist` | `None` | List of sample IDs matching the order in X. Those should correspond to eids in the covariate table. |
| `--model` | `linear` | Model used for covariate adjustment (`linear` or `exponential`). |
| `--cov` | `None` | Covariate table containing the columns `eid`, `assessment_age`, and `sex`. |
| `--out` | — | Output prefix. |