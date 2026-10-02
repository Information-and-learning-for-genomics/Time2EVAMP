# Benchmarking pipelines

This folder contains example scripts for reproducing the benchmarking analyses. It includes scripts for running the vampW, CoxPH and DeepSurv methods, and calculating prediction and variable-selection performance metrics.

To construct FDR–TPR curves, a `.npy` file containing the threshold values must be provided (e.g., `fdr_tpr_thresholds.npy`).

## Training vampW model

`vampw_pipeline.py` — Training the vampW model and reporting prediction metrics.

### Example
```bash
srun python3 vampw_pipeline.py \
    --X-train /path/to/X_train \
    --phen-train /path/to/phen_train \
    --X-test /path/to/X_test \
    --phen-test /path/to/phen_test \
    --rho 0.3 \
    --adaptive-rho 0 \
    --rho-wu-it -1 \
    --rho-autotune 0 \
    --alpha 3 \
    --mu 0 \
    --learn-mu 1 \
    --probs 0.7,0.3 \
    --vars 0,0.01 \
    --n-hutchinson-samples 8 \
    --iterations 20 \
    --out /path/to/output
```

### Input parameters

| Parameter | Default | Description |
|---|---:|---|
| `--X-train` | `None` | Training design matrix. |
| `--phen-train` | `None` | Training phenotype file. |
| `--X-test` | `None` | Test design matrix. |
| `--phen-test` | `None` | Test phenotype file. |
| `--rho` | `0.3` | Damping parameter |
| `--adaptive-rho` | `0` | Enable adaptive damping |
| `--rho-wu-it` | `-1` | Warm-up iterations for adaptive damping |
| `--rho-autotune` | `0` | Number of damping autotuning iterations |
| `--alpha` | `3` | Initial Weibull shape parameter |
| `--mu` | `0` | Initial value of intercept $\mu$ |
| `--learn-mu` | `1` | Learn $\mu$ |
| `--probs` | `0.7,0.3` | Prior probabilities for mixture components |
| `--vars` | `0,0.01` | Prior variances for mixture components |
| `--n-hutchinson-samples` | `8` | Number of Hutchinson samples (-1 sets exact trace calculation) |
| `--standardise` | `0` | Standardise the phenotype. |
| `--iterations` | `20` | Number of vampW iterations |
| `--beta-true` | `None` | File containing the true effect sizes $\beta$. |
| `--thresholds` | `None` | File containing thresholds used to generate FDR–TPR curves. |

### Output files

The vampW pipeline generates the following output files:

| File | Description |
|---|---|
| `<out>_roc.csv` | FDR, TPR, and thresholds used to construct FDR–TPR curves. |
| `<out>_pips.csv` | Posterior inclusion probabilities (PIPs) for the markers. |

## Training CoxPH model

`coxph_pipeline.py` — Training the CoxPH model and reporting prediction metrics. Supports no penalization, marginal testing, LASSO, and Elastic Net penalization.

### Example
```bash
python3 coxph_pipeline.py \
    --X-train /path/to/X_train \
    --phen-train /path/to/phen_train \
    --X-test /path/to/X_test \
    --phen-test /path/to/phen_test \
    --model coxph \
    --step-size 0.95 \
    --out /path/to/output
```

### Input parameters

| Parameter | Default | Description |
|---|---:|---|
| `--X-train` | `None` | Training design matrix. |
| `--phen-train` | `None` | Training phenotype file. |
| `--X-test` | `None` | Test design matrix. |
| `--phen-test` | `None` | Test phenotype file. |
| `--model` | `coxph` | CoxPH model to use. Options include `coxph`, `coxph_lasso`, `coxph_elastic_net`, and `coxph_marginal`. |
| `--step-size` | `0.95` | Step size used for fitting the non-penalized CoxPH model (only available for `coxph` model). |
| `--penalizer` | `0.001` | Penalization strength for the CoxPH model (only available for `coxph` model). |
| `--subsample-fraction` | `0.75` | Fraction of samples used for each stability-selection subsample (only available for `coxph_elastic_net` and `coxph_lasso` model). |
| `--n-subsamples` | `100` | Number of subsamples used for stability selection (only available for `coxph_elastic_net` and `coxph_lasso` model). |
| `--stab-sel-thr` | `0.9` | Stability-selection threshold (only available for `coxph_elastic_net` and `coxph_lasso` model). |
| `--l1-ratios` | `0.0,0.1,0.5,0.7,0.9,0.95,0.99,1.0` | Comma-separated list of L1 ratios evaluated during penalization (only available for `coxph_elastic_net` model). |
| `--standardise` | `0` | Standardise the phenotype. |
| `--beta-true` | `None` | File containing the true effect sizes $\beta$. |
| `--thresholds` | `None` | File containing thresholds used to generate FDR–TPR curves. |
| `--n-jobs` | `1` | Number of parallel CPU jobs. |
| `--out` | — | Output prefix. |

### Output files

The CoxPH models generate the following output files:

| File | Description |
|---|---|
| `<out>_roc.csv` | FDR, TPR, and thresholds used to construct FDR–TPR curves. |
| `<out>_pvals.csv` | Marker-level p-values. |

## Training DeepSurv model

`deepsurv_pipeline.py` — Training the DeepSurv model and reporting prediction performance metrics.  
`deepsurv_utils.py` — Supporting functions for the DeepSurv pipeline.

### Example
```bash
srun python3 deepsurv_pipeline.py \
    --X-train /path/to/X_train \ # Training design matrix
    --phen-train /path/to/phen_train \ # Training phenotype file
    --X-test /path/to/X_test \ # Test design matrix
    --phen-test /path/to/phen_test \ # Test phenotype file
    --n-jobs 1 \ # Number of parallel CPU jobs.
    --out /path/to/output \ # Output prefix
```

### Hyperparameters

| Parameter | Values |
|---|---|
| `epochs` | `100` |
| `L2_reg` | `0.01`, `1`, `5` |
| `batch_norm` | `True` |
| `dropout` | `0`, `0.1` |
| `hidden_layers_sizes` | `[256]`, `[512]`, `[1024]` |
| `learning_rate` | `0.001` |
| `lr_decay` | `0.001` |
| `momentum` | `0.9` |
| `standardize` | `False` |