# Time-To-Event phenotype simulation
This folder contains example scripts to replicate the simulations of the Time-To-Event (TTE) data. The following scripts simulate TTE phenotypes under different data-generating scenarios:

- `sim_phen_tte.py` — Simulation of TTE phenotypes with linear effects.
- `sim_phen_tte_nonlin.py` — Simulation of TTE phenotypes including non-linear effects.
- `sim_phen_tte_interactions.py` — Simulation of TTE phenotypes including protein-protein interaction effects.
- `sim_phen_tte_competing_risks.py` — Simulation of TTE phenotypes with competing risks.

## Example

```bash
python3 sim_phen_tte.py \
    --X /path/to/design \ # Optional. Synthetic design is generated if not specified
    --h2 0.8 \                         
    --lam 0.1 \                       
    --C 0.9 \                          
    --phen-distribution Weibull \    
    --kappa 1 \                        
    --prior spike_slab \               
    --train-frac 0.9 \                 
    --out /output/prefix         
```

## Input parameters

| Parameter | Default | Description |
|---|---:|---|
| `--X` | `None` | Design matrix in `npy` format. If not specified, a synthetic design matrix is generated (only in `sim_phen_tte.py`). |
| `--h2` | `0.8` | Target heritability (signal-to-noise ratio). |
| `--p-nonlin` | `0.5` | Proportion of the non-linear component (only available in `sim_phen_tte_nonlin.py`). |
| `--tau` | `0.2` | Proportion of the interaction component (only available in `sim_phen_tte_interactions.py`). |
| `--lam` | `0.1` | Proportion of the non-sparse signal region. |
| `--C` | `0.9` | Target censoring rate. |
| `--phen-distribution` | `Weibull` | Phenotype distribution (`Weibull` or `Gamma`). |
| `--kappa` | `1` | Gamma distribution parametrization. |
| `--prior` | `spike_slab` | Prior distribution for signal effects. |
| `--train-frac` | `0.9` | Fraction of samples used for training. |
| `--rho` | `0.5` | Target correlation between effects for the two processes (only available in the `sim_phen_tte_competing_risks.py`). |
| `--interactions` | `None` | Protein-protein interaction file from STRING database (only available in `sim_phen_tte_interactions.py`). |
| `--assay` | `None` | Assay file corresponding to the proteins in the interaction file (only available in `sim_phen_tte_interactions.py`). |
| `--out` | — | Output prefix. |

## Simulation outputs

The simulation scripts generate the following output files:

| File | Description |
|---|---|
| `<out>_X_train.npy` | Training design matrix (only in `sim_phen_tte.py`).  |
| `<out>_X_test.npy` | Test design matrix (only in `sim_phen_tte.py`). |
| `<out>_train.phen` | Training phenotype data, stored as a tab-separated file. |
| `<out>_test.phen` | Test phenotype data, stored as a tab-separated file. |
| `<out>_beta_true.txt` | True simulated effect sizes (`beta`). |