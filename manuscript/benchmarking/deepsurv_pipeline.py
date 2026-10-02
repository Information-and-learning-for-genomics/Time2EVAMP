import numpy as np
import argparse
import pandas as pd
import time
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import os
from deepsurv_utils import *
import itertools
from operator import mul
from functools import reduce
import tempfile

eps = 1e-16
seed = np.random.randint(0, 2**32 - 1)

def phen_std(y):
    zero_pos = (y == 0)
    y[zero_pos] = y[zero_pos] + 1e-4
    logy = np.log(y)
    logy = (logy - np.mean(logy[y > 0])) / np.std(logy[y > 0])
    y = np.exp(logy)
    return y

parser = argparse.ArgumentParser()
parser.add_argument("-X_train", "--X-train", help="Train design matrix", default=None)
parser.add_argument("-phen_train", "--phen-train", help="Train phenotype file", default=None)
parser.add_argument("-X_test", "--X-test", help="Test design matrix", default=None)
parser.add_argument("-phen_test", "--phen-test", help="Test phenotype file", default=None)
parser.add_argument("-standardise", "--standardise", help="Phenotype standardisation", default=False)
parser.add_argument("-n_jobs", "--n-jobs", help="Number of parallel CPU jobs", default=False)
parser.add_argument("-out", "--out", help="Output prefix")
args = parser.parse_args()

X_train_fpath = args.X_train
phen_train_fpath = args.phen_train
X_test_fpath = args.X_test
phen_test_fpath = args.phen_test
standardise = bool(int(args.standardise))
n_jobs = int(args.n_jobs)
out = args.out

print("----- DeepSurv pipeline -----")
print("--X-train", X_train_fpath)
print("--phen-train", phen_train_fpath)
print("--X-test", X_test_fpath)
print("--phen-test", phen_test_fpath)
print("--standardise", standardise)
print("--n_jobs", n_jobs)
print("--out", out)
print("\n", flush=True)

epochs = 100

param_space = {
    "L2_reg": [1e-2, 1, 5],
    "batch_norm": [True],
    "dropout": [0, 0.1],
    "hidden_layers_sizes": [[256],[512],[1024]],
    "learning_rate": [1e-3],
    "lr_decay": [1e-3],
    "momentum": [0.9],
    "standardize": [standardise],
    "epochs": [epochs]
}

X_train = np.load(X_train_fpath)
df_phen_train = pd.read_table(phen_train_fpath, sep="\t")
y_train = np.expand_dims(df_phen_train["Y"].to_numpy().ravel(), axis=-1)
if standardise:
    y_train = phen_std(y_train)
event_train = df_phen_train["EVENT"].to_numpy().ravel()
df_train = pd.DataFrame(X_train)
df_train["time"] = y_train
df_train["event"] = event_train
N_train, M = X_train.shape
event_mask_train = event_train.astype(bool)

# Load the data
X_test = np.load(X_test_fpath)
df_phen_test = pd.read_table(phen_test_fpath, sep="\t")
y_test = np.expand_dims(df_phen_test["Y"].to_numpy().ravel(), axis=-1)
if standardise:
    y_test = phen_std(y_test)
event_test = df_phen_test["EVENT"].to_numpy().ravel()
df_test = pd.DataFrame(X_test)
df_test["time"] = y_test
df_test["event"] = event_test
N_test, _ = X_test.shape
event_mask_test = event_test.astype(bool)

print("Data loaded", flush=True)

X_u_train = X_train[event_train==1]
X_c_train = X_train[event_train==0]
y_u_train = y_train[event_train==1]
Y_c_train = y_train[event_train==0]

X_u_test = X_test[event_test==1]
X_c_test = X_test[event_test==0]
y_u_test = y_test[event_test==1]
Y_c_test = y_test[event_test==0]

print(
    f"Using provided test set: uncens={len(y_u_test)}, cens={len(Y_c_test)}",
    flush=True,
)

X_u_subtrain, X_u_val, y_u_subtrain, y_u_val = train_test_split(
    X_u_train,
    y_u_train,
    test_size=0.1,
    random_state=42
)

X_c_subtrain, X_c_val, y_c_subtrain, y_c_val = train_test_split(
    X_c_train,
    Y_c_train,
    test_size=0.1,
    random_state=42
)
print("Parameter space:", param_space, flush=True)
values = list(param_space.values())
keys = list(param_space.keys())
num_combinations = reduce(mul, (len(v) for v in values), 1)
combos = list(itertools.product(*values))

print("----- Performing Grid Search for Hyperparameters -----", flush=True)
ts = time.time()
def run(i):
    thread_dir = tempfile.mkdtemp(prefix=f"{out}_theano_process_{i}_")
    os.environ["THEANO_FLAGS"] = f"base_compiledir={thread_dir},optimizer_excluding=inplace"
        
    print(f"THREAD {i}: Starting...", flush=True)
    combo = next(itertools.islice(itertools.product(*values), i, None))
    params = dict(zip(keys, combo))
    print(f"THREAD {i}: params: ",params, flush=True)
    n_epochs = params.pop("epochs", None)
    try:
        _, model, correlation_metrics, run_dir = fit_deepsurv_with_progress(
            X_u_subtrain,
            X_c_subtrain,
            y_u_subtrain,
            y_c_subtrain,
            X_u_test=X_u_val,
            X_c_test=X_c_val,
            y_u_test=y_u_val,
            Y_c_test=y_c_val,
            hyperparams=params,
            n_epochs=n_epochs,
            save_root=out,
            run_prefix="deepsurv_run",
            verbose=False,
            calc_corr=True
        )
        params['epochs'] = n_epochs
        print(f"THREAD {i}: Validation CI: {correlation_metrics['test']['ci']}", flush=True)
            
        return correlation_metrics["test"]["ci"], params
    
    except Exception as ex:
        print(f"THREAD {i}: Error in DeepSurv fit: {ex}", flush=True)
        
    return 0, params
        
results = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(run)(i) for i in range(num_combinations)
)
ci_list = [res[0] for res in results]
param_list = [res[1] for res in results]
best_i = np.argmax(ci_list)
best_params = param_list[best_i]
print("Best params:", best_params, flush=True)

n_epochs = best_params.pop("epochs", epochs)

_, model, correlation_metrics, run_dir = fit_deepsurv_with_progress(
    X_u_train,
    X_c_train,
    y_u_train,
    Y_c_train,
    X_u_test=X_u_test,
    X_c_test=X_c_test,
    y_u_test=y_u_test,
    Y_c_test=Y_c_test,
    hyperparams=best_params,
    n_epochs=n_epochs,
    save_root=out,
    run_prefix="deepsurv_run",
    calc_corr=True
)
t_infer = time.time() - ts
print("Inference time: %0.4f s"%(t_infer), flush=True)

test_data = build_deepsurv_data(X_u_test, X_c_test, y_u_test, Y_c_test)
train_data = build_deepsurv_data(X_u_train, X_c_train, y_u_train, Y_c_train)

exp_te, e_te, t_te, e_tr, t_tr, surv_matrix, times, event_time_test_valid = deepsurv_predict_time(
    network=model,
    train_data=train_data,
    test_data=test_data,
    penalizer=0,
    l1_ratio=0.001,
    step_size=0.95,
    standardise=standardise
)
mask = (e_te == 1)

c_index = concordance_index(
    event_times=t_te,
    predicted_scores=exp_te,
    event_observed=e_te
)
mse = mean_squared_error(t_te[mask], exp_te[mask])
print(f"Test scores: RMSE = {np.sqrt(mse)}, C-index = {c_index}")