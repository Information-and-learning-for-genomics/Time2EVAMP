import numpy as np
import argparse
import pandas as pd
import time
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
import sympy
from sklearn.metrics import mean_squared_error
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

emc = float( sympy.S.EulerGamma.n(10) )
eps = 1e-16
seed = np.random.randint(0, 2**32 - 1)

def run_thread_coxph(i, penalizer_value, subsample_fraction, df):
    print(f"THREAD {i}: Starting...", flush=True)
    M = df.shape[1] - 2
    df_shuffled = df.sample(frac=subsample_fraction, random_state=i).reset_index(drop=True)
    if df_shuffled['event'].sum() == 0:
        print(f"THREAD {i}: sum(event) = 0", flush=True)
        return np.zeros(M, dtype=bool)
    y = Surv.from_arrays(event=df_shuffled['event'], time=df_shuffled['time'])
    X = df_shuffled.iloc[:, :-2]
    cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[penalizer_value])
    try:
        cox_lasso.fit(X, y)
    except ArithmeticError as e:
        print(f"THREAD {i}: Arithmetic error", e, flush=True)
        return np.zeros(M, dtype=bool)
    selection = (cox_lasso.coef_ != 0).ravel()
    print(f"THREAD {i}: sum(selection) = {np.sum(selection)}", flush=True)
    return selection

def find_best_penalizer_value(df, cpus):
    y = Surv.from_arrays(
            event=df['event'], 
            time=df['time']
        )
    X = df.iloc[:, :-2]
    
    penalizers = np.logspace(-4, 0, 16)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        estimator=CoxnetSurvivalAnalysis(l1_ratio=1),
        param_grid={"alphas": [[float(a)] for a in penalizers]},
        cv=cv,
        error_score=0.5,
        n_jobs=cpus
    ).fit(X, y)

    cv_results = pd.DataFrame(gcv.cv_results_)
    #print(cv_results[["param_alphas", "mean_test_score"]])
    best_penalizer = cv_results.loc[cv_results["mean_test_score"].idxmax()]["param_alphas"][0]
    return best_penalizer


def find_best_penalizer_value_elastic_net(l1_ratios, alpha_grid, X, time_arr, event_arr, cpus=-1, n_folds=5, cv_tol=1e-5, cv_max_iter=50000):
    """Pick (alpha, l1_ratio) by CV, one independent fit per grid point."""
    time_arr = np.asarray(time_arr).ravel().astype(float)
    event_arr = np.asarray(event_arr).astype(bool)
    y = Surv.from_arrays(event=event_arr, time=time_arr)

    l1_ratios = [float(v) for v in l1_ratios if v > 0]
    if not l1_ratios:
        raise ValueError("At least one l1_ratio must be greater than zero.")

    alpha_grid = np.sort(np.asarray(alpha_grid, dtype=float))[::-1]

    cv = list(KFold(n_splits=n_folds, shuffle=True, random_state=0).split(X))

    print(f"CV: {len(l1_ratios)} l1_ratios x {len(alpha_grid)} alphas x "
          f"{n_folds} folds = {len(l1_ratios)*len(alpha_grid)*n_folds} "
          f"independent fits", flush=True)
    print(f"alpha grid: {alpha_grid}", flush=True)

    def fit_one(li, aj, l1_ratio, alpha, train_idx, test_idx):
        # A single alpha -- no path, so nothing can be truncated.
        model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=[alpha],
            tol=cv_tol,
            max_iter=cv_max_iter,
            fit_baseline_model=False,
        )
        try:
            model.fit(X[train_idx], y[train_idx])
        except Exception:
            return li, aj, 0.5, False

        beta = np.asarray(model.coef_).ravel()
        if not np.any(beta):
            # All-zero: carries no information and must not win the argmax.
            return li, aj, 0.5, False

        try:
            score = concordance_index_censored(
                event_arr[test_idx], time_arr[test_idx], X[test_idx] @ beta
            )[0]
        except (ValueError, ZeroDivisionError):
            return li, aj, 0.5, False

        return li, aj, float(score), True

    jobs = [
        (li, aj, l1, a, tr, te)
        for li, l1 in enumerate(l1_ratios)
        for aj, a in enumerate(alpha_grid)
        for tr, te in cv
    ]

    with threadpool_limits(limits=1):
        out = Parallel(n_jobs=cpus, backend="loky", verbose=5,
                       max_nbytes="100M", mmap_mode="r")(
            delayed(fit_one)(*j) for j in jobs
        )

    n_l1, n_a = len(l1_ratios), len(alpha_grid)
    score_sum = np.zeros((n_l1, n_a))
    usable = np.zeros((n_l1, n_a), dtype=bool)
    for li, aj, score, ok in out:
        score_sum[li, aj] += score
        usable[li, aj] |= ok

    mean_scores = score_sum / n_folds

    print(f"usable (non-degenerate) alphas per l1_ratio: "
          f"{usable.sum(axis=1).tolist()} of {n_a}", flush=True)

    if not usable.any():
        raise ValueError(
            "Every (l1_ratio, alpha) in the grid gave all-zero coefficients. "
            "The grid is too heavily penalised for this trait."
        )

    masked = np.where(usable, mean_scores, -np.inf)
    li, aj = np.unravel_index(np.argmax(masked), masked.shape)

    summary = pd.DataFrame([
        {"l1_ratio": l1_ratios[i], "alpha": alpha_grid[j],
         "mean_test_score": mean_scores[i, j], "usable": usable[i, j]}
        for i in range(n_l1) for j in range(n_a)
    ]).sort_values("mean_test_score", ascending=False)

    print("\n--- Cross-Validation Results Summary ---")
    print(summary[summary["usable"]].head(10).to_string(index=False))
    print("----------------------------------------")

    return float(alpha_grid[aj]), float(l1_ratios[li])

def run_thread_coxph_elastic_net(i, df, subsample_fraction, best_l1_ratio, penalizer_value):
    print(f"THREAD {i}: Starting...", flush=True)

    df_shuffled = df.sample(frac=subsample_fraction,
                            random_state=i).reset_index(drop=True)

    if df_shuffled["event"].sum() == 0:
        return np.zeros(M, dtype=bool)

    y = Surv.from_arrays(event=df_shuffled["event"], time=df_shuffled["time"])

    model = CoxnetSurvivalAnalysis(l1_ratio=best_l1_ratio,
                                   alphas=[penalizer_value])
    try:
        model.fit(df_shuffled.iloc[:, :-2], y)
    except ArithmeticError as e:
        print(f"THREAD {i}: Arithmetic error {e}", flush=True)
        return np.zeros(M, dtype=bool)

    return (model.coef_ != 0).ravel()

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
parser.add_argument("-model", "--model", help="coxph,coxph_lasso,coxph_marginal,coxph_elastic_net", default="coxph")
parser.add_argument("-beta_true", "--beta-true", help="Beta true file", default=None)
parser.add_argument("-step_size", "--step-size", help="Step size", default=0.95)
parser.add_argument("-penalizer", "--penalizer", help="Penalizer for coxph", default=0.001)
parser.add_argument("-subsample_fraction", "--subsample-fraction", help="Subsample fraction for stability selection", default=0.75)
parser.add_argument("-n_subsamples", "--n-subsamples", help="Number of subsamples for stability selection", default=100)
parser.add_argument("-stab_sel_thr", "--stab-sel-thr", help="Stability selection threshold", default=0.9)
parser.add_argument("-l1_ratios", "--l1-ratios", help="Comma-separated list of L1 ratios to search over.", default="0.0,0.1,0.5,0.7,0.9,0.95,0.99,1.0")
parser.add_argument("-standardise", "--standardise", help="Phenotype standardisation", default=False)
parser.add_argument("-thresholds", "--thresholds", help="File containing thresholds for FDR-TPR curves")
parser.add_argument("-n_jobs", "--n-jobs", help="Number of parallel CPU jobs", default=1)
parser.add_argument("-out", "--out", help="Output prefix")
args = parser.parse_args()

X_train_fpath = args.X_train
phen_train_fpath = args.phen_train
X_test_fpath = args.X_test
phen_test_fpath = args.phen_test
beta_true_fpath = args.beta_true
model = args.model
step_size = float(args.step_size)
penalizer = float(args.penalizer)
subsample_fraction = float(args.subsample_fraction)
n_subsamples = int(args.n_subsamples)
stab_sel_thr = float(args.stab_sel_thr)
l1_ratios_list = [float(x.strip()) for x in args.l1_ratios.split(",")]
standardise = bool(int(args.standardise))
thrs_fpath = args.thresholds
n_jobs = int(args.n_jobs)
out = args.out

print("----- CoxPH pipeline.py -----")
print("--X-train", X_train_fpath)
print("--phen-train", phen_train_fpath)
print("--X-test", X_test_fpath)
print("--phen-test", phen_test_fpath)
print("--beta-true", beta_true_fpath)
print("--model", model)
print("--penalizer", penalizer)
print("--step-size", step_size)
print("--standardise", standardise)
print("--thresholds", thrs_fpath)
print("--out", out)
print("\n", flush=True)

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

if model == "coxph":
    ts = time.time()
    fit_options = {"step_size": step_size, "precision": 1e-07, "max_steps": 500}
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_train, duration_col='time', event_col='event', fit_options=fit_options, show_progress=True)
    t_infer = time.time() - ts
    print("Inference time: %0.4f s"%(t_infer), flush=True)

    pvals = cph.summary["p"]
    beta = cph.summary["coef"]

    pred_hazard = cph.predict_partial_hazard(df_test)
    y_hat = cph.predict_expectation(X_test)

    if not standardise:
        t_min = cph.baseline_survival_.index.min()
        y_hat = y_hat + t_min

elif model == "coxph_lasso":

    penalizer_value = find_best_penalizer_value(df_train, n_jobs)
    print(f"Chosen penalizer value: {penalizer_value}", flush=True)

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(run_thread_coxph)(i, penalizer_value, subsample_fraction, df_train) for i in range(n_subsamples)
    )
    selection_counts = np.sum(results, axis=0)
    probs = selection_counts / n_subsamples
    stable_features = (probs >= stab_sel_thr)
    print("shape(stable_features) =", np.shape(stable_features), flush=True)
    print(f"Number of selected stable features = {np.sum(stable_features)}", flush=True)

    cph = CoxPHFitter()
    cph.fit(df_train.iloc[:, np.append(stable_features, [True, True])], duration_col='time', event_col='event', show_progress=True)

    beta = np.zeros(M)
    pvals = np.ones(M)
    stable_features_mask = np.ravel(stable_features)
    beta[stable_features_mask] = cph.summary["coef"]
    pvals[stable_features_mask] = cph.summary["p"]

    pred_hazard = cph.predict_partial_hazard(df_test.iloc[:, np.append(stable_features_mask, [False, False])])
    y_hat = cph.predict_expectation(df_test.iloc[:, np.append(stable_features_mask, [False, False])])

    if not standardise:
        t_min = cph.baseline_survival_.index.min()
        y_hat = y_hat + t_min

elif model == "coxph_elastic_net":
    ts = time.time()

    alpha_grid = np.logspace(-4, 0, 16)

    penalizer_value, best_l1_ratio = find_best_penalizer_value_elastic_net(
        l1_ratios_list,
        alpha_grid,
        X_train,
        y_train,
        event_train,
        cpus=n_jobs
    )   

    print(
        f"Optimal parameters found --> L1 Ratio: {best_l1_ratio}, "
        f"Penalizer (Alpha): {penalizer_value}",
        flush=True
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky"
    )(
        delayed(run_thread_coxph_elastic_net)(i, df_train, subsample_fraction, best_l1_ratio, penalizer_value) for i in range(n_subsamples)
    )

    selection_counts = np.sum(results, axis=0)
    probs = selection_counts / n_subsamples
    stable_features = (probs >= stab_sel_thr)

    print(
        "shape(stable_features) =",
        np.shape(stable_features),
        flush=True
    )

    print(
        f"Number of selected stable features = "
        f"{np.sum(stable_features)}",
        flush=True
    )

    cph = CoxPHFitter()

    cph.fit(
    df_train.iloc[:, np.append(stable_features, [True, True])],
        duration_col="time",
        event_col="event",
        show_progress=True
    )

    t_infer = time.time() - ts

    print(
        "Inference time: %0.4f s" % t_infer,
        flush=True
    )

    beta = np.zeros(M)
    pvals = np.ones(M)
    stable_features_mask = np.ravel(stable_features)

    beta[stable_features_mask] = cph.summary["coef"]
    pvals[stable_features_mask] = cph.summary["p"]

    pred_hazard = cph.predict_partial_hazard(df_test.iloc[:, np.append(stable_features, [False, False])])
    y_hat = cph.predict_expectation(df_test.iloc[:, np.append(stable_features, [False, False])])

    if not standardise:
        t_min = cph.baseline_survival_.index.min()
        y_hat = y_hat + t_min

elif model == "coxph_marginal":
    print("Running marginal testing!")
    ts = time.time()
    fit_options = {"step_size": step_size, "precision": 1e-07, "max_steps": 500}
    pvals = []
    for j in range(M):
        df_j = df_train.loc[:,[j, "event", "time"]]
        cph = CoxPHFitter(penalizer=0)
        cph.fit(df_j, duration_col='time', event_col='event', fit_options=fit_options)
        pvals.append(cph.summary["p"].values[0])
    t_infer = time.time() - ts
    print("Inference time: %0.4f s"%(t_infer), flush=True)

    y_hat = np.zeros(N_test)
    pred_hazard = np.zeros(N_test)

pvals_out_fpath = f"{out}_pval.tsv"
df_pval = pd.DataFrame({"P": pvals})
df_pval.to_csv(pvals_out_fpath, index=False)
print(f"P-values stored in: {pvals_out_fpath}")

c_index = concordance_index(
        event_times=df_test["time"],
        predicted_scores=-pred_hazard,   # negative because higher hazard → worse survival
        event_observed=df_test["event"]
    )
mse = mean_squared_error(y_test[event_mask_test], y_hat[event_mask_test])
print(f"Test scores: RMSE = {np.sqrt(mse)}, C-index = {c_index}")

if beta_true_fpath is not None:

    beta_true = np.loadtxt(beta_true_fpath, dtype=float)
    true_mask = (np.abs(beta_true) > 0) * 1

    thrs = np.load(thrs_fpath)

    fdr = []
    tpr = []  
    for thr in thrs:
        est = (pvals <= thr) * 1 
        tn, fp, fn, tp = confusion_matrix(true_mask, est).ravel()
        fdr.append(fp / (fp + tp + eps))
        tpr.append(tp / (tp + fn + eps))

    df = pd.DataFrame({"FDR": fdr, "TPR": tpr, "THR": thrs})
    df.to_csv(f"{out}_roc.csv", index=False, sep="\t")