from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from sksurv.util import Surv
import numpy as np
import pandas as pd
import csv

# Configure THEANO_FLAGS before importing any Theano-dependent libraries (theano/lasagne/deepsurv).
_theano_flags_raw = os.environ.get("THEANO_FLAGS", "")
_theano_flags = [f.strip() for f in _theano_flags_raw.split(",") if f.strip()]
_has_device = any(f.startswith("device=") for f in _theano_flags)
_has_floatx = any(f.startswith("floatX=") for f in _theano_flags)

if (not _has_device) or (not _has_floatx):
    # Only probe for a GPU if device isn't explicitly set already.
    if not _has_device:
        try:
            has_gpu = (
                subprocess.run(
                    ["nvidia-smi"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).returncode
                == 0
            )
        except Exception:
            has_gpu = os.path.exists("/proc/driver/nvidia/version")

        device_flag = "device=cuda" if has_gpu else "device=cpu"
        print(f"Using {'GPU' if has_gpu else 'CPU'}", flush=True)
        _theano_flags = [f for f in _theano_flags if not f.startswith("device=")]
        _theano_flags.append(device_flag)

    # Default to float32 unless explicitly specified.
    if not _has_floatx:
        _theano_flags = [f for f in _theano_flags if not f.startswith("floatX=")]
        _theano_flags.append("floatX=float32")

    os.environ["THEANO_FLAGS"] = ",".join(_theano_flags)
    print(f"Theano flags: {os.environ['THEANO_FLAGS']}", flush=True)

# After setting env flags, import Theano to confirm runtime config.
from theano import config as th_config

print(
    f"Theano runtime device: {th_config.device}, floatX={th_config.floatX}",
    flush=True,
)

import lasagne
from deepsurv import DeepSurv
from deepsurv.deepsurv_logger import TensorboardLogger
from lifelines import CoxPHFitter
from scipy.stats import pearsonr, spearmanr


def cosine_similarity(y, y_hat):
    return (np.dot(y_hat.transpose(), y) / np.linalg.norm(y) / np.linalg.norm(y_hat)).flatten()[0]

def fmt(m):
    return (
        "None"
        if m is None
        else f"Concordance Index={m['ci']:.3f} Pearson={m['pearson_uncens']:.3f} Spearman={m['spearman_uncens']:.3f} "
    )


def build_deepsurv_data(X_u, X_c, y_u, Y_c):
    X_u = np.asarray(X_u, dtype=np.float32)
    X_c = np.asarray(X_c, dtype=np.float32)
    y_u = np.asarray(y_u, dtype=np.float32).reshape(-1)
    Y_c = np.asarray(Y_c, dtype=np.float32).reshape(-1)

    if X_u.shape[0] != y_u.shape[0]:
        raise ValueError(
            f"Uncensored features/time mismatch: X_u={X_u.shape}, y_u={y_u.shape}"
        )
    if X_c.shape[0] != Y_c.shape[0]:
        raise ValueError(
            f"Censored features/time mismatch: X_c={X_c.shape}, Y_c={Y_c.shape}"
        )

    x_all = np.vstack([X_u, X_c]) if X_c.size else X_u.copy()
    t_all = np.concatenate([y_u, Y_c]).astype(np.float32, copy=False)
    e_all = np.concatenate(
        [
            np.ones(y_u.shape[0], dtype=np.int32),
            np.zeros(Y_c.shape[0], dtype=np.int32),
        ]
    )

    return {
        "x": x_all.astype(np.float32),
        "t": t_all.astype(np.float32),
        "e": e_all.astype(np.int32),
    }

def phen_std(y):
    print("Phenotype standardization...", flush=True)
    zero_pos = (y == 0)
    y[zero_pos] = y[zero_pos] + 1e-4
    logy = np.log(y)
    mu = np.mean(logy[y > 0])
    sigma = np.std(logy[y > 0])
    logy = (logy - np.mean(logy[y > 0])) / np.std(logy[y > 0])
    logy = np.nan_to_num(logy, nan=0.0)
    y = np.exp(logy)
    return y, mu, sigma

def get_split(
    X,
    iids,
    colnames,
    phen_fpath,
    standardise=False
):
    with open(phen_fpath, "r") as f:
        sample = f.read(2048)
        has_header = csv.Sniffer().has_header(sample)

        if has_header:
            df_split = pd.read_table(phen_fpath, sep="\s+")
            df_split.columns = df_split.columns.str.upper()
        else:
            df_split = pd.read_table(phen_fpath, header=None, names=["IID", "FID", "Y", "EVENT"], sep="\s+")
            print("No header")
    
    df_split["IID"] = df_split["IID"].astype(str)
    df_split["FID"] = df_split["FID"].astype(str)

    if standardise:
        df_split["Y"], mu, sigma = phen_std(df_split["Y"].values)
    else:
        _, mu, sigma = phen_std(df_split["Y"].values)

    df_split["Y"] = pd.to_numeric(df_split["Y"], errors="coerce")
    df_split["EVENT"] = pd.to_numeric(df_split["EVENT"], errors="coerce").astype(
        "Int64"
    )
    if df_split["Y"].isna().any():
        raise ValueError("Found non-numeric values in Y.")
    if (~df_split["EVENT"].isin([0, 1])).any():
        bad = df_split.loc[~df_split["EVENT"].isin([0, 1]), ["IID", "EVENT"]]
        raise ValueError(f"EVENT must be 0/1; offending rows:\n{bad}")
    # --- Align X to df_train by IID ---
    # Ensure shape, and that colnames length matches features
    N, M = X.shape
    assert len(colnames) == M, f"colnames ({len(colnames)}) != X columns ({M})"
    # Sanity check uniqueness of iids (row ids of X)
    if len(set(iids)) != len(iids):
        # Not fatal, but warn: duplicate IIDs in X row index can cause ambiguous alignment
        dup = pd.Series(iids).value_counts()
        print(f"[warn] Duplicate IIDs in indlist (X rows): {dup[dup > 1].to_dict()}")
    # Build mapping from IID -> row index in X
    iid_to_idx = {iid: idx for idx, iid in enumerate(iids)}
    # Keep only phenotypes whose IID exists in X
    mask_in_X = df_split["IID"].isin(iid_to_idx)
    df_split_aligned = df_split.loc[mask_in_X].copy()
    # Optional: warn if anything was dropped or if duplicates exist
    dropped = (~mask_in_X).sum()
    if dropped:
        print(f"[warn] Dropped {dropped} phenotype rows with IIDs not found in X.")
    dup_iids = df_split_aligned["IID"][df_split_aligned["IID"].duplicated()].unique()
    if len(dup_iids) > 0:
        print(
            f"[warn] Duplicate IIDs in phenotypes: {list(dup_iids)}; "
            "they will create repeated rows in X_train."
        )
    if df_split_aligned.empty:
        raise ValueError("No overlapping IIDs between phenotype and X.")
    # Reorder X to match df_train_aligned order
    row_indices = np.array(
        [iid_to_idx[i] for i in df_split_aligned["IID"].values], dtype=int
    )
    X_split = X[row_indices, :]
    # Subset y and event consistently
    y_split = df_split_aligned["Y"].to_numpy(dtype=float)
    e_split = df_split_aligned["EVENT"].to_numpy(dtype=int)
    # Final sizes
    N_split = len(y_split)
    assert (
        X_split.shape[0] == N_split == len(e_split)
    ), "Length mismatch after alignment"
    # Split the data into uncensored and censored
    # X_u_train ~ predictors for individuals with uncensored outcome
    # X_c_train ~ predictors for individuals with censored outcome
    # y_u_train ~ outcome time for uncensored individuals
    # Y_c_train ~ study dropout time for censored individuals
    # e_train ~ event indicator; 1 for uncensored, 0 for censored
    e_split = df_split_aligned["EVENT"].astype(int).values
    X_u_split = X_split[e_split == 1, :]
    X_c_split = X_split[e_split == 0, :]
    y_u_split = df_split_aligned.loc[df_split_aligned["EVENT"] == 1, "Y"].values
    Y_c_split = df_split_aligned.loc[df_split_aligned["EVENT"] == 0, "Y"].values
    return {
        "X_u": X_u_split,
        "y_u": y_u_split,
        "X_c": X_c_split,
        "y_c": Y_c_split,
        "e": e_split,
        "X_split": X_split,
        "y_split": y_split
    }, mu, sigma


def prep_logrisk_e_t(network, data):
    x_prep, e_prep, t_prep = network.prepare_data(data)
    log_risk = np.asarray(network.predict_risk(x_prep)).ravel()
    return log_risk, np.asarray(e_prep), np.asarray(t_prep)


def corr_on_uncensored(exp, t, e):
    exp = np.asarray(exp).ravel()
    t = np.asarray(t).astype(float).ravel()
    e = np.asarray(e).astype(int).ravel()

    mask = (e == 1) & np.isfinite(exp) & np.isfinite(t)
    if mask.sum() >= 2:
        r_s = spearmanr(exp[mask], t[mask], nan_policy="omit").correlation
        r_p = pearsonr(exp[mask], t[mask])[0]
        cs = cosine_similarity(t[mask], exp[mask])
        return r_s, r_p, cs
    return np.nan, np.nan, np.nan


def expectation_corr_deepsurv(
    network,
    train_data,
    test_data=None,
    run_dir=None,
    print_every=True,
    penalizer=0.0,
    l1_ratio=0.0,
    step_size=0.95
):
    """
    Calibrate DeepSurv log-risk with lifelines' CoxPH (Breslow baseline),
    compute E[T|x] via predict_expectation, then Pearson/Spearman/Cosine similarity vs. observed times
    on uncensored subjects.

    Returns: (r_s_tr, r_p_tr, cs_tr, r_s_te, r_p_te, cs_te, cph)  # test corr are np.nan if no test_data
    """
    if print_every:
        print(
            "Fitting 1D CoxPH on DeepSurv log-risk to learn baseline (Breslow)...",
            flush=True,
        )

    # Defaults if something goes wrong / no test data
    r_s_tr = np.nan
    r_p_tr = np.nan
    cs_tr  = np.nan
    r_s_te = np.nan
    r_p_te = np.nan
    cs_te  = np.nan
    cph = None

    # --- TRAIN ---
    log_risk_tr, e_tr, t_tr = prep_logrisk_e_t(network, train_data)

    n_events = int(e_tr.sum())
    if n_events == 0:
        print(
            "No events in TRAIN. Cannot estimate a Cox baseline; correlations set to NaN.",
            flush=True,
        )
        return r_s_tr, r_p_tr, r_s_te, r_p_te, cph

    df_tr = pd.DataFrame(
        {
            "T": t_tr.astype(float),
            "E": e_tr.astype(int),
            "z": log_risk_tr.astype(float),
        }
    )
    fit_options = {"step_size": step_size, "precision": 1e-07, "max_steps": 500}
    cph = CoxPHFitter(
        baseline_estimation_method="breslow",
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )
    cph.fit(df_tr, duration_col="T", event_col="E", formula="z", fit_options=fit_options)

    # Save baseline for reproducibility
    if run_dir is not None:
        bh = cph.baseline_cumulative_hazard_.copy()
        baseline_times = bh.index.values.astype(float)
        baseline_cumhaz = bh.values.ravel().astype(float)
        np.save(os.path.join(run_dir, "baseline_times.npy"), baseline_times)
        np.save(os.path.join(run_dir, "baseline_cumhaz.npy"), baseline_cumhaz)

    exp_tr = cph.predict_expectation(df_tr[["z"]]).to_numpy().ravel()
    r_s_tr, r_p_tr, cs_tr = corr_on_uncensored(exp_tr, df_tr["T"].values, df_tr["E"].values)

    # --- TEST ---
    if test_data is not None:
        log_risk_te, e_te, t_te = prep_logrisk_e_t(network, test_data)
        df_te = pd.DataFrame({"z": log_risk_te.astype(float)})
        exp_te = cph.predict_expectation(df_te).to_numpy().ravel()
        r_s_te, r_p_te, cs_te = corr_on_uncensored(exp_te, t_te, e_te)

    return r_s_tr, r_p_tr, cs_tr, r_s_te, r_p_te, cs_te, cph

def deepsurv_predict_time(
    network,
    train_data,
    test_data=None,
    penalizer=0.0,
    l1_ratio=0.0,
    step_size=0.95,
    standardise=False
):

    # --- TRAIN ---
    log_risk_tr, e_tr, t_tr = prep_logrisk_e_t(network, train_data)

    n_events = int(e_tr.sum())

    df_tr = pd.DataFrame(
        {
            "T": t_tr.astype(float),
            "E": e_tr.astype(int),
            "z": log_risk_tr.astype(float),
        }
    )
    fit_options = {"step_size": step_size, "precision": 1e-07, "max_steps": 500}
    cph = CoxPHFitter(
        baseline_estimation_method="breslow",
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )
    cph.fit(df_tr, duration_col="T", event_col="E", formula="z", fit_options=fit_options)
    # Get coefficients
    print("Coefficients (beta):")
    print(cph.params_)  

    log_risk_te, e_te, t_te = prep_logrisk_e_t(network, test_data)
    df_te = pd.DataFrame({"z": log_risk_te.astype(float)})
    exp_te = cph.predict_expectation(df_te).to_numpy().ravel()

    if not standardise:
        t_min = cph.baseline_survival_.index.min()
        exp_te = exp_te + t_min

    t_max_train = max(t_tr) # training set max follow-up
    t_max_safe = t_max_train - 1e-5
    valid_mask = t_te < t_max_safe
    n_dropped = len(t_te) - valid_mask.sum()

    if n_dropped > 0:
        print(f"WARNING: Dropping {n_dropped} test samples because their time exceeds the max training time ({t_max_train:.4f}).")

    df = pd.DataFrame({"event": e_te, "time": t_te, "z": log_risk_te.astype(float)})
    X_test_valid = pd.DataFrame(log_risk_te.astype(float)[valid_mask], columns=["z"])

    df_valid = df[valid_mask].copy()
    event_time_test_valid = Surv.from_dataframe("event", "time", df_valid)

    t_min = df_valid["time"].min()            # test set min follow-up
    t_max_test_valid = df_valid["time"].max()       # test set max follow-up
    t_max = min(t_max_test_valid, t_max_safe) - 1e-5

    times = np.linspace(t_min, t_max, 20)
    print(f"IBS Time Grid: {times[0]:.4f} to {times[-1]:.4f} (Max Train: {t_max_train:.4f})", flush=True)

    surv_df = cph.predict_survival_function(X_test_valid, times=times)
    surv_matrix = surv_df.T.values

    return exp_te, e_te, t_te, e_tr, t_tr, surv_matrix, times, event_time_test_valid

def _metrics_from_dataset(network, data):
    # Standardize and sort consistently with training
    x, e, t = network.prepare_data(data)
    # Predict risks on the same standardized/sorted x
    risk = network.predict_risk(x)
    risk = np.asarray(risk).ravel()
    t = np.asarray(t).ravel().astype(np.float32)
    e = np.asarray(e).ravel().astype(np.int32)
    ci = network.get_concordance_index(x=x, t=t, e=e)
    return {"ci": ci}

def fit_deepsurv_with_progress(
    X_u_train,
    X_c_train,
    y_u_train,
    Y_c_train,
    X_u_test=None,
    X_c_test=None,
    y_u_test=None,
    Y_c_test=None,
    hyperparams=None,
    n_epochs=500,
    patience=2000,
    improvement_threshold=0.99999,
    patience_increase=2,
    save_root=".",
    run_prefix="deepsurv_run",
    verbose=True,
    comment="",
    update_fn=lasagne.updates.adam,
    calc_corr=False
):
    """
    Train DeepSurv with built-in LR decay/momentum/patience (single call).
    Computes train/valid/test metrics, returns predictions and saves outputs.
    Returns (risk_test, network, metrics_dict, run_dir).
    """
    # Coerce arrays and shapes
    train_data = build_deepsurv_data(X_u_train, X_c_train, y_u_train, Y_c_train)
    test_data = build_deepsurv_data(X_u_test, X_c_test, y_u_test, Y_c_test)
    
    # Dynamically set n_in corresponding to the number of features in the design matrix
    hyperparams["n_in"] = int(train_data["x"].shape[1])

    network = DeepSurv(**hyperparams)
    # Timestamped run directory
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    run_dir = os.path.join(save_root, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    if verbose:
        print(f"Run directory: {run_dir}", flush=True)
    # Train once to leverage built-in LR schedule/momentum/patience
    t0 = time.perf_counter()
    tb_logdir = os.path.join(run_dir, "tb")
    os.makedirs(tb_logdir, exist_ok=True)
    tb_logger = TensorboardLogger("DeepSurv", tb_logdir)
    if verbose:
        print(
            f"Starting training for {n_epochs} epochs (patience={patience}, threshold={improvement_threshold}, x{patience_increase})...",
            flush=True,
        )
    history = network.train(
        train_data,
        None,  # validation data
        n_epochs=n_epochs,
        validation_frequency=1,
        patience=patience,
        improvement_threshold=improvement_threshold,
        patience_increase=patience_increase,
        update_fn=update_fn,  # Adam is used in the original paper
        verbose=verbose,
        logger=tb_logger,
    )
    # Persist model, predictions, and summary
    if verbose:
        print("Saving model artifacts...", flush=True)
    network.save_model(
        os.path.join(run_dir, "model.json"),
        os.path.join(run_dir, "weights.h5"),
    )

    t_train = time.perf_counter() - t0
    if verbose:
        print(f"Training finished in {t_train:.2f}s", flush=True)

    # Final metrics on train/valid (post-training)
    train_m = _metrics_from_dataset(network, train_data)

    # Build test dataset if provided and compute metrics
    risk_test = None
    test_m = None
    has_any_test = any(v is not None for v in (X_u_test, X_c_test, y_u_test, Y_c_test))
    has_all_test = all(v is not None for v in (X_u_test, X_c_test, y_u_test, Y_c_test))
    if has_any_test and (not has_all_test):
        raise ValueError(
            "If providing a test set, you must provide all of: X_u_test, X_c_test, y_u_test, Y_c_test."
        )

    test_data = None
    if has_all_test:
        test_data = build_deepsurv_data(X_u_test, X_c_test, y_u_test, Y_c_test)
        test_m = _metrics_from_dataset(network, test_data)

    # Predictions (risk) on test if available
    t_infer = 0.0
    if test_data is not None:
        t1 = time.perf_counter()
        x_test_prep, e_test_prep, t_test_prep = network.prepare_data(test_data)
        risk_test = network.predict_risk(x_test_prep)
        risk_test = np.asarray(risk_test).ravel()
        np.save(os.path.join(run_dir, "risk_test.npy"), risk_test)
        t_infer = time.perf_counter() - t1
        if verbose:
            print(f"Inference on test finished in {t_infer:.2f}s", flush=True)

    if calc_corr:
        r_s_tr, r_p_tr, cs_tr, r_s_te, r_p_te, cs_te, cph = expectation_corr_deepsurv(
            network=network,
            train_data=train_data,
            test_data=test_data,
            run_dir=run_dir,
            print_every=verbose,
            penalizer=0.001,
            l1_ratio=0.0,
            step_size=0.5
        )
    else:
        r_s_tr, r_p_tr, cs_tr, r_s_te, r_p_te, cs_te = 0, 0, 0, 0, 0, 0
    train_m["spearman_uncens"] = float(r_s_tr)
    train_m["pearson_uncens"] = float(r_p_tr)
    train_m["cs"] = float(cs_tr)
    if test_m is not None:
        test_m["spearman_uncens"] = float(r_s_te)
        test_m["pearson_uncens"] = float(r_p_te)
        test_m["cs"] = float(cs_te)

    summary = {
        "timestamp": timestamp,
        "hyperparams": hyperparams,
        "n_in": int(train_data["x"].shape[1]),
        "train_size_uncensored": int(X_u_train.shape[0]),
        "train_size_censored": int(X_c_train.shape[0]),
        "test_size_uncensored": int(X_u_test.shape[0]) if has_all_test else 0,
        "test_size_censored": int(X_c_test.shape[0]) if has_all_test else 0,
        "time_train_seconds": t_train,
        "time_infer_seconds": t_infer,
        "metrics": {
            "train": train_m,
            "test": test_m,
        },
        "comment": comment,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[Final] Train: {fmt(train_m)} | Test: {fmt(test_m)}")

    return risk_test, network, {"train": train_m, "test": test_m}, run_dir
