import numpy as np
import argparse
import pandas as pd
from vampw.vamp import *
import pickle
import time
from lifelines.utils import concordance_index
import sympy
from sklearn.metrics import mean_squared_error, confusion_matrix

eps = 1e-16

def pip_calc(r1, gam1, omegas, sigmas, la):
    r1 = np.asmatrix(r1)
    sigmas_max = np.max(sigmas)
    omegas = np.asmatrix(omegas)
    sigmas = np.asmatrix(sigmas)
    gam1inv = 1.0/gam1
    beta_tilde = np.multiply(np.exp(- np.power(r1,2) / 2 @ (1/(sigmas + gam1inv))), la * omegas / np.sqrt(gam1inv + sigmas))
    sum_beta_tilde = beta_tilde.sum(axis=1)
    pi = 1.0 / ( 1.0 + (1-la) * np.exp(-np.power(r1, 2) / 2 * gam1 ) / np.sqrt(gam1inv) / sum_beta_tilde )
    return np.asarray(pi).squeeze()

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
parser.add_argument("-beta_true", "--beta-true", help="Beta true file", default=None)
parser.add_argument("-rho", "--rho", help="Damping", default=0.3)
parser.add_argument("-adaptive-rho", "--adaptive-rho", help="Adaptive damping", default=False)
parser.add_argument("-rho-wu-it", "--rho-wu-it", help="Warm up iterations for adaptive damping", default=-1)
parser.add_argument("-rho-autotune", "--rho-autotune", help="Number of autotuning iterations", default=0)
parser.add_argument("-alpha", "--alpha", help="", default=3)
parser.add_argument("-mu", "--mu", help="", default=0)
parser.add_argument("-learn-mu", "--learn-mu", help="Learn mu", default=True)
parser.add_argument("-probs", "--probs", help="Prior probabilities", default="0.7,0.3")
parser.add_argument("-vars", "--vars", help="Prior variances", default="0,0.01")
parser.add_argument("-n_hutchinson_samples", "--n-hutchinson-samples", help="Hutchinson samples", default=8)
parser.add_argument("-iterations", "--iterations", help="Number of iterations", default=20)
parser.add_argument("-standardise", "--standardise", help="Phenotype standardisation", default=False)
parser.add_argument("-thresholds", "--thresholds", help="File containing thresholds for FDR-TPR curves")
parser.add_argument("-out", "--out", help="Output prefix")
args = parser.parse_args()

X_train_fpath = args.X_train
phen_train_fpath = args.phen_train
X_test_fpath = args.X_test
phen_test_fpath = args.phen_test
beta_true_fpath = args.beta_true
rho = float(args.rho)
adaptive_rho = bool(int(args.adaptive_rho))
rho_wu_it = int(args.rho_wu_it)
rho_autotune = int(args.rho_autotune)
vars = [float(v) for v in args.vars.split(",")]
probs = [float(p) for p in args.probs.split(",")]
alpha_initial = float(args.alpha)
mu = float(args.mu)
learn_mu = bool(args.learn_mu)
n_hutchinson_samples = int(args.n_hutchinson_samples)
iterations = int(args.iterations)
standardise = bool(int(args.standardise))
thrs_fpath = args.thresholds
out = args.out

print("----- vampW pipeline -----")
print("--X-train", X_train_fpath)
print("--phen-train", phen_train_fpath)
print("--X-test", X_test_fpath)
print("--phen-test", phen_test_fpath)
print("--beta-true", beta_true_fpath)
print("--rho", rho)
print("--adaptive-rho", adaptive_rho)
print("--rho-wu-it", rho_wu_it)
print("--rho-autotune", rho_autotune)
print("--vars", vars)
print("--probs", probs)
print("--alpha", alpha_initial)
print("--learn-mu", learn_mu)
print("--n-hutchinson-samples", n_hutchinson_samples)
print("--standardise", standardise)
print("--thresholds", thrs_fpath)
print("--out", out)
print("\n", flush=True)

# Load the train data
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

# Load the test data
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

sigma_initial = vars[1:]
la_initial = sum(probs[1:])
omega_initial = list(np.array(probs[1:]) / la_initial)
print("omega_initial = ", omega_initial)
print("sigma_initial = ", sigma_initial)
print("lambda_initial = ", la_initial)

mode = "censoring"
model = 'Weibull'

Hyp = Hyperparams(model, mu, alpha_initial)

problem = Problem(n=N_train, 
                  m=M, 
                  la=la_initial, 
                  sigmas=sigma_initial, 
                  omegas=omega_initial, 
                  model=model, 
                  hyperparams=Hyp)

pref={'update_mu': learn_mu, 
      'update_alpha': True, 
      'start_at_mu': mu, 
      'start_at_alpha': 3,
      'mode': "censoring", 
      'dampen_coeff': rho, 
      'dampen_wu_it': rho_wu_it, 
      'dampen_autotune': rho_autotune, 
      'dampen_scale': 0.9, 
      'save': False, 
      'print_results': True, 
      'tau1':1e-4, 
      'num_hutchinson_samples': n_hutchinson_samples}

data = {'X': X_train[event_train==1], 
        'y': y_train[event_train==1], 
        'X_c': X_train[event_train==0], 
        'y_c': y_train[event_train==0]}

seed = np.random.randint(0, 2**32 - 1)

ts = time.time()
vampw_model = infere(data, problem, pref, seed, iterations)
t_infer = time.time() - ts
print("Inference time: %0.4f s"%(t_infer), flush=True)

mus = vampw_model['mus']
alphas = vampw_model['alphas']
r1s = vampw_model['r1s']
gam1s = vampw_model['gam1s']
omegas = vampw_model['omegas']
sigmas = vampw_model['sigmas']
lambdas = vampw_model['lambdas']
ci_train_list = vampw_model['ci_train']

best_it = np.argmax(ci_train_list[5:]) + 5
print(f"Best iteration: {best_it}", flush=True)

y_hat = np.exp(mus[best_it] + X_test @ vampw_model["x1_hats"][best_it])

c_index = concordance_index(
    event_times=df_test["time"],
    predicted_scores=y_hat, 
    event_observed=df_test["event"]
)
mse = mean_squared_error(y_test[event_mask_test], y_hat[event_mask_test])
pip = pip_calc(r1s[best_it], gam1s[best_it-1], omegas[best_it-1], sigmas[best_it-1], lambdas[best_it-1])

pips_out_fpath = f"{out}_pip.tsv"
df_pip = pd.DataFrame({"PIP": pip})
df_pip.to_csv(pips_out_fpath, index=False)

print(f"Test scores: RMSE = {np.sqrt(mse)}, C-index = {c_index}")
print(f"Posterior Inclusion Probabilities stored in: {pips_out_fpath}")

if beta_true_fpath is not None:

    beta_true = np.loadtxt(beta_true_fpath, dtype=float)
    true_mask = (np.abs(beta_true) > 0) * 1

    thrs = np.load(thrs_fpath)

    fdr = []
    tpr = []  
    for thr in thrs:
        est = (pip >= thr) * 1 
        tn, fp, fn, tp = confusion_matrix(true_mask, est).ravel()
        fdr.append(fp / (fp + tp + eps))
        tpr.append(tp / (tp + fn + eps))

    df_roc = pd.DataFrame({"FDR": fdr, "TPR": tpr, "THR": thrs})
    df_roc.to_csv(f"{out}_roc.csv", index=False, sep="\t")