import numpy as np
import argparse
import pandas as pd
from vampw.simulations import *
emc = float( sympy.S.EulerGamma.n(10) )
import random 
import sympy
import scipy

parser = argparse.ArgumentParser()
parser.add_argument("-X", "--X", help="Design matrix", default=None)
parser.add_argument("-h2", "--h2", help="Target heritability (Signal-to-noise-ratio)", default=0.8)
parser.add_argument("-p-nonlin", "--p-nonlin", help="Proportion of non-linear part", default=0.5)
parser.add_argument("-lam", "--lam", help="Proportion of non-sparse region", default=0.1)
parser.add_argument("-C", "--C", help="Target censorship", default=0.9)
parser.add_argument("-phen_distribution", "--phen-distribution", help="Phenotype distribution (Weibull, Gamma)", default="Weibull")
parser.add_argument("-kappa", "--kappa", help="Gamma distribution parametrization", default=1)
parser.add_argument("-prior", "--prior", help="Prior distribution for signals", default="spike_slab")
parser.add_argument("-train_frac", "--train-frac", help="Train samples fraction", default=0.9)
parser.add_argument("-out", "--out", help="Output prefix", default="example")
args = parser.parse_args()

X_fpath = args.X
h2 = float(args.h2)
p_nonlin = float(args.p_nonlin)
la = float(args.lam)
censoring = float(args.C)
phen_distribution = args.phen_distribution
kappa = float(args.kappa)
prior = args.prior
train_fraction = float(args.train_frac)
out = args.out

print("----- sim_phen_tte_nonlin.py -----")
print("--X", X_fpath)
print("--h2", h2)
print("--p-nonlin", p_nonlin)
print("--lam", la)
print("--C", censoring)
print("--phen-distribution", phen_distribution)
print("--kappa", kappa)
print("--prior", prior)
print("--train-frac", train_fraction)
print("--out", out)
print("\n", flush=True)

censor_min = 0
censor_max = 1.0
mu = 0

# Load the data
X = np.load(X_fpath)
X_quad = X**2
N,M = X.shape
iids = [f"ID{i:06d}" for i in range(1, N + 1)]

h2_quad = p_nonlin * h2
h2_lin = h2 - h2_quad
sigma_lin = h2_lin / M / la

sparse_mask = np.random.binomial(1, la, size=[M,1]) # We have same causal positions across linear and quadratic effects
causal = sparse_mask.ravel().astype(bool)
var_x2 = np.var(X_quad[:,causal], axis=0)
sigma_quad = h2_quad / np.sum(var_x2)

if prior == "spike_slab":
    beta_lin = np.random.normal(loc=0.0, scale=np.sqrt(sigma_lin), size=[M, 1]) 
    beta_quad = np.random.normal(loc=0.0, scale=np.sqrt(sigma_quad), size=[M, 1])
elif prior == "laplace":
    b_scale_lin = np.sqrt(sigma_lin / 2)
    b_scale_quad = np.sqrt(sigma_quad / 2)
    beta_lin = np.random.laplace(loc=0, scale=b_scale_lin, size=[M,1])
    beta_quad = np.random.laplace(loc=0, scale=b_scale_quad, size=[M,1])

beta_lin *= sparse_mask
beta_quad *= sparse_mask

g_lin = np.matmul(X, beta_lin)
g_quad = np.matmul(X_quad, beta_quad)
lower = np.percentile(g_quad, 0.1)
upper = np.percentile(g_quad, 99.9)
g_quad = np.clip(g_quad, lower, upper) # Truncating extreme values
g = g_lin + g_quad
sigmaG = np.var(g)

if phen_distribution == "Weibull":
    varwi = np.pi * np.pi / 6
    c = np.sqrt((1/h2-1) * sigmaG / varwi)
    wi = -mathematica_evd(n=N, loc=-0, scale=1.0)
    y = np.exp(mu + g + c * (wi + emc))
    dist_param = alpha = 1.0 / c

elif phen_distribution == "Gamma":
    sigmaE =  (1/h2-1) * sigmaG
    theta = np.sqrt( sigmaE / scipy.special.polygamma(1, kappa))
    digamma_kappa = scipy.special.polygamma(0, kappa)
    w = np.log(np.random.gamma(shape=kappa, scale=1.0, size=[N,1]))
    mut = mu + g - theta * digamma_kappa
    y = np.exp(mut + theta * w)
    dist_param = theta

event = np.ones(N)
if censoring > 0:
    censor_len = int(censoring * N)
    # Generate random factors between censor_min and censor_max
    random_factors = np.random.uniform(censor_min, censor_max, censor_len)
    # Randomly select indices to censor
    random_indices = np.random.choice(N, censor_len, replace=False)
    # Apply censoring to the selected positions
    y[random_indices, 0] = y[random_indices, 0] * random_factors
    # Create mask vector
    event[random_indices] = 0

print("Distribution parameter:", dist_param, flush=True)

y = y.squeeze()
event = event.astype(int)
event_mask = event.astype(bool)

print("var(beta_lin) =", np.var(beta_lin))
print("var(beta_quad) =", np.var(beta_quad))
print("var(g_quad) =", np.var(g_quad))
print("var(g_lin) =", np.var(g_lin))
print("var(g) =", np.var(g))
print("var(y) =", np.var(y[event_mask]))
print("var(log(y)) =", np.var(np.log(y[event_mask])))
print("mean(g_quad) =", np.mean(g_quad))
print("sum(event) =", np.sum(event), flush=True)
print("shape(beta_lin) =", np.shape(beta_lin))
print("shape(y) =", np.shape(y))
print("shape(event) =", np.shape(event), flush=True)

# Save true signals
np.savetxt(f"{out}_beta_true.txt", beta_lin)
np.savetxt(f"{out}_beta_true_quad.txt", beta_quad)

# Create DataFrame
df = pd.DataFrame({"IID": iids, "FID": iids, "Y": y,  "EVENT": event})

# Sample train and test parts
df_train = df.sample(frac=train_fraction, replace=False).sort_index()
df_test = df.drop(df_train.index).sort_index()

print("Number of train samples: ", len(df_train))
print("Number of test samples: ", len(df_test), flush=True)

# Save TTE phenotype
df_train.to_csv(f"{out}_train.phen", header=None, index=False, sep="\t")
df_test.to_csv(f"{out}_test.phen", header=None, index=False, sep="\t")