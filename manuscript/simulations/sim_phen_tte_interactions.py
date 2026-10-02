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
parser.add_argument("-tau", "--tau", help="Proportion of interaction term", default=0.2)
parser.add_argument("-lam", "--lam", help="Proportion of non-sparse region", default=0.1)
parser.add_argument("-C", "--C", help="Target censorship", default=0.9)
parser.add_argument("-phen_distribution", "--phen-distribution", help="Phenotype distribution (Weibull, Gamma)", default="Weibull")
parser.add_argument("-kappa", "--kappa", help="Gamma distribution parametrization", default=1)
parser.add_argument("-prior", "--prior", help="Prior distribution for signals", default="spike_slab")
parser.add_argument("-train_frac", "--train-frac", help="Train samples fraction", default=0.9)
parser.add_argument("-interactions", "--interactions", help="Interaction file", default=None)
parser.add_argument("-assay", "--assay", help="Assay file", default=None)
parser.add_argument("-out", "--out", help="Output prefix", default="example")
args = parser.parse_args()

X_fpath = args.X
h2 = float(args.h2)
tau = float(args.tau)
la = float(args.lam)
censoring = float(args.C)
phen_distribution = args.phen_distribution
kappa = float(args.kappa)
prior = args.prior
train_fraction = float(args.train_frac)
interactions_fpath = args.interactions
assay_fpath = args.assay
out = args.out

print("----- sim_phen_tte_interactions.py -----")
print("--X", X_fpath)
print("--h2", h2)
print("--tau", tau)
print("--lam", la)
print("--C", censoring)
print("--phen-distribution", phen_distribution)
print("--kappa", kappa)
print("--prior", prior)
print("--train-frac", train_fraction)
print("--interactions", interactions_fpath)
print("--assay", assay_fpath)
print("--out", out)
print("\n", flush=True)

censor_min = 0
censor_max = 1.0
mu = 0

# Load the data
X = np.load(X_fpath)
N,M = X.shape
iids = [f"ID{i:06d}" for i in range(1, N + 1)]

h2_inter = tau * h2
h2_dir = h2 - h2_inter

sparse_mask = np.random.binomial(1, la, size=[M,1])
causal = sparse_mask.ravel().astype(bool)

df_assay = pd.read_table(assay_fpath, sep="\t")
df_assay_causal = df_assay.loc[causal]

df_interact = pd.read_table(interactions_fpath, sep="\t")
df_interact = df_interact[df_interact["combined_score"] > 500]

causal_proteins = set(df_assay_causal["UniProt"])
df_interact_causal = df_interact[
    df_interact["UniProt1"].isin(causal_proteins) &
    df_interact["UniProt2"].isin(causal_proteins)
].reset_index(drop=True)

df_interact_causal = (
    df_interact_causal
    .assign(
        p1=df_interact_causal[["UniProt1", "UniProt2"]].min(axis=1),
        p2=df_interact_causal[["UniProt1", "UniProt2"]].max(axis=1)
    )
    .drop_duplicates(subset=["p1", "p2"])
    .drop(columns=["p1", "p2"])
    .reset_index(drop=True)
)

K = len(df_interact_causal)
print("K=", K)

protein_to_idx = {p: i for i, p in enumerate(df_assay["UniProt"])}
interactions = [
    (protein_to_idx[p1], protein_to_idx[p2])
    for p1, p2 in zip(df_interact_causal["UniProt1"], df_interact_causal["UniProt2"])
    if p1 in protein_to_idx and p2 in protein_to_idx
]

K = len(interactions)
print("K=", K, flush=True)

X_int = np.column_stack([
    X[:, i] * X[:, j]
    for i, j in interactions
])

var_int = np.var(X_int, axis=0)
sigma_inter = h2_inter / np.sum(var_int)
sigma = h2_dir / M / la

if prior == "spike_slab":
    beta = np.random.normal(loc=0.0, scale=np.sqrt(sigma), size=[M, 1]) 
    beta_inter = np.random.normal(loc=0.0, scale=np.sqrt(sigma_inter), size=[K, 1])
elif prior == "laplace":
    b_scale = np.sqrt(sigma / 2)
    b_scale_inter = np.sqrt(sigma_inter / 2)
    beta = np.random.laplace(loc=0, scale=b_scale, size=[M,1])
    beta_inter = np.random.laplace(loc=0, scale=b_scale_inter, size=[K,1])

beta *= sparse_mask

g_dir = np.matmul(X, beta)
g_inter = np.matmul(X_int, beta_inter)
g = g_dir + g_inter
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

print("var(beta) =", np.var(beta))
print("var(beta_inter) =", np.var(beta_inter))
print("var(g_dir) =", np.var(g_dir))
print("var(g_inter) =", np.var(g_inter))
print("var(g) =", np.var(g))
print("var(y) =", np.var(y[event_mask]))
print("var(log(y)) =", np.var(np.log(y[event_mask])))
print("mean(g_inter) =", np.mean(g_inter))
print("sum(event) =", np.sum(event), flush=True)
print("shape(beta) =", np.shape(beta))
print("shape(y) =", np.shape(y))
print("shape(event) =", np.shape(event), flush=True)

# Save true signals
np.savetxt(f"{out}_beta_true.txt", beta)

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