import numpy as np
import argparse
import pandas as pd
from vampw.simulations import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-X", "--X", help="Design matrix", default=None)
parser.add_argument("-h2", "--h2", help="Target heritability (Signal-to-noise-ratio)", default=0.8)
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
la = float(args.lam)
C = float(args.C)
phen_distribution = args.phen_distribution
kappa = float(args.kappa)
prior = args.prior
train_fraction = float(args.train_frac)
out = args.out

print("----- sim_phen_tte.py -----")
print("--X", X_fpath)
print("--h2", h2)
print("--lam", la)
print("--C", C)
print("--phen-distribution", phen_distribution)
print("--kappa", kappa)
print("--prior", prior)
print("--train-frac", train_fraction)
print("--out", out)
print("\n", flush=True)

# Load the data
if X_fpath == None:
    X = sim_geno(n=5000, m=300, p=0.4, distr="Gaussian")
else:    
    X = np.load(X_fpath)

N,M = X.shape
sigma= h2 / M / la

# Simulate betas, y, and event
problem = Problem(n=N, m=M, la=la, sigmas = [sigma], omegas=[1.0], model=phen_distribution)
beta, y, dist_param, event = sim_model_real_geno(problem, X, h2, kappa=kappa, censoring=C, beta_dist=prior)

print("Distribution parameter:", dist_param, flush=True)

y = y.squeeze()
event = event.astype(int)

iids = [f"ID{i:06d}" for i in range(1, N + 1)]
df_phen = pd.DataFrame({"IID": iids, "FID": iids, "Y": y.squeeze(), "EVENT": event.astype(int)})

# Train-test split
X_train, X_test, df_phen_train, df_phen_test = train_test_split(X, df_phen, test_size=(1 - train_fraction))

# Save output files
np.save(f"{out}_X_train.npy", X_train)
np.save(f"{out}_X_test.npy", X_test)
df_phen_train.to_csv(f"{out}_train.phen", index=False, header=True, sep="\t")
df_phen_test.to_csv(f"{out}_test.phen", index=False, header=True, sep="\t")
np.savetxt(f"{args.out}_beta_true.txt", beta)