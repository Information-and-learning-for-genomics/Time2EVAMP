import numpy as np
import argparse
import pandas as pd
from vampw.simulations import *

def standardize_log_y_by_uncensored_events(y, event):
    """
    Standardize log(Y) using only uncensored / primary-event samples EVENT == 1.

    Transformation:

        log_y_std = (log_y - mean(log_y[event == 1])) / std(log_y[event == 1])
        y_std = exp(log_y_std)

    This keeps Y positive while making:

        mean(log Y | EVENT == 1) = 0
        var(log Y | EVENT == 1) = 1
    """

    y = np.asarray(y).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(int)

    uncensored_mask = event == 1

    if np.sum(uncensored_mask) < 2:
        raise ValueError(
            "Cannot standardize log(Y): fewer than 2 uncensored/event==1 samples."
        )

    log_y = np.log(y)

    mu_event = np.mean(log_y[uncensored_mask])
    sd_event = np.std(log_y[uncensored_mask])

    if sd_event <= 0:
        raise ValueError(
            "Cannot standardize log(Y): std(log Y | EVENT==1) is zero."
        )

    log_y_std = (log_y - mu_event) / sd_event
    y_std = np.exp(log_y_std)

    return y_std, mu_event, sd_event

def draw_correlated_slab_effects(m, sigma, rho, prior):
    """
    Draw paired slab effects for the primary and competing processes.

    Parameters
    ----------
    m : int
        Number of variants.
    sigma : float
        Target marginal variance of each nonzero effect.
    rho : float
        Target correlation between effects for the two processes.
    prior : {"spike_slab", "laplace"}
        If "spike_slab", nonzero effects follow a correlated Gaussian slab.
        If "laplace", nonzero effects follow a correlated Laplace slab.

    Returns
    -------
    slab : ndarray of shape (m, 2)
        Paired effects for the primary and competing processes.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1.")

    covariance = sigma * np.array(
        [
            [1.0, rho],
            [rho, 1.0],
        ]
    )

    if prior == "spike_slab":
        # Gaussian slab with marginal variance sigma.
        slab = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=covariance,
            size=m,
        )

    elif prior == "laplace":

        gaussian_effects = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=covariance,
            size=m,
        )

        mixing_weights = np.random.exponential(
            scale=1.0,
            size=(m, 1),
        )

        slab = np.sqrt(mixing_weights) * gaussian_effects

    else:
        raise ValueError(
            f"Invalid prior '{prior}'. "
            "Expected either 'spike_slab' or 'laplace'."
        )

    return slab

def sim_model_competing_risks(problem, X, h2, C, rho=0.5, kappa=None, mu=0, prior="spike_slab", print_results=True):
    """
    Simulates competing risks (T1 vs T2) across arbitrary distribution models
    ('Weibull', 'Gamma', 'LogNormal') while guaranteeing exactly C proportion of censorship.
    """
    n, m = problem.n, problem.m
    la = problem.prior_instance.la
    sigma = problem.prior_instance.sigmas[0]
    mu_vec = np.full((n, 1), mu)

    # 1. Simulate correlated spike-and-slab genetic effects.
    #
    # The primary and competing processes share the same causal positions.
    # Their nonzero effects follow either a Gaussian slab (--prior spike_slab)
    # or a Laplace slab (--prior laplace).
    is_active = np.random.rand(m) < la

    slab = draw_correlated_slab_effects(
        m=m,
        sigma=sigma,
        rho=rho,
        prior=prior,
    )

    beta_1 = np.zeros((m, 1))
    beta_2 = np.zeros((m, 1))

    beta_1[is_active, 0] = slab[is_active, 0]
    beta_2[is_active, 0] = slab[is_active, 1]

    if print_results:
        slab_name = "Gaussian" if prior == "spike_slab" else "Laplace"
        print(f"Simulating competing risks under model: {problem.model}")
        print(f"Prior: spike-and-slab with a {slab_name} slab")
        print(f"Number of shared causal variants: {np.sum(is_active)}")
        print(
            f"Target heritability h2: {h2} | "
            f"Target censorship C: {C} | "
            f"Genetic correlation rho: {rho}"
        )

    # 2. Draw raw phenotypes using distribution-specific noise calibration
    model = problem.model

    if model == 'Weibull':
        # Match AFT Gumbel noise calibration exactly
        g1, g2 = X @ beta_1, X @ beta_2
        varwi = np.pi * np.pi / 6.0
        c1 = np.sqrt((1.0 / h2 - 1.0) * np.var(g1) / varwi)
        c2 = np.sqrt((1.0 / h2 - 1.0) * np.var(g2) / varwi)

        w1 = -mathematica_evd(n=n, loc=0.0, scale=1.0)
        w2 = -mathematica_evd(n=n, loc=0.0, scale=1.0)

        log_T1 = mu_vec + g1 + c1 * (w1 + emc)
        log_T2_raw = mu_vec + g2 + c2 * (w2 + emc)
        dist_param = 1.0 / c1

    elif model == 'Gamma':
        # Match ExpGamma noise calibration exactly
        if kappa is None:
            kappa = 1.0

        g1, g2 = X @ beta_1, X @ beta_2
        sigmaE_1 = np.sqrt((1.0 / h2 - 1.0) * np.var(g1))
        sigmaE_2 = np.sqrt((1.0 / h2 - 1.0) * np.var(g2))

        theta_1 = sigmaE_1 / scipy.special.polygamma(1, kappa)
        theta_2 = sigmaE_2 / scipy.special.polygamma(1, kappa)

        mut_1 = mu_vec + g1 - theta_1 * scipy.special.polygamma(0, kappa)
        mut_2 = mu_vec + g2 - theta_2 * scipy.special.polygamma(0, kappa)

        log_T1 = np.random.gamma(shape=kappa, scale=theta_1, size=(n, 1)) + mut_1
        log_T2_raw = np.random.gamma(shape=kappa, scale=theta_2, size=(n, 1)) + mut_2
        dist_param = kappa

    elif model == 'LogNormal':
        # Match LogNormal Gaussian noise calibration exactly
        g1, g2 = X @ beta_1, X @ beta_2
        sigma_1 = np.sqrt((1.0 / h2 - 1.0) * np.var(g1))
        sigma_2 = np.sqrt((1.0 / h2 - 1.0) * np.var(g2))

        w1 = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        w2 = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))

        log_T1 = mu_vec + g1 + sigma_1 * w1
        log_T2_raw = mu_vec + g2 + sigma_2 * w2
        dist_param = sigma_1

    else:
        raise Exception(f"{model} is not a valid model. Allowed: 'Weibull', 'Gamma', 'LogNormal'")

    if print_results:
        print(f"Latent Var(log T1) [Before Competing Truncation]: {np.var(log_T1):.4f}")

    # 3. Exact Censorship Calibration via Log-Difference Quantile Shift
    log_diff = log_T1 - log_T2_raw
    req_quantile = (1.0 - C) * 100.0
    delta = np.percentile(log_diff, req_quantile)

    log_T2 = log_T2_raw + delta

    # Convert back to raw survival times
    T1 = np.exp(log_T1)
    T2 = np.exp(log_T2)

    # 4. Resolve competing events
    Y = np.minimum(T1, T2)

    # Use direct comparison instead of floating-point equality Y == T1
    mask = (T1 <= T2).astype(int)

    return beta_1, beta_2, Y, dist_param, mask


parser = argparse.ArgumentParser()
parser.add_argument("-X", "--X", required=True)
parser.add_argument("-h2", "--h2", default=0.8, type=float)
parser.add_argument("-lam", "--lam", default=0.1, type=float)
parser.add_argument("-C", "--C", default=0.9, type=float)
parser.add_argument("-phen_distribution", "--phen-distribution", default="Weibull")
parser.add_argument("-kappa", "--kappa", default=1.0, type=float)
parser.add_argument("-train_frac", "--train-frac", default=0.9, type=float)
parser.add_argument("-rho", "--rho", default=0.5, type=float)
parser.add_argument("-prior", "--prior", help="Prior distribution for signals", default="spike_slab")
parser.add_argument("-out", "--out", required=True, default="example")
args = parser.parse_args()

X = np.load(args.X)
N, M = X.shape
iids = [f"ID{i:06d}" for i in range(1, N + 1)]

sigma = args.h2 / M / args.lam

problem = Problem(
    n=N,
    m=M,
    la=args.lam,
    sigmas=[sigma],
    omegas=[1.0],
    model=args.phen_distribution
)

beta, beta_comp, y, dist_param, event = sim_model_competing_risks(
        problem,
        X,
        h2=args.h2,
        C=args.C,
        rho=args.rho,
        kappa=args.kappa,
        prior=args.prior
    )

y = y.squeeze()
event = event.squeeze().astype(int)
uncensored_mask = event == 1

print("Before final log-time standardization:")
print(f"  Mean(log y | event==1): {np.mean(np.log(y[uncensored_mask])):.6f}")
print(f"  Var(log y | event==1):  {np.var(np.log(y[uncensored_mask])):.6f}")
print(f"  Mean(y | event==1):     {np.mean(y[uncensored_mask]):.6f}")
print(f"  Var(y | event==1):      {np.var(y[uncensored_mask]):.6f}")

y, log_y_event_mean_before, log_y_event_sd_before = standardize_log_y_by_uncensored_events(
    y=y,
    event=event
)

print("After final log-time standardization:")
print(f"  Used mean(log y | event==1): {log_y_event_mean_before:.6f}")
print(f"  Used sd(log y | event==1):   {log_y_event_sd_before:.6f}")
print(f"  Mean(log y | event==1):      {np.mean(np.log(y[uncensored_mask])):.6f}")
print(f"  Var(log y | event==1):       {np.var(np.log(y[uncensored_mask])):.6f}")
print(f"  Mean(y | event==1):          {np.mean(y[uncensored_mask]):.6f}")
print(f"  Var(y | event==1):           {np.var(y[uncensored_mask]):.6f}")

print(f"Distribution: {args.phen_distribution} | Calibrated param: {dist_param:.4f}")
print(f"Overall Var(log y): {np.var(np.log(y)):.4f} | Overall Var(y): {np.var(y):.4f}")
print(f"Overall Mean(log y): {np.mean(np.log(y)):.4f} | Overall Mean(y): {np.mean(y):.4f}")
print(f"Primary Events (event==1): {np.sum(uncensored_mask)} ({np.mean(uncensored_mask)*100:.1f}%)")

np.savetxt(f"{args.out}_beta_true.txt", beta)
np.savetxt(f"{args.out}_beta_comp_true.txt", beta_comp)

df = pd.DataFrame({
    "IID": iids,
    "FID": iids,
    "Y": y,
    "EVENT": event
})

df_train = df.sample(frac=args.train_frac, replace=False).sort_index()
df_test = df.drop(df_train.index).sort_index()

df_train.to_csv(f"{args.out}_train.phen", header=None, index=False, sep="\t")
df_test.to_csv(f"{args.out}_test.phen", header=None, index=False, sep="\t")