import numpy as np
import sympy

emc = float(sympy.S.EulerGamma.n(10))

# Hazard function of Gumbel distribution
def hazard_func_Weibull(x): 
    return np.exp(x)

# Define the gradient of the negative log-likelihood function
def censoring_LMMSE_loss_Weibull(beta, gam2, r2, tau2, X, p2, mu, alpha, X_c, y_c):
    n, m = X.shape
    nc, _ = X_c.shape
    r2.resize((m,1))
    p2.resize((n,1))
    y_c.resize((nc,1))
    beta.resize((m,1))
    term1 = 2 * gam2 * (beta - r2)
    term2 = 2 * tau2 * X.T @ (X @ beta - p2)

    tt = alpha * (np.log(y_c) - mu - X_c @ beta) - emc
    XcBeta = X_c @ beta
    logyc = np.log(y_c)
    # print(f'XcBeta.shape = {XcBeta.shape}')
    # print(f'logyc.shape = {logyc.shape}')
    # print(f'tt.shape = {tt.shape}')
    # print(f'mu = {mu}')
    # print(f'emc = {emc}')
    # print(f'alpha = {alpha}')
    sub = hazard_func_Weibull(alpha * (np.log(y_c) - mu - X_c @ beta) - emc)
    term3 = alpha * X_c.T @ sub
    term = term1 + term2
    term -= term3
    term.resize((m,))
    # return np.sum(term**2)
    return term

def censoring_LMMSE_loss_Weibull_grad(beta, gam2, r2, tau2, X, p2, mu, alpha, X_c, y_c):
    r2 = r2.squeeze(-1)
    p2 = p2.squeeze(-1)
    y_c = y_c.squeeze(-1)
    _, m = X.shape
    term1 = 2 * (gam2 * np.identity(m) + tau2*X.T@X)
    hazard_eval = hazard_func_Weibull(alpha*(np.log(y_c)- mu - X_c@beta) - emc)
    Dh = np.diag(np.diag(hazard_eval))
    term2 = alpha**2 * X_c.T @ Dh @ X_c
    term = term1 + term2 
    return term