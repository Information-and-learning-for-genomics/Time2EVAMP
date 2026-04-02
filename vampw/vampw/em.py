import scipy
import numpy as np
import math
import sympy 

### CONSTANTS ###
# Euler -Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

### WEIBULL MODEL ###
def update_Weibull_alpha_eq(alpha, y, mu, z_hat, xi):
    n,_ = y.shape
    out = np.zeros(n)
    res = np.log(y) - mu - z_hat
    out = n / alpha + np.sum(res) - np.exp(-emc) * np.sum( np.exp(alpha * res + (alpha**2)/2/xi) * (res + alpha/xi) )
    return out

def update_Weibull_alpha(y, mu, z_hat, alpha_old, xi):
    # y.shape = [n,1]
    # z_hat.shape = [n,1]
    alpha_new = scipy.optimize.fsolve(update_Weibull_alpha_eq, x0 = alpha_old, args=(y, mu, z_hat, xi))
    if isinstance(alpha_new, np.ndarray) or isinstance(alpha_new, list): alpha_new = float(alpha_new[0])
    return alpha_new

def update_Weibull_mu(y, z_hat, alpha, xi):
    # y.shape = [n,1]
    # z_hat.shape = [n,1]
    n,_ = y.shape
    mu_new = - np.log(n) / alpha - emc/alpha + alpha / 2 / xi + 1 / alpha * np.log(np.sum(np.exp(alpha*(np.log(y) - z_hat))))
    if isinstance(mu_new, np.ndarray) or isinstance(mu_new, list): mu_new = float(mu_new[0])
    return mu_new

## LOGNORMAL MODEL ###
#computes the EM update non-linear equation for mu in the LogNormal model
#y, mu, z_hat should be np.arrays
def update_LogNormal_mu_eq(mu, y, z_hat, sigma, censored):
    n,_ = y.shape
    out = np.zeros(n)
    #contribution of non-censored individuals
    out[censored==0] = (np.log(y[censored==0])-mu-z_hat[censored==0])/sigma/sigma
    #contribution of censored individuals (for such corresponding values of z_hat = x_i^T * beta_hat and y = censoring time)
    out[censored==1] = np.exp(-np.power(mu + z1_hat[censored==1]-np.log(y[censored==1]),2) / 2 / sigma / sigma) * np.sqrt(2/math.pi) / sigma / (1+math.erf(mu + z1+hat[censored==1]-np.log(y[censored==1])))
    out = np.sum(out)
    return out   

#find an EM-update for mu parameter in LogNormal model
def update_LogNormal_mu(y, mu, z_hat, sigma, censored):
    out = scipy.optimize.fsolve(update_LogNormal_mu_eq, x0 = mu, args=(y, z_hat, sigma, censored))
    return out

#y, mu, z_hat should be np.arrays
def update_LogNormal_sigma_eq(sigma, y, mu, z_hat, xi, censored):
    n,_ = y.shape
    out = np.zeros(n)
    # since from both expression one can extract sigma in the denominator we use the formulae without this additional sigma in the denominator 
    #contribution of non-censored individuals
    out[censored==0] = ( np.power( (np.log(y[censored==0])-mu-z_hat[censored==0]), 2) + 1/xi )/ np.power(sigma,2) - 1 
    #contribution of censored individuals (for such corresponding values of z_hat = x_i^T * beta_hat and y = censoring time)
    out[censored==1] = 1/np.power(sigma,2) / xi + (1 - np.sqrt(2/math.pi) ) / sigma * (np.log(y[censored==1]) - mu - z_hat[censored==1]) / (1+math.erf(mu + z1+hat[censored==1]-np.log(y[censored==1]))) \
        * np.exp(-np.power((np.log(y[censored==1]) - mu - z_hat[censored==1]),2)/2/sigma/sigma) - 1
    out = np.sum(out)
    return out       

#find an EM-update for sigma parameter in LogNormal model
def update_LogNormal_sigma(y, mu, z_hat, sigma, xi, censored):
    out = scipy.optimize.fsolve(update_LogNormal_sigma_eq, x0 = sigma, args=(y, mu, z_hat, xi, censored))
    return out

## EXPGAMMA MODEL ### (no censoring is taken into account)
def update_ExpGamma_mu(y, z_hat, mu, kappa, theta, xi):
    n,_ = y.shape
    out = - theta * np.log(n*kappa) + theta * scipy.special.polygamma(0, kappa) + 0.5 / theta / xi + np.logaddexp.reduce((np.log(y) - z_hat)/theta, dtype=np.float64)
    return out

def update_ExpGamma_theta_eq(theta, y, mu, z_hat, kappa, xi):
    out = kappa * np.sum(np.log(y) - mu - z_hat) - np.exp(scipy.special.polygamma(0, kappa) + 0.5 / theta / theta / xi)  * np.sum( np.multiply( (np.log(y) - mu - z_hat + 1.0/theta / xi), np.exp((np.log(y) - mu - z_hat)/theta) ) )    
    return out

def update_ExpGamma_theta(y, z_hat, mu, kappa, theta, xi): 
    out = scipy.optimize.fsolve(update_ExpGamma_theta_eq, x0 = theta, args=(y, mu, z_hat, kappa, xi))
    return out

def update_ExpGamma_kappa_eq(kappa, y, mu, z_hat, theta, xi):
    n,_ = y.shape
    out = np.sum(np.log(y) - mu - z1_hat)/theta - n * scipy.special.polygamma(0, kappa) + scipy.special.polygamma(1, kappa) * (n * kappa - np.sum( (np.log(y) - mu - z_hat)/theta +  \
        scipy.special.polygamma(0, kappa) + 0.5/theta/xi) )   
    return out

def update_ExpGamma_kappa(y, z_hat, mu, kappa, theta, xi): 
    out = scipy.optimize.fsolve(update_ExpGamma_kappa_eq, x0 = kappa, args=(y, mu, z_hat, theta, xi))
    return out

# performs the update of the prior distribution
def update_Prior(old_prior, r1, gam1):
    prior = old_prior
    r1 = np.asmatrix(r1)
    omegas = np.asmatrix(old_prior.omegas)
    sigmas = np.asmatrix(old_prior.sigmas)
    sigmas_max = old_prior.sigmas.max()
    gam1inv = 1.0/gam1
    # np.exp( - np.power(np.transpose(r1),2) / 2 @ (sigmas_max - sigmas) / (sigmas_max + gam1inv) / (sigmas + gam1inv)) has shape = (P,L) and  omegas / np.sqrt(gam1inv + sigmas) has shape = (1, L)
    beta_tilde=np.multiply( np.exp( - np.power(np.transpose(r1),2) / 2 @ (sigmas_max - sigmas) / (sigmas_max + gam1inv) / (sigmas + gam1inv)), omegas / np.sqrt(gam1inv + sigmas) )
    sum_beta_tilde = beta_tilde.sum(axis=1)
    beta_tilde=beta_tilde / sum_beta_tilde
    # pi.shape = (P, 1)
    pi = 1.0 / ( 1.0 + (1-prior.la * np.exp(-np.power(np.transpose(r1),2) / 2 * sigmas_max * gam1 / (sigmas_max + gam1inv) ) / np.sqrt(gam1inv) ) / sum_beta_tilde )
    gamma = np.divide(np.transpose(r1) * gam1, gam1 + 1.0/sigmas )
    # v.shape = (1,L)
    v = 1.0 / (gam1 + 1.0/sigmas)

    #updating sparsity level
    prior.la = np.mean(pi)
    #updating variances in the mixture
    prior.sigmas = (np.transpose(pi) @ np.multiply( beta_tilde , (np.power(gamma,2) + v)) ) / (np.transpose(pi) @ beta_tilde)
    #updating prior probabilities in the mixture
    prior.omegas = (np.transpose(pi) @ beta_tilde ) / np.sum(pi)
    
    return prior

