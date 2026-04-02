from scipy.stats import norm
import numpy as np
import sympy
import scipy
from .problem import *
from scipy.integrate import quad, quad_vec
from scipy.stats import gumbel_l
K = sympy.EulerGamma.evalf()

# definition of Euler-Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

def den_beta_gaussian(r,gam1,problem):
    prior = problem.prior_instance
    return r * prior.sigmas[0] / (prior.sigmas[0] + 1.0/gam1)

def der_den_beta_gaussian(r,gam1,problem):
    prior = problem.prior_instance
    return prior.sigmas[0] / (prior.sigmas[0] + 1.0/gam1)

# denoiser of the signal beta (for L > 1) based on the gVAMP paper
def denoiser_beta(r, gam1, problem):
    prior = problem.prior_instance
    sigmas = np.asmatrix(prior.sigmas)
    omegas = np.asmatrix(prior.omegas)
    r = np.asmatrix(r)
    sigma_max = np.max(sigmas)

    A = (1 - prior.la) * np.sqrt(gam1) * np.exp(-np.power(r,2) / 2 * sigma_max * gam1 / (1/gam1 + sigma_max))
    term1 = (sigma_max - sigmas) / (1/gam1 + sigmas) / (1/gam1 + sigma_max)
    EXP = np.exp(-np.power(r, 2) / 2 @ term1)
    Num = prior.la * np.sum(np.multiply(EXP, ((r @ (sigmas / np.power(1/gam1 + sigmas, 3/2))))), axis=1)
    Den = A + prior.la * (EXP @ np.transpose(omegas / np.sqrt(1/gam1 + sigmas)))
    return np.array(np.divide(Num, Den))

# derivative of the denoiser above (based on the Onsager correction part in the gVAMP paper)
def der_denoiser_beta(r, gam1, problem):
    prior = problem.prior_instance
    sigmas = np.asmatrix(prior.sigmas)
    omegas = np.asmatrix(prior.omegas)
    r = np.asmatrix(r)
    sigma_max = np.max(sigmas)

    A = (1 - prior.la) * np.sqrt(gam1) * np.exp(-np.power(r,2) / 2 * sigma_max * gam1 / (1/gam1 + sigma_max))
    term1 = (sigma_max - sigmas) / (1/gam1 + sigmas) / (1/gam1 + sigma_max)
    EXP = np.exp(-(np.power(r, 2) / 2) @ term1)
    Num = prior.la * np.sum(np.multiply(EXP, ((r @ (sigmas / np.power(1/gam1 + sigmas, 3/2))))), axis=1)
    Den = A + prior.la * (EXP @ np.transpose(omegas / np.sqrt(1/gam1 + sigmas)))

    term2 = (1 - np.power(r, 2) @ term1)
    Num_der = prior.la * (np.multiply(EXP, term2) @ np.transpose(np.divide(np.multiply(omegas, sigmas), np.power(1/gam1 + sigmas, 3/2))))
    Den_der = - np.multiply(r, (prior.la * (EXP @ np.transpose(np.multiply(np.divide(omegas, np.sqrt(1/gam1 + sigmas)), term1))) + A * (gam1 * sigma_max / (1/gam1 + sigma_max))))

    return np.array(np.divide(Num_der, Den) - np.divide(np.multiply(Num, Den_der), np.power(Den, 2)))

# denoiser of the signal beta
def den_beta(r,gam1,problem): # checked!
    """
    This function returns the conditional expectation of the coefficients beta given the noisy estimate r
    The expectation is of the posterior distribution with the form of Spike and Slab mixture of Gaussians
    """
    prior = problem.prior_instance
    if prior.la == 1:
        return den_beta_gaussian(r,gam1,problem)
    A = (1-prior.la) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1)) # scale = standard deviation
    # Make the variance here match the generated beta
    # h2 / m / la
    # print(f"Denoiser is using sigma: {prior.sigmas[0]}")
    B = prior.la * norm.pdf(r, loc=0, scale=np.sqrt(prior.sigmas[0] + 1.0/gam1))
    ratio = gam1 * r / (gam1 + 1/prior.sigmas[0]) * B / (A + B)
    return ratio

def der_den_beta(r,gam1,problem): # checked!
    prior = problem.prior_instance
    if prior.la == 1:
        # print(f"Gaussian prior, gam1 = {gam1}, sigmas = {prior.sigmas[0]}")
        return der_den_beta_gaussian(r,gam1,problem)
    # Derivative of the Gaussians with respect to r
    A = (1-prior.la) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
    B = prior.la * norm.pdf(r, loc=0, scale=np.sqrt(prior.sigmas[0] + 1.0/gam1))
    # print("B / (A+B) = ", B[1] / (A[1]+B[1]))
    Ader = A * (-r*gam1)
    Bder = B * (-r) / (prior.sigmas[0] + 1.0/gam1)
    BoverAplusBder = ( Bder * A - Ader * B ) / (A+B) / (A+B)
    # print("gam1 / (gam1 + 1/sigma) = ", gam1 / (gam1 + 1/prior.sigmas[0]))
    # print("alpha1 part I = ", gam1 / (gam1 + 1/prior.sigmas[0]) * B[1] / (A[1] + B[1]))
    # print("alpha2 part II = ", BoverAplusBder[1] * r[1] * gam1 / (gam1 + 1.0/prior.sigmas[0]) )
    ratio = gam1 / (gam1 + 1/prior.sigmas[0]) * B / (A + B) + BoverAplusBder * r * gam1 / (gam1 + 1.0/prior.sigmas[0])
    return ratio


# denoiser of z
def den_z(p1, tau1, y, problem, denoiser_type="map"):
    if problem.model == 'Weibull':
        alpha, mu = problem.hyperparams_instance.alpha, problem.hyperparams_instance.mu
        r = np.zeros(shape = p1.shape)
        if denoiser_type == 'map':
            r = den_z_Weibull(p1, tau1, y, alpha, mu)
        elif denoiser_type == 'mmse':
            int_bound = 4
            r = den_z_mmse_Weibull(p1, tau1, int_bound, mu, alpha, np.log(y))
    return r
    #  elif problem.model == 'Gamma':
    #      theta, kappa, mu = problem.hyperparams_instance.theta, problem.hyperparams_instance.kappa, problem.hyperparams_instance.mu
    #      return den_z_Gamma(p1, tau1, y, kappa, theta, mu)
    #  elif problem.model == 'LogNormal':
    #      sigma, mu = problem.hyperparams_instance.sigma, problem.hyperparams_instance.mu
    #      return den_z_LogNormal(p1, tau1, y, sigma, mu)      

def der_den_z(p1, tau1, y, problem, denoiser_type="map"):
    if problem.model == 'Weibull':
        alpha, mu = problem.hyperparams_instance.alpha, problem.hyperparams_instance.mu
        r = np.zeros(shape = p1.shape)
        if denoiser_type == 'map':
            r = der_den_z_Weibull(p1, tau1, y, alpha, mu)
        elif denoiser_type == 'mmse':
            int_bound = 4.0
            r = onsager_z_mmse_Weibull(p1, tau1, int_bound, mu, alpha, np.log(y)) / tau1
        return r
    #  elif problem.model == 'Gamma':
    #      theta, kappa, mu = problem.hyperparams_instance.theta, problem.hyperparams_instance.kappa, problem.hyperparams_instance.mu
    #      return der_den_z_Gamma(p1, tau1, y, kappa, theta, mu)
    #  elif problem.model == 'LogNormal':
    #      sigma, mu = problem.hyperparams_instance.sigma, problem.hyperparams_instance.mu
    #      return der_den_z_LogNormal(p1, tau1, y, sigma, mu) 
         
# Weibull model
def den_z_non_lin_eq_Weibull(z, tau1, p1, y, alpha, mu):
    """
    Performs MAP estimation of z
    Defines the objective to maximize
    Maximizing the expression below is equivalent to maximizing the likelihood of z
    We can treat the components of z as independent under the simplifying assumptions
    """ 
    res = tau1 * (z-p1) + alpha - alpha * np.power(y, alpha) * np.exp(- alpha * (mu + z) - emc)
    return res
    
def den_z_Weibull(p1, tau1, y, alpha, mu): 
    n,_ = p1.shape
    out = np.zeros((n,1))
    for i in range(0, n):
        out[i] = scipy.optimize.fsolve(den_z_non_lin_eq_Weibull, x0 = p1[i], args=(tau1, p1[i], y[i], alpha, mu) )
    # print(f"Out: {out}")
    return out

def der_den_z_Weibull(p1, tau1, y, alpha, mu):
    problem = Problem(model = 'Weibull')
    problem.hyperparams_instance = Hyperparams(problem.model, mu, alpha)
    #problem.hyperparams_instance.alpha, problem.hyperparams_instance.mu = alpha, mu
    z = den_z(p1, tau1, y, problem)
    # print(f"Z: {z}")
    nom = tau1
    den = tau1 + alpha * alpha * np.power(y, alpha) * np.exp(- alpha * (mu + z) - emc)
    return nom / den

# evaluating MMSE denoiser for z under Weibull model
# in the following functions posterior distribution p is proportional to Gumbel(mu + xi^T beta + K/alpha, 1/alpha) * N(z; p1, 1/tau_1)
def den_z_integrand_prob_mass_Weibull(z, mu, alpha, logy, p1, tau1):
    return gumbel_l.pdf(logy, loc=mu+z+K/alpha, scale=1/alpha) * norm.pdf(z, loc=p1, scale=1/np.sqrt(tau1))

def den_z_integrand_expe_Weibull(z, mu, alpha, logy, p1, tau1):
    return z * gumbel_l.pdf(logy, loc=mu+z+K/alpha, scale=1/alpha) * norm.pdf(z, loc=p1, scale=1/np.sqrt(tau1))

def den_z_integrand_second_moment_Weibull(z, mu, alpha, logy, p1, tau1):
    return z * z * gumbel_l.pdf(logy, loc=mu+z+K/alpha, scale=1/alpha) * norm.pdf(z, loc=p1, scale=1/np.sqrt(tau1))

def den_z_mmse_Weibull(p1, tau1, int_bound, mu, alpha, logy):
    """
    This function returns the conditional expectation of the predictor z given the noisy estimate p1 and 
    noise precision tau1 
    """
    Iexpe = quad_vec(den_z_integrand_expe_Weibull, p1-int_bound, p1+int_bound, args=(mu,alpha,logy,p1,tau1))[0]
    Iprob_mass = quad(den_z_integrand_prob_mass_Weibull, p1-int_bound, p1+int_bound, args=(mu,alpha,logy,p1,tau1))[0]
    return np.array( Iexpe / Iprob_mass )

def onsager_z_mmse_Weibull(p1, tau1, int_bound, mu, alpha, logy):
    """
    This function returns the conditional variance of the predictor z given the noisy estimate p1 and 
    noise precision tau1 
    """
    Isecond_moment = quad_vec(den_z_integrand_second_moment_Weibull, p1-int_bound, p1+int_bound, args=(mu,alpha,logy,p1,tau1))[0]
    Iprob_mass = quad_vec(den_z_integrand_prob_mass_Weibull, p1-int_bound, p1+int_bound, args=(mu,alpha,logy,p1,tau1))[0]
    expe = den_z_mmse_Weibull(p1, tau1, int_bound, mu, alpha, logy)
    return np.array( tau1 * (Isecond_moment / Iprob_mass - expe * expe) )