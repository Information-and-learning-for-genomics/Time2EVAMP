import numpy as np
import sympy
import scipy
from .utils import *
from .problem import Problem, Hyperparams
from bed_reader import open_bed
import random  # This ensures you're using Python's built-in random module.
# Euler -Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

#function for simultaing genotype matrix and Weibull distributed phenotypes
def semi_synthetic_data(bed_file, n=15000, m=15000, p=0.4, la=0.05, h2=0.9, mu=0, omega=1, seed=42, model='Weibull', censoring=0, censor_min=0.0, censor_max=1.0):
    sigma=h2/m/la
    problem_instance = Problem(n=n, m=m, la=la, sigmas = [sigma], omegas=[omega], model=model)

    np.random.seed(seed)
    bed = open_bed(bed_file)
    total_columns = 887060
    # Randomly select m columns from the total columns
    random_columns = sorted(random.sample(range(total_columns), m))
    # Read the BED file with a random subset the selected columns
    X = np.array(bed.read(index=(np.s_[:n], random_columns))) 
    X, tot_nans_after, valid_stds = standardize_columnwise(X)
    beta = sim_beta(m, la, sigma)
    y, alpha, mask = sim_pheno_Weibull(X, beta, mu, h2, censoring, censor_min, censor_max)
    print(f"Generated alpha value is: {alpha}")
    print(f"Variance of Xb: {np.var(X@beta)}")
    print(f"Variance of b: {np.var(beta)}")
    problem_instance.hyperparams_instance = Hyperparams(problem_instance.model, mu, alpha)
    return X,beta,y,alpha,problem_instance,tot_nans_after, valid_stds, mask


def synthetic_data(n=800, m=800, p=0.4, la=0.05, h2=0.9, mu=0, omega=1, seed=42, distr = "Gaussian", print_results=True, sigma=None, model='Weibull', censoring=0, censor_min=0.0, censor_max=1.0):
    #  print(f'synthetic_data; received mu: {mu}')
    np.random.seed(seed)
    if sigma == None:
        sigma= h2 / m /la
    # Note: omega is currently unused since the prior is just 1 slab 
    problem_instance = Problem(n=n, m=m, la=la, sigmas = [sigma], omegas=[omega], model=model)
    # note: the sim model should return something model agnostic, needs to be further refactored
    X, beta, y, alpha,tot_nans_after, valid_stds, mask = sim_model(problem_instance, h2, p, None, mu, censoring, censor_min, censor_max, distr, print_results)
    problem_instance.hyperparams_instance = Hyperparams(problem_instance.model, mu, alpha)
    return X, beta, y, alpha, problem_instance, tot_nans_after, valid_stds, mask 

def synthetic_data_real_geno(svd_path, la=0.05, h2=0.9, mu=0, omega=1, seed=42, print_results=True, sigma=None, model='Weibull', censoring=0, censor_min=0.0, censor_max=1.0):
    svd_decomposition = np.load(svd_path)
    U = svd_decomposition["U"]
    S = svd_decomposition["S"]
    VT = svd_decomposition["VT"]
    res = U @ np.diag(S) @ VT
    n, m = res.shape
    
    np.random.seed(seed)
    if sigma == None:
        sigma= h2 / m /la
    # Note: omega is currently unused since the prior is just 1 slab 
    problem_instance = Problem(n=n, m=m, la=la, sigmas = [sigma], omegas=[omega], model=model)
    beta, y, alpha, mask = sim_model_real_geno(problem_instance, res, h2, None, mu, censoring, censor_min, censor_max, print_results)
    problem_instance.hyperparams_instance = Hyperparams(problem_instance.model, mu, alpha)
    X_c, y_c, X, y = res[mask==0], y[mask==0], res[mask==1], y[mask==1]
    data = {'X':X, 'y':y, 'X_c': X_c, 'y_c': y_c, 'beta_true':beta, 'U':U, 'S':S, 'VT':VT}
    return data, alpha, problem_instance


# Simulate the input X
def sim_geno(n,m,p,distr="Binomial"): # checked!
    if distr=="Binomial":
        X = np.random.binomial(2, p, size=[n,m])
        X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) # Standardization
    elif distr=="Gaussian":
        # for debugging purposes we simulate a Gaussian matrix and scale it 
        X = np.random.normal(loc=0.0, scale=1.0, size=[n,m]) / np.sqrt(m) # for biological applications: np.sqrt(n)
    return X

# Simulate the coefficients beta
def sim_beta(m, la, sigma, print_results=True): # checked!
    beta = np.random.normal(loc=0.0, scale=np.sqrt(sigma), size=[m,1]) # scale = standard deviation
    if print_results:
        print(f"Variance of slab part of beta: {np.var(beta)}")
    if la == 1:
        return beta
    beta *= np.random.binomial(1, la, size=[m,1])
    return beta

# Simulate laplace-distributed betas
def sim_beta_laplace(m, la, b_scale, print_results=True):
    beta = np.random.laplace(loc=0, scale=b_scale, size=[m,1])
    if print_results:
        print(f"Variance of slab part of beta: {np.var(beta)}")
    if la == 1:
        return beta
    beta *= np.random.binomial(1, la, size=[m,1])
    return beta

def mathematica_evd(n, loc, scale):
    wi = np.random.gumbel(loc=loc, scale=scale, size=[n, 1])
    return wi

# Simulate the outcomes Y
def sim_pheno_Weibull(X, beta, mu, h2, censoring=0, censor_min=0.0, censor_max=1.0):
    [n,m] = X.shape
    g = np.matmul(X, beta)
    sigmaG = np.var(g)
    varwi = np.pi * np.pi / 6
    c = np.sqrt((1/h2-1) * sigmaG / varwi)
    
    wi = -mathematica_evd(n=n, loc=-0, scale=1.0)
    # wi = -np.random.gumbel(loc=0, scale=1.0, size=[n, 1])

    y = np.exp(mu + g + c * (wi + emc) )
    mask = np.ones(n)
    if censoring > 0:
        censor_len = int(censoring * n)
        # Generate random factors between censor_min and censor_max
        random_factors = np.random.uniform(censor_min, censor_max, censor_len)
        # Randomly select indices to censor
        random_indices = np.random.choice(n, censor_len, replace=False)
        # Apply censoring to the selected positions
        y[random_indices, 0] = y[random_indices, 0] * random_factors
        # Create mask vector
        mask[random_indices] = 0
    # An equivalent formulation would be: 
    # y = np.exp(-mathematica_evd(n=n, loc = -(mu+g+c*emc), scale=c))
    alpha = 1.0 / c
    return y, alpha, mask

def sim_pheno_ExpGamma(X, beta, mu, h2, kappa, censoring=0, censor_min=0.0, censor_max=1.0):
    [n,m] = X.shape
    g = np.matmul(X, beta)
    sigmaG = np.var(g)
    sigmaE =  (1/h2-1) * sigmaG
    theta = np.sqrt( sigmaE / scipy.special.polygamma(1, kappa))
    
    digamma_kappa = scipy.special.polygamma(0, kappa)
    w = np.log(np.random.gamma(shape=kappa, scale=1.0, size=[n,1]))
    mut = mu + g - theta * digamma_kappa
    y = np.exp(mut + theta * w)
    
    mask = np.ones(n)
    if censoring > 0:
        censor_len = int(censoring * n)
        # Generate random factors between censor_min and censor_max
        random_factors = np.random.uniform(censor_min, censor_max, censor_len)
        # Randomly select indices to censor
        random_indices = np.random.choice(n, censor_len, replace=False)
        # Apply censoring to the selected positions
        y[random_indices, 0] = y[random_indices, 0] * random_factors
        # Create mask vector
        mask[random_indices] = 0
    
    return y, theta, mask

def sim_pheno_LogNormal(X, beta, mu, h2):
    # logY_i = mu + xi beta + sigma * wi, wi = standard Normal variable
    # beta is mx1 vector 
    # mu is nx1 vector 
    [n,m] = X.shape
    g = np.matmul(X, beta)
    sigmaG = np.var(g)
    sigma = np.sqrt( (1/h2-1) * sigmaG )
    w = np.random.normal(loc=0.0, scale=1.0, size=[n,1])
    y = np.exp(mu + g + sigma * w)
    return y, sigma
    
def sim_model(problem,h2,p, kappa=None, mu=0, censoring=0, censor_min=0.0, censor_max=1.0, distr="Binomial", print_results=True):
    # print(f'sim_model; received mu: {mu}')
    n, m, la, sigma = problem.n, problem.m, problem.prior_instance.la, problem.prior_instance.sigmas[0]
    mu=np.full((n,1), mu) 
    X = sim_geno(n,m,p,distr)
    #X, tot_nans_after, valid_stds = standardize_columnwise(X)
    tot_nans_after, valid_stds = 0,True
    beta = sim_beta(m, la, sigma, print_results)
    if print_results:
        print(f"Simulating data with sigma: {sigma}")
        print(f"Working under the following model: {problem.model}")
    if problem.model == 'Weibull':
        y, alpha, mask = sim_pheno_Weibull(X, beta, mu, h2, censoring, censor_min, censor_max)
        return X, beta, y, alpha,tot_nans_after, valid_stds, mask 
    elif problem.model == 'Gamma':
        return X, beta, sim_pheno_ExpGamma(X, beta, mu, h2, kappa)
    elif problem.model == 'LogNormal':
        return X, beta, sim_pheno_LogNormal(X, beta, mu, h2)
    else:
        raise Exception(problem.model, " is not a valid model. Allowed models are: 'Weibull', 'Gamma' and 'LogNormal'")
    
def sim_model_real_geno(problem, X, h2, kappa=None, mu=0, censoring=0, censor_min=0.0, censor_max=1.0, beta_dist="spike_slab", print_results=True):
    n, m, la, sigma = problem.n, problem.m, problem.prior_instance.la, problem.prior_instance.sigmas[0]
    mu = np.full((n,1), mu)

    if beta_dist == "spike_slab":
        beta = sim_beta(m, la, sigma, print_results)
    elif beta_dist == "laplace":
        b_scale = np.sqrt(h2 / (2 * int(m * la)))
        beta = sim_beta_laplace(m, la, b_scale, print_results)
    else:
        raise Exception(beta_dist, " is not a valid beta distribution")   
    
    if print_results:
        print(f"Simulating data with sigma: {sigma}")
        print(f"Working under the following model: {problem.model}")
    if problem.model == 'Weibull':
        y, alpha, mask = sim_pheno_Weibull(X, beta, mu, h2, censoring, censor_min, censor_max)
        return beta, y, alpha, mask 
    elif problem.model == 'Gamma':
        y, theta, mask = sim_pheno_ExpGamma(X, beta, mu, h2, kappa, censoring, censor_min, censor_max)
        return beta, y, theta, mask
    elif problem.model == 'LogNormal':
        return beta, sim_pheno_LogNormal(X, beta, mu, h2)
    else:
        raise Exception(problem.model, " is not a valid model. Allowed models are: 'Weibull', 'Gamma' and 'LogNormal'")