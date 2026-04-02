import numpy as np
from .em import *

class Problem:
    n = None
    m = None
    model = None
    learn_params = None
    censored = None

    def __init__(self, n=None, m=None, la=None, sigmas=None, omegas=None, model=None, hyperparams=None, **kwargs):
        self.prior_instance = Prior(la, sigmas, omegas, **kwargs)
        self.model = model
        self.n = n
        self.m = m
        self.censored = np.zeros(()) if n is None else np.zeros(n)
        self.hyperparams_instance = hyperparams
    
class Prior:
    la = 0.5
    sigmas = [1]
    omegas = [1]
    # distribution_parameters = {}

    def __init__(self, la, sigmas, omegas, **kwargs):
        self.la = la
        self.sigmas = sigmas
        self.omegas = omegas

    def update_prior(self, r1, gam1):
        r1 = np.asmatrix(r1)
        omegas_old = np.asmatrix(self.omegas)
        sigmas_old = np.asmatrix(self.sigmas)
        sigmas_max = np.max(self.sigmas)
        gam1inv = 1.0/gam1

        beta_tilde=np.multiply( np.exp( - np.power(r1,2) / 2 @ ((sigmas_max - sigmas_old) / (sigmas_max + gam1inv) / (sigmas_old + gam1inv))), self.la * omegas_old / np.sqrt(gam1inv + sigmas_old) )
        sum_beta_tilde = beta_tilde.sum(axis=1)
        beta_tilde=beta_tilde / sum_beta_tilde
        pi = 1.0 / ( 1.0 + ((1-self.la) * np.exp(-np.power(r1,2) / 2 * sigmas_max * gam1 / (sigmas_max + gam1inv) ) / np.sqrt(gam1inv) ) / sum_beta_tilde )
        gamma = np.multiply(r1 * gam1, 1/(gam1 + 1.0/sigmas_old))
        v = 1.0 / (gam1 + 1.0/sigmas_old)

        #updating sparsity level
        self.la = np.mean(pi)
        #updating variances in the mixture
        self.sigmas = np.array(np.transpose(pi) @ np.multiply( beta_tilde , (np.power(gamma,2) + v)) / (np.transpose(pi) @ beta_tilde))[0]
        #updating prior probabilities in the mixture
        self.omegas = np.array((np.transpose(pi) @ beta_tilde ) / np.sum(pi))[0]

class Hyperparams:

    def __init__(self, model, *args):
        # Storing methods to update the hyperparameters
        self.model = model
        if self.model == 'Weibull':
            # mu = args[0], alpha = args[1]
            self.mu = args[0]
            self.alpha = args[1]
            self.mus = [self.mu]
            self.alphas = [self.alpha]
        elif self.model == 'Gamma':
            # mu = args[0], kappa = args[1], theta = args[2]
            self.mu = args[0]
            self.kappa = args[1]
            self.theta = args[2]
        elif self.model == 'LogNormal':
            # mu = args[0], sigma = args[1]
            self.mu = args[0]
            self.sigma = args[1]
        else:
            raise Exception(Problem.model, " is not a valid model. Allowed models are: 'Weibull', 'Gamma' and 'LogNormal'.")
    
    def update(self, **kwargs):
        if self.model == 'Weibull':
            y = kwargs.get('y')
            z_hat = kwargs.get('z_hat')
            xi = kwargs.get('xi')
            it = kwargs.get('it')
            update_preference = kwargs.get('update_preference')
            update_mu, update_alpha, start_at_alpha, start_at_mu = update_preference['update_mu'], update_preference['update_alpha'], update_preference['start_at_alpha'], update_preference['start_at_mu']
            old_mu = self.mu
            new_mu, new_alpha = None, None
            if it > start_at_mu:
                if update_mu:
                    new_mu = update_Weibull_mu(y, z_hat, self.alpha, xi)
                    self.mu = new_mu
            if it > start_at_alpha:
                if update_alpha:
                    new_alpha = update_Weibull_alpha(y, old_mu, z_hat, self.alpha, xi)
                    self.alpha = new_alpha
            
            if not new_mu: self.mus.append(self.mu)
            else: self.mus.append(new_mu)

            if not new_alpha: self.alphas.append(self.alpha)
            else: self.alphas.append(new_alpha)
