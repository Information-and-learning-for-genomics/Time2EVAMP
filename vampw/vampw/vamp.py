import numpy as np
from .denoisers import *
from .utils import *
from .censoring import *
import sympy
from scipy.sparse.linalg import cg as con_grad
from scipy.optimize import minimize
from numpy.random import binomial
import random  # This ensures you're using Python's built-in random module.
from datetime import datetime
import warnings
import time
warnings.filterwarnings("ignore", category=RuntimeWarning) 

emc = float(sympy.S.EulerGamma.n(10))

def infere_and_visualize(data, problem, pref, seed=42, maxiter=10, h2_title=None, distr="Binomial"):
    inf_out = infere(data, problem, pref, seed, maxiter)
    corrs_x = inf_out['corrs_x']
    l2_errs_x = inf_out['l2_errs_x']
    corrs_z = inf_out['corrs_z']
    l2_errs_z = inf_out['l2_errs_z']
    actual_xis = inf_out['actual_xis']
    predicted_xis = inf_out['predicted_xis']
    r1s = inf_out['r1s']
    gam1s = inf_out['gam1s']
    p1s = inf_out["p1s"]
    tau1s = inf_out["tau1s"]
    z_true = data["X"] @ data["beta_true"]

    mus, alphas = problem.hyperparams_instance.mus, problem.hyperparams_instance.alphas
    n, m = data["X"].shape
    prior = problem.prior_instance
    plot_metrics(corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, actual_xis, predicted_xis, mus[-1], alphas[-1], \
                 n, m, r1s, gam1s, data["beta_true"], p1s, tau1s, z_true, \
                 f"{n}x{m}, heritability: {h2_title}, X distr: " + distr + f", sparsity: {1-prior.la}; mode: {pref['mode']}; dampen_coeff: {pref['dampen_coeff']}")     
    print_summaries(mus, alphas, corrs_x, corrs_z, l2_errs_x, l2_errs_z)

def infere(data, problem, pref, seed=42, maxiter=10, load_svd=False, censoring_fsolve_tol=1e-8, print_fsolve_state=False):

    np.random.seed(seed)

    X = data["X"]
    y = data["y"]
    n,m = X.shape
    is_synthetic = "beta_true" in data.keys()
    beta_true = data["beta_true"] if is_synthetic else None

    # fetching the preferences about the inference process
    mode = pref["mode"] if pref["mode"] is not None else 'svd'
    dampen_coeff = pref["dampen_coeff"] if pref["dampen_coeff"] is not None else 1.0
    adaptive_dampen = False if pref["dampen_wu_it"] is -1 else True
    dampen_wu_it = pref["dampen_wu_it"]
    dampen_autotune = pref["dampen_autotune"]
    dampen_scale = pref["dampen_scale"] if pref["dampen_scale"] is not None else 1.0
    save = pref["save"] if pref["save"] is not None else False
    print_results = pref["print_results"] if pref["print_results"] is not None else False
    if "out_dir" not in pref.keys():
        pref["out_dir"] = "outputs"
    if "tau1" not in pref.keys():
        pref["tau1"] = 1e-1
    if "gam1" not in pref.keys():
        pref["gam1"] = 1e-4
    if "denoiser_type" not in pref.keys():
        pref["denoiser_type"] = "map"
    if "num_hutchinson_samples" not in pref.keys():
        num_hutchinson_samples = -1
    else:
        num_hutchinson_samples = int(pref["num_hutchinson_samples"])

    # starting values of the precisions
    denoiser_type = pref["denoiser_type"]
    gam1=pref["gam1"]        
    tau1=pref["tau1"]
    if print_results:
        print(f"gam1 = {gam1}")
        print(f"tau1 = {tau1}")

    # getting hyperparameters of the model (currnetly only for Weibull)
    mu, alpha = problem.hyperparams_instance.mus[0], problem.hyperparams_instance.alphas[0]

    r1 = np.zeros((m,1))
    p1 = np.zeros((n,1)) 

    ci_train_prev = 0.0
    
    [n,m] = X.shape
    if mode=='svd': 
        #computing SVD decomposition of X
        if print_results:
            print("performing SVD decoposition..")
        if not load_svd:
            u, s, vh = np.linalg.svd(X, full_matrices=False)
        else:
            u, s, vh = data['U'], data['S'], data['VT']
    elif mode=='congrad': 
        corr_uncensored = X.T@X
        Sigma2_u_prev = np.zeros((m,1))
        x2_hat_prev = np.zeros((m,1))
    elif mode=='censoring':
        X_c = data["X_c"]
        y_c = data["y_c"]
        
    
    if is_synthetic: Xbeta_true = X @ beta_true

    #storing measure of recovery quality
    l2_errs_x = []
    l2_errs_x2 = []
    corrs_x = []
    corrs_x2 = []
    l2_errs_z = []
    corrs_z = []
    actual_xis = []
    predicted_xis = []
    z1_hats = []
    x1_hats = []
    x2_hats = []
    gam1s = [gam1]
    tau1s = [tau1]
    r1s = [r1]
    p1s = [p1]
    lams = [problem.prior_instance.la]
    sigmas = [problem.prior_instance.sigmas]
    omegas = [problem.prior_instance.omegas]
    
    for it in range(maxiter):
        start_time = time.time()
        if print_results:
            print("**** iteration = ", it, " **** \n" )
            print("->DENOISING")
        # Conditional expectation of x given r and the parameters of the prior distribution of x
        # This is applied elementwise to r1
        x1_hat_new = denoiser_beta(r1, gam1, problem)
        if it == 0: x1_hat = x1_hat_new
        x1_hat_damped = dampen(x1_hat, x1_hat_new, dampen_coeff)
        ci_train = cindex(x1_hat_damped, mu, data)
        print("C index =", ci_train, flush=True)
        if adaptive_dampen:
            if (it + 1) >= dampen_wu_it:
                print("Adaptive damping...", flush=True)
                i = 0
                dampen_coeff_new = dampen_coeff
                while (ci_train < ci_train_prev) and (i < dampen_autotune):
                    print(f"Adaptive damping iteration {i}...", flush=True)
                    dampen_coeff_new = dampen_coeff_new * dampen_scale
                    print("dampen =", dampen_coeff_new, flush=True)
                    x1_hat_damped = dampen(x1_hat, x1_hat_new, dampen_coeff_new)
                    ci_train = cindex(x1_hat_damped, mu, data)
                    print("C index =", ci_train, flush=True)
                    i += 1
        x1_hat = x1_hat_damped
        ci_train_prev = ci_train
        x1_hats.append(x1_hat)
        if print_results: print("x1_hat[2] = ", x1_hat[2])
        
        ############################################################
        if is_synthetic:
            if np.linalg.norm(x1_hat) != 0:
                # Cosine similarity
                # Note that this is not exactly a correlation. Instead this is an alignment score
                corr = np.dot(x1_hat.transpose(), beta_true) / np.linalg.norm(x1_hat) / np.linalg.norm(beta_true)
            else:
                corr = np.dot(x1_hat.transpose(), beta_true)
            l2_err = np.linalg.norm(x1_hat - beta_true) / np.linalg.norm(beta_true)
            if print_results: 
                print("corr(x1_hat, beta_true) = ", corr[0][0])
                print("l2 error for x1_hat = ", l2_err)
            corrs_x.append(corr[0][0])
            l2_errs_x.append(l2_err)
        ############################################################
        
        alpha1_new = np.mean(der_denoiser_beta(r1, gam1, problem))
        if it == 0: alpha1 = alpha1_new
        alpha1 = dampen(alpha1, alpha1_new, dampen_coeff)
        gam2 = gam1 * (1-alpha1) / alpha1
        r2 = (x1_hat - alpha1 * r1) / (1-alpha1)
        if print_results:
            print("alpha1_new = ", alpha1_new)
            print("alpha1 = ", alpha1)
            if is_synthetic: print("true gam2 = ", 1.0 / np.var(r2 - beta_true))
            print("gam2 = ", gam2)
        # Denoising z (the genetic predictor)
        z1_hat = den_z(p1, tau1, y, problem, denoiser_type)
        z1_hats.append(z1_hat)
        
        ############################################################
        # Cosine similarity
        # This is an alignment score, not correlation
        if is_synthetic:
            corr = np.dot(z1_hat.transpose(), Xbeta_true) / np.linalg.norm(z1_hat) / np.linalg.norm(Xbeta_true) 
            corrs_z.append(corr[0][0])
            l2_err = np.linalg.norm(z1_hat - Xbeta_true) / np.linalg.norm(Xbeta_true)
            if print_results:
                print("corr(z1_hat, X*beta_true) = ", corr[0][0])
                print("l2 error for z1_hat = ", l2_err)
            l2_errs_z.append(l2_err)
        ############################################################
        
        beta_1 = np.mean(der_den_z(p1, tau1, y, problem, denoiser_type))
        tau2 = tau1 * (1-beta_1) / beta_1
        p2 = (z1_hat - beta_1 * p1) / (1-beta_1)
        if print_results:
            print("v1 = ", beta_1)
            if is_synthetic: print("true tau2 = ", 1.0 / np.var(p2 - Xbeta_true))
            print("tau2 =", tau2)

        predicted_xi = tau1 / beta_1
        predicted_xis.append(predicted_xi)
        if is_synthetic:
            actual_xi = 1 / np.var(X@beta_true-z1_hat)
            actual_xis.append(actual_xi)
        
        # Update Weibull model's parameters
        problem.hyperparams_instance.update(y=y, z_hat=z1_hat, xi=predicted_xi, it=it, update_preference=pref)
        mu, alpha = problem.hyperparams_instance.mu, problem.hyperparams_instance.alpha
        print(f"Updated mu = {mu} -- update alpha = {alpha}", flush=True)
        
        # LMMSE estimation of x
        if print_results: print("->LMMSE")
        if mode=='svd':
            dk = 1.0 / (tau2 * s * s + gam2)
            x2_hat = vh.transpose() @ np.diag(dk) @ (tau2 * np.diag(s).transpose() @ u.transpose() @ p2 + gam2 * vh @ r2)
            alpha2 = np.sum( gam2 / (tau2 * s * s + gam2) ) / m
            beta2 = (1-alpha2) * m / n

        elif mode=='congrad':
            # Conjugate gradient solver
            # We are solving the system A2x2 = y2;
            # Note: X^T @ X is unavoidable as it is the summary statistic used in sgVAMP
            # This requires 2.8 TB of RAM for 800k uk biobank
            A2 = tau2*corr_uncensored + gam2*np.eye(X.shape[1])
            y2 = tau2*X.transpose()@p2 + gam2*r2
            x2_hat, ret = con_grad(A2, y2, maxiter=500, x0=x2_hat_prev)
            # x2_hat.resize((m,1))
            x2_hat_prev = x2_hat
            if ret > 0: 
                print(f"WARNING: CG 1 convergence after {ret} iterations not achieved!")
            x2_hat.resize((m,1))
            u = binomial(p=1/2, n=1, size=m) * 2 - 1
            # Hutchinson trace estimator
            # Sigma2 = (gamw * R + gam2 * I)^(-1)
            # Conjugate gradient for solving linear system (gamw * R + gam2 * I)^(-1) @ u
            Sigma2_u, ret = con_grad(A2,u, maxiter=500, x0=Sigma2_u_prev)
            Sigma2_u_prev = Sigma2_u
            if ret > 0: 
                print(f"WARNING: CG 2 convergence after {ret} iterations not achieved!")
            TrSigma2 = u.T @ Sigma2_u # Tr[Sigma2] = u^T @ Sigma2 @ u 
            alpha2 = gam2 * TrSigma2 / m
            beta2 = (1-alpha2) * m / n

        elif mode=='censoring':
            initial_guess, _ = con_grad(gam2 * np.eye(m) + tau2 * X.T@X, tau2 * X.T @ p2 + gam2 * r2, maxiter=500, x0=np.zeros(m))
            x2_hat, info, status, message = scipy.optimize.fsolve(censoring_LMMSE_loss_Weibull, x0=initial_guess, args=(gam2, r2, tau2, X, p2, mu, alpha, X_c, y_c),\
            full_output=1 , xtol=censoring_fsolve_tol)
            if print_fsolve_state:
                print(f"fsolve status : {status}")
                print(f"fsolve message : {message}", flush=True)
            x2_hat.resize((m,1))
            x2_hats.append(x2_hat)
            if print_results: print(f"Norm of x2_hat = {np.linalg.norm(x2_hat)}", flush=True)
            ###########################################################################
            if is_synthetic:
                corr = np.dot(x2_hat.transpose(), beta_true) / np.linalg.norm(x2_hat) / np.linalg.norm(beta_true)
                l2_err_x2 = np.linalg.norm(x2_hat - beta_true) / np.linalg.norm(beta_true)
                corrs_x2.append(corr[0][0])
                l2_errs_x2.append(l2_err_x2)
                if print_results:
                    print("corr(x2_hat, beta_true) = ", corr[0][0])
                    print("l2 error for x2_hat = ", l2_err)
            ##########################################################################
                
            LMMSE_jacob_censoring = censoring_LMMSE_loss_Weibull_grad(x2_hat, gam2, r2, tau2, X, p2, mu, alpha, X_c, y_c)
            if num_hutchinson_samples > 0:
                alpha2, beta2 = 0.0, 0.0
                for ind_sample in range(num_hutchinson_samples):
                    u = binomial(p=1/2, n=1, size=m) * 2 - 1
                    inv_LMMSE_jacob_censoring_vec, _ = con_grad(LMMSE_jacob_censoring,u, maxiter=500, x0=np.zeros(m))
                    alpha2 += 2 * gam2 * np.dot(u, inv_LMMSE_jacob_censoring_vec) / m
                    u = binomial(p=1/2, n=1, size=n) * 2 - 1
                    inv_X_LMMSE_jacob_censoring_XT_vec, _ = con_grad(X @ LMMSE_jacob_censoring @ X.T,u, maxiter=500, x0=np.zeros(n))
                    beta2 += 2 * tau2 * np.dot(u, inv_X_LMMSE_jacob_censoring_XT_vec)/ n
                alpha2 /= num_hutchinson_samples
                beta2 /= num_hutchinson_samples
            else:
                inv_LMMSE_jacob_censoring = np.linalg.inv(LMMSE_jacob_censoring)
                alpha2 = 2 * gam2 * np.trace( inv_LMMSE_jacob_censoring ) / m
                beta2 = 2 * tau2 * np.trace( X @ inv_LMMSE_jacob_censoring @ X.T ) / n
            print(f"alpha2 = {alpha2}")
            print(f"beta2 = {beta2}")

        
        gam1 = gam2 * (1-alpha2) / alpha2
        gam1s.append(gam1)
        r1 = (x2_hat - alpha2 * r2) / (1-alpha2)
        r1s.append(r1)
        if print_results:
            print("alpha2 = ", alpha2)
            if is_synthetic: print("true gam1 = ", 1.0 / np.var(r1 - beta_true))
            print("gam1 = ", gam1)
        
        # LMMSE estimation of z
        z2_hat = np.matmul(X, x2_hat)
        tau1 = tau2 * (1-beta2) / beta2
        tau1s.append(tau1)
        p1 = (z2_hat - beta2 * p2) / (1-beta2)
        p1s.append(p1)
        if print_results:
            if is_synthetic:
                print("corr(z2_hat, beta_true) = ", np.dot(z2_hat.transpose(), Xbeta_true) / np.linalg.norm(z2_hat) / np.linalg.norm(Xbeta_true))
                print("l2 error for z2_hat = ", np.linalg.norm(z2_hat - Xbeta_true) / np.linalg.norm(Xbeta_true))
                print("true tau1 = ", 1.0 / np.var(p1 - Xbeta_true))
            print("tau1 = ", tau1)
            print("\n")
        
        # EM update function of the prior distribution parameters (sigmas, omegas, and lambda)
        problem.prior_instance.update_prior(r1, gam1)
        lams.append(problem.prior_instance.la)
        sigmas.append(problem.prior_instance.sigmas)
        omegas.append(problem.prior_instance.omegas)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if print_results:
            print("-> EM update of the prior distribution parameters")
            print(f"lambda:{problem.prior_instance.la} -- sigmas: {problem.prior_instance.sigmas} -- omegas: {problem.prior_instance.omegas}", flush=True)
            print(f"Iteration {it} took {elapsed_time:.1f} seconds", flush=True)


    if save:
        save_results(pref['out_dir'], 
                    n,
                    m,
                    x1_hat=x1_hat, 
                    gam1=gam1, 
                    corrs_x=corrs_x, 
                    l2_errs_x=l2_errs_x, 
                    corrs_z=corrs_z, 
                    l2_errs_z=l2_errs_z, 
                    mus=problem.hyperparams_instance.mus, 
                    alphas=problem.hyperparams_instance.alphas, 
                    actual_xis=actual_xis, 
                    predicted_xis=predicted_xis, 
                    z1_hats=z1_hats, 
                    x1_hats=x1_hats)
    
    output = {'x1_hats': x1_hats, 'gam1s': gam1s, 'tau1s': tau1s, 'corrs_x': corrs_x, 'corrs_z': corrs_z, \
              'actual_xis': actual_xis, 'predicted_xis': predicted_xis, 'z1_hats': z1_hats, \
              'r1s': r1s, 'p1s':p1s, 'l2_errs_x': l2_errs_x, 'l2_errs_z': l2_errs_z, 'l2_errs_x2': l2_errs_x2, \
              'x2_hats': x2_hats, 'corrs_x2': corrs_x2, 'lambdas': lams, 'sigmas': sigmas, 'omegas': omegas}

    return output