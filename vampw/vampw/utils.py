import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sympy
import json
import zarr
import scipy
from lifelines.utils import concordance_index

emc = float( sympy.S.EulerGamma.n(10) )

def save_in_gibbs_format(X, beta, y, mu, la, h2, p, alpha, tot_nans, valid_stds, data_type='real', directory='\'./\'', mask=None):
    if not os.path.exists(f'{directory}'):
        os.makedirs(f'{directory}')
    n,m = X.shape()

    y = np.log(y)
    indices =  np.arange(0,n)
    df = pd.DataFrame({'IID': indices, 'FID': indices, 'y': y.squeeze(-1)})
    df.to_csv(f"{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_.phen", sep=' ', index=None, header=None)
    vector = mask
    # status:
    # censored: 0
    # uncensored: 1
    # missing: -1
    np.savetxt(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_status.fail', vector, fmt='%d')
    zarr.save(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_X.zarr', X)
    np.save(f'{directory}beta.npy', beta)
    z_true = X@beta
    np.save(f'{directory}z_true.npy', z_true)
    # Store variables in a dictionary
    variables = {
        "n": n,
        "m": m,
        "p": p,
        "la": la,
        "h2": h2,
        "data_type": data_type,
        "directory": directory,
        "sigma": "h2/m/la",
        "alpha": alpha,
        "mu": float(mu),
        "Variance of Xb": float(np.var(X@beta)),
        "Variance of b": float(np.var(beta)),
        "Scale of y": "log",
        "nans in X": float(tot_nans),
        "validity of stds": f"Boolean: {valid_stds}"
    }

    # Save dictionary as JSON file
    with open(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_hyperparameters.json', 'w') as json_file:
        json.dump(variables, json_file, indent=4)


def standardize_columnwise(X):
    nan_mask = np.isnan(X)
    tot_nans_before = np.sum(nan_mask)
    print(f'The number of NaNs in the matrix before standardization is: {tot_nans_before}')
    column_means = np.nanmean(X, axis=0)
    column_stds = np.nanstd(X, axis=0)
    valid_stds = not 0 in column_stds
    print(f"Are standard deviations valid? {valid_stds}")
    X = (X - column_means) / column_stds
    nan_mask = np.isnan(X)
    tot_nans_after = np.sum(nan_mask)
    print(f'The number of NaNs in the matrix after standardization is: {tot_nans_after}')
    X = np.nan_to_num(X)
    nan_mask = np.isnan(X)
    tot_nans_after = np.sum(nan_mask)
    print(f'The number of NaNs in the matrix after standardization and nan to num is: {tot_nans_after}')
    return X, tot_nans_after, valid_stds

def calculate_score(x1_hat, beta_true, title="corr(x1_hat, beta_true) = "):
    corr = x1_hat.T @ beta_true / np.linalg.norm(x1_hat) / np.linalg.norm(beta_true)
    print(title, corr[0][0])

def benchmark_gibbs(X, beta_true, directory, betas, progress, iters=100):
    # Load the TSV file into a NumPy array
    coefficients= np.loadtxt( f"{directory}{betas}", delimiter='\t')
    df = pd.read_csv(f"{directory}{progress}", sep='\t')

    average_last_100 = np.mean(coefficients[-iters:], axis=0)
    print(f"Average value of mu over the last {iters} iterations: {np.mean(df['mu'][-100:])}")
    print(f"Average value of alpha over the last {iters} iterations: {np.mean(df['alpha'][-100:])}")

    n, m = X.shape
    calculate_score(average_last_100.reshape(m, 1), beta_true, "alignment beta = ")
    calculate_score(X@average_last_100.reshape(m, 1), X@beta_true,"alignment z = ")
    print(f"l2 error on x: {np.linalg.norm(average_last_100.reshape(m, 1) - beta_true) / np.linalg.norm(beta_true)}")
    print(f"l2 error on z: {np.linalg.norm(X@average_last_100.reshape(m, 1) - X@beta_true) / np.linalg.norm(X@beta_true)}")

def plot_gibbs_small(file_path, title=None):
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')

    # Create a figure with 2 subplots in a 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Set font sizes
    label_size = 18
    legend_size = 18
    global_title_size = 20

    # Plot mu vs iter
    axs[0].plot(df['iter'], df['mu'], label='mu', color='b')
    axs[0].set_xlabel('iter', fontsize=label_size)
    axs[0].set_ylabel('mu', fontsize=label_size)
    axs[0].legend(fontsize=legend_size)

    # Plot alpha vs iter
    axs[1].plot(df['iter'], df['alpha'], label='alpha', color='g')
    axs[1].set_xlabel('iter', fontsize=label_size)
    axs[1].set_ylabel('alpha', fontsize=label_size)
    axs[1].legend(fontsize=legend_size)

    # Global title
    if title: fig.suptitle(title, fontsize=global_title_size, y=1.05)

    # Adjust layout and increase overall font size
    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()
    plt.show()


def plot_gibbs(file_path):
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    label_size = 18
    legend_size = 14
    global_title_size = 20
    title_size = 18

    # Create a figure with 4 subplots in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot mu vs iter
    axs[0, 0].plot(df['iter'], df['mu'], label='mu', color='b')
    axs[0, 0].set_ylabel('mu', fontsize=label_size)
    axs[0, 0].set_title('mu vs iter', fontsize=title_size)
    axs[0, 0].legend(fontsize=legend_size)

    # Plot alpha vs iter
    axs[0, 1].plot(df['iter'], df['alpha'], label='alpha', color='g')
    axs[0, 1].set_ylabel('alpha', fontsize=label_size)
    axs[0, 1].set_title('alpha vs iter', fontsize=title_size)
    axs[0, 1].legend(fontsize=legend_size)

    # Plot h2 vs iter
    axs[1, 0].plot(df['iter'], df['h2'], label='h2', color='r')
    axs[1, 0].set_ylabel('h2', fontsize=label_size)
    axs[1, 0].set_title('h2 vs iter', fontsize=title_size)
    axs[1, 0].legend(fontsize=legend_size)

    # Plot num_markers vs iter
    axs[1, 1].plot(df['iter'], df['num_markers'], label='num_markers', color='m')
    axs[1, 1].set_xlabel('iter', fontsize=label_size)
    axs[1, 1].set_ylabel('num_markers', fontsize=label_size)
    axs[1, 1].set_title('num_markers vs iter', fontsize=title_size)
    axs[1, 1].legend(fontsize=legend_size)

    # Set a global title if needed
    fig.suptitle('Gibbs Sampling Plots', fontsize=global_title_size)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def pickle_summary(path='real_vamp_em_15k_h2_09_la_005.pkl'):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    print(f"Summaries for the file: {path}\n")
    print_summaries(data['mus'], data['alphas'], data['corrs_x'], data['corrs_z'], data['l2_errs_x'], data['l2_errs_z'])

def print_summaries(mus, alphas, corrs_x, corrs_z, l2_errs_x, l2_errs_z):
    print(f"Final mu: {mus[-1]}")
    print(f"Final alpha: {alphas[-1]}")
    print(f"Final correlation of x: {corrs_x[-1]}")
    print(f"Final correlation of z: {corrs_z[-1]}")
    print(f"Final l2 error of x: {l2_errs_x[-1]}")
    print(f"Final l2 error of z: {l2_errs_z[-1]}")

def plot_metrics(corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, a, p, correct_mu, correct_alpha, n, m, r1s, gam1s, beta_true, p1s, tau1s, z_true, 
                 title=None, plot_x2_hat=False, corrs_x2=None, l2_errs_x2=None):
    plt.figure(figsize=(20, 10))
    
    nrows = 4
    ncols = 4

    if not title:
        title = f"Working with {n}x{m} matrix"
    # Overall title
    plt.suptitle(title, fontsize=26)

    # Plotting corr_x
    plt.subplot(nrows, ncols, 1)
    plt.ylabel('corr_x')
    plt.plot(range(len(corrs_x)), corrs_x, 'ro-')

    # Plotting l2_err_x
    plt.subplot(nrows, ncols, 2)
    plt.ylabel('l2_err_x')
    plt.plot(range(len(l2_errs_x)), l2_errs_x, 'ro-')

    # Plotting corr_z
    plt.subplot(nrows, ncols, 3)
    plt.ylabel('corr_z')
    plt.plot(range(len(corrs_z)), corrs_z, 'bo-')

    # Plotting l2_err_z
    plt.subplot(nrows, ncols, 4)
    plt.ylabel('l2_err_z')
    plt.plot(range(len(l2_errs_z)), l2_errs_z, 'bo-')

    # Plotting mu evolution
    plt.subplot(nrows, ncols, 5)
    plt.ylabel('mu')
    plt.plot(range(len(mus)), mus, 'go-')
    plt.axhline(y=correct_mu, color='r', linestyle='--', label='Correct mu')
    plt.legend()

    # Plotting alpha evolution
    plt.subplot(nrows, ncols, 6)
    plt.ylabel('alpha')
    plt.plot(range(len(alphas)), alphas, 'go-')
    plt.axhline(y=correct_alpha, color='r', linestyle='--', label='Correct alpha')
    plt.legend()

    # Plotting Actual vs Predicted Scatter Plot in the last cell
    plt.subplot(nrows, ncols, 7)
    indices = range(len(a))
    plt.scatter(indices, a, color='blue', label='Actual')
    plt.scatter(indices, p, color='red', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted Scatter Plot')
    plt.legend()

    # QQ plot to inspect normallity of (r1 - beta) * sqrt(gam1)
    plt.subplot(nrows, ncols, 8)
    # print(r1s[-1].reshape(1,-1)[0])
    scipy.stats.probplot(((r1s[-1]-beta_true)*np.sqrt(gam1s[-1])).reshape(1,-1)[0], dist="norm", plot=plt)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Probability plot for (r1-beta)*sqrt(gam1)')

    # QQ plot to inspect normallity of (p1 - z) * sqrt(tau1)
    plt.subplot(nrows, ncols, 9)
    # print(r1s[-1].reshape(1,-1)[0])
    scipy.stats.probplot(((p1s[-1]-z_true)*np.sqrt(tau1s[-1])).reshape(1,-1)[0], dist="norm", plot=plt)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Probability plot for (p1-z)*sqrt(tau1)')
    
    if plot_x2_hat:
        # Plotting corr_x2
        plt.subplot(nrows, ncols, 10)
        plt.ylabel('corr_x2')
        plt.plot(range(len(corrs_x2)), corrs_x2, 'ro-')

        # Plotting l2_err_x2
        plt.subplot(nrows, ncols, 11)
        plt.ylabel('l2_err_x2')
        plt.plot(range(len(l2_errs_x2)), l2_errs_x2, 'ro-')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the suptitle
    
    # Create output directory if it does not exist
    # output_dir = './ignored/outputs/figures'
    # os.makedirs(output_dir, exist_ok=True)

    # Generate filename with current date and time
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # filename = f"{output_dir}/plot_{current_time}.png"

    # Save the figure
    # plt.savefig(filename)
    # plt.show()
    
def plot_normallity_inspect_iterations_x(r1s, gam1s, beta_true):
    plt.figure(figsize=(20, 10))
    
    iter_num = gam1s.shape[0] - 1
    ncols = 4
    nrows = int(np.ceil(iter_num / ncols))
    plt.suptitle("Inspect Normality over Iterations X", fontsize=26)
    
    for i in range(1, iter_num+1):
        plt.subplot(nrows, ncols, i)
        scipy.stats.probplot(((r1s[i]-beta_true)*np.sqrt(gam1s[i])).reshape(1,-1)[0], dist="norm", plot=plt)
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(f'Probability plot for (r1-beta)*sqrt(gam1) -- iteration {i}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
def plot_normallity_inspect_iterations_z(p1s, tau1s, z_true):
    plt.figure(figsize=(20, 10))
    
    iter_num = tau1s.shape[0] - 1
    ncols = 4
    nrows = int(np.ceil(iter_num / ncols))
    plt.suptitle("Inspect Normality over Iterations Z", fontsize=26)
    
    for i in range(1, iter_num+1):
        plt.subplot(nrows, ncols, i)
        scipy.stats.probplot(((p1s[i]-z_true)*np.sqrt(tau1s[i])).reshape(1,-1)[0], dist="norm", plot=plt)
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(f'Probability plot for (p1-z)*sqrt(tau1) -- iteration {i}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_metrics_poster(l2_errs_x, l2_errs_z, mus, alphas, correct_mu, correct_alpha, title=None):
    plt.figure(figsize=(10, 6))
    # Set font sizes
    label_size = 18
    legend_size = 14
    global_title_size = 20
    title_size = 18
    if title: plt.suptitle(title, fontsize=global_title_size, y=0.98)


    # Plotting l2_err_x
    plt.subplot(2, 2, 1)
    plt.title('L2 Error Beta', fontsize=title_size)
    plt.ylabel('Error', fontsize=label_size)
    plt.plot(range(len(l2_errs_x)), l2_errs_x, 'ro-')

    # Plotting l2_err_z
    plt.subplot(2, 2, 2)
    plt.title('L2 Error Z', fontsize=title_size)
    plt.ylabel('Error', fontsize=label_size)
    plt.plot(range(len(l2_errs_z)), l2_errs_z, 'bo-')

    # Plotting mu evolution
    plt.subplot(2, 2, 3)
    plt.title('Mu Evolution', fontsize=title_size)
    plt.ylabel('mu', fontsize=label_size)
    plt.plot(range(len(mus)), mus, 'go-')
    plt.axhline(y=correct_mu, color='r', linestyle='--', label='Correct mu')
    plt.legend(fontsize=legend_size)

    # Plotting alpha evolution
    plt.subplot(2, 2, 4)
    plt.title('Alpha Evolution', fontsize=title_size)
    plt.ylabel('alpha', fontsize=label_size)
    plt.plot(range(len(alphas)), alphas, 'go-')
    plt.axhline(y=correct_alpha, color='r', linestyle='--', label='Correct alpha')
    plt.legend(fontsize=legend_size)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the suptitle
    
    # Create output directory if it does not exist
    output_dir = './ignored/outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{output_dir}/plot_{current_time}.png"

    # Save the figure
    plt.savefig(filename)
    plt.show()

def dampen(x1_hat, x1_hat_new, damp=0.9):
    if x1_hat is None: ans = x1_hat_new
    else: ans = x1_hat*(1-damp) + x1_hat_new*damp
    return ans

def cindex(x1_hat, mu, data):
    A = np.concatenate((data["X"], data["X_c"]))
    y = np.concatenate((data["y"], data["y_c"]))
    event = np.concatenate((np.ones(len(data["y"])), np.zeros(len(data["y_c"]))))
    y_hat = np.exp(mu + A @ x1_hat)
    c_index = concordance_index(
        event_times=y,
        predicted_scores=y_hat, 
        event_observed=event
    )
    return c_index

def save_results(output_dir, out_name, n, m, **kwargs):
    print("Saving results!!\n\n\n\n")
    """
    Svave results as a pickle file in the specified output directory.

    Parameters:
    output_dir (str): Directory where the results should be saved.
    **kwargs: Results to be saved, passed as keyword arguments.
    """
    output_filepath = os.path.join(output_dir, f'{out_name}.pkl')

    # Save the results dictionary as a pickle file
    with open(output_filepath, 'wb') as f:
        pickle.dump(kwargs, f)