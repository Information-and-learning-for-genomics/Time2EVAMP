# vampW
Python implementation of Time To Event vampW.

We consider a Weibull $(\alpha, \eta)$ model, one of the most common models in survival analysis, we formulate the final model as
$$\log Y_i =\mu + x_i^T \beta + \frac{w_i}{\alpha} + \frac{K}{\alpha},$$
where $w_i$ is Gumbel $(0,1)$ distributed.

To allow for a range of effect sizes from different data types, we select the prior on the marker effects $\beta$ to be distributed as a form of spike and slab distribution, with non-zero signals coming from a mixture of normal distributions,
$$\beta_j \sim ( 1 - \lambda ) \delta_0 + \lambda \sum^L_{i=1} \pi_i \cdot \mathcal{N}(0, \sigma^2_i),$$
where $\pi_i$ is the probability of including the $i$-th mixture out of a total of $L$ components, $\delta_0$ is a dirac spike function at 0, $\lambda$ is the sparsity ratio, and $\{\sigma^2_1, \dots, \sigma^2_L\}$ are mixture specific variances.

# Installation
In order to install the vampW python library, navigate to *vampw* directory, and install the module by:
```
pip install .
```

# Example
Example jupyter notebook can be found in *example* directory. 


## Input Parameters
The primary entry point is the `infere` function, which accepts a `pref` (preferences) dictionary alongside `maxiter` (default values is 10) which describe the number of vampW iterations. Below are the available configuration parameters:

| Parameter Name | Interpretation / Meaning | Default Value |
| :--- | :--- | :--- |
| **mode** | Computational backbone: 'svd', 'congrad' (using conjugate gradients), or 'censoring' (used when some datapoints are censored) | 'svd' |
| **gam1** | Initial precision (inverse variance) for the $\beta$ coefficients | 1e-4 |
| **tau1** | Initial precision for the linear predictors $z = X \beta$ | 1e-1 |
| **dampen_coeff** | Static damping factor to stabilize approximate message passing (values smaller or equal 1.0). | 1.0 |
| **dampen_wu_it** | Iteration index to start adaptive damping. Set to -1 to disable | -1 |
| **dampen_scale** | Factor by which dampen_coeff is reduced during adaptive damping | 1.0 |
| **dampen_autotune** | Maximum number of internal search iterations for adaptive damping | None |
| **num_hutchinson_samples** | Number of samples for Hutchinson trace estimation (used in 'censoring') | -1 (Exact) |
| **print_results** | Boolean flag to print iteration-wise metrics to the console | False |
| **save** | Boolean flag to export results to the output directory | False |
| **out_dir** | Path to the directory where results will be saved | "outputs" |

In addition to this options, one can specify update preference options for the expectation-maximization (EM) updates of hyperparameters. Namly, The `update_preference` dictionary controls the behavior of the **M-step** for the Weibull model. Below are the available options and their recommended default values:

| Option | Type  | Description |
| :--- | :--- | :--- |
| `update_mu` | `bool` | Enables/Disables the update of the scale parameter ($\mu$) |
| `update_alpha` | `bool` | Enables/Disables the update of the shape parameter ($\alpha$) |
| `start_at_mu` | `int` | The EM iteration index to begin calculating new $\mu$ values |
| `start_at_alpha` | `int` | The EM iteration index to begin calculating new $\alpha$ values |

---


### **`infer()` Function Output**

The `infer` function returns a dictionary containing the estimated values, optimized prior parameters, and performance metrics collected across the EM iterations.

| Key | Type | Description |
| :--- | :--- | :--- |
| **`x1_hats`** | `list[np.array]` | Trajectory of the estimated coefficients (posterior mean of $\beta$). |
| **`gam1s`** | `list[float]` | Evolution of the precision (inverse variance) of $\beta$. |
| **`tau1s`** | `list[float]` | Evolution of the precision (inverse variance) of $z$. |
| **`corrs_x`** | `list[float]` | Correlation between the estimated $\hat{\beta}$ and the ground truth. |
| **`z1_hats`** | `list[np.array]` | Trajectory of the latent survival variables $\hat{z}$. |
| **`r1s`** | `list[np.array]` | Evolution of the noisy vector of signal $r_1$. |
| **`p1s`** | `list[np.array]` | Evolution of the noisy vector of signal $p_1$. |
| **`l2_errs_x`** | `list[float]` | $L_2$ error between estimated coefficients and the ground truth. |
| **`lambdas`** | `list[float]` | History of the sparsity level ($\lambda$) optimized in the `Prior` class. |
| **`sigmas`** | `list[np.array]` | History of the mixture variances in the Spike & Slab prior. |
| **`omegas`** | `list[np.array]` | History of the prior probabilities for the mixture components. |

---



