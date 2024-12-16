import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from timeit import timeit
from distributions_sampling import *

# Optimized Sampling Function
def sample_f_xy_optimized(n_samples, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_interp, h_s_interp, g_b_interp, h_b_interp):
    samples_x, samples_y = [], []
    n_batch = 10 * n_samples
    while len(samples_x) < n_samples:
        is_signal = np.random.rand(n_batch) < f_signal
        x_candidates = np.random.uniform(0, 5, n_batch)
        y_candidates = np.random.uniform(0, 10, n_batch)
        if is_signal.sum() > 0:
            signal_acceptance = g_s_interp(x_candidates[is_signal]) * h_s_interp(y_candidates[is_signal])
            accepted_signal = np.random.rand(is_signal.sum()) < signal_acceptance
            samples_x.extend(x_candidates[is_signal][accepted_signal])
            samples_y.extend(y_candidates[is_signal][accepted_signal])
        background_mask = ~is_signal
        if background_mask.sum() > 0:
            background_acceptance = g_b_interp(x_candidates[background_mask]) * h_b_interp(y_candidates[background_mask])
            accepted_background = np.random.rand(background_mask.sum()) < background_acceptance
            samples_x.extend(x_candidates[background_mask][accepted_background])
            samples_y.extend(y_candidates[background_mask][accepted_background])
    return np.array(samples_x[:n_samples]), np.array(samples_y[:n_samples])

# Parallel Sampling Function
def parallel_sample_f_xy_optimized(n_samples, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_interp, h_s_interp, g_b_interp, h_b_interp):
    n_processes = cpu_count()
    batch_size = n_samples // n_processes
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(
            sample_f_xy_optimized,
            [(batch_size, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_interp, h_s_interp, g_b_interp, h_b_interp)] * n_processes
        )
    samples_x = np.concatenate([r[0] for r in results])
    samples_y = np.concatenate([r[1] for r in results])
    return samples_x, samples_y

# Extended Likelihood
def extended_likelihood(params, x_data, y_data, n_total):
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, n_extended = params
    f_values = np.array([f_xy(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal) for x, y in zip(x_data, y_data)])
    return -(n_total * np.log(n_extended) - n_extended + np.sum(np.log(f_values)))

# Perform Fit
def perform_fit(x_data, y_data, n_total):
    initial_params = [3.1, 0.35, 0.9, 1.5, 0.28, -0.2, 2.8, 0.65, n_total]
    bounds = [(2.5, 3.5), (0.1, 0.5), (0.5, 2), (1.1, 2), (0.1, 0.5), (-1, 1), (1, 4), (0.3, 0.8), (n_total * 0.9, n_total * 1.1)]
    result = minimize(extended_likelihood, initial_params, args=(x_data, y_data, n_total), bounds=bounds, method="L-BFGS-B")
    print("Fit Params:", result.x)

# Main Execution Block
if __name__ == "__main__":
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = 3, 0.3, 1, 1.4, 0.3, 0, 2.5, 0.6
    n_samples = 100000
    g_s_interp, h_s_interp, g_b_interp, h_b_interp = precompute_marginals(mu, sigma, beta, m, lambda_s, mu_b, sigma_b)
    x_data, y_data = parallel_sample_f_xy_optimized(n_samples, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_interp, h_s_interp, g_b_interp, h_b_interp)
    perform_fit(x_data, y_data, n_samples)
