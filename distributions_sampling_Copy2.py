
from numba import jit
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import trapezoid

# def parallel_execute(func, args_list, n_jobs=8):
#     """
#     General-purpose parallel executor for functions with arguments.
#     - func: The function to execute.
#     - args_list: A list of argument tuples for the function.
#     - n_jobs: Number of parallel jobs.
#     """
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(func)(*args) for args in args_list
#     )
#     return results


@jit(nopython=True)
def crystal_ball_pdf_vectorized(x, mu, sigma, beta, m):
    z = (x - mu) / sigma
    core = np.exp(-z**2 / 2)
    A = (m / beta) ** m * np.exp(-beta**2 / 2)
    B = m / beta - beta
    tail = A * (B - z) ** -m
    pdf = np.where(z > -beta, core, tail)
    return pdf

@jit(nopython=True)
def crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5, n_samples=1000):
    x_vals = np.linspace(lower, upper, n_samples)
    pdf_vals = crystal_ball_pdf_vectorized(x_vals, mu, sigma, beta, m)
    integral = np.trapz(pdf_vals, x_vals)
    return 1 / integral

@jit(nopython=True)
def g_s_vectorized(x, mu, sigma, beta, m, norm_const):
    mask = (0 <= x) & (x <= 5)
    pdf = crystal_ball_pdf_vectorized(x, mu, sigma, beta, m)
    return np.where(mask, pdf * norm_const, 0)

@jit(nopython=True)
def h_s_vectorized(y, lambda_s):
    norm_const = 1 - np.exp(-lambda_s * 10)
    mask = (0 <= y) & (y <= 10)
    pdf = lambda_s * np.exp(-lambda_s * y) / norm_const
    return np.where(mask, pdf, 0)

@jit(nopython=True)
def erf_approximation(x):
    """
    Approximate the error function using a polynomial approximation.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y

@jit(nopython=True)
def h_b_vectorized(y, mu_b, sigma_b):
    a = (0 - mu_b) / sigma_b
    b = (10 - mu_b) / sigma_b
    norm_const = 0.5 * (erf_approximation(b / np.sqrt(2)) - erf_approximation(a / np.sqrt(2)))
    pdf = (1 / (sigma_b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y - mu_b) / sigma_b) ** 2) / norm_const
    return np.where((0 <= y) & (y <= 10), pdf, 0)

@jit(nopython=True)
def g_b_vectorized(x):
    return np.where((0 <= x) & (x <= 5), 1 / 5, 0)

@jit(nopython=True)
def f_xy_vectorized(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm):
    g_s_vals = g_s_vectorized(x, mu, sigma, beta, m, g_s_norm)
    h_s_vals = h_s_vectorized(y, lambda_s)
    g_b_vals = g_b_vectorized(x)
    h_b_vals = h_b_vectorized(y, mu_b, sigma_b)
    signal = g_s_vals * h_s_vals
    background = g_b_vals * h_b_vals
    return f_signal * signal + (1 - f_signal) * background

def test_normalization(params, n_samples=100):
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

    x_vals = np.linspace(0, 5, n_samples)
    y_vals = np.linspace(0, 10, n_samples)

    g_s_integral = np.trapz(g_s_vectorized(x_vals, mu, sigma, beta, m, g_s_norm), x_vals)
    h_s_integral = np.trapz(h_s_vectorized(y_vals, lambda_s), y_vals)
    g_b_integral = np.trapz(g_b_vectorized(x_vals), x_vals)
    h_b_integral = np.trapz(h_b_vectorized(y_vals, mu_b, sigma_b), y_vals)

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    f_vals = f_xy_vectorized(x_grid, y_grid, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm)
    xy_integral = np.trapz(np.trapz(f_vals, y_vals, axis=1), x_vals)

    return {
        "g_s(X)": g_s_integral,
        "h_s(Y)": h_s_integral,
        "g_b(X)": g_b_integral,
        "h_b(Y)": h_b_integral,
        "f(X, Y)": xy_integral
    }

@jit(nopython=True)
def sample_h_s(size, lambda_s):
    """
    Efficient sampling from the exponential distribution for h_s(Y).
    """
    uniform_randoms = np.random.uniform(0.0, 1.0, size)  # Use positional arguments
    return -np.log(1 - uniform_randoms * (1 - np.exp(-lambda_s * 10))) / lambda_s


@jit(nopython=True)
def sample_g_b(size):
    """
    Efficient sampling from the uniform distribution for g_b(X).
    """
    return np.random.uniform(0.0, 5.0, size)  # Use positional arguments


@jit(nopython=True)
def sample_g_s(size, mu, sigma, beta, m, g_s_norm):
    """
    Efficient sampling from the truncated Crystal Ball distribution for g_s(X).
    """
    x_vals = np.linspace(0, 5, 10000) 
    pdf_vals = crystal_ball_pdf_vectorized(x_vals, mu, sigma, beta, m) * g_s_norm
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals /= cdf_vals[-1]
    uniform_randoms = np.random.uniform(0.0, 1.0, size)  # Use positional arguments
    indices = np.searchsorted(cdf_vals, uniform_randoms)
    return x_vals[indices]

from scipy.stats import truncnorm

def sample_h_b(size, mu_b, sigma_b):
    """
    Efficient sampling from the truncated normal distribution for h_b(Y).
    """
    a, b = (0 - mu_b) / sigma_b, (10 - mu_b) / sigma_b
    return truncnorm.rvs(a, b, loc=mu_b, scale=sigma_b, size=size)

def sample_componentwise(size, params):
    """
    Efficiently sample from the combined signal and background distributions.

    Parameters:
    - size: Number of samples to generate.
    - params: Model parameters (mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal).

    Returns:
    - x_samples: Array of sampled X values.
    - y_samples: Array of sampled Y values.
    """
    # Unpack parameters
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params

    # Compute normalization constant for g_s(X)
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

    # Decide if each sample is signal or background
    is_signal = np.random.uniform(size=size) < f_signal

    # Preallocate arrays
    x_samples = np.zeros(size)
    y_samples = np.zeros(size)

    # Signal samples
    signal_indices = np.where(is_signal)[0]
    x_samples[signal_indices] = sample_g_s(len(signal_indices), mu, sigma, beta, m, g_s_norm)
    y_samples[signal_indices] = sample_h_s(len(signal_indices), lambda_s)

    # Background samples
    background_indices = np.where(~is_signal)[0]
    x_samples[background_indices] = sample_g_b(len(background_indices))
    y_samples[background_indices] = sample_h_b(len(background_indices), mu_b, sigma_b)

    return x_samples, y_samples


