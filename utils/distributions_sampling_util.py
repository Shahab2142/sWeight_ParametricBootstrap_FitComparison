from numba import jit
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import trapezoid
from scipy.stats import truncnorm

@jit(nopython=True)
def crystal_ball_pdf_vectorized(x, mu, sigma, beta, m):
    """
    Compute the Crystal Ball probability density function (PDF).

    Parameters:
    - x : float or array-like
        Input values for which the PDF is evaluated.
    - mu : float
        Mean (location) of the distribution.
    - sigma : float
        Standard deviation (scale) of the Gaussian core.
    - beta : float
        Transition point between Gaussian core and power-law tail.
    - m : float
        Slope parameter for the power-law tail.

    Returns:
    - pdf : float or array-like
        Evaluated PDF values for the input x.
    """
    z = (x - mu) / sigma
    core = np.exp(-z**2 / 2)
    A = (m / beta) ** m * np.exp(-beta**2 / 2)
    B = m / beta - beta
    tail = A * (B - z) ** -m
    pdf = np.where(z > -beta, core, tail)
    return pdf

@jit(nopython=True)
def crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5, n_samples=1000):
    """
    Compute the normalization constant for the truncated Crystal Ball PDF.

    Parameters:
    - mu, sigma, beta, m : float
        Parameters of the Crystal Ball PDF.
    - lower, upper : float
        Truncation bounds.
    - n_samples : int
        Number of samples for numerical integration.

    Returns:
    - norm_const : float
        Inverse of the integral over the truncated region.
    """
    x_vals = np.linspace(lower, upper, n_samples)
    pdf_vals = crystal_ball_pdf_vectorized(x_vals, mu, sigma, beta, m)
    integral = np.trapz(pdf_vals, x_vals)
    return 1 / integral

@jit(nopython=True)
def g_s_vectorized(x, mu, sigma, beta, m, norm_const):
    """
    Compute the truncated and normalized Crystal Ball PDF.

    Parameters:
    - x : array-like
        Input values.
    - mu, sigma, beta, m : float
        Crystal Ball distribution parameters.
    - norm_const : float
        Normalization constant for truncation.

    Returns:
    - pdf : array-like
        Evaluated PDF values.
    """
    mask = (0 <= x) & (x <= 5)
    pdf = crystal_ball_pdf_vectorized(x, mu, sigma, beta, m)
    return np.where(mask, pdf * norm_const, 0)

@jit(nopython=True)
def h_s_vectorized(y, lambda_s):
    """
    Compute the truncated exponential PDF for h_s(Y).

    Parameters:
    - y : array-like
        Input values.
    - lambda_s : float
        Decay constant of the exponential.

    Returns:
    - pdf : array-like
        Evaluated PDF values.
    """
    norm_const = 1 - np.exp(-lambda_s * 10)
    mask = (0 <= y) & (y <= 10)
    pdf = lambda_s * np.exp(-lambda_s * y) / norm_const
    return np.where(mask, pdf, 0)

@jit(nopython=True)
def erf_approximation(x):
    """
    Approximate the error function (erf) using a polynomial approximation.

    Parameters:
    - x : float
        Input value.

    Returns:
    - erf : float
        Approximated value of erf(x).
    """
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y

@jit(nopython=True)
def h_b_vectorized(y, mu_b, sigma_b):
    """
    Compute the truncated normal PDF for h_b(Y).

    Parameters:
    - y : array-like
        Input values.
    - mu_b : float
        Mean of the normal distribution.
    - sigma_b : float
        Standard deviation of the normal distribution.

    Returns:
    - pdf : array-like
        Evaluated PDF values.
    """
    a = (0 - mu_b) / sigma_b
    b = (10 - mu_b) / sigma_b
    norm_const = 0.5 * (erf_approximation(b / np.sqrt(2)) - erf_approximation(a / np.sqrt(2)))
    pdf = (1 / (sigma_b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y - mu_b) / sigma_b) ** 2) / norm_const
    return np.where((0 <= y) & (y <= 10), pdf, 0)

@jit(nopython=True)
def g_b_vectorized(x):
    """
    Compute the uniform PDF for g_b(X).

    Parameters:
    - x : array-like
        Input values.

    Returns:
    - pdf : array-like
        Evaluated PDF values.
    """
    return np.where((0 <= x) & (x <= 5), 1 / 5, 0)

@jit(nopython=True)
def f_xy_vectorized(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm):
    """
    Compute the joint PDF f(X, Y) combining signal and background distributions.

    Parameters:
    - x, y : array-like
        Input values.
    - mu, sigma, beta, m, lambda_s, mu_b, sigma_b : float
        Distribution parameters.
    - f_signal : float
        Signal fraction.
    - g_s_norm : float
        Normalization constant for the Crystal Ball PDF.

    Returns:
    - pdf : array-like
        Evaluated joint PDF values.
    """
    g_s_vals = g_s_vectorized(x, mu, sigma, beta, m, g_s_norm)
    h_s_vals = h_s_vectorized(y, lambda_s)
    g_b_vals = g_b_vectorized(x)
    h_b_vals = h_b_vectorized(y, mu_b, sigma_b)
    signal = g_s_vals * h_s_vals
    background = g_b_vals * h_b_vals
    return f_signal * signal + (1 - f_signal) * background

def sample_componentwise(size, params):
    """
    Generate samples from the combined signal and background distributions.

    Parameters:
    - size : int
        Number of samples to generate.
    - params : tuple
        Model parameters: (mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal).

    Returns:
    - x_samples, y_samples : array-like
        Arrays of sampled X and Y values.
    """
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params

    # Precompute normalization for g_s
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m)

    # Determine signal/background assignment
    is_signal = np.random.uniform(size=size) < f_signal

    x_samples = np.zeros(size)
    y_samples = np.zeros(size)

    # Generate signal samples
    signal_indices = np.where(is_signal)[0]
    x_samples[signal_indices] = sample_g_s(len(signal_indices), mu, sigma, beta, m, g_s_norm)
    y_samples[signal_indices] = sample_h_s(len(signal_indices), lambda_s)

    # Generate background samples
    background_indices = np.where(~is_signal)[0]
    x_samples[background_indices] = sample_g_b(len(background_indices))
    y_samples[background_indices] = sample_h_b(len(background_indices), mu_b, sigma_b)

    return x_samples, y_samples

def sample_h_b(size, mu_b, sigma_b):
    """
    Sample from the truncated normal distribution for h_b(Y).

    Parameters:
    - size : int
        Number of samples to generate.
    - mu_b : float
        Mean of the normal distribution.
    - sigma_b : float
        Standard deviation of the normal distribution.

    Returns:
    - samples : array-like
        Samples drawn from the truncated normal distribution over [0, 10].
    """
    # Compute truncation bounds in standard normal space
    a, b = (0 - mu_b) / sigma_b, (10 - mu_b) / sigma_b
    return truncnorm.rvs(a, b, loc=mu_b, scale=sigma_b, size=size)

@jit(nopython=True)
def sample_g_s(size, mu, sigma, beta, m, g_s_norm):
    """
    Sample from the truncated Crystal Ball distribution for g_s(X).

    Parameters:
    - size : int
        Number of samples to generate.
    - mu, sigma, beta, m : float
        Crystal Ball distribution parameters.
    - g_s_norm : float
        Normalization constant for the truncated Crystal Ball PDF.

    Returns:
    - samples : array-like
        Samples drawn from the truncated Crystal Ball distribution.
    """
    # Generate fine-grained grid of x values
    x_vals = np.linspace(0, 5, 10000)
    # Compute PDF values and normalize
    pdf_vals = crystal_ball_pdf_vectorized(x_vals, mu, sigma, beta, m) * g_s_norm
    # Construct cumulative distribution function (CDF)
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals /= cdf_vals[-1]
    # Sample from the CDF using inverse transform sampling
    uniform_randoms = np.random.uniform(0.0, 1.0, size)
    indices = np.searchsorted(cdf_vals, uniform_randoms)
    return x_vals[indices]

@jit(nopython=True)
def sample_h_s(size, lambda_s):
    """
    Sample from the truncated exponential distribution for h_s(Y).

    Parameters:
    - size : int
        Number of samples to generate.
    - lambda_s : float
        Decay constant of the exponential distribution.

    Returns:
    - samples : array-like
        Samples drawn from the truncated exponential distribution over [0, 10].
    """
    # Uniform random samples for inverse transform sampling
    uniform_randoms = np.random.uniform(0.0, 1.0, size)
    # Inverse CDF of truncated exponential distribution
    return -np.log(1 - uniform_randoms * (1 - np.exp(-lambda_s * 10))) / lambda_s

@jit(nopython=True)
def sample_g_b(size):
    """
    Sample from the uniform distribution for g_b(X).

    Parameters:
    - size : int
        Number of samples to generate.

    Returns:
    - samples : array-like
        Samples drawn uniformly over [0, 5].
    """
    return np.random.uniform(0.0, 5.0, size)


def test_normalization(params, n_samples=100):
    """
    Test the normalization of all component distributions and the joint distribution.

    Parameters:
    - params : tuple
        Model parameters (mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal).
    - n_samples : int, optional
        Number of samples for numerical integration (default is 100).

    Returns:
    - normalization_results : dict
        A dictionary containing the integrals of all component distributions:
        - "g_s(X)" : Integral of g_s(X) over [0, 5].
        - "h_s(Y)" : Integral of h_s(Y) over [0, 10].
        - "g_b(X)" : Integral of g_b(X) over [0, 5].
        - "h_b(Y)" : Integral of h_b(Y) over [0, 10].
        - "f(X, Y)" : Integral of the joint distribution f(X, Y).
    """
    # Unpack parameters
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params

    # Compute normalization constant for g_s
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

    # Generate integration grids
    x_vals = np.linspace(0, 5, n_samples)
    y_vals = np.linspace(0, 10, n_samples)

    # Compute integrals for individual distributions
    g_s_integral = np.trapz(g_s_vectorized(x_vals, mu, sigma, beta, m, g_s_norm), x_vals)
    h_s_integral = np.trapz(h_s_vectorized(y_vals, lambda_s), y_vals)
    g_b_integral = np.trapz(g_b_vectorized(x_vals), x_vals)
    h_b_integral = np.trapz(h_b_vectorized(y_vals, mu_b, sigma_b), y_vals)

    # Compute integral for the joint distribution f(X, Y)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    f_vals = f_xy_vectorized(x_grid, y_grid, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm)
    xy_integral = np.trapz(np.trapz(f_vals, y_vals, axis=1), x_vals)

    # Return results as a dictionary
    return {
        "g_s(X)": g_s_integral,
        "h_s(Y)": h_s_integral,
        "g_b(X)": g_b_integral,
        "h_b(Y)": h_b_integral,
        "f(X, Y)": xy_integral
    }
