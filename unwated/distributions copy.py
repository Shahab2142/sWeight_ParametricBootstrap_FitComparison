import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm, norm
from scipy.optimize import minimize


# Crystal Ball PDF
def crystal_ball_pdf(x, mu, sigma, beta, m):
    z = (x - mu) / sigma
    if z > -beta:
        return np.exp(-z**2 / 2)
    else:
        A = (m / beta) ** m * np.exp(-beta**2 / 2)
        B = m / beta - beta
        return A * (B - z) ** -m


def crystal_ball_normalization(sigma, beta, m):
    """
    Compute the explicit normalization constant for the Crystal Ball PDF.
    """
    term1 = (m / (beta * (m - 1))) * np.exp(-beta**2 / 2)
    term2 = np.sqrt(2 * np.pi) * norm.cdf(beta)
    return 1 / (sigma * (term1 + term2))


# Signal distribution g_s(X)
def g_s(x, mu, sigma, beta, m, norm_const):
    """
    Signal distribution g_s(X) using the explicit normalization constant.
    """
    if 0 <= x <= 5:
        pdf_unnorm = crystal_ball_pdf(x, mu, sigma, beta, m)
        return pdf_unnorm * norm_const
    return 0


# Signal distribution h_s(Y)
def h_s(y, lambda_s, norm_const):
    if 0 <= y <= 10:
        return lambda_s * np.exp(-lambda_s * y) / norm_const
    return 0


def h_s_normalization(lambda_s):
    """
    Compute the normalization constant for h_s(Y).
    """
    return 1 - np.exp(-lambda_s * 10)


# Background distribution g_b(X)
def g_b(x):
    return 1 / 5 if 0 <= x <= 5 else 0


# Background distribution h_b(Y)
def h_b(y, mu_b, sigma_b):
    if 0 <= y <= 10:
        a, b = (0 - mu_b) / sigma_b, (10 - mu_b) / sigma_b
        h_b_dist = truncnorm(a, b, loc=mu_b, scale=sigma_b)
        return h_b_dist.pdf(y)
    return 0


# Combined distribution f(X, Y)
def f_xy(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm):
    signal = g_s(x, mu, sigma, beta, m, g_s_norm) * h_s(y, lambda_s, h_s_norm)
    background = g_b(x) * h_b(y, mu_b, sigma_b)
    return f_signal * signal + (1 - f_signal) * background


# Find the maximum of the PDF
def find_max_pdf(mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm):
    def negative_pdf(coords):
        x, y = coords
        if 0 <= x <= 5 and 0 <= y <= 10:
            return -f_xy(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm)
        else:
            return np.inf

    result = minimize(
        negative_pdf, [2.5, 5.0], bounds=[(0, 5), (0, 10)], method="L-BFGS-B"
    )
    if result.success:
        return -result.fun
    else:
        raise RuntimeError("Failed to find maximum PDF value")


# Generate samples from the joint distribution
def generate_sample_from_joint(n_samples, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm):
    max_density = find_max_pdf(mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm)
    print(f"Maximum PDF value: {max_density}")

    x_samples = np.random.uniform(0, 5, size=n_samples * 2)
    y_samples = np.random.uniform(0, 10, size=n_samples * 2)

    densities = np.array([
        f_xy(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm)
        for x, y in zip(x_samples, y_samples)
    ])

    accept_mask = np.random.uniform(0, max_density, size=len(densities)) < densities
    accepted_samples = np.column_stack((x_samples[accept_mask], y_samples[accept_mask]))
    return accepted_samples[:n_samples]


# Extended likelihood function
def extended_likelihood(params, data, g_s_norm, h_s_norm):
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, n_total = params
    x_data, y_data = data[:, 0], data[:, 1]

    log_likelihood = np.sum(
        np.log(f_xy(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm, h_s_norm) + 1e-12)
        for x, y in zip(x_data, y_data)
    )

    log_likelihood += len(data) * np.log(n_total) - n_total
    return -log_likelihood


# Parameter fitting
def fit_parameters(data, initial_guess, g_s_norm, h_s_norm):
    result = minimize(
        extended_likelihood,
        initial_guess,
        args=(data, g_s_norm, h_s_norm),
        bounds=[
            (2, 4), (0.2, 0.6), (0.5, 2), (1.1, 3), (0.1, 0.5), (-1, 2), (1, 4), (0.3, 0.8), (90000, 110000)
        ],
        method="L-BFGS-B"
    )
    return result.x, result.fun


# Test normalization for a single set of parameters
def test_normalization(mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal):
    h_s_norm = h_s_normalization(lambda_s)

    g_b_norm, _ = quad(g_b, 0, 5)
    h_b_norm, _ = quad(lambda y: h_b(y, mu_b, sigma_b), 0, 10)

    def integrand_x(x):
        result, _ = quad(lambda y: f_xy(x, y, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, 1, h_s_norm), 0, 10)
        return result

    f_xy_norm, _ = quad(integrand_x, 0, 5)

    return h_s_norm, g_b_norm, h_b_norm, f_xy_norm


# Systematic test with random parameter combinations
def systematic_test(num_tests=10):
    results = {
        "h_s(Y)": [],
        "g_b(X)": [],
        "h_b(Y)": [],
        "f(X, Y)": []
    }

    np.random.seed(42)  # For reproducibility

    for _ in range(num_tests):
        mu = np.random.uniform(2, 4)
        sigma = np.random.uniform(0.2, 0.6)
        beta = np.random.uniform(0.5, 2)
        m = np.random.uniform(1.1, 3)
        lambda_s = np.random.uniform(0.1, 0.5)
        mu_b = np.random.uniform(-1, 2)
        sigma_b = np.random.uniform(1, 4)
        f_signal = np.random.uniform(0.3, 0.8)

        h_s_norm, g_b_norm, h_b_norm, f_xy_norm = test_normalization(
            mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal
        )

        results["h_s(Y)"].append(h_s_norm)
        results["g_b(X)"].append(g_b_norm)
        results["h_b(Y)"].append(h_b_norm)
        results["f(X, Y)"].append(f_xy_norm)

    averages = {key: np.mean(values) for key, values in results.items()}
    print("\nAverage normalization values across all tests:")
    for key, value in averages.items():
        print(f"{key}: {value}")

    return results
