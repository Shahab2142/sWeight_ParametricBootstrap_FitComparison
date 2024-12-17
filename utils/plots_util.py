import numpy as np
import matplotlib.pyplot as plt
from utils.distributions_sampling_util import *
from scipy.ndimage import gaussian_filter
from utils.fits_util import *
import seaborn as sns

def plot_distributions(params, resolution=1000):
    """
    Plot the theoretical marginal distributions and the joint PDF for a given set of parameters.

    Parameters:
    - params : list
        List of model parameters: [mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal].
    - resolution : int, optional
        Resolution of the grid for plotting the joint PDF and marginals (default: 1000).

    Returns:
    - None
    """
    # Unpack parameters
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params

    # Precompute normalization constant for the Crystal Ball distribution
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

    # Define grids for X and Y
    x_vals = np.linspace(0, 5, resolution)
    y_vals = np.linspace(0, 10, resolution)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Compute marginal distributions
    g_s_vals = g_s_vectorized(x_vals, mu, sigma, beta, m, g_s_norm)
    g_b_vals = g_b_vectorized(x_vals)
    h_s_vals = h_s_vectorized(y_vals, lambda_s)
    h_b_vals = h_b_vectorized(y_vals, mu_b, sigma_b)

    # Compute the joint distribution
    f_vals = f_xy_vectorized(x_grid, y_grid, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm)

    # ----------------------------------------
    # Plot marginal distribution in X
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, g_s_vals, label='Signal $g_s(X)$', linestyle='--', color='blue')
    plt.plot(x_vals, g_b_vals, label='Background $g_b(X)$', linestyle=':', color='orange')
    plt.plot(x_vals, f_signal * g_s_vals + (1 - f_signal) * g_b_vals,
             label='Marginal $f_X(X)$', color='black', linewidth=2)
    plt.xlabel('$X$')
    plt.ylabel('Probability Density')
    plt.title('Distributions in X: Signal, Background, and Marginal')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("marginal_distribution_X.pdf")
    plt.show()

    # ----------------------------------------
    # Plot marginal distribution in Y
    plt.figure(figsize=(8, 6))
    plt.plot(y_vals, h_s_vals, label='Signal $h_s(Y)$', linestyle='--', color='blue')
    plt.plot(y_vals, h_b_vals, label='Background $h_b(Y)$', linestyle=':', color='orange')
    plt.plot(y_vals, f_signal * h_s_vals + (1 - f_signal) * h_b_vals,
             label='Marginal $f_Y(Y)$', color='black', linewidth=2)
    plt.xlabel('$Y$')
    plt.ylabel('Probability Density')
    plt.title('Distributions in Y: Signal, Background, and Marginal')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("marginal_distribution_Y.pdf")
    plt.show()

    # ----------------------------------------
    # Plot joint distribution f(X, Y)
    plt.figure(figsize=(8, 6))
    plt.contourf(
        x_grid, y_grid, f_vals, levels=100,  # High resolution for smooth contours
        cmap='viridis'
    )
    plt.colorbar(label='Density')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.title('Theoretical 2D Joint PDF $f(X, Y)$')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("theoretical_joint_pdf.pdf")
    plt.show()

def plot_sampled_distributions(x_samples, y_samples, params):
    """
    Plot the sampled joint distribution and compare with theoretical marginal distributions.

    Parameters:
    - x_samples : array-like
        Array of sampled X values.
    - y_samples : array-like
        Array of sampled Y values.
    - params : list
        List of model parameters: [mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal].

    Returns:
    - None
    """
    # Unpack parameters
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params

    # Precompute normalization constant for g_s
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

    # Define X and Y ranges
    x_vals = np.linspace(0, 5, 1000)
    y_vals = np.linspace(0, 10, 1000)

    # Compute marginal distributions
    g_s_vals = g_s_vectorized(x_vals, mu, sigma, beta, m, g_s_norm)
    g_b_vals = g_b_vectorized(x_vals)
    marginal_x = f_signal * g_s_vals + (1 - f_signal) * g_b_vals

    h_s_vals = h_s_vectorized(y_vals, lambda_s)
    h_b_vals = h_b_vectorized(y_vals, mu_b, sigma_b)
    marginal_y = f_signal * h_s_vals + (1 - f_signal) * h_b_vals

    # Plot joint distribution (2D histogram)
    plt.figure(figsize=(8, 6))
    plt.hist2d(x_samples, y_samples, bins=100, cmap='viridis', density=True)
    plt.colorbar(label="Density")
    plt.title("Sampled 2D Joint Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(alpha=0.3)
    plt.savefig("sampled_joint_distribution.pdf")
    plt.show()

    # Plot sampled vs theoretical marginal distribution for X
    plt.figure(figsize=(8, 6))
    plt.hist(x_samples, bins=100, density=True, color='blue', alpha=0.7, label='Sampled Marginal X')
    plt.plot(x_vals, marginal_x, color='red', linewidth=2, label='Theoretical Marginal X')
    plt.title("Sampled Marginal Distribution of X")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("sampled_marginal_X.pdf")
    plt.show()

    # Plot sampled vs theoretical marginal distribution for Y
    plt.figure(figsize=(8, 6))
    plt.hist(y_samples, bins=100, density=True, color='green', alpha=0.7, label='Sampled Marginal Y')
    plt.plot(y_vals, marginal_y, color='red', linewidth=2, label='Theoretical Marginal Y')
    plt.title("Sampled Marginal Distribution of Y")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("sampled_marginal_Y.pdf")
    plt.show()
