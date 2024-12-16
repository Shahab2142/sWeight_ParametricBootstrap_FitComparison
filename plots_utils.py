import numpy as np
import matplotlib.pyplot as plt
from distributions_sampling import *
from scipy.ndimage import gaussian_filter  # Import gaussian_filter
from fits_util import perform_fit
import seaborn as sns


def plot_distributions(params, resolution=1000):
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)
    
    x_vals = np.linspace(0, 5, resolution)
    y_vals = np.linspace(0, 10, resolution)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Marginals
    g_s_vals = g_s_vectorized(x_vals, mu, sigma, beta, m, g_s_norm)
    g_b_vals = g_b_vectorized(x_vals)
    h_s_vals = h_s_vectorized(y_vals, lambda_s)
    h_b_vals = h_b_vectorized(y_vals, mu_b, sigma_b)
    
    # Combined PDF
    f_vals = f_xy_vectorized(x_grid, y_grid, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm)
    
    # Plot Marginals in X
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, g_s_vals, label='Signal $g_s(X)$', linestyle='--', color='blue')
    plt.plot(x_vals, g_b_vals, label='Background $g_b(X)$', linestyle=':', color='orange')
    plt.plot(x_vals, f_signal * g_s_vals + (1 - f_signal) * g_b_vals, label='Total $f_X(X)$', color='black', linewidth=2)
    plt.xlabel('$X$')
    plt.ylabel('Probability Density')
    plt.title('Marginal Distribution in $X$')
    plt.legend()
    plt.grid()

    # Plot Marginals in Y
    plt.subplot(1, 2, 2)
    plt.plot(y_vals, h_s_vals, label='Signal $h_s(Y)$', linestyle='--', color='blue')
    plt.plot(y_vals, h_b_vals, label='Background $h_b(Y)$', linestyle=':', color='orange')
    plt.plot(y_vals, f_signal * h_s_vals + (1 - f_signal) * h_b_vals, label='Total $f_Y(Y)$', color='black', linewidth=2)
    plt.xlabel('$Y$')
    plt.ylabel('Probability Density')
    plt.title('Marginal Distribution in $Y$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot Joint Distribution f(X, Y)
    plt.figure(figsize=(8, 6))
    plt.contourf(
        x_grid, y_grid, f_vals, levels=100,  # Increased levels for finer granularity
        cmap='viridis'  # Raw values, no scaling applied
    )
    plt.colorbar(label='Sampled Joint PDF $f(X, Y)$')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.title('Sampled Joint PDF from $f(X, Y)$')
    plt.grid(False)  # Remove grid for a cleaner look
    plt.show()
    
    
# # Function 3: Plot sampled joint distribution and marginals
# def plot_sampled_distributions(x_samples, y_samples):
#     # Plot the 2D joint distribution
#     plt.figure(figsize=(8, 6))
#     plt.hist2d(x_samples, y_samples, bins=100, cmap='viridis', density=True)
#     plt.colorbar(label="Density")
#     plt.title("2D Joint Distribution of Samples")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.grid(alpha=0.3)
#     plt.show()

#     # Plot the marginal distribution for X
#     plt.figure(figsize=(8, 4))
#     plt.hist(x_samples, bins=100, density=True, color='blue', alpha=0.7, label='Marginal X')
#     plt.title("Marginal Distribution of X")
#     plt.xlabel("X")
#     plt.ylabel("Density")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.show()

#     # Plot the marginal distribution for Y
#     plt.figure(figsize=(8, 4))
#     plt.hist(y_samples, bins=100, density=True, color='green', alpha=0.7, label='Marginal Y')
#     plt.title("Marginal Distribution of Y")
#     plt.xlabel("Y")
#     plt.ylabel("Density")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.show()

def plot_sampled_distributions(x_samples, y_samples, params):
    """
    Plot the sampled joint distribution and overlay the marginal distributions.
    
    Args:
        x_samples (array): Sampled X values.
        y_samples (array): Sampled Y values.
        params (list): List of parameters [mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal].
    """
    # Unpack parameters
    mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = params
    g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

    # Define X and Y ranges
    x_vals = np.linspace(0, 5, 1000)
    y_vals = np.linspace(0, 10, 1000)

    # Compute marginals
    g_s_vals = g_s_vectorized(x_vals, mu, sigma, beta, m, g_s_norm)
    g_b_vals = g_b_vectorized(x_vals)
    marginal_x = f_signal * g_s_vals + (1 - f_signal) * g_b_vals

    h_s_vals = h_s_vectorized(y_vals, lambda_s)
    h_b_vals = h_b_vectorized(y_vals, mu_b, sigma_b)
    marginal_y = f_signal * h_s_vals + (1 - f_signal) * h_b_vals

    # Plot the 2D joint distribution
    plt.figure(figsize=(8, 6))
    plt.hist2d(x_samples, y_samples, bins=100, cmap='viridis', density=True)
    plt.colorbar(label="Density")
    plt.title("2D Joint Distribution of Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(alpha=0.3)
    plt.show()

    # Plot the marginal distribution for X
    plt.figure(figsize=(8, 4))
    plt.hist(x_samples, bins=100, density=True, color='blue', alpha=0.7, label='Sampled Marginal X')
    plt.plot(x_vals, marginal_x, color='red', linewidth=2, label='Actual Marginal X')
    plt.title("Marginal Distribution of X")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # Plot the marginal distribution for Y
    plt.figure(figsize=(8, 4))
    plt.hist(y_samples, bins=100, density=True, color='green', alpha=0.7, label='Sampled Marginal Y')
    plt.plot(y_vals, marginal_y, color='red', linewidth=2, label='Actual Marginal Y')
    plt.title("Marginal Distribution of Y")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()



# # Function 4: Plot smoothed sampled surface and fitted surface
# def plot_surface_comparison(x_samples, y_samples, fitted_params, bins=200, sigma=2):
#     # Generate the fitted surface
#     mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal = fitted_params[:8]
#     g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

#     # Define grids for x and y
#     x_vals = np.linspace(0, 5, 100)
#     y_vals = np.linspace(0, 10, 100)
#     x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

#     # Evaluate the joint PDF f(x, y) using fitted parameters
#     fitted_f_vals = f_xy_vectorized(x_grid, y_grid, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm)

#     # Sampled data as a heatmap
#     hist, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=bins, density=True)
#     x_mid = (xedges[:-1] + xedges[1:]) / 2
#     y_mid = (yedges[:-1] + yedges[1:]) / 2
#     x_mid_grid, y_mid_grid = np.meshgrid(x_mid, y_mid, indexing="ij")

#     # Apply Gaussian filter for smoothing
#     hist_smooth = gaussian_filter(hist, sigma=sigma)

#     # Plot the smoothed sampled surface
#     fig1 = plt.figure(figsize=(12, 8))
#     ax1 = fig1.add_subplot(111, projection='3d')
#     ax1.plot_surface(x_mid_grid, y_mid_grid, hist_smooth, cmap='viridis', alpha=0.8)
#     ax1.set_title("Smoothed Sampled Surface")
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("Y")
#     ax1.set_zlabel("Density")

#     # Plot the fitted surface
#     fig2 = plt.figure(figsize=(12, 8))
#     ax2 = fig2.add_subplot(111, projection='3d')
#     ax2.plot_surface(x_grid, y_grid, fitted_f_vals, cmap='coolwarm', alpha=0.8)
#     fit_params_str = f"mu={mu:.2f}, sigma={sigma:.2f}, beta={beta:.2f}, m={m:.2f},\n" \
#                      f"lambda_s={lambda_s:.2f}, mu_b={mu_b:.2f}, sigma_b={sigma_b:.2f}, f_signal={f_signal:.2f}"
#     ax2.set_title(f"Fitted Surface\n{fit_params_str}")
#     ax2.set_xlabel("X")
#     ax2.set_ylabel("Y")
#     ax2.set_zlabel("Density")

#     plt.show()
