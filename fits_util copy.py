from distributions_sampling import *
from scipy.optimize import minimize
import numpy as np

def perform_fit(sample, bounds):
    """
    Perform an extended maximum likelihood fit without relying on a predefined initial guess.
    """
    x_vals, y_vals = sample
    n_events = len(x_vals)

    # Define the negative log-likelihood function
    def negative_log_likelihood(params):
        # Unpack parameters
        mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, mu_total = params
        
        # Dynamically recalculate g_s_norm for current parameters
        g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

        # Evaluate the joint PDF f(X, Y)
        f_vals = f_xy_vectorized(
            x_vals, y_vals, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm
        )

        # Clamp PDF values for numerical stability
        f_vals_safe = np.maximum(f_vals, 1e-8)

        # Compute the negative log-likelihood
        poisson_term = -mu_total + n_events * np.log(mu_total)  # Poisson term for total events
        pdf_term = np.sum(np.log(f_vals_safe))  # Log-likelihood for the PDF
        nll = -(poisson_term + pdf_term)  # Negate for minimization

        return nll

    # Dynamically generate a random initial guess within bounds
    initial_guess = [np.random.uniform(low, high) for low, high in bounds]

    # Minimize the negative log-likelihood
    result = minimize(
        negative_log_likelihood,
        initial_guess,
        method="L-BFGS-B",  # Default optimizer
        bounds=bounds,
        options={'ftol': 1e-6, 'gtol': 1e-6, 'disp': True}
    )

    # Process the fit results
    if result.success:
        try:
            # Compute parameter uncertainties if Hessian is available
            hessian_inv = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else None
            parameter_uncertainties = np.sqrt(np.diag(hessian_inv)) if hessian_inv is not None else [None] * len(initial_guess)
        except Exception as e:
            print(f"Error calculating uncertainties: {e}")
            hessian_inv = None
            parameter_uncertainties = [None] * len(initial_guess)
    else:
        hessian_inv = None
        parameter_uncertainties = [None] * len(initial_guess)

    # Print detailed fit diagnostics
    print("\nFit Diagnostics:")
    for i, (param_name, value, uncertainty) in enumerate(
        zip(["mu", "sigma", "beta", "m", "lambda_s", "mu_b", "sigma_b", "f_signal", "mu_total"],
            result.x, parameter_uncertainties)
    ):
        print(f"{param_name:>10} : {value:>10.4f} ± {uncertainty if uncertainty is not None else 'N/A':>10}")

    return result.x, parameter_uncertainties, result.success, result.fun, hessian_inv
