from iminuit import Minuit
from distributions_sampling import crystal_ball_normalization_truncated, f_xy_vectorized
from numba import jit
import numpy as np

# @jit(nopython=True)
def perform_fit(x_samples, y_samples, bounds, initial_guess, return_minuit=False):
    """
    Perform an extended likelihood fit to the given samples and return the results.

    Args:
        x_samples (array): Array of X samples.
        y_samples (array): Array of Y samples.
        bounds (dict): A dictionary of parameter bounds for the fit.
        initial_guess (dict): A dictionary of initial parameter guesses.

    Returns:
        dict: Fitted parameters, uncertainties, and covariance matrix.
    """
    # Nested function for extended negative log-likelihood
    def extended_negative_log_likelihood(mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, N_expected):
        try:
            # Precompute normalization constant for the signal PDF
            g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

            # Evaluate the joint PDF at each sample point
            pdf_vals = f_xy_vectorized(
                x_samples, y_samples, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm
            )

            # Ensure numerical stability (avoid log(0))
            pdf_vals += 1e-10

            # Observed sample size
            N_observed = len(x_samples)

            # Extended likelihood terms
            log_likelihood = (
                -N_expected
                + N_observed * np.log(N_expected)
                + np.sum(np.log(pdf_vals))
            )

            # Return negative log-likelihood for minimization
            return -log_likelihood

        except Exception as e:
            print(f"Error in likelihood calculation: {e}")
            return 1e10

    # Initialize Minuit object
    m = Minuit(extended_negative_log_likelihood, **initial_guess)

    # Set parameter bounds
    for param, bound in bounds.items():
        m.limits[param] = bound

    # Perform the fit
    try:
        m.migrad()  # Minimize the negative log-likelihood
        m.hesse()   # Estimate uncertainties

        # Check if the fit converged
        if not m.valid:
            print("Fit did not converge. Check Minuit diagnostics:")
            print(m.fmin)

        # Collect results
        result = {
            "parameters": {name: value for name, value in zip(m.parameters, m.values)},
            "uncertainties": {name: error for name, error in zip(m.parameters, m.errors)},
            "converged": m.valid,
            "covariance": m.covariance if m.valid else None,  # Collect covariance object
        }

    except Exception as e:
        print(f"Error during fit: {e}")
        result = {"converged": False, "covariance": None}

    # Debug: Print fit results
    if result["converged"]:
        print("Fit converged successfully.")
        print(f"Fitted parameters: {result['parameters']}")
        print(f"Uncertainties: {result['uncertainties']}")
    else:
        print("Fit failed.")

    return (result, m) if return_minuit else result


