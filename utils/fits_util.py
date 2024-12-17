from iminuit import Minuit
from utils.distributions_sampling_util import *
import numpy as np

def perform_fit(x_samples, y_samples, bounds, initial_guess, return_minuit=False):
    """
    Perform an extended likelihood fit to the given (X, Y) samples using Minuit.

    The fit minimizes the extended negative log-likelihood, which includes:
    1. Poisson term for the expected and observed number of events.
    2. Log-sum of the joint probability density function evaluated at each sample point.

    Parameters:
    - x_samples : array-like
        Array of X samples.
    - y_samples : array-like
        Array of Y samples.
    - bounds : dict
        Dictionary specifying bounds for each fit parameter in the format:
        {"param_name": (lower_bound, upper_bound)}.
    - initial_guess : dict
        Dictionary specifying initial guesses for each parameter.
        Example: {"mu": 3, "sigma": 0.3, ...}
    - return_minuit : bool, optional
        If True, return the Minuit object along with the fit results.

    Returns:
    - result : dict
        A dictionary containing:
        - "parameters": Best-fit parameter values.
        - "uncertainties": Estimated uncertainties of the parameters.
        - "converged": Boolean indicating fit convergence.
        - "covariance": Covariance matrix of the parameters (if fit converged).
    - m : Minuit object (optional)
        The Minuit object used for the fit (returned only if return_minuit=True).
    """
    # Nested function: Extended negative log-likelihood
    def extended_negative_log_likelihood(mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, N_expected):
        """
        Compute the extended negative log-likelihood for the given parameters.

        Parameters:
        - mu, sigma, beta, m : float
            Crystal Ball PDF parameters.
        - lambda_s : float
            Decay constant for the exponential PDF (signal).
        - mu_b, sigma_b : float
            Mean and standard deviation of the truncated normal PDF (background).
        - f_signal : float
            Signal fraction parameter.
        - N_expected : float
            Expected total number of events.

        Returns:
        - nll : float
            Negative log-likelihood value to be minimized.
        """
        try:
            # Precompute normalization constant for the signal PDF
            g_s_norm = crystal_ball_normalization_truncated(mu, sigma, beta, m, lower=0, upper=5)

            # Evaluate the joint PDF for the given parameters at each sample point
            pdf_vals = f_xy_vectorized(
                x_samples, y_samples, mu, sigma, beta, m, lambda_s, mu_b, sigma_b, f_signal, g_s_norm
            )

            # Ensure numerical stability (prevent log(0) errors)
            pdf_vals += 1e-10

            # Observed sample size
            N_observed = len(x_samples)

            # Compute the extended negative log-likelihood
            log_likelihood = (
                -N_expected
                + N_observed * np.log(N_expected)
                + np.sum(np.log(pdf_vals))
            )

            return -log_likelihood  # Return the negative log-likelihood for minimization

        except Exception as e:
            print(f"Error in likelihood calculation: {e}")
            return 1e10  # Return large value to signal failure

    # Initialize Minuit for the negative log-likelihood function
    m = Minuit(extended_negative_log_likelihood, **initial_guess)

    # Apply parameter bounds
    for param, bound in bounds.items():
        m.limits[param] = bound

    # Perform the fit
    try:
        m.migrad()  # Run the MIGRAD algorithm for minimization
        m.hesse()   # Compute uncertainties using the HESSE algorithm

        # Check for fit convergence
        if not m.valid:
            print("Fit did not converge. Check Minuit diagnostics:")
            print(m.fmin)

        # Collect fit results
        result = {
            "parameters": {name: value for name, value in zip(m.parameters, m.values)},
            "uncertainties": {name: error for name, error in zip(m.parameters, m.errors)},
            "converged": m.valid,
            "covariance": m.covariance if m.valid else None,
        }

    except Exception as e:
        # Handle any errors that occur during the fit process
        print(f"Error during fit: {e}")
        result = {"converged": False, "covariance": None}

    # Debug output: Print fit results
    if result["converged"]:
        print("Fit converged successfully.")
        print(f"Fitted parameters: {result['parameters']}")
        print(f"Uncertainties: {result['uncertainties']}")
    else:
        print("Fit failed.")

    # Return results, optionally including the Minuit object
    return (result, m) if return_minuit else result
