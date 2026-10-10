"""Geometric Brownian motion (Black-Scholes) simulator.

Risk-neutral dynamics: dS/S = r dt + sigma dW.
Uses the exact discretization, so there is no time-stepping bias.
"""

import numpy as np


def simulate_gbm(n_paths, n_steps, T, S0, r, sigma, seed=None):
    """Simulate risk-neutral GBM paths.

    Parameters
    ----------
    n_paths : int
    n_steps : int
    T : float
        Maturity in years.
    S0 : float
        Spot price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility.
    seed : int, optional

    Returns
    -------
    S : np.ndarray, shape (n_paths, n_steps + 1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    # Exact solution of the SDE over each step.
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.log(S0) + np.cumsum(log_returns, axis=1)
    s = np.exp(log_s)
    return np.concatenate([np.full((n_paths, 1), S0), s], axis=1)
