"""Heston stochastic-volatility simulator.

Risk-neutral dynamics::

    dS/S = r dt + sqrt(v) dW1
    dv   = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho dt

Discretized with a full-truncation Euler scheme, which keeps the variance
process non-negative and is the standard workhorse for this model.
"""

import numpy as np


def simulate_heston(
    n_paths, n_steps, T, S0, r, v0, kappa, theta, xi, rho, seed=None
):
    """Simulate risk-neutral Heston paths.

    Returns
    -------
    S : np.ndarray, shape (n_paths, n_steps + 1)
    v : np.ndarray, shape (n_paths, n_steps + 1)
        Instantaneous variance paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    s = np.full((n_paths, n_steps + 1), S0, dtype=float)
    v = np.full((n_paths, n_steps + 1), v0, dtype=float)

    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = sqrt_dt * z1
        w2 = sqrt_dt * (rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2)

        v_pos = np.maximum(v[:, i], 0.0)
        sqrt_v = np.sqrt(v_pos)
        v[:, i + 1] = v[:, i] + kappa * (theta - v_pos) * dt + xi * sqrt_v * w2
        # Log-Euler update for the spot keeps prices positive.
        s[:, i + 1] = s[:, i] * np.exp((r - 0.5 * v_pos) * dt + sqrt_v * w1)

    v = np.maximum(v, 0.0)
    return s, v
