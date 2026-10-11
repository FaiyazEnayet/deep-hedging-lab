"""Merton jump-diffusion price scenario generator.

Simulates paths of a risky asset as GBM plus a compound Poisson jump
component with log-normal jump sizes. The drift is adjusted by the jump
compensator so the scenarios stay consistent with the assumed mean
growth rate. This is a scenario generator, not a pricing model: it
produces price paths with realistic sudden moves, nothing more.
"""

import numpy as np


def simulate_merton(
    n_paths, n_steps, T, S0, r, sigma, lam, mu_j, sigma_j, seed=None
):
    """Simulate Merton jump-diffusion price scenarios.

    Parameters
    ----------
    r : float
        Assumed mean growth rate of the scenario (continuously
        compounded). Set to the cash rate for a driftless baseline.
    lam : float
        Jump intensity (jumps per year).
    mu_j, sigma_j : float
        Mean and std of the log jump size.

    Returns
    -------
    S : np.ndarray, shape (n_paths, n_steps + 1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # E[e^J - 1]: the compensator that keeps the mean growth consistent with r.
    kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift = (r - lam * kappa_j - 0.5 * sigma**2) * dt

    s = np.full((n_paths, n_steps + 1), S0, dtype=float)
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, n_paths)
        # Sum of N log-jumps ~ N(mu_j, sigma_j^2) each; exact in distribution.
        jump_sum = (
            n_jumps * mu_j
            + sigma_j * np.sqrt(np.maximum(n_jumps, 0)) * rng.standard_normal(n_paths)
        )
        s[:, i + 1] = s[:, i] * np.exp(drift + sigma * sqrt_dt * z + jump_sum)
    return s
