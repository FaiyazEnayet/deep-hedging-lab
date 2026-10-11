"""Heston stochastic-volatility price scenario generator.

Simulates paths of a risky asset whose variance mean-reverts::

    dS/S = mu dt + sqrt(v) dW1
    dv   = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho dt

This is a scenario generator, not a pricing model: it produces
realistic price paths with volatility clustering and leverage
effects, nothing more.

Two discretizations are available via ``scheme``:

- ``"euler"`` (default): full-truncation Euler, the standard workhorse.
- ``"qe"``: Andersen's (2008) Quadratic-Exponential scheme. The variance
  step samples from the exact conditional mean/variance of the CIR
  process: a squared Gaussian when the normalized variance ``psi`` is
  small, an atom at zero plus an exponential tail when it is large.
  Variance stays non-negative by construction and the scheme is far
  more accurate than Euler for large vol-of-vol or long steps.
"""

import numpy as np

_PSI_CRIT = 1.5  # Andersen's cutoff between the quadratic and exponential regimes


def _qe_variance_step(v, kappa, theta, xi, dt, rng):
    """One Andersen QE step of the CIR variance process. Always >= 0."""
    n = v.shape[0]
    exp_kd = np.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kd
    m = np.maximum(m, 1e-12)
    s2 = (
        v * xi**2 * exp_kd / kappa * (1.0 - exp_kd)
        + theta * xi**2 / (2.0 * kappa) * (1.0 - exp_kd) ** 2
    )
    psi = s2 / m**2

    v_next = np.empty(n)
    zv = rng.standard_normal(n)  # Gaussian draw used in both branches

    quad = psi <= _PSI_CRIT
    if np.any(quad):
        psi_q = psi[quad]
        b2 = 2.0 / psi_q - 1.0 + np.sqrt(2.0 / psi_q * (2.0 / psi_q - 1.0))
        b = np.sqrt(b2)
        a = m[quad] / (1.0 + b2)
        v_next[quad] = a * (b + zv[quad]) ** 2

    if np.any(~quad):
        psi_e = psi[~quad]
        p = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p) / m[~quad]
        u = rng.random((~quad).sum())
        draw = np.where(u <= p, 0.0, np.log((1.0 - p) / (1.0 - u)) / beta)
        v_next[~quad] = draw
        # Fresh Gaussian for the spot's correlated shock in this branch.
        zv[~quad] = rng.standard_normal((~quad).sum())

    return v_next, zv


def simulate_heston(
    n_paths, n_steps, T, S0, r, v0, kappa, theta, xi, rho, seed=None,
    scheme="euler",
):
    """Simulate Heston price scenarios.

    Parameters
    ----------
    r : float
        Assumed mean growth rate of the scenario (continuously
        compounded). Set to the cash rate for a driftless baseline.
    scheme : {"euler", "qe"}
        Discretization of the variance process (see module docstring).

    Returns
    -------
    S : np.ndarray, shape (n_paths, n_steps + 1)
    v : np.ndarray, shape (n_paths, n_steps + 1)
        Instantaneous variance paths.
    """
    if scheme not in ("euler", "qe"):
        raise ValueError(f"unknown Heston scheme '{scheme}'")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    s = np.full((n_paths, n_steps + 1), S0, dtype=float)
    v = np.full((n_paths, n_steps + 1), v0, dtype=float)

    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = sqrt_dt * z1

        if scheme == "euler":
            v_pos = np.maximum(v[:, i], 0.0)
            sqrt_v = np.sqrt(v_pos)
            v[:, i + 1] = v[:, i] + kappa * (theta - v_pos) * dt + xi * sqrt_v * (
                sqrt_dt * (rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2)
            )
            # Log-Euler update for the spot keeps prices positive.
            s[:, i + 1] = s[:, i] * np.exp((r - 0.5 * v_pos) * dt + sqrt_v * w1)
        else:
            v_next, zv = _qe_variance_step(v[:, i], kappa, theta, xi, dt, rng)
            v[:, i + 1] = v_next
            # Log-Euler spot update on the left variance. Using v[:, i]
            # (known at time i) rather than a trapezoidal average keeps the
            # mean growth exactly consistent with r: conditional on F_i, zv
            # and z2 are independent N(0,1), so the shock's mgf cancels the
            # -0.5 v dt convexity term. zv is the same Gaussian that drove
            # the variance draw, preserving the leverage correlation.
            v_i = np.maximum(v[:, i], 0.0)
            spot_shock = rho * zv + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
            s[:, i + 1] = s[:, i] * np.exp(
                (r - 0.5 * v_i) * dt + np.sqrt(v_i) * spot_shock * sqrt_dt
            )

    if scheme == "euler":
        v = np.maximum(v, 0.0)
    return s, v
