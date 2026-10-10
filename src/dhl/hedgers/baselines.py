"""Classical delta-hedging baselines.

Black-Scholes delta hedging is the textbook baseline: rebalance to the
model delta at each date. Leland (1985) adjusts the volatility upward to
account for proportional transaction costs, widening the delta and
reducing turnover.
"""

import math

import numpy as np

from dhl.derivatives.black_scholes import bs_call_delta


def bs_delta_hedge(S, K, r, sigma, dt):
    """Black-Scholes delta hedge evaluated along each path.

    Returns deltas of shape (n_paths, n_steps) as a numpy array.
    """
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    T = n_steps * dt
    taus = T - np.arange(n_steps) * dt  # time to maturity at each date
    return bs_call_delta(S[:, :-1], K, taus[None, :], r, sigma)


def leland_delta_hedge(S, K, r, sigma, dt, cost_rate):
    """Leland (1985) transaction-cost-adjusted delta hedge."""
    sigma_adj = sigma * math.sqrt(
        1.0 + math.sqrt(2.0 / math.pi) * cost_rate / (sigma * math.sqrt(dt))
    )
    return bs_delta_hedge(S, K, r, sigma_adj, dt)
