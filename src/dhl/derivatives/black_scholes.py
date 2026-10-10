"""Closed-form Black-Scholes formulas.

Used to price the premium, initialize the deep hedger's initial capital,
and provide the classical delta-hedging baseline.
"""

import math

import numpy as np


def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _d1_d2(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def bs_call_price(S, K, tau, r, sigma):
    """Black-Scholes European call price. ``tau`` may be 0 (intrinsic value)."""
    S = np.asarray(S, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    tau_pos = np.maximum(tau, 1e-12)
    d1, d2 = _d1_d2(S, K, tau_pos, r, sigma)
    price = S * _norm_cdf(d1) - K * np.exp(-r * tau_pos) * _norm_cdf(d2)
    return np.where(np.asarray(tau) <= 0, intrinsic, price)


def bs_call_delta(S, K, tau, r, sigma):
    """Black-Scholes European call delta, clipped to [0, 1]."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, _ = _d1_d2(S, K, np.maximum(tau, 1e-12), r, sigma)
    delta = _norm_cdf(d1)
    # At expiry the delta is a step function.
    expired = tau <= 0
    delta = np.where(expired, (S > K).astype(float), delta)
    return np.clip(delta, 0.0, 1.0)


def bs_put_price(S, K, tau, r, sigma):
    call = bs_call_price(S, K, tau, r, sigma)
    return call - S + K * np.exp(-r * np.maximum(tau, 0.0))
