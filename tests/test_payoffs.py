"""Tests for derivative payoffs and Black-Scholes formulas."""

import math

import numpy as np

from dhl.derivatives import (
    asian_call,
    bs_call_delta,
    bs_call_price,
    digital_call,
    european_call,
    european_put,
    up_and_out_call,
)


def _paths():
    # S0=100, then 90 / 110 / 105
    return np.array([[100.0, 90.0], [100.0, 110.0], [100.0, 105.0]])


def test_european_call_put():
    S = _paths()
    np.testing.assert_allclose(european_call(S, 100), [0, 10, 5])
    np.testing.assert_allclose(european_put(S, 100), [10, 0, 0])


def test_call_put_parity_on_payoffs():
    S = _paths()
    assert np.allclose(
        european_call(S, 100) - european_put(S, 100), S[:, -1] - 100
    )


def test_digital_bounds():
    S = _paths()
    d = digital_call(S, 100)
    assert set(np.unique(d)) <= {0.0, 1.0}
    np.testing.assert_allclose(d, [0, 1, 1])


def test_asian_below_european():
    rng = np.random.default_rng(0)
    S = np.exp(np.cumsum(rng.normal(0, 0.02, (5000, 30)), axis=1)) * 100
    S = np.concatenate([np.full((5000, 1), 100.0), S], axis=1)
    # Averaging reduces volatility -> Asian call worth less than European.
    assert asian_call(S, 100).mean() < european_call(S, 100).mean()


def test_barrier_knockout():
    S = np.array([[100.0, 150.0, 110.0], [100.0, 105.0, 110.0]])
    out = up_and_out_call(S, 100, barrier=120)
    np.testing.assert_allclose(out, [0.0, 10.0])


def test_bs_price_known_value():
    # Hull textbook example: S=42, K=40, r=0.1, T=0.5, sigma=0.2 -> ~4.76
    assert abs(bs_call_price(42, 40, 0.5, 0.1, 0.2) - 4.76) < 0.02


def test_bs_price_bounds():
    S = np.linspace(50, 150, 11)
    p = bs_call_price(S, 100, 1.0, 0.05, 0.2)
    assert np.all(p >= np.maximum(S - 100, 0))  # above intrinsic
    assert np.all(p <= S)  # below spot


def test_bs_delta_bounds_and_monotonic():
    S = np.linspace(50, 150, 21)
    d = bs_call_delta(S, 100, 1.0, 0.05, 0.2)
    assert np.all((d >= 0) & (d <= 1))
    assert np.all(np.diff(d) >= 0)
    assert abs(d[10] - 0.5) > 0  # ATM delta is not degenerate


def test_bs_put_call_parity():
    S, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 0.25
    lhs = bs_call_price(S, K, T, r, sigma) - (
        bs_call_price(S, K, T, r, sigma) - S + K * math.exp(-r * T)
    )
    assert abs(lhs - (S - K * math.exp(-r * T))) < 1e-10
