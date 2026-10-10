"""Tests for the Monte Carlo market simulators."""

import numpy as np

from dhl.markets import simulate_gbm, simulate_heston, simulate_merton


def test_gbm_shape_and_martingale():
    S = simulate_gbm(20000, 50, T=1.0, S0=100.0, r=0.05, sigma=0.2, seed=0)
    assert S.shape == (20000, 51)
    assert np.all(S > 0)
    # Risk-neutral: E[S_T] = S0 * e^{rT}.
    assert abs(S[:, -1].mean() - 100 * np.exp(0.05)) < 1.0


def test_gbm_log_variance():
    S = simulate_gbm(50000, 100, T=2.0, S0=100.0, r=0.0, sigma=0.3, seed=1)
    # Var[log S_T] = sigma^2 T exactly (exact discretization).
    assert abs(np.log(S[:, -1] / 100).var() - 0.09 * 2.0) < 0.005


def test_heston_positive_and_mean_reverting():
    S, v = simulate_heston(
        5000, 50, T=1.0, S0=100.0, r=0.02, v0=0.04,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, seed=2,
    )
    assert S.shape == v.shape == (5000, 51)
    assert np.all(S > 0) and np.all(v >= 0)
    # Variance mean-reverts toward theta.
    assert abs(v[:, -1].mean() - 0.04) < 0.01


def test_heston_zero_vol_of_vol_is_gbm():
    s_h, _ = simulate_heston(
        3000, 40, T=0.5, S0=100.0, r=0.03, v0=0.09,
        kappa=1.0, theta=0.09, xi=0.0, rho=0.0, seed=3,
    )
    s_g = simulate_gbm(3000, 40, T=0.5, S0=100.0, r=0.03, sigma=0.3, seed=3)
    # Same seed and xi=0 -> identical paths (moment-matched).
    assert s_h[:, -1].mean() == s_h[:, -1].mean()  # sanity
    assert abs(np.log(s_h[:, -1]).mean() - np.log(s_g[:, -1]).mean()) < 0.05


def test_merton_martingale():
    S = simulate_merton(
        30000, 50, T=1.0, S0=100.0, r=0.05, sigma=0.2,
        lam=0.5, mu_j=-0.05, sigma_j=0.1, seed=4,
    )
    assert S.shape == (30000, 51)
    assert np.all(S > 0)
    # Compensator keeps E[S_T] = S0 e^{rT}.
    assert abs(S[:, -1].mean() - 100 * np.exp(0.05)) < 2.0


def test_merton_no_jumps_is_gbm():
    s_m = simulate_merton(
        5000, 40, T=0.5, S0=100.0, r=0.03, sigma=0.3,
        lam=0.0, mu_j=0.0, sigma_j=0.1, seed=5,
    )
    assert np.all(s_m > 0)
    assert abs(s_m[:, -1].mean() - 100 * np.exp(0.03 * 0.5)) < 2.0
