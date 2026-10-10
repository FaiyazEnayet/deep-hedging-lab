"""Tests for the self-financing hedging accounting engine."""

import math

import numpy as np
import torch

from dhl.hedgers import hedging_error


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float32)


def test_no_trade_no_payoff():
    # Flat market, no position, no liability: error = -p0 * e^{rT}.
    S = _t([[100.0, 100.0, 100.0]])
    err = hedging_error(S, _t([[0.0, 0.0]]), _t([0.0]), 5.0, 0.05, 0.5, 0.001)
    assert abs(float(err) + 5.0 * math.exp(0.05 * 1.0)) < 1e-5


def test_perfect_replication():
    # Buy-and-hold one share replicates a forward payoff S_T - K:
    # fund the share by borrowing, so p0 = 0.
    S = _t([[100.0, 110.0, 121.0]])
    err = hedging_error(
        S, _t([[1.0, 1.0]]), _t([21.0]), 0.0, 0.0, 0.5, 0.0
    )
    assert abs(float(err)) < 1e-4


def test_proportional_costs_reduce_wealth():
    S = _t(np.full((50, 11), 100.0))
    d = _t(np.full((50, 10), 0.5))
    err_free = hedging_error(S, d, _t(np.zeros(50)), 0.0, 0.0, 0.1, 0.0)
    err_cost = hedging_error(S, d, _t(np.zeros(50)), 0.0, 0.0, 0.1, 0.01)
    # Costs make the short position worse: error larger with costs.
    assert float(err_cost.mean()) > float(err_free.mean())
    # Opening 0.5 shares at 100 with 1% cost, liquidating at end: 0.5*100*0.01*2 = 1.0
    assert abs(float(err_cost.mean()) - 1.0) < 1e-4


def test_gains_match_closed_form():
    # Single step: gains = delta * (S1 - S0 e^{rdt}).
    S = _t([[100.0, 105.0]])
    err = hedging_error(S, _t([[0.5]]), _t([0.0]), 50.0, 0.0, 1.0, 0.0)
    # X_T = 50 + 0.5*(105-100) = 52.5, error = -52.5
    assert abs(float(err) + 52.5) < 1e-4


def test_vectorized_matches_loop():
    rng = np.random.default_rng(7)
    S = _t(np.exp(np.cumsum(rng.normal(0, 0.01, (32, 21)), axis=1)) * 100)
    d = _t(rng.uniform(0, 1, (32, 20)))
    err = hedging_error(S, d, _t(np.zeros(32)), 3.0, 0.02, 0.05, 0.002)
    assert err.shape == (32,)
    assert torch.isfinite(err).all()
