"""Tests for the hedging policies: bounds, no-lookahead, and a training smoke test."""

import numpy as np
import torch

from dhl.derivatives import european_call
from dhl.hedgers import DeepHedger, bs_delta_hedge, leland_delta_hedge, train_deep_hedger
from dhl.markets import simulate_gbm
from dhl.risk import cvar


def test_bs_delta_bounds():
    S = simulate_gbm(200, 10, T=0.25, S0=100.0, r=0.02, sigma=0.25, seed=0)
    d = bs_delta_hedge(S, 100.0, 0.02, 0.25, 0.025)
    assert d.shape == (200, 10)
    assert np.all((d >= 0) & (d <= 1))


def test_leland_more_conservative_without_costs():
    # Leland only differs from BS when cost_rate > 0.
    S = simulate_gbm(200, 10, T=0.25, S0=100.0, r=0.02, sigma=0.25, seed=1)
    d_bs = bs_delta_hedge(S, 100.0, 0.02, 0.25, 0.025)
    d_zero = leland_delta_hedge(S, 100.0, 0.02, 0.25, 0.025, 0.0)
    np.testing.assert_allclose(d_bs, d_zero, rtol=1e-10)


def test_deep_hedger_output_bounds():
    torch.manual_seed(0)
    for arch in ("mlp", "lstm"):
        model = DeepHedger(hidden_dims=(8, 8) if arch == "mlp" else (8,), architecture=arch)
        S = torch.as_tensor(
            simulate_gbm(16, 5, T=0.25, S0=100.0, r=0.02, sigma=0.25, seed=2),
            dtype=torch.float32,
        )
        d = model(S, 100.0, 0.05)
        assert d.shape == (16, 5)
        assert bool(((d >= 0) & (d <= 1)).all())


def test_no_lookahead():
    """Perturbing future prices must not change earlier holdings."""
    torch.manual_seed(3)
    model = DeepHedger(hidden_dims=(8, 8), architecture="mlp")
    S = torch.as_tensor(
        simulate_gbm(8, 6, T=0.25, S0=100.0, r=0.02, sigma=0.25, seed=4),
        dtype=torch.float32,
    )
    d1 = model(S, 100.0, 0.25 / 6).detach()
    S2 = S.clone()
    S2[:, 4:] *= 1.5  # shock the future
    d2 = model(S2, 100.0, 0.25 / 6).detach()
    # Holdings at dates 0..3 depend only on information up to each date.
    torch.testing.assert_close(d1[:, :4], d2[:, :4], rtol=1e-5, atol=1e-6)


def test_training_reduces_objective():
    torch.manual_seed(5)
    model = DeepHedger(hidden_dims=(16, 16), architecture="mlp", p0_init=3.0)

    def sample(n, seed):
        return simulate_gbm(n, 8, T=0.25, S0=100.0, r=0.02, sigma=0.25, seed=seed)

    history = train_deep_hedger(
        model,
        sample,
        lambda S: european_call(S, 100.0),
        K=100.0,
        r=0.02,
        dt=0.25 / 8,
        cost_rate=0.0,
        risk_fn=lambda L: cvar(L, 0.9),
        n_epochs=15,
        batch_size=512,
        lr=5e-3,
        seed=5,
        log_every=0,
    )
    assert len(history) == 15
    assert all(np.isfinite(history))
    assert history[-1] < history[0], "training did not improve the objective"
