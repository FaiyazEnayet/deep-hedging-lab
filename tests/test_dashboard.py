"""Tests for the dashboard helpers (dhl.viz.dashboard).

No network: the committed CSVs back every data read. Tiny training
settings keep these fast.
"""

import numpy as np
import torch

from dhl.viz import dashboard as D


def test_train_demo_policy_runs():
    pol = D.train_demo_policy(n_epochs=2, batch_size=64, horizon_days=10,
                              seed=0)
    torch.manual_seed(0)
    w = pol(torch.rand(16), torch.rand(16), torch.rand(16) * 0.3)
    assert w.shape == (16,)
    assert bool(((w >= 0) & (w <= 1)).all())  # long-only, cap 1.0


def test_policy_weight_grid_shapes():
    pol = D.train_demo_policy(n_epochs=1, batch_size=64, horizon_days=10,
                              seed=0)
    g = D.policy_weight_grid(pol, n_cushion=11, n_time=7)
    assert g["weights"].shape == (11, 7)
    assert bool(((g["weights"] >= 0) & (g["weights"] <= 1)).all())
    assert g["cushions"][0] < 0 < g["cushions"][-1]  # spans the floor
    assert g["vol_fixed"] > 0


def test_simulate_wealth_shapes_and_metrics():
    pol = D.train_demo_policy(n_epochs=1, batch_size=64, horizon_days=10,
                              seed=0)
    sim = D.simulate_wealth(pol, horizon_days=10, n_paths=64, seed=3)
    for name in ("learned", "classic"):
        s = sim[name]
        assert s["paths"].shape == (64, 11)
        assert bool(np.isfinite(s["paths"]).all())
        assert s["mean_wealth"] > 0
        assert 0.0 <= s["p_breach"] <= 1.0
        assert (s["p10"] <= s["p50"]).all() and (s["p50"] <= s["p90"]).all()


def test_train_demo_multi_runs_and_sums_to_one():
    pol = D.train_demo_multi(n_epochs=2, batch_size=64, horizon_days=10,
                             seed=0)
    torch.manual_seed(0)
    w = pol(torch.rand(8), torch.rand(8) * 2 - 1,
            torch.rand(8, 3) * 0.3, torch.randn(8, 3) * 0.02)
    assert w.shape == (8, 4)
    assert bool(((w >= 0) & (w <= 1)).all())
    assert torch.allclose(w.sum(dim=1), torch.ones(8), atol=1e-5)


def test_multi_weight_grid_sums_to_one():
    pol = D.train_demo_multi(n_epochs=1, batch_size=64, horizon_days=10,
                             seed=0)
    g = D.multi_weight_grid(pol, n_cushion=6, n_time=4)
    assert g["weights"].shape == (6, 4, 4)
    assert g["labels"] == ["SPY", "TLT", "GLD", "cash"]
    sums = g["weights"].sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_repo_metrics_missing_or_valid():
    m = D.repo_metrics(root="/tmp/definitely-not-a-repo")
    assert m is None
    m = D.repo_metrics()  # repo root: runs/ may or may not exist
    assert m is None or isinstance(m, dict)
