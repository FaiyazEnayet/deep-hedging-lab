"""Tests for the CPPI portfolio-insurance experiment.

No network: synthetic price paths only. Fast by design.
"""

import math

import numpy as np
import torch

from dhl.experiments.cppi_learned import (
    CAP,
    COST_RATE,
    FLOOR,
    R,
    V0,
    CPPIPolicy,
    _vol_features,
    classic_cppi_weight,
    rollout,
    train_cppi_policy,
)


def _prices(start, rets):
    p = [start]
    for r in rets:
        p.append(p[-1] * math.exp(r))
    return np.array([p], dtype=np.float64)


def _const_weight(w):
    def fn(i, V, tau_frac, cushion, vol):
        return torch.full_like(V, w)

    return fn


def test_buy_and_hold_matches_spot():
    # w = 1, no costs: wealth tracks the risky asset exactly.
    S = torch.as_tensor(_prices(100.0, [0.01] * 30), dtype=torch.float32)
    V_T = rollout(S, _const_weight(1.0), cost_rate=0.0)
    assert abs(float(V_T[0]) - 100.0 * float(S[0, -1] / S[0, 0])) < 1e-3


def test_all_cash_earns_risk_free():
    # w = 0, no costs: wealth compounds at r.
    S = torch.as_tensor(_prices(100.0, [0.01] * 30), dtype=torch.float32)
    dt = 1.0 / 252.0
    V_T = rollout(S, _const_weight(0.0), cost_rate=0.0, r=R, dt=dt)
    assert abs(float(V_T[0]) - V0 * math.exp(R * 30 * dt)) < 1e-3


def test_classic_cppi_never_breaches_on_rising_path():
    # Steadily rising market: cushion only grows, floor never touched.
    S = torch.as_tensor(_prices(100.0, [0.008] * 30), dtype=torch.float32)
    seen = []

    def fn(i, V, tau_frac, cushion, vol):
        w = classic_cppi_weight(V, multiplier=4.0)
        seen.append(w.clone())
        return w

    V_T, path = rollout(S, fn, return_path=True)
    assert float(path.min()) >= FLOOR - 1e-6
    for w in seen:
        assert bool(((w >= 0) & (w <= CAP)).all())


def test_classic_cppi_derisks_after_breach():
    # An overnight gap blows through the floor (gap risk is real: no
    # daily rule can de-risk inside a single jump). Afterwards the
    # weight must be pinned at exactly zero.
    rets = [0.0] * 5 + [-0.51] + [0.0] * 24  # ~ -40% in one day
    S = torch.as_tensor(_prices(100.0, rets), dtype=torch.float32)
    seen = []

    def fn(i, V, tau_frac, cushion, vol):
        w = classic_cppi_weight(V, multiplier=4.0)
        seen.append(w.clone())
        return w

    V_T, path = rollout(S, fn, return_path=True)
    assert float(path.min()) < FLOOR  # the gap really breached
    for w in seen[7:]:
        assert float(w[0]) == 0.0
    for w in seen:
        assert bool(((w >= 0) & (w <= CAP)).all())


def test_classic_weight_formula():
    # m * cushion / V, capped; zero cushion gives zero weight.
    assert classic_cppi_weight(100.0, multiplier=4.0) == 4.0 * 10.0 / 100.0
    assert classic_cppi_weight(85.0, multiplier=4.0) == 0.0
    assert classic_cppi_weight(200.0, multiplier=5.0) <= CAP


def test_policy_output_bounds():
    torch.manual_seed(0)
    policy = CPPIPolicy()
    tau = torch.rand(64)
    cushion = torch.rand(64) * 2 - 1
    vol = torch.rand(64) * 0.5
    w = policy(tau, cushion, vol)
    assert w.shape == (64,)
    assert bool(((w >= 0) & (w <= CAP)).all())


def test_training_reduces_objective():
    # Tiny deterministic check: a few epochs on synthetic returns must
    # improve the penalized objective on a fixed batch.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.012, size=2000)

    from dhl.markets.real import block_bootstrap_paths

    horizon = 10
    S_np = block_bootstrap_paths(rets, 256, horizon, 21, S0=V0, seed=123)
    S = torch.as_tensor(S_np, dtype=torch.float32)

    policy = CPPIPolicy(hidden_dims=(8, 8))

    def objective():
        from dhl.experiments.cppi_learned import _learned_weight_fn

        with torch.no_grad():
            V_T = rollout(S, _learned_weight_fn(policy))
            shortfall = torch.relu(FLOOR - V_T)
            return float(-(V_T.mean() - 30.0 * shortfall.mean()))

    before = objective()
    sampler = lambda n, s: block_bootstrap_paths(rets, n, horizon, 21,
                                                 S0=V0, seed=s)
    train_cppi_policy(policy, sampler, horizon, n_epochs=6, batch_size=256,
                      lr=5e-3, log_every=0, seed=0)
    after = objective()
    assert after < before, f"objective did not improve: {before} -> {after}"


def test_vol_features_are_causal():
    # Shifting price i+1 must not move any feature at times <= i.
    torch.manual_seed(0)
    S = torch.exp(torch.randn(4, 31).cumsum(dim=1) * 0.02)
    base = _vol_features(S)
    for i in [0, 5, 15, 29]:
        bumped = S.clone()
        bumped[:, i + 1] *= 1.1
        got = _vol_features(bumped)
        assert torch.equal(got[:, : i + 1], base[:, : i + 1]), f"leak at i={i}"


def test_vol_features_first_step_is_zero():
    # No history at i=0, so the feature is exactly 0, not a nan cleanup.
    torch.manual_seed(1)
    S = torch.exp(torch.randn(3, 21).cumsum(dim=1) * 0.02)
    vols = _vol_features(S)
    assert torch.all(vols[:, 0] == 0.0)
    assert torch.isfinite(vols).all()
