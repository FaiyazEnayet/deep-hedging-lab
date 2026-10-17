"""Tests for crash-robustness: causal features, CPPI safety layer,
crash-weighted loss, and the extended (2007+) market history.

Every crash feature must be strictly causal: the value at step ``i``
may only depend on prices up to and including bar ``i``. These tests
pin that down by perturbing future bars and asserting past features
don't move.
"""

import math

import numpy as np
import torch

from dhl.experiments.cppi_learned import (
    CAP,
    FLOOR,
    CPPIPolicy,
    _crash_features,
    _crash_weights,
    _learned_weight_fn,
    classic_cppi_weight,
    cppi_constrained_weight_fn,
)


def _trending_prices(n_paths=4, steps=80, drift=0.001, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=(n_paths, steps))
    S = np.empty((n_paths, steps + 1))
    S[:, 0] = 100.0
    S[:, 1:] = 100.0 * np.exp(np.cumsum(rets, axis=1))
    return torch.as_tensor(S, dtype=torch.float32)


def test_crash_features_keys_and_shapes():
    S = _trending_prices()
    feats = _crash_features(S)
    assert set(feats) == {"vol", "vol_ratio", "drawdown", "panic_z"}
    n, steps_p1 = S.shape
    for v in feats.values():
        assert v.shape == (n, steps_p1 - 1)


def test_crash_features_strictly_causal():
    # Perturb bars AFTER step i; features at step i must not change.
    S = _trending_prices(n_paths=2, steps=60, seed=1)
    S2 = S.clone()
    S2[:, 40:] *= 1.5  # huge future move
    f1 = _crash_features(S)
    f2 = _crash_features(S2)
    for name, v1 in f1.items():
        v2 = f2[name]
        # steps 0..38 only see bars <= 39 < 40, so must be identical
        assert torch.allclose(v1[:, :39], v2[:, :39], atol=1e-6), name
        # and at least one feature must react to the future shock later
    assert not torch.allclose(f1["vol"][:, 50:], f2["vol"][:, 50:])


def test_vol_ratio_spikes_on_vol_regime_change():
    # Calm then violent: vol_ratio should exceed 1 in the violent part.
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0, 0.005, size=(2, 100))
    violent = rng.normal(0.0, 0.04, size=(2, 40))
    rets = np.concatenate([calm, violent], axis=1)
    S = torch.as_tensor(
        np.concatenate([np.full((2, 1), 100.0),
                        100.0 * np.exp(np.cumsum(rets, axis=1))], axis=1),
        dtype=torch.float32)
    feats = _crash_features(S)
    # Early in the violent regime, 5d vol >> 60d vol (which still
    # mostly holds calm history). Later the 60d window catches up
    # and the ratio decays -- that is the correct behavior.
    assert float(feats["vol_ratio"][:, 110].mean()) > 1.2


def test_drawdown_tracks_price_fall():
    # Steady 1% daily drops for 30 days: drawdown should be large.
    rets = np.array([[-0.01] * 40])
    S = torch.as_tensor(
        np.concatenate([np.full((1, 1), 100.0),
                        100.0 * np.exp(np.cumsum(rets, axis=1))], axis=1),
        dtype=torch.float32)
    feats = _crash_features(S)
    # After 30 down days, drawdown ~ 1 - exp(-0.3) ~= 0.26
    assert float(feats["drawdown"][0, 35]) > 0.20


def test_panic_z_negative_on_sharp_drop():
    # One -8% day after calm: panic_z strongly negative at the NEXT
    # decision step (the shock return is applied after weights are
    # chosen, so the feature sees it one step later -- correctly
    # causal).
    rng = np.random.default_rng(2)
    rets = np.concatenate(
        [rng.normal(0.0, 0.005, size=(1, 58)),
         np.array([[-0.08]]),
         rng.normal(0.0, 0.005, size=(1, 5))], axis=1)
    S = torch.as_tensor(
        np.concatenate([np.full((1, 1), 100.0),
                        100.0 * np.exp(np.cumsum(rets, axis=1))], axis=1),
        dtype=torch.float32)
    feats = _crash_features(S)
    # Step 59 is the first decision after the shock return (index 58).
    assert float(feats["panic_z"][0, 59]) < -2.0


def test_cppi_layer_never_exceeds_budget():
    torch.manual_seed(0)
    policy = CPPIPolicy(feature_names=("vol",))
    learned = _learned_weight_fn(policy)
    safe = cppi_constrained_weight_fn(learned, multiplier=4.0)
    S = _trending_prices(n_paths=8, steps=30, drift=-0.005, seed=3)
    from dhl.experiments.cppi_learned import rollout
    # Capture proposed vs constrained weights.
    proposed, constrained = [], []

    def spy(i, V, tau_frac, cushion, feats):
        w_p = learned(i, V, tau_frac, cushion, feats)
        w_c = safe(i, V, tau_frac, cushion, feats)
        proposed.append(w_p.clone())
        constrained.append(w_c.clone())
        return w_c

    rollout(S, spy)
    for wp, wc in zip(proposed, constrained):
        assert bool((wc <= wp + 1e-6).all())
        assert bool(((wc >= 0) & (wc <= CAP)).all())


def test_cppi_layer_scales_down_example():
    # Hand-check: V=100, floor=90, m=4 -> budget = 4*10/100 = 0.4.
    # A network proposing 0.7 must come back as 0.4.
    def big_bet(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, 0.7)

    safe = cppi_constrained_weight_fn(big_bet, multiplier=4.0)
    V = torch.full((5,), 100.0)
    feats = {"vol": torch.full((5,), 0.2),
             "vol_ratio": torch.ones(5),
             "drawdown": torch.zeros(5),
             "panic_z": torch.zeros(5)}
    w = safe(0, V, 1.0, torch.full((5,), 0.1), feats)
    assert torch.allclose(w, torch.full((5,), 0.4), atol=1e-6)


def test_cppi_layer_zero_cushion_gives_zero():
    def big_bet(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, 0.9)

    safe = cppi_constrained_weight_fn(big_bet, multiplier=4.0)
    V = torch.full((3,), 85.0)  # below the 90 floor: no budget
    feats = {"vol": torch.full((3,), 0.2),
             "vol_ratio": torch.ones(3),
             "drawdown": torch.zeros(3),
             "panic_z": torch.zeros(3)}
    w = safe(0, V, 0.5, torch.full((3,), -0.05), feats)
    assert bool((w == 0.0).all())


def test_cppi_layer_emergency_tightens_budget():
    def big_bet(i, V, tau_frac, cushion, feats):
        return torch.full_like(V, 1.0)

    safe = cppi_constrained_weight_fn(
        big_bet, multiplier=4.0, emergency_vol=0.30,
        emergency_multiplier=2.0)
    V = torch.full((4,), 100.0)
    feats_calm = {"vol": torch.full((4,), 0.15),
                  "vol_ratio": torch.ones(4),
                  "drawdown": torch.zeros(4),
                  "panic_z": torch.zeros(4)}
    feats_hot = {"vol": torch.full((4,), 0.50),
                 "vol_ratio": torch.ones(4),
                 "drawdown": torch.zeros(4),
                 "panic_z": torch.zeros(4)}
    w_calm = safe(0, V, 1.0, torch.full((4,), 0.1), feats_calm)
    w_hot = safe(0, V, 1.0, torch.full((4,), 0.1), feats_hot)
    # calm: budget 0.4; hot: emergency m=2 -> budget 0.2
    assert torch.allclose(w_calm, torch.full((4,), 0.4), atol=1e-6)
    assert torch.allclose(w_hot, torch.full((4,), 0.2), atol=1e-6)


def test_crash_weights_flat_below_cap():
    torch.manual_seed(0)
    # Calm paths: all weights should be ~1.
    rets = torch.randn(16, 63) * 0.008
    S = torch.empty(16, 64)
    S[:, 0] = 100.0
    S[:, 1:] = 100.0 * torch.exp(torch.cumsum(rets, dim=1))
    w = _crash_weights(S, crash_mult=3.0, vol_cap=0.30)
    assert bool((w >= 1.0).all())
    # Calm 0.8% daily vol ~= 12.7% ann < 30% cap -> weights exactly 1
    assert torch.allclose(w, torch.ones_like(w), atol=1e-6)


def test_crash_weights_upweight_violent_paths():
    torch.manual_seed(1)
    calm = torch.randn(8, 63) * 0.008
    violent = torch.randn(8, 63) * 0.035  # ~55% ann vol
    rets = torch.cat([calm, violent], dim=0)
    S = torch.empty(16, 64)
    S[:, 0] = 100.0
    S[:, 1:] = 100.0 * torch.exp(torch.cumsum(rets, dim=1))
    w = _crash_weights(S, crash_mult=3.0, vol_cap=0.30)
    assert float(w[:8].mean()) == 1.0
    assert float(w[8:].mean()) > 2.0


def test_crash_weights_zero_mult_is_uniform():
    S = _trending_prices(n_paths=4, steps=63, vol=0.03, seed=5)
    w = _crash_weights(S, crash_mult=0.0, vol_cap=0.30)
    assert torch.allclose(w, torch.ones(4), atol=1e-6)


def test_spy_history_covers_2008():
    from dhl.markets.real import load_spy_daily
    d = load_spy_daily()
    assert d["dates"][0] < "2008-01-01", d["dates"][0]
    # The crisis months are present.
    assert "2008-09-15" in d["dates"]  # Lehman Monday
    assert "2008-10-10" in d["dates"]


def test_vix_lagged_has_no_lookahead():
    # aligned_vix_lagged(d) must never use VIX from day d itself.
    from dhl.markets.history_ext import (
        aligned_vix_lagged, load_vix_daily)
    from dhl.markets.real import load_spy_daily
    spy = load_spy_daily()
    vix = load_vix_daily()
    lagged = aligned_vix_lagged(spy["dates"], vix, lag_days=1)
    # Spot check: lagged[1] == VIX close on spy date[0].
    vix_by_date = dict(zip(vix["dates"], vix["close"]))
    assert abs(lagged[1] - vix_by_date[spy["dates"][0]]) < 1e-9
    # And lagged[i] != same-day VIX close (unless coincidentally equal,
    # so check the dates differ by construction on a volatile day).
    assert abs(lagged[2] - vix_by_date[spy["dates"][1]]) < 1e-9
