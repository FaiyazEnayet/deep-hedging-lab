"""Dashboard helpers: fast demo training and plot-ready arrays.

This module holds everything the Streamlit app (``app.py``) needs that
is not Streamlit itself: small-policy training on the committed real
data, policy-surface grids, wealth-path simulation, and multi-asset
allocation summaries. It imports no web framework, so it is unit
tested headless.

The demo policies are deliberately small and train for few epochs.
They illustrate the mechanics; the repo's published numbers come from
the full experiments in ``dhl.experiments``.
"""

import json
import math
import os

import numpy as np
import torch

from dhl.experiments.cppi_learned import (
    DT,
    R,
    V0,
    CPPIPolicy,
    _learned_weight_fn,
    classic_cppi_weight,
    rollout,
)
from dhl.experiments.cppi_multi import (
    TICKERS,
    MultiAssetPolicy,
    _learned_multi_weight_fn,
    block_bootstrap_paths_multi,
    rollout_multi,
)
from dhl.markets.real import (
    block_bootstrap_paths,
    load_aligned_returns,
    load_spy_daily,
    log_returns,
)

BLOCK_LEN = 21
FLOOR_PENALTY = 30.0
CAP = 1.0  # dashboard demos are long-only, no leverage
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def demo_epochs(default):
    """Epoch count override for fast smoke tests.

    ``NEURAL_CPPI_DEMO_EPOCHS`` set to a positive int replaces the
    default; anything else keeps the default.
    """
    try:
        n = int(os.environ.get("NEURAL_CPPI_DEMO_EPOCHS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else default


def _spy_returns():
    data = load_spy_daily()
    return log_returns(data["adjclose"])


def _single_sampler(returns, horizon_days):
    def sampler(n, s):
        return block_bootstrap_paths(
            returns, n, horizon_days, BLOCK_LEN, S0=V0, seed=s
        )

    return sampler


def _median_vol():
    """Median annualized 21-day rolling vol of SPY, for the policy grid."""
    r = _spy_returns()
    vols = np.array(
        [r[i : i + BLOCK_LEN].std(ddof=0) for i in range(len(r) - BLOCK_LEN)]
    ) * math.sqrt(252.0)
    return float(np.median(vols))


def train_demo_policy(floor=90.0, horizon_days=63, cost_rate=0.002,
                      n_epochs=30, batch_size=512, seed=7):
    """Train a small single-asset policy on bootstrapped SPY paths.

    Mirrors ``train_cppi_policy`` but parameterized by floor and cost
    for the dashboard sliders. Long-only (cap 1.0).
    """
    n_epochs = demo_epochs(n_epochs)
    torch.manual_seed(seed)
    returns = _spy_returns()
    sampler = _single_sampler(returns, horizon_days)
    policy = CPPIPolicy(hidden_dims=(16, 16), cap=CAP)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        S = torch.as_tensor(
            sampler(batch_size, seed + epoch), dtype=torch.float32
        )
        V_T = rollout(
            S, _learned_weight_fn(policy), floor=floor, cost_rate=cost_rate
        )
        shortfall = torch.relu(torch.full_like(V_T, floor) - V_T)
        objective = -(V_T.mean() - FLOOR_PENALTY * shortfall.mean())
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return policy


def policy_weight_grid(policy, n_cushion=41, n_time=25):
    """Risky weight on a (cushion, time-left) grid at median vol.

    Returns dict with ``cushions`` (fraction of wealth above the floor,
    can go negative past the floor), ``taus`` (fraction of time left),
    and ``weights`` shaped ``(n_cushion, n_time)``.
    """
    cushions = np.linspace(-0.1, 1.0, n_cushion)
    taus = np.linspace(0.0, 1.0, n_time)
    vol_fixed = _median_vol()
    with torch.no_grad():
        T, C = np.meshgrid(taus, cushions)
        w = policy(
            torch.as_tensor(T.ravel(), dtype=torch.float32),
            torch.as_tensor(C.ravel(), dtype=torch.float32),
            torch.full((T.size,), vol_fixed, dtype=torch.float32),
        )
    return {
        "cushions": cushions,
        "taus": taus,
        "weights": w.numpy().reshape(n_cushion, n_time),
        "vol_fixed": vol_fixed,
    }


def _classic_fn(floor, multiplier):
    def fn(i, V, tau_frac, cushion, vol):
        return classic_cppi_weight(V, floor=floor, multiplier=multiplier,
                                   cap=CAP)

    return fn


def simulate_wealth(policy, floor=90.0, horizon_days=63, cost_rate=0.002,
                    multiplier=4.0, n_paths=500, seed=999):
    """Roll learned and classic policies over the same test paths.

    Returns dict with wealth paths (``n_paths x steps+1``) for both
    strategies, per-step percentile bands, and summary metrics.
    """
    returns = _spy_returns()
    S = torch.as_tensor(
        _single_sampler(returns, horizon_days)(n_paths, seed),
        dtype=torch.float32,
    )
    out = {}
    with torch.no_grad():
        for name, fn in (
            ("learned", _learned_weight_fn(policy)),
            ("classic", _classic_fn(floor, multiplier)),
        ):
            V_T, path = rollout(
                S, fn, floor=floor, cost_rate=cost_rate, return_path=True
            )
            P = path.numpy()
            V = V_T.numpy()
            out[name] = {
                "paths": P,
                "mean_wealth": float(V.mean()),
                "std_wealth": float(V.std()),
                "p_breach": float((V < floor).mean()),
                "p10": np.percentile(P, 10, axis=0),
                "p50": np.percentile(P, 50, axis=0),
                "p90": np.percentile(P, 90, axis=0),
            }
    out["horizon_days"] = horizon_days
    return out


def _multi_sampler(rets, horizon_days):
    def sampler(n, s):
        return block_bootstrap_paths_multi(
            rets, TICKERS, n, horizon_days, BLOCK_LEN, S0=V0, seed=s
        )

    return sampler


def train_demo_multi(floor=90.0, horizon_days=63, cost_rate=0.002,
                     n_epochs=20, batch_size=512, seed=11):
    """Train a small 3-asset policy (SPY/TLT/GLD + cash) for the demo."""
    n_epochs = demo_epochs(n_epochs)
    torch.manual_seed(seed)
    _, rets = load_aligned_returns(TICKERS)
    sampler = _multi_sampler(rets, horizon_days)
    policy = MultiAssetPolicy(len(TICKERS), hidden_dims=(16, 16))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        P = torch.as_tensor(
            sampler(batch_size, seed + epoch), dtype=torch.float32
        )
        V_T, W, F = rollout_multi(
            P, _learned_multi_weight_fn(policy), floor_mode="fixed",
            floor=floor, cost_rate=cost_rate, return_path=True,
        )
        shortfall = torch.relu(F - W).amax(dim=1)
        objective = -(V_T.mean() - FLOOR_PENALTY * shortfall.mean())
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return policy


def _median_multi_features():
    """Median per-asset vol and trailing return, for the allocation grid."""
    _, rets = load_aligned_returns(TICKERS)
    vols, trails = [], []
    for t in TICKERS:
        r = rets[t]
        v = np.array(
            [r[i : i + BLOCK_LEN].std(ddof=0)
             for i in range(len(r) - BLOCK_LEN)]
        ) * math.sqrt(252.0)
        vols.append(float(np.median(v)))
        trails.append(float(np.median(r)))
    return vols, trails


def multi_weight_grid(policy, n_cushion=41, n_time=25):
    """Softmax weights on a (cushion, time-left) grid at median features.

    Returns dict with ``cushions``, ``taus``, ``labels``
    (SPY/TLT/GLD/cash) and ``weights`` shaped
    ``(n_cushion, n_time, n_assets + 1)``.
    """
    cushions = np.linspace(-0.1, 1.0, n_cushion)
    taus = np.linspace(0.0, 1.0, n_time)
    vols_med, trail_med = _median_multi_features()
    n = n_cushion * n_time
    with torch.no_grad():
        T, C = np.meshgrid(taus, cushions)
        w = policy(
            torch.as_tensor(T.ravel(), dtype=torch.float32),
            torch.as_tensor(C.ravel(), dtype=torch.float32),
            torch.as_tensor([vols_med] * n, dtype=torch.float32),
            torch.as_tensor([trail_med] * n, dtype=torch.float32),
        )
    return {
        "cushions": cushions,
        "taus": taus,
        "labels": list(TICKERS) + ["cash"],
        "weights": w.numpy().reshape(n_cushion, n_time, len(TICKERS) + 1),
    }


def repo_metrics(root=REPO_ROOT):
    """Headline metrics written by the full experiments, if present.

    ``runs/`` is gitignored, so a fresh clone has no metrics until the
    experiments run. Returns None when nothing is on disk.
    """
    root = os.path.abspath(root)
    found = {}
    for name, rel in (
        ("single_asset", os.path.join("runs", "cppi", "metrics.json")),
        ("multi_asset", os.path.join("runs", "cppi_multi", "metrics.json")),
        ("drawdown_63d",
         os.path.join("runs", "cppi_multi", "drawdown_63d.json")),
    ):
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path) as f:
                found[name] = json.load(f)
    return found or None
