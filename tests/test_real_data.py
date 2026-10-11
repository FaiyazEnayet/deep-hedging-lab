"""Tests for the real-data module (SPY daily bars + block bootstrap).

No network in tests: they run against the committed ``data/spy_daily_*.csv``
parts and synthetic return series.
"""

import numpy as np

from dhl.markets.real import (
    block_bootstrap_paths,
    load_spy_daily,
    log_returns,
)

def test_load_spy_daily():
    data = load_spy_daily()
    n = len(data["dates"])
    assert n > 2000  # multi-year daily history
    for key in ("open", "high", "low", "close", "adjclose", "volume"):
        assert data[key].shape == (n,)
    assert np.all(data["adjclose"] > 0)
    assert np.all(data["high"] >= data["low"])
    # Adjusted for splits/dividends: adjclose <= close in the distant past.
    assert data["adjclose"][0] < data["close"][0]


def test_log_returns():
    r = log_returns(np.array([100.0, 110.0, 99.0]))
    assert r.shape == (2,)
    assert abs(r[0] - np.log(1.1)) < 1e-12


def test_block_bootstrap_shape_and_positivity():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0004, 0.01, size=1000)
    S = block_bootstrap_paths(rets, n_paths=500, path_len=30, block_len=21,
                              S0=100.0, seed=3)
    assert S.shape == (500, 31)
    assert np.all(S[:, 0] == 100.0)
    assert np.all(S > 0)


def test_block_bootstrap_deterministic():
    rets = np.random.default_rng(1).normal(0, 0.01, size=500)
    S1 = block_bootstrap_paths(rets, 100, 30, seed=42)
    S2 = block_bootstrap_paths(rets, 100, 30, seed=42)
    assert np.array_equal(S1, S2)


def test_block_bootstrap_preserves_blocks():
    # Every sampled return must be a genuine historical return, and
    # consecutive returns inside a block must be consecutive in history
    # (circularly): that is the whole point of the block bootstrap.
    rets = np.arange(200, dtype=float) * 0.001  # 0, 0.001, ..., 0.199
    S = block_bootstrap_paths(rets, n_paths=200, path_len=30, block_len=10,
                              S0=1.0, seed=5)
    sampled = np.diff(np.log(S), axis=1).round(9)
    assert set(np.unique(sampled)).issubset(set(np.round(rets, 9)))
    # Within each 10-step block, values step by 0.001 (mod 0.2, circular).
    for b in range(3):
        blk = sampled[:, b * 10:(b + 1) * 10]
        steps = np.round(np.diff(blk, axis=1) * 1000).astype(int) % 200
        assert np.all(steps == 1)


def test_block_bootstrap_needs_enough_history():
    import pytest

    with pytest.raises(ValueError, match="block_len"):
        block_bootstrap_paths(np.ones(5), 10, 30, block_len=21, seed=0)


def test_spy_bootstrap_smoke():
    data = load_spy_daily()
    rets = log_returns(data["adjclose"])
    S = block_bootstrap_paths(rets, 2000, 30, block_len=21, S0=100.0, seed=6)
    assert S.shape == (2000, 31) and np.all(S > 0)
    # Bootstrapped daily vol should resemble historical daily vol.
    hist_vol = rets.std()
    boot_vol = np.diff(np.log(S), axis=1).std()
    assert abs(boot_vol - hist_vol) / hist_vol < 0.15
