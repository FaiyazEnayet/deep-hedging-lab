"""Tests for chronological out-of-time evaluation.

Fast by design: synthetic series and tiny real-data slices only.
No training here -- that lives in the experiment script.
"""

import numpy as np
import torch

from dhl.eval.chrono import (
    contiguous_paths,
    max_drawdown,
    paired_evaluate,
    split_returns_by_date,
    window_paths_in_range,
)
from dhl.experiments.cppi_learned import (
    FLOOR,
    V0,
    _classic_weight_fn,
    rollout,
)
from dhl.markets.real import load_spy_daily


def _toy_series(n=500, start="2018-01-01"):
    import datetime
    base = datetime.date.fromisoformat(start)
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(n)]
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    rets = np.log(prices[1:] / prices[:-1])
    return dates, prices, rets


def test_split_has_no_date_leakage():
    dates, _, rets = _toy_series(800, "2018-01-01")
    splits, sdates = split_returns_by_date(
        dates, rets, "2019-01-01", "2019-07-01")
    assert max(sdates["train"]) < "2019-01-01"
    assert min(sdates["val"]) >= "2019-01-01"
    assert max(sdates["val"]) < "2019-07-01"
    assert min(sdates["test"]) >= "2019-07-01"
    total = sum(len(v) for v in splits.values())
    assert total == len(rets)


def test_split_rejects_misaligned_inputs():
    dates, _, rets = _toy_series(100)
    try:
        split_returns_by_date(dates[:-1], rets, "2019-01-01", "2019-07-01")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for misaligned dates")


def test_contiguous_paths_match_history_exactly():
    _, prices, _ = _toy_series(200)
    paths, starts = contiguous_paths(prices, 30, stride=7, s0=100.0)
    assert paths.shape[1] == 31
    for k, s in enumerate(starts):
        expected = 100.0 * prices[s:s + 31] / prices[s]
        assert np.allclose(paths[k], expected), f"window {k} not contiguous"


def test_contiguous_paths_rejects_short_series():
    try:
        contiguous_paths(np.ones(10), 30)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for short series")


def test_crash_window_stays_inside_range():
    dates, prices, _ = _toy_series(400, "2019-01-01")
    paths, wdates = window_paths_in_range(
        dates, prices, "2019-06-01", "2019-08-01", 20, stride=3)
    assert all("2019-06-01" <= d < "2019-08-01" for d in wdates)
    assert paths.shape[1] == 21
    # every window fully inside the range
    assert len(wdates) == len(paths)


def test_real_covid_window_contains_crash():
    data = load_spy_daily()
    dates, prices = data["dates"], data["adjclose"]
    paths, wdates = window_paths_in_range(
        dates, prices, "2020-02-19", "2020-05-01", 30, stride=1)
    assert len(paths) > 5, "expected a handful of crash windows"
    # the crash itself: SPY fell ~30% Feb 19 -> Mar 23, 2020
    trough = paths[:, 1:].min(axis=1) / 100.0
    assert trough.min() < 0.80, "no window captured the crash drawdown"


def test_max_drawdown_on_known_path():
    # peak 110, trough 70, recover to 90 -> max DD = 40/110
    W = np.array([[100.0, 110.0, 70.0, 90.0]])
    dd = max_drawdown(W)
    assert dd.shape == (1,)
    assert abs(dd[0] - 40.0 / 110.0) < 1e-9


def test_max_drawdown_zero_for_monotone_gain():
    W = np.array([[100.0, 101.0, 105.0, 110.0]])
    assert max_drawdown(W)[0] == 0.0


def test_paired_evaluate_identical_strategies_agree():
    rng = np.random.default_rng(1)
    S = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (50, 63)), axis=1))
    S = np.concatenate([np.full((50, 1), 100.0), S], axis=1)
    res = paired_evaluate(
        {"a": _classic_weight_fn(4.0), "b": _classic_weight_fn(4.0)},
        S, rollout, FLOOR, reference="a")
    assert res["a"]["mean_wealth"] == res["b"]["mean_wealth"]
    # gap of identical strategies is exactly zero on every path
    assert res["b"]["paired_gap_vs_a"]["mean"] == 0.0
    assert res["b"]["paired_gap_vs_a"]["std"] == 0.0
    assert res["b"]["paired_gap_vs_a"]["frac_paths_better"] == 0.0


def test_paired_evaluate_uses_identical_paths():
    # A strategy that is long iff the first-day return is positive would
    # behave differently under resampling; with paired paths both
    # strategies must see the same first-day move. We verify indirectly:
    # the reference and challenger share n_paths and the gap std is
    # finite and deterministic across two calls with the same S.
    rng = np.random.default_rng(2)
    S = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (30, 30)), axis=1))
    S = np.concatenate([np.full((30, 1), 100.0), S], axis=1)
    kw = dict(strategies={"m3": _classic_weight_fn(3.0),
                          "m5": _classic_weight_fn(5.0)},
              S_np=S, rollout_fn=rollout, floor=FLOOR, reference="m3")
    r1 = paired_evaluate(**kw)
    r2 = paired_evaluate(**kw)
    assert r1["m5"]["paired_gap_vs_m3"]["mean"] == \
        r2["m5"]["paired_gap_vs_m3"]["mean"]
    assert r1["m5"]["n_paths"] == 30


def test_paired_evaluate_reports_breach_and_drawdown():
    # Gap crash: flat at 100, then a single-day -40% gap. CPPI holds
    # 0.4 risky into the gap (cushion 10 * m4 / 100), so wealth drops
    # to ~84 < 90 floor: a genuine breach the metrics must catch.
    n, steps = 40, 30
    S = np.full((n, steps + 1), 100.0)
    S[:, -1] = 60.0
    res = paired_evaluate({"m4": _classic_weight_fn(4.0)}, S, rollout,
                          FLOOR)
    m = res["m4"]
    assert m["p_breach_anytime"] > 0.0
    assert m["mean_max_drawdown"] > 0.0
    assert m["worst_drawdown"] >= m["mean_max_drawdown"]
    for k in ("mean_wealth", "std_wealth", "p_breach_terminal",
              "mean_max_drawdown"):
        assert k in m


def test_paired_evaluate_rejects_bad_reference():
    S = np.full((5, 11), 100.0)
    try:
        paired_evaluate({"a": _classic_weight_fn(4.0)}, S, rollout,
                        FLOOR, reference="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad reference")
