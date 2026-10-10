"""Config-driven experiment runner.

An experiment:

1. simulates training paths from a market model,
2. trains a deep hedger to minimize a convex risk measure,
3. evaluates it on fresh test paths against Black-Scholes and Leland
   delta-hedging baselines,
4. writes ``metrics.json`` and plots to ``runs/<name>/``.

Example
-------
>>> run_experiment("configs/gbm_call.yaml")
"""

import json
import math
import os
import time

import numpy as np
import torch
import yaml

from dhl.derivatives import PAYOFFS, bs_call_price
from dhl.hedgers import (
    DeepHedger,
    bs_delta_hedge,
    hedging_error,
    leland_delta_hedge,
    train_deep_hedger,
)
from dhl.markets import simulate_gbm, simulate_heston, simulate_merton
from dhl.risk import RISK_MEASURES


def _make_sampler(market_cfg, n_steps, T):
    name = market_cfg["name"]
    p = dict(market_cfg.get("params", {}))
    p.setdefault("T", T)  # configs carry T in params; don't pass it twice

    if name == "gbm":
        def sample(n_paths, seed):
            return simulate_gbm(n_paths, n_steps, seed=seed, **p)
    elif name == "heston":
        def sample(n_paths, seed):
            S, _ = simulate_heston(n_paths, n_steps, seed=seed, **p)
            return S
    elif name == "merton":
        def sample(n_paths, seed):
            return simulate_merton(n_paths, n_steps, seed=seed, **p)
    else:
        raise ValueError(f"unknown market '{name}'")
    return sample


def _payoff_fn(deriv_cfg):
    name = deriv_cfg["name"]
    fn = PAYOFFS[name]
    kwargs = deriv_cfg.get("params", {})

    def payoff(S):
        return fn(S, **kwargs)

    return payoff


def _summarize(name, errors):
    e = np.asarray(errors, dtype=float)
    return {
        "strategy": name,
        "mean": float(e.mean()),
        "std": float(e.std()),
        "cvar_95": float(np.mean(np.sort(e)[-max(1, int(0.05 * len(e))) :])) if len(e) else float("nan"),
        "worst": float(e.max()),
        "prob_profit": float((e < 0).mean()),
    }


def run_experiment(config_path, outdir=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    mkt, der, hdg = cfg["market"], cfg["derivative"], cfg["hedge"]
    tr = cfg["training"]
    n_steps, T = hdg["n_steps"], hdg["T"]
    dt = T / n_steps
    K = der["params"]["K"]
    r = mkt["params"]["r"]
    cost_rate = hdg.get("cost_rate", 0.0)

    sample = _make_sampler(mkt, n_steps, T)
    payoff_fn = _payoff_fn(der)

    # --- train the deep hedger -------------------------------------------
    arch = tr.get("architecture", "mlp")
    p0_init = tr.get("p0_init", "bs")
    if p0_init == "bs":
        p0_init = float(
            bs_call_price(
                mkt["params"]["S0"], K, T, r, mkt["params"].get("sigma", 0.2)
            )
        )
    model = DeepHedger(
        hidden_dims=tuple(tr.get("hidden_dims", [32, 32])),
        architecture=arch,
        p0_init=float(p0_init),
    )
    risk_name = tr.get("risk", "cvar")
    risk_kwargs = tr.get("risk_kwargs", {})
    risk_fn = lambda L: RISK_MEASURES[risk_name](L, **risk_kwargs)  # noqa: E731

    t0 = time.time()
    history = train_deep_hedger(
        model,
        sample,
        payoff_fn,
        K,
        r,
        dt,
        cost_rate,
        risk_fn,
        n_epochs=tr.get("n_epochs", 200),
        batch_size=tr.get("batch_size", 4096),
        lr=tr.get("lr", 1e-3),
        seed=cfg.get("seed", 0),
        log_every=tr.get("log_every", 20),
    )
    train_time = time.time() - t0

    # --- evaluate on fresh paths ------------------------------------------
    n_test = cfg.get("n_test", 20000)
    S_test = sample(n_test, cfg.get("seed", 0) + 10**6)
    payoff_test = payoff_fn(S_test)
    S_t = torch.as_tensor(S_test, dtype=torch.float32)
    payoff_t = torch.as_tensor(payoff_test, dtype=torch.float32)

    results = {}
    with torch.no_grad():
        d_deep = model(S_t, K, dt).cpu().numpy()
        err_deep = hedging_error(S_t, torch.as_tensor(d_deep), payoff_t, float(model.p0), r, dt, cost_rate).numpy()
    results["deep_hedge"] = _summarize(f"deep_hedge ({arch})", err_deep)
    results["deep_hedge"]["p0_learned"] = float(model.p0.detach().cpu())

    sigma = mkt["params"].get("sigma", hdg.get("baseline_sigma"))
    if sigma is not None and der["name"] == "european_call":
        for label, fn in (
            ("bs_delta", lambda S: bs_delta_hedge(S, K, r, sigma, dt)),
            ("leland", lambda S: leland_delta_hedge(S, K, r, sigma, dt, cost_rate)),
        ):
            d = fn(S_test)
            err = hedging_error(
                S_t, torch.as_tensor(d, dtype=torch.float32), payoff_t,
                float(bs_call_price(mkt["params"]["S0"], K, T, r, sigma)), r, dt, cost_rate,
            ).numpy()
            results[label] = _summarize(label, err)

    # --- persist ------------------------------------------------------------
    name = cfg.get("name", os.path.splitext(os.path.basename(config_path))[0])
    outdir = outdir or os.path.join("runs", name)
    os.makedirs(outdir, exist_ok=True)
    metrics = {
        "config": config_path,
        "train_time_s": train_time,
        "history": history,
        "results": results,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        from dhl.viz.plots import plot_experiment, plot_training

        errors = {"deep_hedge": err_deep}  # hedging error = payoff - X_T
        plot_experiment(S_test, errors, {"deep_hedge": d_deep}, K, dt, outdir)
        plot_training(history, outdir)
    except Exception as exc:  # plotting is best-effort
        print(f"[warn] plotting failed: {exc}")

    print(f"\nsaved metrics + plots to {outdir}")
    for label, stats in results.items():
        print(
            f"{label:20s} mean={stats['mean']:+.4f} std={stats['std']:.4f} "
            f"cvar95={stats['cvar_95']:+.4f} P(profit)={stats['prob_profit']:.2%}"
        )
    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m dhl.experiments.runner <config.yaml>")
        raise SystemExit(1)
    run_experiment(sys.argv[1])
