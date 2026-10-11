"""Real-data evaluation: deep hedger vs Black-Scholes on bootstrapped SPY.

The deep hedger is trained exactly as in ``configs/gbm_call.yaml``
(risk-neutral GBM, 30-day ATM call, 20 bps costs, CVaR95 objective).
It is then evaluated, without retraining, on 20k paths block-bootstrapped
from real SPY daily log returns (Yahoo Finance, 2015-01-02 to 2026-10-09,
dividend/split adjusted). This tests out-of-sample robustness: can a
hedger trained in the lab survive contact with real return dynamics
(volatility clustering, fat tails, drift)?

Run: ``python -m dhl.experiments.real_data_eval [--retrain]``

Writes ``runs/real_data/metrics.json`` and prints the comparison table.
"""

import argparse
import json
import os

import numpy as np
import torch

from dhl.derivatives import bs_call_price, european_call
from dhl.hedgers import (
    DeepHedger,
    bs_delta_hedge,
    hedging_error,
    leland_delta_hedge,
    train_deep_hedger,
)
from dhl.markets import (
    block_bootstrap_paths,
    load_spy_daily,
    log_returns,
    simulate_gbm,
)
from dhl.risk import cvar

# --- experiment settings (mirror configs/gbm_call.yaml) -------------------
S0, K, T, R = 100.0, 100.0, 0.25, 0.02
N_STEPS = 30
DT = T / N_STEPS
COST_RATE = 0.002  # 20 bps
SIGMA_TRAIN = 0.25
N_EPOCHS = 300
BATCH_SIZE = 4096
N_TEST = 20000
BLOCK_LEN = 21  # trading days; keeps volatility clustering
SEED = 0

OUTDIR = os.path.join("runs", "real_data")
WEIGHTS = os.path.join(OUTDIR, "deep_hedger.pt")


def _summarize(name, errors):
    e = np.asarray(errors, dtype=float)
    return {
        "strategy": name,
        "mean": float(e.mean()),
        "std": float(e.std()),
        "cvar_95": float(np.mean(np.sort(e)[-max(1, int(0.05 * len(e))) :])),
        "worst": float(e.max()),
        "prob_profit": float((e < 0).mean()),
    }


def train_or_load(retrain=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    p0_init = float(bs_call_price(S0, K, T, R, SIGMA_TRAIN))
    model = DeepHedger(hidden_dims=(32, 32), architecture="mlp", p0_init=p0_init)
    if os.path.exists(WEIGHTS) and not retrain:
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        print(f"loaded trained hedger from {WEIGHTS}")
        return model
    print("training deep hedger on GBM (same setup as configs/gbm_call.yaml)...")
    train_deep_hedger(
        model,
        sample_paths=lambda n, seed: simulate_gbm(
            n, N_STEPS, T=T, S0=S0, r=R, sigma=SIGMA_TRAIN, seed=seed
        ),
        payoff_fn=lambda S: european_call(S, K),
        K=K,
        r=R,
        dt=DT,
        cost_rate=COST_RATE,
        risk_fn=lambda L: cvar(L, 0.95),
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        seed=SEED,
        log_every=50,
    )
    os.makedirs(OUTDIR, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    print(f"saved weights to {WEIGHTS}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the hedger even if weights exist")
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--bootstrap-seed", type=int, default=SEED + 999,
                    help="seed for the block bootstrap (robustness checks)")
    args = ap.parse_args()

    # --- real data ---------------------------------------------------------
    data = load_spy_daily()
    rets = log_returns(data["adjclose"])
    sigma_hist = float(rets.std() * np.sqrt(252))
    print(
        f"SPY daily bars: {len(data['dates'])} "
        f"({data['dates'][0]} -> {data['dates'][-1]}), "
        f"realized vol {sigma_hist:.3f}"
    )

    # --- model --------------------------------------------------------------
    model = train_or_load(retrain=args.retrain)

    # --- block-bootstrap test paths -----------------------------------------
    print(f"bootstrapping {args.n_test} x {N_STEPS}-day paths "
          f"(block_len={BLOCK_LEN})...")
    S_test = block_bootstrap_paths(
        rets, args.n_test, N_STEPS, block_len=BLOCK_LEN, S0=S0,
        seed=args.bootstrap_seed,
    )
    payoff_test = european_call(S_test, K)
    S_t = torch.as_tensor(S_test, dtype=torch.float32)
    payoff_t = torch.as_tensor(payoff_test, dtype=torch.float32)

    results = {}
    with torch.no_grad():
        d_deep = model(S_t, K, DT).cpu().numpy()
        err_deep = hedging_error(
            S_t, torch.as_tensor(d_deep), payoff_t,
            float(model.p0), R, DT, COST_RATE,
        ).numpy()
    results["deep_hedge"] = _summarize("deep_hedge (mlp)", err_deep)
    results["deep_hedge"]["p0_learned"] = float(model.p0.detach().cpu())

    p0_bs = float(bs_call_price(S0, K, T, R, sigma_hist))
    for label, fn in (
        ("bs_delta", lambda S: bs_delta_hedge(S, K, R, sigma_hist, DT)),
        ("leland", lambda S: leland_delta_hedge(S, K, R, sigma_hist, DT, COST_RATE)),
    ):
        d = fn(S_test)
        err = hedging_error(
            S_t, torch.as_tensor(d, dtype=torch.float32), payoff_t,
            p0_bs, R, DT, COST_RATE,
        ).numpy()
        results[label] = _summarize(label, err)
    results["bs_delta"]["sigma_hist"] = sigma_hist
    results["meta"] = {
        "data": f"SPY daily {data['dates'][0]} -> {data['dates'][-1]}",
        "n_bars": len(data["dates"]),
        "block_len": BLOCK_LEN,
        "realized_vol_ann": sigma_hist,
        "note": "physical measure: paths carry the equity risk premium, "
                "so absolute losses are not comparable to risk-neutral tables",
    }

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n30-day ATM call on bootstrapped SPY paths, 20 bps costs:")
    print(f"{'strategy':12s} {'mean':>8s} {'cvar95':>8s} {'P(profit)':>9s}")
    for label in ("deep_hedge", "bs_delta", "leland"):
        s = results[label]
        print(f"{label:12s} {s['mean']:+8.4f} {s['cvar_95']:+8.4f} "
              f"{s['prob_profit']:8.2%}")
    print(f"\nsaved metrics to {OUTDIR}/metrics.json")


if __name__ == "__main__":
    main()
