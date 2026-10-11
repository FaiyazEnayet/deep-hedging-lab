"""Portfolio insurance: classic CPPI vs a learned dynamic allocator.

This reframes the lab's machinery around risk management instead of
derivatives. CPPI (constant proportion portfolio insurance) protects a
floor under wealth by holding ``multiplier * cushion`` in the risky
asset, where ``cushion = wealth - floor``. Here a small neural policy
learns the allocation directly: it maps (time left, cushion ratio,
recent volatility) to a risky weight, trained to maximize expected
terminal wealth with a penalty for breaching the floor. No options
anywhere. Just a portfolio, a floor, and a network that learns not to
fall through it.

Training uses block-bootstrapped real SPY paths (same 21-day blocks as
``real_data_eval``), fresh every epoch. Evaluation is on held-out
bootstrap seeds.

Run: ``python -m dhl.experiments.cppi_learned [--epochs 200]``

Writes ``runs/cppi/metrics.json`` and prints the comparison table.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from dhl.markets.real import block_bootstrap_paths, load_spy_daily, log_returns

# --- experiment settings -------------------------------------------------
V0 = 100.0
FLOOR = 90.0  # 90% of initial capital, fixed (not grown at r)
R = 0.02  # cash earns 2% annually
DT = 1.0 / 252.0  # daily rebalancing
COST_RATE = 0.002  # 20 bps, same as the hedging experiments
CAP = 1.5  # max risky weight for every strategy (apples to apples)
BLOCK_LEN = 21
HORIZONS = (30, 63)  # trading days
MULTIPLIERS = (3, 4, 5)
N_EPOCHS = 200
BATCH_SIZE = 2048
N_TEST = 20000
FLOOR_PENALTY = 30.0  # weight on expected shortfall below the floor
SEED = 7

OUTDIR = os.path.join("runs", "cppi")


def classic_cppi_weight(V, floor=FLOOR, multiplier=4.0, cap=CAP):
    """Classic CPPI risky weight: multiplier * cushion / wealth, capped.

    Works on floats or torch tensors; cushion is floored at zero so the
    strategy fully de-risks once the floor is breached.
    """
    if torch.is_tensor(V):
        cushion = torch.clamp(V - floor, min=0.0)
        w = multiplier * cushion / V.clamp_min(1e-8)
        return torch.clamp(w, max=cap)
    cushion = max(float(V) - floor, 0.0)
    return min(multiplier * cushion / max(float(V), 1e-8), cap)


class CPPIPolicy(nn.Module):
    """Small MLP: (time_left_frac, cushion_ratio, vol_est) -> [0, cap].

    cushion_ratio = (V - floor) / V, clipped to [-1, 1]; it goes
    negative if the floor is breached, which the network is allowed
    to see. vol_est is the annualized rolling std of the last 21
    daily log returns.
    """

    def __init__(self, hidden_dims=(16, 16), cap=CAP):
        super().__init__()
        self.cap = cap
        layers = []
        dim = 3
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.Tanh()]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tau_frac, cushion_ratio, vol):
        x = torch.stack(
            [
                torch.as_tensor(tau_frac, dtype=torch.float32).expand_as(cushion_ratio),
                cushion_ratio,
                vol,
            ],
            dim=1,
        )
        return (self.cap * torch.sigmoid(self.net(x))).squeeze(1)


def _vol_features(S, window=21):
    """Annualized rolling std of log returns, shape (n_paths, n_steps).

    Pure function of prices (no gradients needed); expanding window
    for the first ``window`` steps.
    """
    n, steps_p1 = S.shape
    steps = steps_p1 - 1
    lr = torch.log(S[:, 1:] / S[:, :-1])
    vols = torch.empty_like(lr)
    for i in range(steps):
        j = max(0, i - window + 1)
        w = lr[:, j : i + 1]
        vols[:, i] = w.std(dim=1, unbiased=False) * math.sqrt(252.0)
    return torch.nan_to_num(vols, nan=0.0)


def rollout(S, weight_fn, floor=FLOOR, r=R, dt=DT, cost_rate=COST_RATE,
            return_path=False):
    """Self-financing portfolio wealth under a dynamic allocation.

    Parameters
    ----------
    S : torch.Tensor, shape (n_paths, n_steps + 1)
    weight_fn : callable ``(i, V, tau_frac, cushion_ratio, vol) -> w``
        Risky weight in [0, cap] chosen at each rebalancing date.
    return_path : if True, also return wealth at every date.

    Returns
    -------
    V_T : torch.Tensor, shape (n_paths,), after liquidating at T.
    """
    n, steps_p1 = S.shape
    steps = steps_p1 - 1
    T = steps * dt
    device, dtype = S.device, S.dtype
    V = torch.full((n,), V0, device=device, dtype=dtype)
    D = torch.zeros(n, device=device, dtype=dtype)  # risky dollars held
    growth = math.exp(r * dt)
    rets = S[:, 1:] / S[:, :-1]
    vols = _vol_features(S)
    path = [V.clone()] if return_path else None

    for i in range(steps):
        tau_frac = (T - i * dt) / T
        cushion = torch.clamp((V - floor) / V.clamp_min(1e-8), -1.0, 1.0)
        w = weight_fn(i, V, tau_frac, cushion, vols[:, i])
        prev_grown = D if i == 0 else D * rets[:, i - 1]
        target = w * V
        V = V - cost_rate * (target - prev_grown).abs()
        D = w * V
        V = (V - D) * growth + D * rets[:, i]
        if return_path:
            path.append(V.clone())

    V = V - cost_rate * D.abs()  # liquidate the risky leg at T
    if return_path:
        return V, torch.stack(path, dim=1)
    return V


def _learned_weight_fn(policy):
    def fn(i, V, tau_frac, cushion, vol):
        t = torch.full_like(V, tau_frac)
        return policy(t, cushion, vol)

    return fn


def _classic_weight_fn(multiplier):
    def fn(i, V, tau_frac, cushion, vol):
        return classic_cppi_weight(V, multiplier=multiplier)

    return fn


def train_cppi_policy(policy, sampler, horizon_days, n_epochs=N_EPOCHS,
                       batch_size=BATCH_SIZE, lr=1e-3,
                       floor_penalty=FLOOR_PENALTY, seed=SEED, log_every=20):
    """Train on fresh bootstrapped paths every epoch (infinite data).

    Objective: maximize E[V_T] - floor_penalty * E[max(floor - V_T, 0)].
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    dt = DT
    history = []
    for epoch in range(n_epochs):
        S_np = sampler(batch_size, seed + epoch)
        S = torch.as_tensor(S_np, dtype=torch.float32)
        V_T = rollout(S, _learned_weight_fn(policy), dt=dt)
        shortfall = torch.relu(FLOOR - V_T)
        objective = -(V_T.mean() - floor_penalty * shortfall.mean())
        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        history.append(float(objective.detach()))
        if log_every and (epoch + 1) % log_every == 0:
            with torch.no_grad():
                ev = float(V_T.mean())
                pb = float((V_T < FLOOR).float().mean())
            print(
                f"epoch {epoch + 1:4d}/{n_epochs}  objective={history[-1]:.4f}  "
                f"E[V_T]={ev:.2f}  P(breach)={pb:.3f}",
                flush=True,
            )
    return history


def evaluate(weight_fn, S_np, floor=FLOOR):
    """Metrics on held-out paths: mean/std wealth, breach probabilities."""
    S = torch.as_tensor(np.asarray(S_np), dtype=torch.float32)
    with torch.no_grad():
        V_T, path = rollout(S, weight_fn, return_path=True)
    V_T = V_T.numpy()
    wealth_min = path.numpy().min(axis=1)
    return {
        "mean_wealth": float(V_T.mean()),
        "std_wealth": float(V_T.std()),
        "p_breach_terminal": float((V_T < floor).mean()),
        "p_breach_anytime": float((wealth_min < floor).mean()),
        "mean_min_wealth": float(wealth_min.mean()),
    }


def run_experiment(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, n_test=N_TEST,
                   seed=SEED):
    data = load_spy_daily()
    returns = log_returns(data["adjclose"])
    print(f"loaded {len(returns)} SPY daily log returns", flush=True)

    results = {}
    for horizon in HORIZONS:
        print(f"\n=== horizon {horizon} days ===", flush=True)

        def make_sampler(h=horizon):
            return lambda n, s: block_bootstrap_paths(
                returns, n, h, BLOCK_LEN, S0=V0, seed=s
            )

        sampler_h = make_sampler()
        S_test = sampler_h(n_test, seed + 999)

        strategies = {}
        for m in MULTIPLIERS:
            strategies[f"classic_m{m}"] = _classic_weight_fn(m)

        policy = CPPIPolicy()
        train_cppi_policy(policy, sampler_h, horizon, n_epochs=n_epochs,
                          batch_size=batch_size, seed=seed, log_every=50)
        strategies["learned"] = _learned_weight_fn(policy)

        horizon_results = {}
        for name, fn in strategies.items():
            metrics = evaluate(fn, S_test)
            horizon_results[name] = metrics
            print(
                f"{name:12s}  E[V_T]={metrics['mean_wealth']:7.2f}  "
                f"sd={metrics['std_wealth']:6.2f}  "
                f"P(breach)={metrics['p_breach_terminal']:.3f}  "
                f"P(any)={metrics['p_breach_anytime']:.3f}",
                flush=True,
            )
        results[f"{horizon}d"] = horizon_results

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTDIR}/metrics.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="CPPI vs learned allocator")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_experiment(n_epochs=args.epochs, batch_size=args.batch_size,
                   n_test=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
