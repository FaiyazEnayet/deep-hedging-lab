# deep-hedging-lab

**I teach neural networks to manage financial risk.**

Hedging options under transaction costs, stochastic volatility, and jumps. Insuring portfolios with learned dynamic allocation. Same question throughout: can a network learn risk management better than the textbook?

I kept running into the same wall: every "learn quant finance" project is either a moving-average backtest on Yahoo data or Black-Scholes re-implemented for the hundredth time. I wanted something at the intersection I actually care about: risk management and machine learning. So I built the thing I wished existed: deep hedging (Buehler et al., 2019), from scratch, no magic, no 500-line `train.py` you can't read. Then I pointed the same machinery at portfolio insurance, because hedging a derivative and protecting a portfolio are the same problem wearing different clothes.

The idea is simple and kind of beautiful: stop deriving hedging formulas. Simulate a market, let a neural net trade in it, and punish it with a risk measure until it learns to hedge. Under transaction costs and stochastic vol the textbook delta is *wrong*. The network figures out something better entirely on its own.

```python
from dhl import DeepHedger, train_deep_hedger, simulate_heston, european_call
from dhl.risk import cvar

model = DeepHedger(hidden_dims=(32, 32), architecture="mlp", p0_init=3.6)
train_deep_hedger(
    model,
    sample_paths=lambda n, seed: simulate_heston(n, 30, 0.25, 100.0, 0.02, seed=seed, **heston_params)[0],
    payoff_fn=lambda S: european_call(S, 100.0),
    K=100.0, r=0.02, dt=0.25 / 30, cost_rate=0.002,
    risk_fn=lambda L: cvar(L, 0.95),
    n_epochs=300,
)
```

## Things I learned building this

- **The accounting has to be exactly right or nothing converges.** I burned an embarrassing amount of time on sign errors in the financing terms before the P&L identity checked out against closed forms. The test suite now pins it down. `test_accounting.py` is the file I'm proudest of.
- **CVaR as a loss function is weirdly well-behaved.** Top-k gradients just work. I expected to need all kinds of smoothing tricks; I didn't.
- **The MLP discovers no-trade regions by itself.** Nobody told it about Leland (1985). It just... stops trading when costs eat the edge. That was the moment this project felt worth it.
- **LSTMs are overkill for GBM but earn their keep under Heston**, where the vol regime matters and a memoryless policy can't see it.
- Letting the net use leverage (deltas outside [0,1]) trains fine and explodes out of sample. Ask me how I know. Outputs are sigmoid-constrained now.

## What's in here

- **Market simulators:** GBM (exact discretization), Heston stochastic vol (full-truncation Euler or Andersen's QE scheme), Merton jump diffusion. All risk-neutral, all with martingale tests.
- **Real data:** SPY daily bars from Yahoo (2015 to 2026, committed under `data/`) and a circular block bootstrap that turns history into test paths while keeping volatility clustering. The GBM-trained hedger gets evaluated on 20k bootstrapped real paths with no retraining: `python -m dhl.experiments.real_data_eval`.
- **Payoffs:** European call/put, digital, arithmetic Asian, up-and-out barrier.
- **Classical baselines:** Black-Scholes delta hedge and Leland's cost-adjusted vol, so the neural net has something honest to beat.
- **Deep hedgers:** MLP policy with previous-holding memory, LSTM policy, learnable initial capital.
- **Risk measures:** variance, empirical CVaR, entropic risk, all differentiable, all usable as training objectives.
- **Portfolio insurance:** classic CPPI (constant proportion portfolio insurance) vs a neural allocator learned on bootstrapped real paths. Floor at 90% of capital, 20 bps costs, no derivatives involved: `python -m dhl.experiments.cppi_learned`.
- **Exact accounting:** self-financing wealth, discrete rebalancing, proportional costs compounded to maturity, terminal liquidation.
- **No-lookahead guarantee:** tested: shocking future prices can't change earlier holdings.
- **Experiment runner:** YAML configs, seeded runs, `metrics.json` + plots under `runs/`.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                       # 42 tests, about a minute
python -m dhl.experiments.runner configs/gbm_call.yaml
python -m dhl.experiments.real_data_eval   # real SPY paths, no retraining
python -m dhl.experiments.cppi_learned     # portfolio insurance: CPPI vs learned
```

Start with the notebooks: `01_baselines.ipynb` → `02_deep_hedging.ipynb` → `03_stochastic_vol.ipynb`. They read like a lab journal, because that's what they are.

## Honest limitations

- Heston defaults to Euler discretization; Andersen's QE scheme is available via `scheme="qe"` and keeps variance non-negative by construction. Fine for hedging research, don't price exotics to the 4th decimal with either.
- Real data, with caveats. SPY daily bars live in the repo and the trained hedger is evaluated on block-bootstrapped real paths. It is still daily bars on a single name. The bootstrap assumes history is a fair deck to deal from. The paths carry the equity risk premium, so absolute losses do not compare to the risk-neutral tables. It is a robustness check, not a trading system.
- Risk-neutral simulation only. Hedging is about replication, not alpha. If you want drift, add it yourself.
- CPU-friendly by design. A full run takes minutes, not hours.

## Results

European call, GBM, 30 rebalancing dates, 20 bps costs, CVaR@95% objective (`configs/gbm_call.yaml`, seed 0, 20k fresh test paths):

| Strategy   | Mean loss | CVaR95 | P(profit) |
|------------|-----------|--------|-----------|
| BS delta   | 0.56      | 2.54   | 23%       |
| Leland     | 0.55      | 2.46   | 23%       |
| Deep hedge | 0.25      | 2.45   | 38%       |

Two things worth noticing. The network learns a *higher fair premium* (5.46 vs 5.22 Black-Scholes), because hedging under costs genuinely costs more. And it discovers cost-aware no-trade regions by itself. No formula told it any of this.

Under Heston stochastic vol (`configs/heston_call.yaml`, LSTM policy, seed 0, 20k fresh test paths), where the BS delta is genuinely misspecified:

| Strategy   | Mean loss | CVaR95 | P(profit) |
|------------|-----------|--------|-----------|
| BS delta   | 0.51      | 3.26   | 30%       |
| Leland     | 0.51      | 3.17   | 29%       |
| Deep hedge | 0.06      | 3.39   | 46%       |

The learned policy wins on average and lands in profit far more often; the tail is roughly a draw. An honest result, not a sweep. Notebook 03 walks through it.

### Real data

Same GBM-trained MLP hedger, no retraining, evaluated on 20,000 paths block-bootstrapped from real SPY daily returns (2015-01-02 to 2026-10-09, 21-day blocks, seed 999). BS delta uses the sample realized vol of 17.55%. 30-day ATM call, 20 bps costs:

| Strategy   | Mean loss | CVaR95 | P(profit) |
|------------|-----------|--------|-----------|
| Deep hedge | -2.70     | 1.33   | 96%       |
| BS delta   | -0.83     | 2.86   | 86%       |
| Leland     | -0.83     | 2.87   | 85%       |

Negative mean losses are the equity risk premium at work. These paths drift about 13% a year, so do not compare these numbers to the risk-neutral tables above. The honest decomposition: the deep hedger's learned premium is 5.46, priced for a 25% vol world, while realized vol is 17.55% (BS premium 3.75). At a common 3.75 premium the hedging contest is a draw: mean -0.98 vs -0.83, CVaR95 3.05 vs 2.86, profit probability 82% vs 86%. So the transfer win is mostly the premium, not the deltas. The policy holds up out of sample, which is more than I expected. A second bootstrap seed gives the same picture.

### Portfolio insurance

Same machinery, no derivatives. A CPPI strategy protects a floor at 90% of capital by holding multiplier times cushion in SPY, rebalanced daily with 20 bps costs. I trained a small network to choose the allocation directly: it sees time left, cushion ratio, and recent volatility, and outputs a risky weight in [0, 1.5]. The objective is expected terminal wealth minus 30 times the expected shortfall below the floor. Training and testing both use block-bootstrapped real SPY paths, 20k held-out paths on seeds the network never saw.

30-day horizon:

| Strategy | E[wealth] | Std  | P(breach) |
|----------|-----------|------|-----------|
| CPPI m=3 | 100.52    | 1.61 | 0.000     |
| CPPI m=4 | 100.58    | 2.14 | 0.000     |
| CPPI m=5 | 100.64    | 2.67 | 0.000     |
| Learned  | 100.65    | 2.03 | 0.001     |

63-day horizon:

| Strategy | E[wealth] | Std  | P(breach) |
|----------|-----------|------|-----------|
| CPPI m=3 | 101.22    | 2.46 | 0.000     |
| CPPI m=4 | 101.41    | 3.31 | 0.000     |
| CPPI m=5 | 101.57    | 4.18 | 0.000     |
| Learned  | 102.00    | 4.28 | 0.004     |

The network matches the most aggressive classic multiplier on expected wealth with less wobble at 30 days, and beats it outright at 63 days. The price is a small breach probability where classic CPPI has none on these paths. It learned to trade a little floor risk for a little extra return. Whether that trade is worth it is your call, not the network's.

Numbers move with seeds. Rerun the config and check my work.

## Theory in 60 seconds

You're short an option paying `Z` at `T`. You trade the stock at discrete dates, paying proportional costs. Self-financing gives your terminal wealth `X_T`; your loss is `Z − X_T`. Pick a convex risk measure `ρ` and solve `min_δ ρ(Z − X_T(δ))` over adapted strategies. A neural net parametrizes `δ`; Monte Carlo gives unbiased gradients. That's the whole paper. The rest is engineering.

## Reference

Hans Buehler, Lukas Gonon, Josef Teichmann, Ben Wood, *"Deep hedging"*, Quantitative Finance 19(8), 2019. [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)

## Where I'd take it next

- Basket hedging with cross-gamma features (multi-asset is where this gets real)
- Rough Bergomi simulator: hedging under rough vol is still open territory
- A limit-order-book execution layer on top of the hedge
- Distributional risk measures, actor-critic training

Issues and PRs welcome. If you break it in an interesting way, tell me about it.
