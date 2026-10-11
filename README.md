# neural-cppi

**I teach neural networks to protect a portfolio.**

A network learns how to allocate between risky assets and cash, day by day, guarding a floor under wealth while transaction costs eat every rebalance. The benchmark is the textbook rule, classic CPPI. The question: can the network learn a better trade between growth and safety?

I built this lab around a simple question: can a network learn to protect a portfolio? What remains is risk management in its cleanest form: scenarios, a floor, a cost for every trade, and a policy that has to survive all of it. Just a portfolio, a floor, and a network that learns not to fall through it. Stocks and cash only. No derivatives of any kind.

```python
from dhl import CPPIPolicy, train_cppi_policy, block_bootstrap_paths, load_spy_daily, log_returns

data = load_spy_daily()
returns = log_returns(data["adjclose"])
sampler = lambda n, s: block_bootstrap_paths(returns, n, 30, 21, S0=100.0, seed=s)

policy = CPPIPolicy(hidden_dims=(16, 16))
train_cppi_policy(policy, sampler, horizon_days=30, n_epochs=200)
```

## How it works

Portfolio insurance is simple to state and hard to do well. You start with 100. You promise yourself you will never finish below 90. Every day you choose how much of your wealth sits in the risky asset. Classic CPPI answers with a formula: hold the multiplier times your cushion above the floor. It is elegant and it is rigid. It cannot see that volatility just spiked or that the horizon is almost over.

The network sees three things: how much time is left, how far wealth sits above the floor, and recent volatility. It outputs a risky weight between 0 and 1.5. It trains on fresh market scenarios every epoch, maximizing expected terminal wealth minus 30 times the expected shortfall below the floor. Infinite training data, no memorization possible. The scenarios come from real SPY history, resampled in 21 day blocks so volatility clustering survives.

## What is in here

- **Price scenario generators:** GBM, Heston stochastic volatility with full truncation Euler or Andersen QE, Merton jump diffusion. These are SDEs that produce paths. Nothing more. Plus real daily bars from Yahoo (SPY, TLT, GLD, 2015 to 2026, committed under `data/`) and a circular block bootstrap that turns history into test scenarios while keeping volatility clustering. The multi asset bootstrap draws the same blocks for every asset, so cross asset correlation survives.
- **Risk measures:** variance, empirical CVaR, entropic risk. Differentiable, usable as training objectives.
- **The experiment:** classic CPPI at multipliers 3, 4, 5 against the learned allocator, on 20k held out bootstrapped SPY paths the network never saw. Floor 90, 20 bps costs, daily rebalancing, cash earns 2%. Run it: `python -m dhl.experiments.cppi_learned`.
- **The multi asset experiment:** the policy outputs softmax weights over SPY, TLT, GLD, and cash. Softmax means long only and at most fully invested: no leverage, no shorting, by construction. It adds drawdown floors (the floor ratchets with the running wealth peak) and a cross decade stress test (train before 2020, evaluate on 2020+ paths that include the COVID crash). Run it: `python -m dhl.experiments.cppi_multi`.
- **Exact accounting:** self financing wealth, proportional costs on every rebalance, terminal liquidation. Tested.

## Quickstart

```bash
pip install -e .
pytest
python -m dhl.experiments.cppi_learned   # trains and evaluates, writes runs/cppi/metrics.json
python -m dhl.experiments.cppi_multi     # multi-asset, drawdown floor, cross-decade; writes runs/cppi_multi/metrics.json
```

## Results

30 day horizon, 20,000 held out paths, seed 7:

| Strategy | E[wealth] | Std  | P(breach) |
|----------|-----------|------|-----------|
| CPPI m=3 | 100.52    | 1.61 | 0.000     |
| CPPI m=4 | 100.58    | 2.14 | 0.000     |
| CPPI m=5 | 100.64    | 2.67 | 0.000     |
| Learned  | 100.65    | 2.03 | 0.001     |

63 day horizon:

| Strategy | E[wealth] | Std  | P(breach) |
|----------|-----------|------|-----------|
| CPPI m=3 | 101.22    | 2.46 | 0.000     |
| CPPI m=4 | 101.41    | 3.31 | 0.000     |
| CPPI m=5 | 101.57    | 4.18 | 0.000     |
| Learned  | 102.00    | 4.28 | 0.004     |

The network matches the most aggressive classic multiplier on expected wealth with less wobble at 30 days, and beats it outright at 63 days. The price is a small breach probability where classic CPPI has none on these paths. It learned to trade a little floor risk for a little extra return. Whether that trade is worth it is your call, not the network's.

Numbers move with seeds. Rerun it and check my work.

## What I added next

Three extensions, all on 10,000 held out paths, seed 11. One honest note first: the new comparisons cap the risky allocation at 1.0 (no leverage, long only). The single asset numbers above used a 1.5 cap. The new single asset baseline below was retrained under the same no leverage rule, so the comparisons are fair.

**Multi asset.** The learned allocator splits across SPY, TLT, and GLD plus cash. The classic baseline holds its cushion in an equal weight mix of the three. The single asset learner only sees SPY.

30 day horizon:

| Strategy | E[wealth] | Std | P(breach) |
|---|---|---|---|
| Learned multi (3 assets) | 100.79 | 2.85 | 0.001 |
| Learned single (SPY) | 100.61 | 1.96 | 0.001 |
| Classic m=4 (mix) | 100.36 | 1.37 | 0.000 |

63 day horizon:

| Strategy | E[wealth] | Std | P(breach) |
|---|---|---|---|
| Learned multi (3 assets) | 101.97 | 3.85 | 0.001 |
| Learned single (SPY) | 101.59 | 3.01 | 0.003 |
| Classic m=4 (mix) | 100.97 | 2.11 | 0.000 |

Diversification helps the learner: more return at both horizons, at higher volatility. Classic never breaches. Same trade as before, now across assets.

**Drawdown floor.** Instead of a fixed 90, the floor ratchets: floor_t = 0.90 times the running wealth peak. Stricter and more realistic. 63 day horizon:

| Strategy | Fixed floor | Drawdown floor |
|---|---|---|
| Learned multi | 101.91 / 3.74 / 0.004 | 101.84 / 3.42 / 0.003 |
| Classic m=4 (mix) | 100.97 / 2.11 / 0.000 | 100.93 / 1.87 / 0.000 |

(Each cell: E[wealth] / Std / P(breach anytime).)

Honest read: the ratchet works mechanically (the test suite pins the ratcheting down), but it barely binds on 30 to 63 day paths. The drawdown policy is a touch more conservative at 63 days and nothing else changes much. The feature matters more on longer horizons, where peaks have time to build. I report the small effect instead of overselling it.

**Cross decade stress test.** I trained the single asset learner on pre-2020 paths only, then evaluated on 2020+ paths that include the COVID crash. A regime the network never saw.

30 day horizon, test on 2020+:

| Strategy | E[wealth] | Std | P(breach) |
|---|---|---|---|
| Learned (train pre-2020) | 101.26 | 4.57 | 0.020 |
| Learned (train all years) | 100.74 | 2.20 | 0.002 |
| Classic m=4 | 100.69 | 2.43 | 0.000 |

This is the most interesting table in the repo. The pre-2020 policy earned the most on average and breached ten times as often. It never saw a crash, so it learned to be aggressive, and the COVID regime punished exactly that. The all-years policy saw crashes in training and behaved. Classic CPPI does not care about regimes: a formula has no memory to overfit. The lesson is not that learning fails. It is that a learned policy is only as worldly as its training data.

## Honest limitations

- The scenarios are bootstrapped from a kind decade. SPY from 2015 to 2026 is mostly up. A different decade gives different numbers. The cross decade test shows how much the regime matters.
- Three assets is not a universe. SPY, TLT, and GLD diversify each other, but a real allocator faces dozens of assets, currencies, and frictions I do not model.
- The joint bootstrap preserves each block's correlation, but 21 day blocks cannot invent a correlation regime that never happened.
- COVID is one crash. Passing one out of distribution test is evidence, not proof.
- Gap risk is real. An overnight crash bigger than the cushion blows through any daily rule, learned or classic. The test suite pins this down on purpose.
- The fixed floor is 90, not grown at the cash rate. The drawdown floor only bites on longer horizons. A stricter floor changes the trade.
- The network's breach probability is small but not zero. Classic CPPI breaches never on these paths because the formula de risks to exactly zero at the floor. The learned policy accepts the tail to buy return.
- CPU friendly by design. Training takes minutes, not hours.

## Where I would take it next

- Regime aware features: credit spreads, term structure slope
- More assets and real rebalancing frictions: bid ask spreads, market impact
- Longer horizons where the drawdown floor binds hard
