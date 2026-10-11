# deep-hedging-lab

**I teach neural networks to protect a portfolio.**

A network learns how to allocate between a risky asset and cash, day by day, guarding a floor under wealth while transaction costs eat every rebalance. The benchmark is the textbook rule, classic CPPI. The question: can the network learn a better trade between growth and safety?

I started this lab hedging derivatives and then asked myself what the machinery was really for. Strip away the payoff and what remains is risk management: scenarios, a floor, a cost for every trade, and a policy that has to survive all of it. That is a cleaner problem and an honest one. So I rebuilt the repo around it. No options. No derivatives of any kind. Just a portfolio, a floor, and a network that learns not to fall through it.

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

- **Price scenario generators:** GBM, Heston stochastic volatility with full truncation Euler or Andersen QE, Merton jump diffusion. These are SDEs that produce paths. Nothing more. Plus real SPY daily bars from Yahoo (2015 to 2026, committed under `data/`) and a circular block bootstrap that turns history into test scenarios while keeping volatility clustering.
- **Risk measures:** variance, empirical CVaR, entropic risk. Differentiable, usable as training objectives.
- **The experiment:** classic CPPI at multipliers 3, 4, 5 against the learned allocator, on 20k held out bootstrapped SPY paths the network never saw. Floor 90, 20 bps costs, daily rebalancing, cash earns 2%. Run it: `python -m dhl.experiments.cppi_learned`.
- **Exact accounting:** self financing wealth, proportional costs on every rebalance, terminal liquidation. Tested.

## Quickstart

```bash
pip install -e .
pytest
python -m dhl.experiments.cppi_learned   # trains and evaluates, writes runs/cppi/metrics.json
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

## Honest limitations

- The scenarios are bootstrapped from one asset's history. SPY from 2015 to 2026 is a kind decade. A different decade gives different numbers.
- Gap risk is real. An overnight crash bigger than the cushion blows through any daily rule, learned or classic. The test suite pins this down on purpose.
- The floor is fixed at 90, not grown at the cash rate. A stricter floor changes the trade.
- The network's breach probability is small but not zero. Classic CPPI breaches never on these paths because the formula de risks to exactly zero at the floor. The learned policy accepts the tail to buy return.
- CPU friendly by design. Training takes minutes, not hours.

## Where I would take it next

- Multi asset allocation with a correlation aware policy
- A drawdown constraint instead of a fixed floor
- Regime aware features: credit spreads, term structure slope
- Comparing learned policies across different historical decades
