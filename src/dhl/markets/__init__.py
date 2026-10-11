"""Price scenario generators for a risky asset.

All simulators return paths of shape ``(n_paths, n_steps + 1)`` with the
starting price in column 0. These are SDEs used to generate scenarios
for portfolio experiments: GBM, Heston stochastic volatility, Merton
jump diffusion, and a block bootstrap over real SPY history. They
produce price paths, nothing more.
"""

from dhl.markets.gbm import simulate_gbm
from dhl.markets.heston import simulate_heston
from dhl.markets.jump import simulate_merton
from dhl.markets.real import (
    block_bootstrap_paths,
    fetch_spy_daily_yahoo,
    load_spy_daily,
    log_returns,
)

__all__ = [
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton",
    "block_bootstrap_paths",
    "fetch_spy_daily_yahoo",
    "load_spy_daily",
    "log_returns",
]
