"""Risk-neutral Monte Carlo simulators for the underlying asset.

All simulators return paths of shape ``(n_paths, n_steps + 1)`` with the
spot price in column 0. Dynamics are simulated under the risk-neutral
measure so that discounted asset prices are martingales.
"""

from dhl.markets.gbm import simulate_gbm
from dhl.markets.heston import simulate_heston
from dhl.markets.jump import simulate_merton

__all__ = ["simulate_gbm", "simulate_heston", "simulate_merton"]
