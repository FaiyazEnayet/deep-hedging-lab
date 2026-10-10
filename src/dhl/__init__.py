"""deep-hedging-lab: train neural networks to hedge derivatives.

A from-scratch implementation of deep hedging (Buehler et al., 2019):
optimal hedging under market frictions, learned directly from simulated
paths by minimizing a convex risk measure of the terminal hedging error.
"""

__version__ = "0.1.0"

from dhl.markets import simulate_gbm, simulate_heston, simulate_merton
from dhl.derivatives import (
    european_call,
    european_put,
    digital_call,
    asian_call,
    up_and_out_call,
    bs_call_price,
    bs_call_delta,
)
from dhl.risk import cvar, entropic_risk, variance
from dhl.hedgers import (
    hedging_error,
    DeepHedger,
    train_deep_hedger,
    bs_delta_hedge,
    leland_delta_hedge,
)

__all__ = [
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton",
    "european_call",
    "european_put",
    "digital_call",
    "asian_call",
    "up_and_out_call",
    "bs_call_price",
    "bs_call_delta",
    "cvar",
    "entropic_risk",
    "variance",
    "hedging_error",
    "DeepHedger",
    "train_deep_hedger",
    "bs_delta_hedge",
    "leland_delta_hedge",
]
