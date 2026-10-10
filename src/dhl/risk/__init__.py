"""Convex risk measures for the terminal hedging error.

All measures take a 1-D torch tensor of losses ``L = payoff - X_T``
(the hedger's shortfall) and return a scalar. They are differentiable
almost everywhere so they can be used directly as training objectives.
"""

import math

import torch

from dhl.risk.measures import RISK_MEASURES, cvar, entropic_risk, variance

__all__ = ["RISK_MEASURES", "cvar", "entropic_risk", "variance"]
