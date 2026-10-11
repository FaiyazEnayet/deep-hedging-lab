"""Convex risk measures for portfolio losses.

All measures take a 1-D torch tensor of losses (for example a
portfolio's shortfall below its floor across scenarios) and return a
scalar. They are differentiable almost everywhere so they can be used
directly as training objectives.
"""

import math

import torch

from dhl.risk.measures import RISK_MEASURES, cvar, entropic_risk, variance

__all__ = ["RISK_MEASURES", "cvar", "entropic_risk", "variance"]
