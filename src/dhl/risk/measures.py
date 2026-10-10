"""Differentiable convex risk measures (PyTorch).

The deep hedger minimizes one of these applied to the terminal hedging
error ``L = payoff - X_T`` over a batch of simulated paths.
"""

import math

import torch


def variance(losses):
    """Var[L]: penalizes dispersion symmetrically."""
    return losses.var(unbiased=False)


def cvar(losses, alpha=0.95):
    """Empirical CVaR_alpha[L]: mean of the worst (1 - alpha) fraction.

    Uses ``torch.topk``, whose gradient flows to the selected tail
    observations, so this is a valid training objective. Surprisingly
    well-behaved in practice. No smoothing tricks needed.
    """
    losses = losses.reshape(-1)
    n = losses.numel()
    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    tail, _ = torch.topk(losses, k)
    return tail.mean()


def entropic_risk(losses, gamma=1.0):
    """Entropic risk: (1/gamma) log E[exp(gamma L)].

    Interpolates between mean (gamma -> 0) and worst case (gamma -> inf).
    Computed via log-sum-exp for numerical stability.
    """
    losses = losses.reshape(-1)
    n = losses.numel()
    return (torch.logsumexp(gamma * losses, dim=0) - math.log(n)) / gamma


RISK_MEASURES = {
    "variance": variance,
    "cvar": cvar,
    "entropic": entropic_risk,
}
