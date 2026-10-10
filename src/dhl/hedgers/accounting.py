"""Self-financing hedging accounting with transaction costs.

We consider a hedger who is short one derivative: they receive a premium
``p0`` at time 0 and owe ``payoff`` at maturity ``T``. Trading happens at
``N`` discrete dates with step ``dt``. The strategy ``deltas`` holds
``delta_i`` shares over ``[t_i, t_{i+1})`` and liquidates at ``T``.

Terminal wealth of a self-financing strategy::

    X_T = p0 * e^{rT}
        + sum_i delta_i * (S_{i+1} - S_i * e^{r dt})      # trading gains
        - sum_i c * |delta_i - delta_{i-1}| * S_i * e^{r(T - t_i)}  # costs

with ``delta_{-1} = delta_N = 0``. The hedger's loss is ``payoff - X_T``;
every strategy in this library is trained/evaluated by applying a convex
risk measure to that loss.

Get the signs wrong here and nothing converges. I know from experience.
The test suite pins every term down against closed forms.
"""

import math

import torch


def hedging_error(S, deltas, payoff, p0, r, dt, cost_rate):
    """Terminal hedging error of a short position.

    Parameters
    ----------
    S : torch.Tensor, shape (n_paths, n_steps + 1)
    deltas : torch.Tensor, shape (n_paths, n_steps)
    payoff : torch.Tensor, shape (n_paths,)
    p0 : float or torch.Tensor
        Premium received at time 0.
    r, dt, cost_rate : float

    Returns
    -------
    torch.Tensor, shape (n_paths,): ``payoff - X_T`` (the hedger's loss).
    """
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    T = n_steps * dt
    device, dtype = S.device, S.dtype

    times = torch.arange(n_steps_plus1, device=device, dtype=dtype) * dt
    growth_to_T = torch.exp(r * (T - times))  # compounds costs to maturity
    growth_step = math.exp(r * dt)

    # Discounted trading gains of the stock position.
    gains = (deltas * (S[:, 1:] - S[:, :-1] * growth_step)).sum(dim=1)

    # Trades include opening (from 0) and liquidating (to 0) at maturity.
    zeros = torch.zeros((n_paths, 1), device=device, dtype=dtype)
    positions = torch.cat([zeros, deltas, zeros], dim=1)
    trades = positions[:, 1:] - positions[:, :-1]
    costs = (trades.abs() * S * growth_to_T * cost_rate).sum(dim=1)

    wealth_T = p0 * math.exp(r * T) + gains - costs
    return payoff - wealth_T
