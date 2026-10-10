"""Deep hedgers: neural networks that learn optimal hedging strategies.

Following Buehler et al. (2019), the hedger is a parametric policy mapping
market information available at each rebalancing date to a position in the
underlying. The policy and the initial capital ``p0`` are trained
end-to-end by minimizing a convex risk measure of the terminal hedging
error over freshly simulated paths (infinite data, no overfitting).

Two architectures are provided:

- ``mlp``: a feed-forward network shared across time steps, taking
  ``[log-moneyness, sqrt(time-to-maturity), previous holding]`` as input.
  The recurrent dependence on the previous holding lets it learn
  no-trade regions under transaction costs.
- ``lstm``: an LSTM cell whose hidden state carries the trading history;
  input is ``[log-moneyness, sqrt(time-to-maturity)]``.

Outputs are squashed through a sigmoid into ``position_bounds``
(default ``[0, 1]``, the no-arbitrage range of a call delta).
Unconstrained outputs train fine and blow up out of sample. Learned
that one the hard way.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class _MLPPolicy(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        dim = 3
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.Tanh()]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


class DeepHedger(nn.Module):
    """Neural hedging policy with learnable initial capital."""

    def __init__(
        self,
        hidden_dims=(32, 32),
        architecture="mlp",
        position_bounds=(0.0, 1.0),
        p0_init=0.0,
    ):
        super().__init__()
        assert architecture in ("mlp", "lstm")
        self.architecture = architecture
        self.lo, self.hi = position_bounds
        self.p0 = nn.Parameter(torch.tensor(float(p0_init)))

        if architecture == "mlp":
            self.policy = _MLPPolicy(hidden_dims)
            self._lstm = None
            self._head = None
        else:
            hidden = hidden_dims[0] if hidden_dims else 32
            self._lstm = nn.LSTMCell(2, hidden)
            self._head = nn.Sequential(nn.Tanh(), nn.Linear(hidden, 1))
            self.policy = None
        self._hidden = hidden_dims[0] if hidden_dims else 32

    def _features(self, s, K, tau, T):
        # tau, T are Python floats; features come back as (n_paths, 1).
        log_moneyness = torch.log(s / K)
        time_feat = torch.full_like(log_moneyness, math.sqrt(max(tau / T, 0.0)))
        return log_moneyness, time_feat

    def forward(self, S, K, dt):
        """Compute holdings for every path and rebalancing date.

        Parameters
        ----------
        S : torch.Tensor, shape (n_paths, n_steps + 1)
        K : float
        dt : float

        Returns
        -------
        torch.Tensor, shape (n_paths, n_steps)
        """
        n_paths, n_steps_plus1 = S.shape
        n_steps = n_steps_plus1 - 1
        T = n_steps * dt
        device, dtype = S.device, S.dtype

        deltas = []
        if self.architecture == "mlp":
            prev = torch.zeros((n_paths, 1), device=device, dtype=dtype)
            for i in range(n_steps):
                tau = T - i * dt
                m, t = self._features(S[:, i : i + 1], K, tau, T)
                raw = torch.sigmoid(self.policy(torch.cat([m, t, prev], dim=1)))
                delta = self.lo + (self.hi - self.lo) * raw
                deltas.append(delta)
                prev = delta
        else:
            hx = torch.zeros((n_paths, self._hidden), device=device, dtype=dtype)
            cx = torch.zeros((n_paths, self._hidden), device=device, dtype=dtype)
            for i in range(n_steps):
                tau = T - i * dt
                m, t = self._features(S[:, i : i + 1], K, tau, T)
                hx, cx = self._lstm(torch.cat([m, t], dim=1), (hx, cx))
                raw = torch.sigmoid(self._head(hx))
                deltas.append(self.lo + (self.hi - self.lo) * raw)
        return torch.cat(deltas, dim=1)


def train_deep_hedger(
    model,
    sample_paths,
    payoff_fn,
    K,
    r,
    dt,
    cost_rate,
    risk_fn,
    n_epochs=200,
    batch_size=4096,
    lr=1e-3,
    seed=0,
    log_every=20,
    grad_clip=1.0,
    device="cpu",
):
    """Train a deep hedger on freshly simulated paths.

    Parameters
    ----------
    model : DeepHedger
    sample_paths : callable ``(n_paths, seed) -> np.ndarray``
    payoff_fn : callable ``(S: np.ndarray) -> np.ndarray``
    risk_fn : callable ``(losses: torch.Tensor) -> torch.Tensor``

    Returns
    -------
    history : list of float
        Training objective per epoch.
    """
    from dhl.hedgers.accounting import hedging_error

    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(n_epochs):
        S_np = sample_paths(batch_size, seed + epoch)
        S = torch.as_tensor(S_np, dtype=torch.float32, device=device)
        payoff = torch.as_tensor(payoff_fn(S_np), dtype=torch.float32, device=device)

        deltas = model(S, K, dt)
        losses = hedging_error(S, deltas, payoff, model.p0, r, dt, cost_rate)
        objective = risk_fn(losses)

        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        history.append(float(objective.detach().cpu()))
        if log_every and (epoch + 1) % log_every == 0:
            print(
                f"epoch {epoch + 1:4d}/{n_epochs}  objective={history[-1]:.6f}  "
                f"p0={float(model.p0.detach().cpu()):.4f}",
                flush=True,
            )
    return history
