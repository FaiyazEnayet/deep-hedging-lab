"""Option payoffs evaluated on simulated paths.

Each payoff takes a path array ``S`` of shape ``(n_paths, n_steps + 1)``
and returns an array of shape ``(n_paths,)`` with the undiscounted
time-T payoff.
"""

import numpy as np


def european_call(S, K):
    return np.maximum(S[:, -1] - K, 0.0)


def european_put(S, K):
    return np.maximum(K - S[:, -1], 0.0)


def digital_call(S, K):
    return (S[:, -1] > K).astype(float)


def asian_call(S, K):
    """Arithmetic Asian call on the average of the fixing dates (excl. S0)."""
    return np.maximum(S[:, 1:].mean(axis=1) - K, 0.0)


def up_and_out_call(S, K, barrier):
    """Up-and-out barrier call: knocked out if S ever touches the barrier."""
    knocked_out = S.max(axis=1) >= barrier
    payoff = np.maximum(S[:, -1] - K, 0.0)
    return np.where(knocked_out, 0.0, payoff)


PAYOFFS = {
    "european_call": european_call,
    "european_put": european_put,
    "digital_call": digital_call,
    "asian_call": asian_call,
    "up_and_out_call": up_and_out_call,
}
