"""Matplotlib figures saved alongside experiment metrics."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment(S, errors_dict, deltas_dict, K, dt, outdir):
    """Save P&L histograms and a sample hedging path.

    Parameters
    ----------
    S : np.ndarray, shape (n_paths, n_steps + 1)
    errors_dict : mapping label -> np.ndarray of terminal hedging errors
        (payoff - X_T); negative means the hedge made money.
    deltas_dict : mapping label -> np.ndarray deltas, shape (n_paths, n_steps)
    """
    n_steps = S.shape[1] - 1

    # --- hedging-error distribution ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, err in errors_dict.items():
        ax.hist(-np.asarray(err), bins=80, alpha=0.45, label=label, density=True)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("terminal hedging P&L  (X_T - payoff)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pnl_hist.png"), dpi=120)
    plt.close(fig)

    # --- sample path with holdings ----------------------------------------
    i = 0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    times = np.arange(n_steps + 1) * dt
    ax1.plot(times, S[i], lw=1.5)
    ax1.axhline(K, color="k", ls="--", lw=1)
    ax1.set_ylabel("spot")
    for label, d in deltas_dict.items():
        ax2.step(times[:-1], d[i], where="post", label=label)
    ax2.set_xlabel("time (years)")
    ax2.set_ylabel("holding")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sample_path.png"), dpi=120)
    plt.close(fig)


def plot_training(history, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training objective")
    ax.set_title("Deep hedger training curve")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "training.png"), dpi=120)
    plt.close(fig)
