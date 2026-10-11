"""Make an animated GIF demo of the dashboard: docs/demo.gif.

Shows the median and p10/p90 wealth bands of the learned allocator vs
classic CPPI accumulating day by day on bootstrapped SPY paths. Every
frame is computed from this repo's own code and committed data: nothing
is faked.

Run:  python scripts/make_demo_gif.py [--fast] [--out docs/demo.gif]
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from dhl.viz import dashboard as D

HORIZON = 63
FLOOR = 90.0
MULTIPLIER = 4.0


def build(out_path, fast=False):
    n_epochs = 2 if fast else 30
    n_paths = 64 if fast else 800
    print("training demo policy...", flush=True)
    policy = D.train_demo_policy(floor=FLOOR, horizon_days=HORIZON,
                                 n_epochs=n_epochs, seed=7)
    print("simulating wealth paths...", flush=True)
    sim = D.simulate_wealth(policy, floor=FLOOR, horizon_days=HORIZON,
                            multiplier=MULTIPLIER, n_paths=n_paths,
                            seed=999)
    days = np.arange(HORIZON + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=90)
    ax.set_xlim(0, HORIZON)
    lo = min(sim["learned"]["p10"].min(), sim["classic"]["p10"].min())
    hi = max(sim["learned"]["p90"].max(), sim["classic"]["p90"].max())
    ax.set_ylim(lo - 1, hi + 1)
    ax.axhline(FLOOR, color="black", linestyle=":", linewidth=1)
    ax.text(HORIZON, FLOOR + 0.3, "floor 90", ha="right", fontsize=9)
    ax.set_xlabel("day")
    ax.set_ylabel("wealth")
    ax.set_title("Learned allocator vs classic CPPI (median, p10-p90)")
    ln_l, = ax.plot([], [], color="#1f77b4", lw=2, label="learned median")
    band_l = ax.fill_between([], [], [], color="#1f77b4", alpha=0.18,
                             label="learned p10-p90")
    ln_c, = ax.plot([], [], color="#ff7f0e", lw=2, label="classic median")
    band_c = ax.fill_between([], [], [], color="#ff7f0e", alpha=0.18,
                             label="classic p10-p90")
    ax.legend(loc="upper left", fontsize=9)
    day_text = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right",
                       fontsize=11, weight="bold")

    def frame(f):
        nonlocal band_l, band_c
        i = f + 1
        d = days[:i]
        for coll in (band_l, band_c):
            coll.remove()
        s = sim["learned"]
        ln_l.set_data(d, s["p50"][:i])
        band_l = ax.fill_between(d, s["p10"][:i], s["p90"][:i],
                                 color="#1f77b4", alpha=0.18)
        s = sim["classic"]
        ln_c.set_data(d, s["p50"][:i])
        band_c = ax.fill_between(d, s["p10"][:i], s["p90"][:i],
                                 color="#ff7f0e", alpha=0.18)
        day_text.set_text(f"day {f}")
        return ln_l, ln_c, day_text

    n_frames = HORIZON + 1
    anim = FuncAnimation(fig, frame, frames=n_frames, interval=90,
                         blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build docs/demo.gif")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="docs/demo.gif")
    args = ap.parse_args()
    build(args.out, fast=args.fast)


if __name__ == "__main__":
    main()
