"""Real market data: SPY daily bars and block-bootstrap path generation.

The committed ``data/spy_daily_2015.csv`` and ``data/spy_daily_2021.csv``
hold Yahoo Finance daily SPY bars (adjusted closes, 2015-01-02 to
2026-10-09), split in two only to keep each file small.
``fetch_spy_daily_yahoo`` refreshes them; it needs no API key.

``block_bootstrap_paths`` turns the historical log-return series into
synthetic price paths by resampling blocks of consecutive returns
(Kunsch, 1989). Blocks preserve the short-horizon dependence real
returns have (volatility clustering, mild autocorrelation) that iid
resampling would destroy. Circular wrapping avoids edge effects.
"""

import csv
import glob
import json
import os
import urllib.request

import numpy as np

YAHOO_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    "?period1=1420070400&period2=9999999999&interval=1d"
)

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data"
)


def _write_parts(rows, data_dir=DATA_DIR):
    """Write rows to the two ``spy_daily_*.csv`` part files, split in half."""
    mid = len(rows) // 2
    paths = []
    for name, rs in (("spy_daily_2015.csv", rows[:mid]),
                     ("spy_daily_2021.csv", rows[mid:])):
        path = os.path.join(data_dir, name)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
            )
            writer.writerows(rs)
        paths.append(path)
    return paths


def fetch_spy_daily_yahoo(data_dir=DATA_DIR):
    """Download SPY daily bars from Yahoo Finance and save as CSV parts.

    Returns the paths written. No API key needed. Raises on HTTP errors.
    """
    req = urllib.request.Request(
        YAHOO_CHART_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; deep-hedging-lab)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]

    import datetime

    rows = []
    for i, t in enumerate(stamps):
        if quote["close"][i] is None or adj[i] is None:
            continue
        day = datetime.datetime.fromtimestamp(
            t, datetime.timezone.utc
        ).date().isoformat()
        rows.append(
            [
                day,
                f"{quote['open'][i]:.2f}",
                f"{quote['high'][i]:.2f}",
                f"{quote['low'][i]:.2f}",
                f"{quote['close'][i]:.2f}",
                f"{adj[i]:.2f}",
                quote["volume"][i],
            ]
        )
    return _write_parts(rows, data_dir)


def _part_files(data_dir=DATA_DIR):
    parts = sorted(glob.glob(os.path.join(data_dir, "spy_daily_*.csv")))
    if not parts:
        raise FileNotFoundError(f"no spy_daily_*.csv parts in {data_dir}")
    return parts


def load_spy_daily(path=None, data_dir=DATA_DIR):
    """Load the committed SPY daily CSV parts.

    Returns a dict with ``dates`` (list of str) and float arrays
    ``open``, ``high``, ``low``, ``close``, ``adjclose``, ``volume``.
    Pass ``path`` to load a single CSV file instead of the parts.
    """
    dates = []
    cols = {k: [] for k in ("open", "high", "low", "close", "adjclose", "volume")}
    files = [path] if path is not None else _part_files(data_dir)
    for fp in files:
        with open(fp, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dates.append(row["Date"])
                cols["open"].append(float(row["Open"]))
                cols["high"].append(float(row["High"]))
                cols["low"].append(float(row["Low"]))
                cols["close"].append(float(row["Close"]))
                cols["adjclose"].append(float(row["AdjClose"]))
                cols["volume"].append(float(row["Volume"]))
    out = {"dates": dates}
    out.update({k: np.asarray(v) for k, v in cols.items()})
    return out


def log_returns(prices):
    """Log returns of a price series (length n - 1)."""
    p = np.asarray(prices, dtype=float)
    return np.log(p[1:] / p[:-1])


def block_bootstrap_paths(
    returns, n_paths, path_len, block_len=21, S0=100.0, seed=None
):
    """Generate price paths by circular block bootstrap of log returns.

    Parameters
    ----------
    returns : array_like, shape (n,)
        Historical log returns (e.g. daily SPY log returns).
    n_paths : int
        Number of paths to generate.
    path_len : int
        Number of steps per path (path has ``path_len + 1`` prices).
    block_len : int
        Block length in steps; 21 trading days ~= one month, long enough
        to keep volatility clustering, short enough to mix regimes.
    S0 : float
        Starting price; paths are normalized so only returns matter.
    seed : int, optional

    Returns
    -------
    S : np.ndarray, shape (n_paths, path_len + 1)
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < block_len:
        raise ValueError(f"need at least block_len={block_len} returns, got {n}")
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(path_len / block_len))
    # Random block start indices; modulo n implements circular wrapping.
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    offsets = np.arange(block_len)[None, None, :]
    idx = (starts[:, :, None] + offsets) % n  # (n_paths, n_blocks, block_len)
    blocks = r[idx]  # (n_paths, n_blocks, block_len)
    sampled = blocks.reshape(n_paths, n_blocks * block_len)[:, :path_len]

    S = np.empty((n_paths, path_len + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(sampled, axis=1))
    return S
