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
    return fetch_ticker_daily_yahoo("SPY", data_dir=data_dir)


def fetch_ticker_daily_yahoo(ticker, data_dir=DATA_DIR):
    """Download daily bars for any Yahoo ticker, saved as CSV parts.

    Writes ``{ticker}_daily_2015.csv`` and ``{ticker}_daily_2021.csv``
    (lowercase ticker), split in half like the SPY parts. No API key.
    """
    t = ticker.upper()
    req = urllib.request.Request(
        YAHOO_CHART_URL.replace("/SPY?", f"/{t}?"),
        headers={"User-Agent": "Mozilla/5.0 (compatible; neural-cppi)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]

    import datetime

    rows = []
    for i, ts in enumerate(stamps):
        if quote["close"][i] is None or adj[i] is None:
            continue
        day = datetime.datetime.fromtimestamp(
            ts, datetime.timezone.utc
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
    mid = len(rows) // 2
    paths = []
    for name, rs in (
        (f"{t.lower()}_daily_2015.csv", rows[:mid]),
        (f"{t.lower()}_daily_2021.csv", rows[mid:]),
    ):
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


def load_ticker_daily(ticker, data_dir=DATA_DIR):
    """Load the committed daily CSV parts for one ticker.

    Same return format as :func:`load_spy_daily`.
    """
    t = ticker.lower()
    files = sorted(
        glob.glob(os.path.join(data_dir, f"{t}_daily_*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"no {t}_daily_*.csv parts in {data_dir}")
    return _load_parts(files)


def _load_parts(files):
    dates = []
    cols = {k: [] for k in ("open", "high", "low", "close", "adjclose", "volume")}
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


def load_aligned_returns(tickers, data_dir=DATA_DIR):
    """Load daily log returns for several tickers on their common dates.

    Returns ``(dates, rets)`` where ``dates`` is the sorted list of common
    trading dates and ``rets`` maps each ticker to its log-return array
    (length ``len(dates) - 1``). Inner join: a date must exist for every
    ticker. US-listed ETFs share a calendar, so almost nothing is lost.
    """
    series = {}
    for t in tickers:
        d = load_ticker_daily(t, data_dir=data_dir)
        series[t] = dict(zip(d["dates"], d["adjclose"]))
    common = sorted(set.intersection(*(set(s) for s in series.values())))
    rets = {}
    for t in tickers:
        prices = np.array([series[t][d] for d in common])
        rets[t] = log_returns(prices)
    return common, rets


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
    """Price paths from a circular block bootstrap of log returns.

    Blocks of ``block_len`` steps (21 trading days ~= a month) keep
    volatility clustering while mixing regimes. Returns an array of
    shape (n_paths, path_len + 1) starting at S0.
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
