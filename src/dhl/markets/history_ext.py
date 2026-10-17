"""Extended market history: pre-2015 SPY (2008 crisis) and VIX.

The committed ``data/spy_daily_2015.csv`` / ``spy_daily_2021.csv`` start
in 2015, so the 2008 financial crisis -- the cleanest pre-2020 crash
regime for training -- is missing. This module fetches it (plus VIX,
the market's crash gauge) and stores it alongside the existing parts.

Files written (all under ``data/``):
- ``spy_daily_2007.csv``: SPY daily bars 2007-01-01 onward. The existing
  ``_part_files`` glob (``spy_daily_*.csv``) picks it up automatically
  and sorts it before the 2015 part, so ``load_spy_daily`` returns one
  continuous 2007+ series with no code changes elsewhere.
- ``vix_daily.csv``: Cboe VIX daily closes, same date range.

VIX timing discipline (no look-ahead): a decision made at the close of
day ``t`` after seeing that day's price action may NOT use the VIX
close of day ``t`` (published after the decision point in a live
setting). Features must use VIX from ``t-1`` or earlier. The
``aligned_vix_lagged`` helper enforces this by shifting one day.
"""

import csv
import datetime
import json
import os
import urllib.request

import numpy as np

from dhl.markets.real import DATA_DIR, YAHOO_PERIOD1_2007

_VIX_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
    f"?period1={YAHOO_PERIOD1_2007}&period2=9999999999&interval=1d"
)
_UA = {"User-Agent": "Mozilla/5.0 (compatible; neural-cppi)"}


def _download_chart(url):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    return result


def fetch_spy_2007(data_dir=DATA_DIR):
    """Download SPY daily bars from 2007-01-01, save as ``spy_daily_2007.csv``.

    Returns the path written. Idempotent: re-running refreshes the file.
    """
    result = _download_chart(
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
        f"?period1={YAHOO_PERIOD1_2007}&period2=9999999999&interval=1d"
    )
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]

    rows = []
    for i, ts in enumerate(stamps):
        if quote["close"][i] is None or adj[i] is None:
            continue
        day = datetime.datetime.fromtimestamp(
            ts, datetime.timezone.utc).date().isoformat()
        # Keep only pre-2015 bars; the committed parts cover 2015+.
        if day >= "2015-01-01":
            continue
        rows.append([
            day,
            f"{quote['open'][i]:.2f}",
            f"{quote['high'][i]:.2f}",
            f"{quote['low'][i]:.2f}",
            f"{quote['close'][i]:.2f}",
            f"{adj[i]:.2f}",
            quote["volume"][i],
        ])
    path = os.path.join(data_dir, "spy_daily_2007.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"])
        writer.writerows(rows)
    return path


def fetch_vix_daily(data_dir=DATA_DIR):
    """Download Cboe VIX daily closes, save as ``vix_daily.csv``.

    Returns the path written. Idempotent.
    """
    result = _download_chart(_VIX_URL)
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    rows = []
    for i, ts in enumerate(stamps):
        if quote["close"][i] is None:
            continue
        day = datetime.datetime.fromtimestamp(
            ts, datetime.timezone.utc).date().isoformat()
        rows.append([day, f"{quote['close'][i]:.2f}"])
    path = os.path.join(data_dir, "vix_daily.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Close"])
        writer.writerows(rows)
    return path


def load_vix_daily(data_dir=DATA_DIR):
    """Load ``vix_daily.csv``. Returns dict with ``dates`` and ``close``.

    Raises FileNotFoundError if not yet fetched (see
    :func:`fetch_vix_daily`).
    """
    path = os.path.join(data_dir, "vix_daily.csv")
    dates, closes = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dates.append(row["Date"])
            closes.append(float(row["Close"]))
    return {"dates": dates, "close": np.asarray(closes)}


def aligned_vix_lagged(spy_dates, vix_data, lag_days=1):
    """VIX closes aligned to SPY dates, lagged to prevent look-ahead.

    For each SPY date ``d``, returns the VIX close from ``lag_days``
    trading days before ``d`` (default 1: yesterday's VIX, the latest
    value knowable before today's close). Dates with no available
    lagged VIX get NaN.

    This is the ONLY sanctioned way to feed VIX into features: the raw
    same-day close must never be used at decision time.
    """
    spy_dates = list(spy_dates)
    vix_dates = list(vix_data["dates"])
    vix_close = np.asarray(vix_data["close"], dtype=float)
    vix_pos = {d: j for j, d in enumerate(vix_dates)}

    out = np.full(len(spy_dates), np.nan)
    for i, d in enumerate(spy_dates):
        j = vix_pos.get(d)
        if j is not None and j >= lag_days:
            out[i] = vix_close[j - lag_days]
    return out
