"""Tests for the artifact generators (scripts/make_dashboard.py etc.).

Fast mode only: enough to prove the pages build, not enough to learn
anything real.
"""

import os
import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup

REPO = os.path.join(os.path.dirname(__file__), "..")


def _run(script, out):
    return subprocess.run(
        [sys.executable, script, "--fast", "--out", out],
        cwd=os.path.abspath(REPO),
        capture_output=True, text=True, timeout=1200,
    )



def _assert_offline_dashboard(path):
    # Embedded fonts can precede the scripts by more than 200 KB. Inspect the
    # complete generated document instead of assuming a library byte offset.
    html = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    assert soup.h1 is not None
    assert soup.h1.get_text(strip=True) == "Neural-CPPI"
    scripts = soup.find_all("script")
    assert not any(script.get("src") for script in scripts)
    assert sum("plotly.js v" in (script.string or "").lower()
               for script in scripts) == 1
    assert "Plotly.newPlot(" in html
    assert soup.select_one("a[href='#wealth']") is not None
    assert soup.select_one("a[href='#policy']") is not None


def test_make_dashboard_fast(tmp_path):
    out = str(tmp_path / "dashboard.html")
    r = _run(os.path.join("scripts", "make_dashboard.py"), out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(out)
    assert os.path.getsize(out) > 100_000  # plotly.js is embedded
    _assert_offline_dashboard(out)


def test_make_demo_gif_fast(tmp_path):
    out = str(tmp_path / "demo.gif")
    r = _run(os.path.join("scripts", "make_demo_gif.py"), out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(out)
    with open(out, "rb") as f:
        assert f.read(6) == b"GIF89a"
