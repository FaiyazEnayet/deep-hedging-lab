"""Tests for the artifact generators (scripts/make_dashboard.py etc.).

Fast mode only: enough to prove the pages build, not enough to learn
anything real.
"""

import os
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), "..")


def _run(script, out):
    return subprocess.run(
        [sys.executable, script, "--fast", "--out", out],
        cwd=os.path.abspath(REPO),
        capture_output=True, text=True, timeout=1200,
    )


def test_make_dashboard_fast(tmp_path):
    out = str(tmp_path / "dashboard.html")
    r = _run(os.path.join("scripts", "make_dashboard.py"), out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(out)
    assert os.path.getsize(out) > 100_000  # plotly.js is embedded
    with open(out) as f:
        html = f.read(200000)
    assert "plotly" in html.lower()
    assert "neural-cppi lab" in html


def test_make_demo_gif_fast(tmp_path):
    out = str(tmp_path / "demo.gif")
    r = _run(os.path.join("scripts", "make_demo_gif.py"), out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(out)
    with open(out, "rb") as f:
        assert f.read(6) == b"GIF89a"
