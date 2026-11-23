"""Smoke test: every dashboard tab renders without errors on synthetic data."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "swingbot" / "dashboard" / "app.py")

FILE = """
name = "dash"
runs_dir = "{runs}"
[data]
provider = "synthetic"
[risk]
max_positions = 5
max_position_pct = 0.40
[[strategy]]
name = "dip"
entry = "close > sma(close, 200) and rsi(close, 3) < 10"
exit = "rsi(close, 3) > 70"
stop = "2.5 * atr(14)"
score = "roc(close, 126)"
quantity = "risk 1%"
[[strategy]]
name = "brk"
type = "breakout"
quantity = "equity 20%"
priority = 2
"""


def test_dashboard_renders_all_tabs(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text(FILE.format(runs=tmp_path / "runs"))
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["file"] = str(f)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert [t.label for t in at.tabs] == ["Backtest", "Paper trading", "Live", "Signals today", "Strategy file"]
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Max positions held"]) <= 5

    run_paper = next(b for b in at.button if b.label == "Run paper trading now")
    run_paper.click().run()
    assert not at.exception and not at.error
    assert (tmp_path / "runs" / "dash" / "paper" / "book.json").exists()

    next(b for b in at.button if b.label == "Validate").click().run()
    assert any("Valid" in s.value for s in at.success)

    at.text_area[0].set_value(f.read_text().replace('quantity = "risk 1%"', 'quantity = "all of it"'))
    next(b for b in at.button if b.label == "Save").click().run()
    assert any("Not saved" in e.value for e in at.error)
    assert 'quantity = "risk 1%"' in f.read_text()  # invalid edit never reaches disk
