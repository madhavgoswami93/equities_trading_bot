import numpy as np
import pandas as pd

from swingbot import indicators as ind
from swingbot.data import synthetic_panel
from swingbot.engine.metrics import equity_stats


def frame(x):
    return pd.DataFrame({"A": np.asarray(x, float)})


def test_rsi_extremes_and_range():
    up = frame(np.arange(1, 50))
    assert ind.rsi(up, 14).iloc[-1, 0] == 100.0
    p = synthetic_panel(5, 500)
    r = ind.rsi(p.frame("close"), 3).stack().dropna()
    assert r.between(0, 100).all()


def test_atr_constant_range():
    c = frame(np.full(50, 100.0))
    h, l = c + 2, c - 2
    assert np.isclose(ind.atr(h, l, c, 14).iloc[-1, 0], 4.0)


def test_indicators_are_causal():
    p = synthetic_panel(5, 600, seed=1)
    c, h, l = p.frame("close"), p.frame("high"), p.frame("low")
    full = ind.supertrend(h, l, c, 10, 3)[1]
    part = ind.supertrend(h.iloc[:400], l.iloc[:400], c.iloc[:400], 10, 3)[1]
    pd.testing.assert_frame_equal(full.iloc[:400], part)
    pd.testing.assert_frame_equal(ind.rsi(c, 3).iloc[:400], ind.rsi(c.iloc[:400], 3))


def test_supertrend_direction_follows_trend():
    x = np.r_[np.linspace(100, 200, 100), np.linspace(200, 100, 100)]
    c = frame(x)
    _, d = ind.supertrend(c + 1, c - 1, c, 10, 3)
    assert d.iloc[95, 0] == 1 and d.iloc[-1, 0] == -1


def test_equity_stats():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(100 * 1.1 ** (np.arange(253) / 252), index=idx)
    s = equity_stats(eq)
    assert np.isclose(s["cagr"], 0.10, atol=1e-6)
    assert s["max_drawdown"] == 0
