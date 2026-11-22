import os
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from swingbot import Backtest, CostModel, EngineConfig, Panel, SupertrendSwing, TrendPullback, VolumeBreakout
from swingbot.data import synthetic_panel
from swingbot.strategies.base import Signals, Strategy


# ------------------------------------------------------------------ helpers
def make_panel(close, open_=None, high=None, low=None, volume=1e7):
    close = np.asarray(close, float).reshape(len(close), -1)
    T, N = close.shape
    open_ = close.copy() if open_ is None else np.asarray(open_, float).reshape(T, N)
    high = np.maximum(open_, close) if high is None else np.asarray(high, float).reshape(T, N)
    low = np.minimum(open_, close) if low is None else np.asarray(low, float).reshape(T, N)
    dates = pd.bdate_range("2024-01-01", periods=T)
    return Panel(dates, [f"S{j}" for j in range(N)], open_, high, low, close, np.full((T, N), volume))


@dataclass
class Scripted(Strategy):
    """Strategy that replays hand-written signal arrays."""

    name = "scripted"
    entry: np.ndarray = field(default=None)
    stop: float = 10.0
    exit: np.ndarray = field(default=None)
    target_r: float = 0.0
    trail: float = 0.0
    max_hold: int = 15
    regime_sma: int = 0

    def generate(self, p):
        T, N = p.shape
        return Signals(
            entry=self.entry.reshape(T, N),
            stop_dist=np.full((T, N), self.stop),
            exit=None if self.exit is None else self.exit.reshape(T, N),
            trail_dist=np.full((T, N), self.trail) if self.trail else None,
            target_r=self.target_r, min_hold=self.min_hold, max_hold=self.max_hold,
        )


FREE = EngineConfig(initial_cash=100_000, risk_per_trade=0.01, max_positions=5, max_position_pct=1.0,
                    max_adv_pct=0, max_entry_gap=0, slippage_bps=0, costs=CostModel.free())


def run(panel, **kw):
    return Backtest(panel, Scripted(**kw), FREE).run()


def onehot(T, *idx):
    a = np.zeros(T, bool)
    a[list(idx)] = True
    return a


# -------------------------------------------------------------------- tests
def test_entry_fills_at_next_open_and_time_stop():
    T = 30
    close = np.full(T, 100.0)
    open_ = np.full(T, 101.0)
    r = run(make_panel(close, open_), entry=onehot(T, 5), max_hold=3)
    t = r.trades.iloc[0]
    assert t.entry_date == r.equity.index[6]  # signal at close of 5 -> fill at open of 6
    assert t.entry_price == 101.0
    assert t.exit_reason == "time"
    assert t.bars_held == 3  # held sessions 6,7,8 -> exits at open of 9
    assert t.exit_date == r.equity.index[9]
    # 1% risk of 100k = 1000 Rs; stop 10 Rs/share -> 100 shares
    assert t.qty == 100


def test_stop_fills_at_level_or_gap_open():
    T = 20
    close = np.full(T, 100.0)
    low = close.copy()
    low[8] = 85.0  # intraday breach -> fill at the stop (90)
    r = run(make_panel(close, low=low), entry=onehot(T, 5))
    t = r.trades.iloc[0]
    assert (t.exit_reason, t.exit_price, t.exit_date) == ("stop", 90.0, r.equity.index[8])

    open_ = close.copy()
    open_[8] = 80.0  # gap below the stop -> fill at the open, not the stop
    r = run(make_panel(close, open_), entry=onehot(T, 5))
    assert r.trades.iloc[0].exit_price == 80.0
    assert np.isclose(r.trades.iloc[0].r_multiple, -2.0)


def test_target_and_stop_same_bar_is_pessimistic():
    T = 20
    close = np.full(T, 100.0)
    high, low = close.copy(), close.copy()
    high[9], low[9] = 125.0, 85.0
    r = run(make_panel(close, high=high, low=low), entry=onehot(T, 5), target_r=2.0)
    assert r.trades.iloc[0].exit_reason == "stop"
    high[9], low[9] = 125.0, 95.0
    r = run(make_panel(close, high=high, low=low), entry=onehot(T, 5), target_r=2.0)
    assert (r.trades.iloc[0].exit_reason, r.trades.iloc[0].exit_price) == ("target", 120.0)


def test_trailing_stop_ratchets_up():
    close = np.array([100.0] * 6 + [110, 120, 130, 125, 118, 118, 118, 118, 118])
    T = len(close)
    open_ = np.r_[close[0], close[:-1]]  # open at the prior close, so the drop to 118 is intraday
    r = run(make_panel(close, open_), entry=onehot(T, 5), trail=10.0, stop=20.0)
    t = r.trades.iloc[0]
    assert t.exit_reason == "trail"
    assert t.exit_price == 120.0  # highest close 130 - 10


def test_signal_exit_next_open_and_min_hold():
    T = 20
    close = np.full(T, 100.0)
    r = run(make_panel(close), entry=onehot(T, 5), exit=onehot(T, 7))
    assert (r.trades.iloc[0].exit_reason, r.trades.iloc[0].exit_date) == ("signal", r.equity.index[8])
    s = Scripted(entry=onehot(T, 5), exit=onehot(T, 7), min_hold=5, max_hold=8)
    r = Backtest(make_panel(close), s, FREE).run()
    assert r.trades.iloc[0].exit_reason == "time"  # early exit signal ignored


def test_accounting_identity_with_costs():
    p = synthetic_panel(40, 1500, seed=3)
    cfg = EngineConfig(costs=CostModel.groww(), slippage_bps=15)
    r = Backtest(p, VolumeBreakout(), cfg).run()
    assert len(r.trades) > 50
    assert np.isclose(r.equity.iloc[-1], cfg.initial_cash + r.trades.pnl.sum(), rtol=1e-9)
    assert (r.positions <= cfg.max_positions).all()
    assert (r.cash >= -1e-6).all()


def test_no_lookahead_truncation_invariance():
    """Deleting the future must not change any decision made before it."""
    p = synthetic_panel(60, 1500, seed=11)
    cut = 1100
    short = Panel(p.dates[:cut], p.symbols, *(getattr(p, f)[:cut] for f in ("open", "high", "low", "close", "volume")),
                  benchmark=p.benchmark[:cut])
    for strat in (TrendPullback(), VolumeBreakout(), SupertrendSwing()):
        full = Backtest(p, strat).run().trades
        part = Backtest(short, strat).run().trades
        last = p.dates[cut - 2]
        a = full[full.exit_date <= last].reset_index(drop=True)
        b = part[part.exit_date <= last].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        assert np.allclose(Backtest(p, strat).run().equity.iloc[: cut - 1], Backtest(short, strat).run().equity.iloc[: cut - 1])


def test_window_run_and_determinism():
    p = synthetic_panel(30, 1200, seed=5)
    bt = Backtest(p, TrendPullback())
    r1, r2 = bt.run("2011-01-01", "2013-12-31"), bt.run("2011-01-01", "2013-12-31")
    assert r1.equity.index[0] >= pd.Timestamp("2011-01-01")
    assert r1.equity.index[-1] <= pd.Timestamp("2013-12-31")
    pd.testing.assert_frame_equal(r1.trades, r2.trades)


def test_max_positions_and_ranking():
    T = 20
    close = np.full((T, 4), 100.0)
    entry = np.zeros((T, 4), bool)
    entry[5] = True

    @dataclass
    class Ranked(Scripted):
        def generate(self, p):
            s = super().generate(p)
            s.score = np.tile(np.array([1.0, 4.0, 3.0, 2.0]), (p.shape[0], 1))
            return s

    cfg = EngineConfig(**{**FREE.__dict__, "max_positions": 2})
    r = Backtest(make_panel(close), Ranked(entry=entry), cfg).run()
    assert sorted(r.trades.symbol) == ["S1", "S2"]


def test_cost_model_zerodha():
    cm = CostModel.zerodha()
    # 1 lakh delivery buy: STT 100 + txn 2.97 + SEBI 0.10 + stamp 15 + GST on (2.97 + 0.10)
    assert np.isclose(cm.buy(100_000), 100 + 2.97 + 0.10 + 15 + 0.18 * 3.07)
    assert np.isclose(cm.sell(100_000), 100 + 2.97 + 0.10 + 0.18 * 3.07 + 15.93)
    g = CostModel.groww()
    assert np.isclose(g.buy(1_000), CostModel.zerodha().buy(1_000) + 5 * 1.18)  # min brokerage
    assert np.isclose(g.buy(1_000_000) - CostModel.zerodha().buy(1_000_000), 20 * 1.18)  # capped


@pytest.mark.slow
def test_pure_python_fallback_matches_numba():
    code = (
        "from swingbot import Backtest, VolumeBreakout;from swingbot.data import synthetic_panel;"
        "r=Backtest(synthetic_panel(15,400,seed=2),VolumeBreakout()).run();print(repr(r.equity.iloc[-1]),len(r.trades))"
    )
    env_jit = {**os.environ}
    env_jit.pop("SWINGBOT_NO_JIT", None)
    env_py = {**os.environ, "SWINGBOT_NO_JIT": "1"}
    out_jit = subprocess.check_output([sys.executable, "-c", code], env=env_jit, text=True)
    out_py = subprocess.check_output([sys.executable, "-c", code], env=env_py, text=True)
    assert out_jit == out_py
