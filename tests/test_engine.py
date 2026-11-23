import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import Scripted, make_panel, onehot

from swingbot import (
    Backtest,
    CostModel,
    Quantity,
    RiskConfig,
    StrategySpec,
    SupertrendSwing,
    TrendPullback,
    VolumeBreakout,
)
from swingbot.data import synthetic_panel


def run(panel, risk, **kw):
    return Backtest(panel, Scripted(**kw), risk).run()


# ---------------------------------------------------------------- fills
def test_entry_fills_at_next_open_and_time_stop(free_risk):
    T = 30
    r = run(make_panel(np.full(T, 100.0), np.full(T, 101.0)), free_risk, entry=onehot(T, 5), max_hold=3)
    t = r.trades.iloc[0]
    assert t.entry_date == r.equity.index[6] and t.entry_price == 101.0
    assert (t.exit_reason, t.bars_held, t.exit_date) == ("time", 3, r.equity.index[9])
    assert t.qty == 100  # 1% of 100k = 1000 Rs risk / 10 Rs stop


def test_stop_fills_at_level_or_gap_open(free_risk):
    T = 20
    close = np.full(T, 100.0)
    low = close.copy()
    low[8] = 85.0
    t = run(make_panel(close, low=low), free_risk, entry=onehot(T, 5)).trades.iloc[0]
    assert (t.exit_reason, t.exit_price) == ("stop", 90.0)
    open_ = close.copy()
    open_[8] = 80.0
    t = run(make_panel(close, open_), free_risk, entry=onehot(T, 5)).trades.iloc[0]
    assert t.exit_price == 80.0 and np.isclose(t.r_multiple, -2.0)


def test_target_vs_stop_same_bar_is_pessimistic(free_risk):
    T = 20
    close = np.full(T, 100.0)
    high, low = close.copy(), close.copy()
    high[9], low[9] = 125.0, 85.0
    r = run(make_panel(close, high=high, low=low), free_risk, entry=onehot(T, 5), target_r=2.0)
    assert r.trades.iloc[0].exit_reason == "stop"
    low[9] = 95.0
    t = run(make_panel(close, high=high, low=low), free_risk, entry=onehot(T, 5), target_r=2.0).trades.iloc[0]
    assert (t.exit_reason, t.exit_price) == ("target", 120.0)


def test_trailing_stop_ratchets_up(free_risk):
    close = np.array([100.0] * 6 + [110, 120, 130, 125, 118, 118, 118, 118, 118])
    open_ = np.r_[close[0], close[:-1]]
    t = run(make_panel(close, open_), free_risk, entry=onehot(len(close), 5), trail=10.0, stop=20.0).trades.iloc[0]
    assert (t.exit_reason, t.exit_price) == ("trail", 120.0)


def test_signal_exit_and_min_hold(free_risk):
    T = 20
    p = make_panel(np.full(T, 100.0))
    r = run(p, free_risk, entry=onehot(T, 5), exit=onehot(T, 7))
    assert (r.trades.iloc[0].exit_reason, r.trades.iloc[0].exit_date) == ("signal", r.equity.index[8])
    r = Backtest(p, Scripted(entry=onehot(T, 5), exit=onehot(T, 7), min_hold=5, max_hold=8), free_risk).run()
    assert r.trades.iloc[0].exit_reason == "time"


def test_entry_gap_limit_skips_big_gap_up(free_risk):
    T = 20
    open_ = np.full(T, 100.0)
    open_[6] = 106.0  # > 5% above the signal close
    risk = RiskConfig(**{**free_risk.__dict__, "max_entry_gap": 0.05})
    assert run(make_panel(np.full(T, 100.0), open_), risk, entry=onehot(T, 5)).trades.empty
    open_[6] = 104.0
    assert len(run(make_panel(np.full(T, 100.0), open_), risk, entry=onehot(T, 5)).trades) == 1


# -------------------------------------------------------- risk + sizing
def test_position_cap_40pct_and_max_5_positions():
    T, N = 30, 8
    p = make_panel(np.full((T, N), 100.0))
    entry = np.zeros((T, N), bool)
    entry[5] = True
    risk = RiskConfig(capital=100_000, max_positions=5, max_position_pct=0.40, max_adv_pct=0,
                      max_entry_gap=0, slippage_bps=0, costs="free")
    spec = StrategySpec("all_in", Scripted(entry=entry, max_hold=10), quantity=Quantity.parse("equity 100%"))
    r = Backtest(p, [spec], risk).run()
    # 100% requested -> capped at 40% each; the remaining cash (minus a 0.3% reserve
    # for charges per order) buys the third
    assert list(r.trades.qty) == [400, 400, 197]
    spec = StrategySpec("small", Scripted(entry=entry, max_hold=10), quantity=Quantity.parse("equity 10%"))
    r = Backtest(p, [spec], risk).run()
    assert len(r.trades) == 5 and r.positions.max() == 5  # 8 signals, only 5 slots


def test_quantity_modes(free_risk):
    T = 20
    p = make_panel(np.full(T, 100.0))
    for q, expect in (("shares 7", 7), ("amount 50000", 500), ("risk 2%", 200), ("equity 30%", 300)):
        spec = StrategySpec("x", Scripted(entry=onehot(T, 5)), quantity=Quantity.parse(q))
        assert Backtest(p, [spec], free_risk).run().trades.iloc[0].qty == expect, q


def test_multi_strategy_priority_and_per_strategy_cap(free_risk):
    T, N = 20, 4
    p = make_panel(np.full((T, N), 100.0))
    e = np.zeros((T, N), bool)
    e[5] = True
    a = StrategySpec("a", Scripted(entry=e), priority=2, max_positions=1)
    b = StrategySpec("b", Scripted(entry=e), priority=1, max_positions=2)
    tr = Backtest(p, [a, b], free_risk).run().trades
    assert sorted(tr.strategy) == ["a", "b", "b"]
    assert set(tr[tr.strategy == "b"].symbol) == {"S0", "S1"}  # b is served first


def test_drawdown_halt_pauses_then_resumes():
    T = 60
    close = np.full(T, 100.0)
    close[7:] = 50.0  # crash -> 50% drawdown at the close of day 7
    risk = RiskConfig(capital=100_000, max_position_pct=1.0, max_adv_pct=0, max_entry_gap=0, slippage_bps=0,
                      costs="free", max_drawdown_halt=0.10, halt_cooldown=20)
    spec = StrategySpec("x", Scripted(entry=onehot(T, 5, 25, 40), stop=60.0, max_hold=15),
                        quantity=Quantity.parse("equity 100%"))
    tr = Backtest(make_panel(close), [spec], risk).run().trades
    dates = pd.bdate_range("2024-01-01", periods=T)
    assert list(tr.entry_date) == [dates[6], dates[41]]  # day-25 signal blocked by the halt (until day 27)


# ---------------------------------------------------- integrity checks
def test_accounting_identity_with_costs():
    p = synthetic_panel(40, 1500, seed=3)
    risk = RiskConfig(costs=CostModel.groww(), slippage_bps=15)
    r = Backtest(p, [TrendPullback(), VolumeBreakout()], risk).run()
    assert len(r.trades) > 50
    assert np.isclose(r.equity.iloc[-1], risk.capital + r.trades.pnl.sum(), rtol=1e-9)
    assert (r.positions <= risk.max_positions).all() and (r.cash >= -1e-6).all()


def test_no_lookahead_truncation_invariance():
    p = synthetic_panel(60, 1500, seed=11)
    cut = 1100
    for strat in (TrendPullback(), VolumeBreakout(), SupertrendSwing()):
        full, part = Backtest(p, strat).run(), Backtest(p.head(cut), strat).run()
        last = p.dates[cut - 2]
        a = full.trades[full.trades.exit_date <= last].reset_index(drop=True)
        b = part.trades[part.trades.exit_date <= last].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        assert np.allclose(full.equity.iloc[: cut - 1], part.equity.iloc[: cut - 1])


def test_window_run_and_determinism():
    bt = Backtest(synthetic_panel(30, 1200, seed=5), TrendPullback())
    r1, r2 = bt.run("2011-01-01", "2013-12-31"), bt.run("2011-01-01", "2013-12-31")
    assert r1.equity.index[0] >= pd.Timestamp("2011-01-01") and r1.equity.index[-1] <= pd.Timestamp("2013-12-31")
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
    assert np.isclose(r1.equity.iloc[-1], 1_000_000 + r1.trades.pnl.sum())


def test_cost_model_zerodha_and_groww():
    cm = CostModel.zerodha()
    assert np.isclose(cm.buy(100_000), 100 + 2.97 + 0.10 + 15 + 0.18 * 3.07)
    assert np.isclose(cm.sell(100_000), 100 + 2.97 + 0.10 + 0.18 * 3.07 + 15.93)
    assert np.isclose(CostModel.groww().buy(1_000) - cm.buy(1_000), 5 * 1.18)
    assert np.isclose(CostModel.groww().buy(1_000_000) - cm.buy(1_000_000), 20 * 1.18)


@pytest.mark.slow
def test_pure_python_fallback_matches_numba():
    code = ("from swingbot import Backtest, VolumeBreakout, TrendPullback;from swingbot.data import synthetic_panel;"
            "r=Backtest(synthetic_panel(15,400,seed=2),[VolumeBreakout(),TrendPullback()]).run();"
            "print(repr(r.equity.iloc[-1]),len(r.trades))")
    env = {k: v for k, v in os.environ.items() if k != "SWINGBOT_NO_JIT"}
    a = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    b = subprocess.check_output([sys.executable, "-c", code], env={**env, "SWINGBOT_NO_JIT": "1"}, text=True)
    assert a == b
