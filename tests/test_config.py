from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swingbot import Backtest, Quantity, RuleStrategy, load_config
from swingbot import indicators as ind
from swingbot.data import synthetic_panel
from swingbot.engine import kernel as K
from swingbot.strategies.expr import Evaluator, ExprError

ROOT = Path(__file__).resolve().parents[1]


def test_example_file_loads():
    cfg = load_config(ROOT / "strategies.example.toml")
    assert cfg.risk.max_positions == 5 and cfg.risk.max_position_pct == 0.40
    assert [s.name for s in cfg.active] == ["trend_dip", "volume_breakout"]
    assert cfg.strategy("trend_dip").strategy.params == {"rsi_entry": 5}
    assert cfg.live.dry_run is True


def test_unknown_setting_is_rejected(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text('[[strategy]]\nname="x"\ntype="breakout"\nlookbak=20\n')
    with pytest.raises(ValueError, match="lookbak"):
        load_config(f)


@pytest.mark.parametrize("text,mode,value", [
    ("risk", K.Q_RISK, 0.0), ("risk 0.5%", K.Q_RISK, 0.005), ("equity 20%", K.Q_EQUITY, 0.20),
    ("shares 50", K.Q_SHARES, 50), ("amount 1,00,000", K.Q_AMOUNT, 100000), ("Equity 25", K.Q_EQUITY, 0.25),
])
def test_quantity_parse(text, mode, value):
    q = Quantity.parse(text)
    assert q.mode == mode and np.isclose(q.value, value)


def test_quantity_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Quantity.parse("all of it")


def test_expressions_match_indicators():
    p = synthetic_panel(5, 400)
    ev = Evaluator(p, {"n": 3})
    c = p.frame("close")
    got = ev("close > sma(close, 20) and rsi(close, n) < 30")
    want = (c > ind.sma(c, 20)) & (ind.rsi(c, 3) < 30)
    pd.testing.assert_frame_equal(got, want)
    pd.testing.assert_frame_equal(ev("2 * atr(14)"), 2 * ind.atr(p.frame("high"), p.frame("low"), c, 14))
    cross = ev("crossover(close, sma(close, 10))")
    assert cross.dtypes.eq(bool).all()


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')", "close.__class__", "open('x')", "close[0]", "lambda: 1",
    "unknown_var > 1", "sma(close, 2.5)",
])
def test_expressions_are_sandboxed(bad):
    with pytest.raises(ExprError):
        Evaluator(synthetic_panel(3, 100))(bad)


def test_rule_strategy_matches_builtin_pullback():
    """The same rules written in the file reproduce the built-in strategy exactly."""
    from swingbot import TrendPullback

    p = synthetic_panel(30, 1500, seed=9)
    builtin = TrendPullback(rsi_entry=10, max_hold=10)
    rules = RuleStrategy(
        entry="close > sma(close, 200) and sma(close, 50) > sma(close, 200) and rsi(close, 3) < x",
        exit="rsi(close, 3) > 70", stop="2.5 * atr(14)", score="roc(close, 126)", max_hold=10,
        params={"x": 10})
    a, b = Backtest(p, builtin).run().trades, Backtest(p, rules).run().trades
    pd.testing.assert_frame_equal(a.drop(columns="strategy"), b.drop(columns="strategy"))
    assert rules.with_params(x=5).params == {"x": 5}


def test_single_broker_drives_data_and_orders(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text('broker = "groww"\n[data]\nuniverse = "nifty50"\n[[strategy]]\nname = "x"\ntype = "breakout"\n')
    cfg = load_config(f)
    assert (cfg.data.provider, cfg.live.broker, cfg.data.history_provider) == ("groww", "groww", "yahoo")
    f.write_text('broker = "kite"\n[data]\nhistory_provider = "yahoo"\n[[strategy]]\nname = "x"\ntype = "breakout"\n')
    cfg = load_config(f)
    assert (cfg.data.provider, cfg.live.broker, cfg.data.history_provider) == ("kite", "kite", "yahoo")
    f.write_text('broker = "kite"\n[data]\nprovider = "yahoo"\n[[strategy]]\nname = "x"\ntype = "breakout"\n')
    assert load_config(f).data.provider == "kite"  # the single-broker line wins over a leftover provider
    f.write_text('broker = "upstox"\n[[strategy]]\nname = "x"\ntype = "breakout"\n')
    with pytest.raises(ValueError, match="broker must be"):
        load_config(f)
