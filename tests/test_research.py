import numpy as np
import pandas as pd

from swingbot import RiskConfig, StrategySpec, VolumeBreakout, TrendPullback
from swingbot import research as R
from swingbot.data import synthetic_panel
from swingbot.data.store import MarketDB


def test_signal_study_tags_and_buckets():
    p = synthetic_panel(40, 900, seed=8)
    meta = pd.DataFrame({"cap": ["large"] * 10 + ["mid"] * 10 + ["small"] * 20,
                         "industry": ["IT"] * 20 + ["Banks"] * 20}, index=p.symbols)
    specs = [StrategySpec("brk", VolumeBreakout()), StrategySpec("pull", TrendPullback())]
    study = R.signal_study(p, specs, RiskConfig(), meta=meta)
    assert len(study) > 100 and set(study.strategy) == {"brk", "pull"}
    # every trade is an independent Rs 1 lakh position (no slot / cash limits)
    assert (study.qty * study.entry_price).between(90_000, 100_000).mean() > 0.95
    # tags come from the signal day = the session before the entry fill
    row = study.iloc[0]
    i = p.dates.get_loc(row.entry_date) - 1
    assert np.isclose(row.price, p.close[i, p.symbols.index(row.symbol)])
    assert set(study.cap) <= {"large", "mid", "small"}
    for f in ("price", "turnover", "volume", "cap", "industry", "volatility", "momentum", "trend", "market"):
        tb = R.bucket_table(study, f)
        assert tb.trades.sum() == len(study.dropna(subset=[f])) if f not in ("cap", "industry", "market") \
            else tb.trades.sum() == len(study)
    tb = R.bucket_table(study, "cap", by_strategy=True)
    assert set(tb.strategy) == {"brk", "pull"}
    board = R.leaderboard(p, specs, RiskConfig(), study)
    assert list(board.columns[:2]) == ["strategy", "signal_trades"]


def test_intraday_table_roundtrip(tmp_path):
    db = MarketDB(tmp_path / "m.duckdb")
    ts = pd.date_range("2025-01-02 09:15", periods=75, freq="5min")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}, index=ts)
    assert db.upsert_intraday("AAA", "5minute", df, "kite") == 75
    assert db.upsert_intraday("AAA", "5minute", df.iloc[-5:], "kite") == 5  # re-insert: no duplicates
    got = db.read_intraday("AAA", "5minute")
    assert len(got) == 75 and got.index[0] == ts[0]
    assert db.last_ts("AAA", "5minute") == ts[-1]
    assert db.intraday_coverage().bars[0] == 75
