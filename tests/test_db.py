"""Local market database: backfill, incremental updates, corporate-action refresh."""

import numpy as np
import pandas as pd

from swingbot.data.base import DataProvider
from swingbot.data.store import BENCHMARK, MarketDB, load_panel, sync_symbol, sync_universe


class FakeProvider(DataProvider):
    def __init__(self, name, frames, benchmark="IDX"):
        self.name, self.frames, self.benchmark_symbol, self.calls = name, frames, benchmark, []

    def daily(self, symbol, start, end):
        self.calls.append((symbol, pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        df = self.frames.get(symbol)
        if df is None:
            return self.normalize(None)
        return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def bars(start, n, price=100.0):
    idx = pd.bdate_range(start, periods=n)
    c = np.full(n, price) + np.arange(n)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6}, index=idx)


def test_backfill_then_incremental_update(tmp_path):
    db = MarketDB(tmp_path / "m.duckdb")
    full = bars("2020-01-01", 300)
    hist = FakeProvider("yahoo", {"AAA": full.iloc[:250], "IDX": full.iloc[:250]})
    upd = FakeProvider("groww", {"AAA": full}, benchmark=None)
    sync_universe(db, ["AAA"], upd, hist, pd.Timestamp("2020-01-01").date(), full.index[249].date())
    assert db.coverage().set_index("symbol").loc["AAA", "bars"] == 250
    assert db.read(BENCHMARK) is not None  # benchmark came from the history provider
    n = sync_symbol(db, "AAA", upd, hist, pd.Timestamp("2020-01-01").date(), full.index[-1].date())
    got = db.read("AAA")
    assert len(got) == 300 and n > 0
    assert set(db.coverage().sources[0].split(",")) == {"yahoo", "groww"}
    assert db.sql("select count(*) n from fetch_log").n[0] >= 3


def test_split_triggers_full_refresh(tmp_path):
    db = MarketDB(tmp_path / "m.duckdb")
    raw = bars("2021-01-01", 200)
    hist = FakeProvider("yahoo", {"AAA": raw.iloc[:150]})
    sync_symbol(db, "AAA", hist, hist, pd.Timestamp("2021-01-01").date(), raw.index[149].date())
    adjusted = raw.copy()
    adjusted[["open", "high", "low", "close"]] /= 2  # a 1:1 bonus restates all history
    hist.frames["AAA"] = adjusted
    sync_symbol(db, "AAA", hist, hist, pd.Timestamp("2021-01-01").date(), raw.index[-1].date())
    got = db.read("AAA")
    assert len(got) == 200 and np.allclose(got.close.to_numpy(), adjusted.close.to_numpy())
    assert "mismatch" in set(db.sql("select status from fetch_log").status)


def test_load_panel_from_db(tmp_path):
    db = MarketDB(tmp_path / "m.duckdb")
    db.upsert("AAA", bars("2022-01-03", 50), "yahoo")
    db.upsert("BBB", bars("2022-01-10", 40, 50.0), "yahoo")
    db.upsert(BENCHMARK, bars("2022-01-03", 50, 1000.0), "yahoo")
    p = load_panel(["AAA", "BBB", "ZZZ"], db=db)
    assert p.symbols == ["AAA", "BBB"] and p.benchmark is not None
    assert np.isnan(p.close[0, 1]) and p.close[-1, 0] == 149.0
