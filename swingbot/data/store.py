"""Local market database: one DuckDB file holds every daily bar you have ever downloaded.

    data/market.duckdb
      bars_daily     symbol, date, open, high, low, close, volume, source, fetched_at   (PK symbol, date)
      bars_intraday  symbol, interval, ts (IST), open, high, low, close, volume, source  (e.g. 5-minute)
      fetch_log    logged_at, provider, symbol, from_date, to_date, rows, status, message

Why a database and not files: one file to back up, SQL for ad-hoc questions
(`swingbot db sql FILE "select ..."`), atomic upserts, and the history survives
switching brokers. DuckDB is embedded (no server) and reads columns fast.

Two providers fill it:
  * the history provider backfills long history once (default: free Yahoo, 15+ years)
  * the update provider appends new sessions every day (your broker: Groww or Kite)

Corporate actions: brokers serve raw prices, Yahoo serves split-adjusted ones.
Every update re-downloads the last few sessions and compares them with what is
stored; if they disagree by more than 1% (a split or bonus restated history),
the symbol's whole history is re-downloaded from the history provider.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..panel import Panel
from .base import OHLCV, DataProvider

log = logging.getLogger(__name__)
DEFAULT_DB = Path("data/market.duckdb")
BENCHMARK = "NIFTY50"  # canonical name of the benchmark index inside the database

_SCHEMA = """
create table if not exists bars_daily (
    symbol varchar not null, date date not null,
    open double, high double, low double, close double, volume double,
    source varchar, fetched_at timestamp default current_timestamp,
    primary key (symbol, date));
create table if not exists bars_intraday (
    symbol varchar not null, interval varchar not null, ts timestamp not null,
    open double, high double, low double, close double, volume double,
    source varchar, primary key (symbol, interval, ts));
create table if not exists fetch_log (
    logged_at timestamp default current_timestamp, provider varchar, symbol varchar,
    from_date date, to_date date, rows integer, status varchar, message varchar);
"""


class MarketDB:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self, read_only: bool = False):
        import duckdb  # imported lazily so the engine itself does not need it

        if read_only and not self.path.exists():
            read_only = False
        c = duckdb.connect(str(self.path), read_only=read_only)
        try:
            yield c
        finally:
            c.close()  # short-lived connections: the dashboard and the CLI can take turns

    # ---------------------------------------------------------------- read
    def read(self, symbol: str, start=None, end=None) -> pd.DataFrame | None:
        q = "select date, open, high, low, close, volume from bars_daily where symbol = ?"
        args: list = [symbol]
        if start is not None:
            q += " and date >= ?"
            args.append(pd.Timestamp(start).date())
        if end is not None:
            q += " and date <= ?"
            args.append(pd.Timestamp(end).date())
        with self._conn(read_only=True) as c:
            df = c.execute(q + " order by date", args).df()
        if df.empty:
            return None
        return df.set_index(pd.DatetimeIndex(df.pop("date"), name="date"))

    def read_many(self, symbols: list[str], start=None, end=None) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        q = "select symbol, date, open, high, low, close, volume from bars_daily where symbol in (select unnest(?))"
        args: list = [list(symbols)]
        if start is not None:
            q += " and date >= ?"
            args.append(pd.Timestamp(start).date())
        if end is not None:
            q += " and date <= ?"
            args.append(pd.Timestamp(end).date())
        with self._conn(read_only=True) as c:
            df = c.execute(q + " order by symbol, date", args).df()
        out = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.drop(columns="symbol")
            out[sym] = g.set_index(pd.DatetimeIndex(g.pop("date"), name="date"))
        return out

    def last_date(self, symbol: str) -> date | None:
        with self._conn(read_only=True) as c:
            r = c.execute("select max(date) from bars_daily where symbol = ?", [symbol]).fetchone()
        return r[0] if r and r[0] else None

    def coverage(self) -> pd.DataFrame:
        with self._conn(read_only=True) as c:
            return c.execute("""
                select symbol, min(date) as first, max(date) as last, count(*) as bars,
                       string_agg(distinct source, ',') as sources, max(fetched_at) as updated
                from bars_daily group by symbol order by symbol""").df()

    def sql(self, query: str) -> pd.DataFrame:
        with self._conn(read_only=True) as c:
            return c.execute(query).df()

    # --------------------------------------------------------------- write
    def upsert(self, symbol: str, df: pd.DataFrame, source: str, replace: bool = False) -> int:
        if df is None or df.empty:
            return 0
        rows = df[OHLCV].copy()
        rows.insert(0, "date", pd.DatetimeIndex(rows.index).date)
        rows.insert(0, "symbol", symbol)
        rows["source"] = source
        with self._conn() as c:
            c.register("incoming", rows.reset_index(drop=True))
            c.execute("begin")
            if replace:
                c.execute("delete from bars_daily where symbol = ?", [symbol])
            c.execute("""
                insert or replace into bars_daily (symbol, date, open, high, low, close, volume, source)
                select symbol, date, open, high, low, close, volume, source from incoming""")
            c.execute("commit")
        return len(rows)

    # ------------------------------------------------------------ intraday
    def upsert_intraday(self, symbol: str, interval: str, df: pd.DataFrame, source: str) -> int:
        if df is None or df.empty:
            return 0
        rows = df[OHLCV].copy()
        rows.insert(0, "ts", pd.DatetimeIndex(rows.index))
        rows.insert(0, "interval", interval)
        rows.insert(0, "symbol", symbol)
        rows["source"] = source
        with self._conn() as c:
            c.register("incoming", rows.reset_index(drop=True))
            c.execute("insert or replace into bars_intraday select * from incoming")
        return len(rows)

    def last_ts(self, symbol: str, interval: str):
        with self._conn(read_only=True) as c:
            r = c.execute("select max(ts) from bars_intraday where symbol = ? and interval = ?",
                          [symbol, interval]).fetchone()
        return r[0] if r and r[0] else None

    def read_intraday(self, symbol: str, interval: str = "5minute", start=None, end=None) -> pd.DataFrame | None:
        q = "select ts, open, high, low, close, volume from bars_intraday where symbol = ? and interval = ?"
        args: list = [symbol, interval]
        if start is not None:
            q += " and ts >= ?"
            args.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            q += " and ts < ?"
            args.append((pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime())
        with self._conn(read_only=True) as c:
            df = c.execute(q + " order by ts", args).df()
        return None if df.empty else df.set_index(pd.DatetimeIndex(df.pop("ts"), name="ts"))

    def intraday_coverage(self) -> pd.DataFrame:
        with self._conn(read_only=True) as c:
            return c.execute("""select interval, count(distinct symbol) symbols, min(ts) first, max(ts) last,
                                count(*) bars from bars_intraday group by interval""").df()

    def log_fetch(self, provider, symbol, start, end, rows, status, message=""):
        with self._conn() as c:
            c.execute("insert into fetch_log (provider, symbol, from_date, to_date, rows, status, message) "
                      "values (?, ?, ?, ?, ?, ?, ?)",
                      [provider, symbol, start, end, rows, status, str(message)[:500]])

    def delete(self, symbol: str) -> None:
        with self._conn() as c:
            c.execute("delete from bars_daily where symbol = ?", [symbol])


# ------------------------------------------------------------------ syncing
def _provider_symbol(provider: DataProvider, symbol: str) -> str | None:
    """Map the canonical benchmark name to the provider's own symbol."""
    return provider.benchmark_symbol if symbol == BENCHMARK else symbol


def sync_symbol(db: MarketDB, symbol: str, update: DataProvider, history: DataProvider | None,
                start: date, end: date | None = None, full: bool = False) -> int:
    """Bring one symbol up to date. Returns the number of rows written."""
    end = end or date.today()
    history = history or update
    last = None if full else db.last_date(symbol)

    if last is None:  # never seen: backfill from the history provider
        src = history if _provider_symbol(history, symbol) else update
        psym = _provider_symbol(src, symbol)
        if not psym:
            raise ValueError("no provider can serve this symbol")
        df = src.daily(psym, start, end)
        if df.empty:
            raise ValueError("no data returned (renamed/delisted symbol?)")
        n = db.upsert(symbol, df, src.name, replace=True)
        db.log_fetch(src.name, symbol, start, end, n, "backfill")
        return n

    src = update if _provider_symbol(update, symbol) else history
    psym = _provider_symbol(src, symbol)
    since = last - timedelta(days=7)  # overlap: the last stored candle may have been partial
    new = src.daily(psym, since, end)
    if new.empty:
        db.log_fetch(src.name, symbol, since, end, 0, "empty")
        return 0
    old = db.read(symbol, start=since)
    if old is not None:
        ov = old.index.intersection(new.index)
        if len(ov) and (abs(old.loc[ov, "close"] / new.loc[ov, "close"] - 1) > 0.01).any():
            log.warning("%s: stored prices disagree with %s (split/bonus?) -- re-downloading full history",
                        symbol, src.name)
            db.log_fetch(src.name, symbol, since, end, 0, "mismatch", "corporate action suspected")
            return sync_symbol(db, symbol, update, history, start, end, full=True)
    n = db.upsert(symbol, new, src.name)
    db.log_fetch(src.name, symbol, since, end, n, "update")
    return n


class _Progress:
    def __init__(self, total: int, what: str):
        self.total, self.what, self.done, self.t0 = total, what, 0, time.monotonic()

    def step(self, symbol: str, msg: str) -> None:
        self.done += 1
        el = time.monotonic() - self.t0
        eta = el / self.done * (self.total - self.done)
        log.info("[%d/%d %s] %s %s  (elapsed %s, ~%s left)", self.done, self.total, self.what, symbol, msg,
                 _hms(el), _hms(eta))


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def sync_universe(db: MarketDB, symbols: list[str], update: DataProvider, history: DataProvider | None,
                  start: date, end: date | None = None, full: bool = False) -> list[str]:
    """Update every symbol plus the benchmark. Returns the symbols that failed.

    Symbols never seen before are backfilled in bulk when the history provider has a bulk
    endpoint (Yahoo: ~100 tickers per request); the rest are updated one by one. Safe to
    interrupt: everything already written stays, and the next run continues from there."""
    end = end or date.today()
    history = history or update
    todo = list(dict.fromkeys(list(symbols) + [BENCHMARK]))
    have = set() if full else set(db.coverage().symbol)
    missing = [s for s in todo if s not in have and s != BENCHMARK and _provider_symbol(history, s)]
    failed: list[str] = []
    if missing and type(history).daily_many is not DataProvider.daily_many:
        log.info("bulk backfill of %d symbols from %s", len(missing), history.name)
        got = history.daily_many(missing, start, end)
        for s in missing:
            df = got.get(s)
            if df is None or df.empty:
                db.log_fetch(history.name, s, start, end, 0, "error", "no data returned")
                failed.append(s)
            else:
                db.log_fetch(history.name, s, start, end, db.upsert(s, df, history.name, replace=True), "backfill")
        todo = [s for s in todo if s not in missing]
    prog = _Progress(len(todo), "daily")
    for s in todo:
        try:
            n = sync_symbol(db, s, update, history, start, end, full=full)
            prog.step(s, f"+{n} rows")
        except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the run
            prog.step(s, f"FAILED: {e}")
            db.log_fetch(getattr(update, "name", "?"), s, start, end, 0, "error", e)
            failed.append(s)
    return failed


def sync_intraday(db: MarketDB, symbols: list[str], provider: DataProvider, interval: str,
                  start: date, end: date | None = None) -> list[str]:
    """Download intraday candles (e.g. 5-minute) for every symbol, resuming from the last
    stored candle. Interrupt and re-run at will. Returns symbols that failed."""
    end = end or date.today()
    failed = []
    prog = _Progress(len(symbols), interval)
    for s in symbols:
        try:
            last = db.last_ts(s, interval)
            since = pd.Timestamp(last).date() if last is not None else pd.Timestamp(start).date()
            if since > end:
                prog.step(s, "up to date")
                continue
            df = provider.intraday(s, since, end, interval)
            n = db.upsert_intraday(s, interval, df, provider.name)
            db.log_fetch(provider.name, s, since, end, n, f"intraday:{interval}")
            prog.step(s, f"+{n} candles")
        except Exception as e:  # noqa: BLE001
            prog.step(s, f"FAILED: {e}")
            db.log_fetch(provider.name, s, start, end, 0, "error", e)
            failed.append(s)
    return failed


def load_panel(symbols: list[str], start=None, end=None, db: MarketDB | None = None,
               benchmark: bool = True) -> Panel:
    """Build a Panel from the local database (no network)."""
    db = db or MarketDB()
    frames = db.read_many(symbols, start, end)
    missing = [s for s in symbols if s not in frames]
    if missing:
        log.warning("no data in %s for %d symbols: %s", db.path, len(missing), ", ".join(missing[:10]))
    if not frames:
        raise ValueError(f"no price data in {db.path} -- run `python -m swingbot fetch <strategy file>` first")
    bench = None
    if benchmark:
        b = db.read(BENCHMARK, start, end)
        if b is not None:
            bench = b["close"]
    panel = Panel.from_frames(frames, benchmark=bench, benchmark_name=BENCHMARK, start=start, end=end)
    if benchmark and bench is None:
        log.info("no %s index in %s (Groww has no index candles): using an equal-weight index of the "
                 "universe as the benchmark", BENCHMARK, db.path)
        panel.benchmark, panel.benchmark_name = panel.equal_weight_index(), "EQUAL_WEIGHT"
    return panel


def import_parquet(db: MarketDB, cache_root: str | Path = "data/cache") -> int:
    """One-off migration from the old Parquet cache (data/cache/<provider>/*.parquet)."""
    n = 0
    for f in sorted(Path(cache_root).glob("*/*.parquet")):
        provider = f.parent.name
        name = f.stem
        symbol = BENCHMARK if name in ("IDX_NSEI", "NIFTY_50") else name.replace("_", " ")
        n += db.upsert(symbol, pd.read_parquet(f), provider, replace=True)
    return n
