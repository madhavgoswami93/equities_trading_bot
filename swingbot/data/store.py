"""Local Parquet cache: one file per symbol per provider, updated incrementally.

    data/cache/<provider>/<SYMBOL>.parquet

Re-running `fetch` only downloads bars after the last cached date, so a daily
refresh of a 500-stock universe is a few hundred tiny requests.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..panel import Panel
from .base import OHLCV, DataProvider

log = logging.getLogger(__name__)
DEFAULT_ROOT = Path("data/cache")


def _fname(symbol: str) -> str:
    return symbol.replace(" ", "_").replace("/", "_").replace("^", "IDX_") + ".parquet"


class ParquetStore:
    def __init__(self, provider: str, root: str | Path = DEFAULT_ROOT):
        self.dir = Path(root) / provider
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str) -> Path:
        return self.dir / _fname(symbol)

    def read(self, symbol: str) -> pd.DataFrame | None:
        p = self.path(symbol)
        return pd.read_parquet(p) if p.exists() else None

    def write(self, symbol: str, df: pd.DataFrame) -> None:
        df[OHLCV].to_parquet(self.path(symbol))

    def update(self, provider: DataProvider, symbol: str, start: date, end: date | None = None,
               full_refresh: bool = False) -> pd.DataFrame:
        end = end or date.today()
        old = None if full_refresh else self.read(symbol)
        fetch_from = pd.Timestamp(start).date()
        if old is not None and len(old):
            # refetch the last few bars too: the latest candle may have been partial
            fetch_from = max(fetch_from, (old.index[-1] - timedelta(days=5)).date())
        new = provider.daily(symbol, fetch_from, end)
        df = new if old is None else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if old is not None and len(old) and len(new):
            # a corporate action restates history -> the overlap no longer matches
            ov = old.index.intersection(new.index)
            if len(ov) and (abs(old.loc[ov, "close"] / new.loc[ov, "close"] - 1) > 0.01).any():
                log.warning("%s: history changed (split/bonus?) -- doing a full refresh", symbol)
                return self.update(provider, symbol, start, end, full_refresh=True)
        self.write(symbol, df)
        return df


def fetch_universe(provider: DataProvider, symbols: list[str], start: date, end: date | None = None,
                   root: str | Path = DEFAULT_ROOT) -> list[str]:
    """Update the cache for every symbol (+ benchmark). Returns symbols that failed."""
    store = ParquetStore(provider.name, root)
    failed = []
    todo = list(symbols) + ([provider.benchmark_symbol] if provider.benchmark_symbol else [])
    for i, s in enumerate(todo, 1):
        try:
            df = store.update(provider, s, start, end)
            if df.empty:
                store.path(s).unlink(missing_ok=True)
                raise ValueError("no data returned (renamed/delisted symbol?)")
            log.info("[%d/%d] %s: %d bars", i, len(todo), s, len(df))
        except Exception as e:  # keep going -- one delisted symbol should not stop the run
            log.warning("[%d/%d] %s failed: %s", i, len(todo), s, e)
            failed.append(s)
    return failed


def load_panel(provider: str, symbols: list[str], start=None, end=None, benchmark: str | None = None,
               benchmark_provider: str | None = None, root: str | Path = DEFAULT_ROOT) -> Panel:
    """Build a Panel from cached Parquet files (no network). The benchmark can
    come from another provider's cache (e.g. Groww stocks + Yahoo Nifty index)."""
    store = ParquetStore(provider, root)
    frames, missing = {}, []
    for s in symbols:
        df = store.read(s)
        if df is None or df.empty:
            missing.append(s)
        else:
            frames[s] = df
    if missing:
        log.warning("no cached data for %d symbols: %s", len(missing), ", ".join(missing[:10]))
    bench = None
    if benchmark:
        b = ParquetStore(benchmark_provider or provider, root).read(benchmark)
        if b is not None:
            bench = b["close"]
        else:
            log.warning("benchmark %s not cached -- regime filter disabled", benchmark)
    return Panel.from_frames(frames, benchmark=bench, benchmark_name=benchmark, start=start, end=end)
