from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
log = logging.getLogger(__name__)


def with_retry(fn, *args, tries: int = 5, base_delay: float = 1.0, **kw):
    """Call a broker API with exponential backoff -- long downloads hit rate limits
    (HTTP 429) and network blips; one failure must not kill a 90-minute run."""
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            permanent = any(w in msg for w in ("permission", "invalid", "not found", "token", "insufficient"))
            if permanent or i == tries - 1:
                raise
            delay = base_delay * 2 ** i
            log.info("retry %d/%d in %.0fs after: %s", i + 1, tries, delay, e)
            time.sleep(delay)


class DataProvider(ABC):
    """Daily OHLCV source. Implementations return a DataFrame indexed by a
    tz-naive DatetimeIndex (one row per session) with columns OHLCV."""

    name: str = "base"
    benchmark_symbol: str | None = None  # the provider's symbol for the Nifty 50 index

    @abstractmethod
    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...

    def daily_many(self, symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """Several symbols at once; providers with a bulk endpoint override this."""
        out = {}
        for s in symbols:
            try:
                out[s] = self.daily(s, start, end)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: %s", s, e)
        return out

    def intraday(self, symbol: str, start: date, end: date, interval: str = "5minute") -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not provide intraday candles")

    @staticmethod
    def normalize_intraday(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="ts"))
        df = df.rename(columns=str.lower)
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)  # store exchange time (IST), naive
        df.index = idx
        df.index.name = "ts"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[OHLCV].astype("float64")

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="date"))
        df = df.rename(columns=str.lower)
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        df.index = idx.normalize()
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[OHLCV].astype("float64")


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so python-dotenv is not required.
    Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def require_env(*names: str) -> list[str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing environment variable(s): {', '.join(missing)} (see .env.example)")
    return [os.environ[n] for n in names]
