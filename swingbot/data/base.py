from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    """Daily OHLCV source. Implementations return a DataFrame indexed by a
    tz-naive DatetimeIndex (one row per session) with columns OHLCV."""

    name: str = "base"
    benchmark_symbol: str | None = None  # the provider's symbol for the Nifty 50 index

    @abstractmethod
    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...

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
