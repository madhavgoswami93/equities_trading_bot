from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataProvider, load_env
from .store import ParquetStore, fetch_universe, load_panel
from .synthetic import synthetic_panel

UNIVERSE_DIR = Path(__file__).resolve().parents[2] / "data" / "universes"


def get_provider(name: str) -> DataProvider:
    name = name.lower()
    if name == "kite":
        from .kite import KiteProvider

        return KiteProvider()
    if name == "groww":
        from .groww import GrowwProvider

        return GrowwProvider()
    if name == "yahoo":
        from .yahoo import YahooProvider

        return YahooProvider()
    raise ValueError(f"unknown provider {name!r}; use kite | groww | yahoo")


def benchmark_for(provider: str) -> str | None:
    return {"kite": "NIFTY 50", "yahoo": "^NSEI"}.get(provider)


def load_universe(name_or_path: str) -> list[str]:
    """A universe is a CSV with a `symbol` column (data/universes/<name>.csv),
    or a comma-separated list of symbols."""
    p = Path(name_or_path)
    if not p.exists():
        p = UNIVERSE_DIR / f"{name_or_path}.csv"
    if p.exists():
        return pd.read_csv(p)["symbol"].dropna().str.strip().tolist()
    if "," in name_or_path or name_or_path.isupper():
        return [s.strip() for s in name_or_path.split(",") if s.strip()]
    raise FileNotFoundError(f"universe {name_or_path!r} not found in {UNIVERSE_DIR}")


__all__ = ["DataProvider", "ParquetStore", "benchmark_for", "fetch_universe", "get_provider",
           "load_env", "load_panel", "load_universe", "synthetic_panel"]
