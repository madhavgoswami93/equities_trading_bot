"""Yahoo Finance provider (pip install yfinance) -- free, no account needed.

Good enough for strategy research (split/bonus-adjusted, 20+ years for large
caps) but unofficial and occasionally patchy; validate final numbers on Kite
data before trading real money.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .base import DataProvider


class YahooProvider(DataProvider):
    name = "yahoo"
    benchmark_symbol = "^NSEI"

    def __init__(self, suffix: str = ".NS"):
        self.suffix = suffix

    def ticker(self, symbol: str) -> str:
        return symbol if symbol.startswith("^") or "." in symbol else symbol + self.suffix

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # we report failures ourselves, once

        df = yf.Ticker(self.ticker(symbol)).history(
            start=str(start), end=str(pd.Timestamp(end).date() + timedelta(days=1)), auto_adjust=True, actions=False
        )
        return self.normalize(df)
