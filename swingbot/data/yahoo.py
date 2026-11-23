"""Yahoo Finance provider (pip install yfinance) -- free, no account needed.

Good enough for strategy research (20+ years for large caps) but unofficial and
occasionally patchy; validate final numbers on Kite data before trading real money.

Prices are adjusted for splits and bonuses only, not for dividends (Yahoo's
"auto adjust"), so they match the prices you actually trade at and Kite's
adjustment. Yahoo adjusts some demergers (ITC Hotels, Jan 2025) but not others
(Tata Motors, Oct 2025: shows as a fake crash in the quality report); Kite adjusts them.
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
            start=str(start), end=str(pd.Timestamp(end).date() + timedelta(days=1)), auto_adjust=False, actions=False
        )
        return self.normalize(df)

    def daily_many(self, symbols: list[str], start: date, end: date, chunk: int = 100) -> dict[str, pd.DataFrame]:
        """Bulk download (one request per ~100 tickers) -- minutes instead of an hour for all of NSE."""
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), chunk):
            part = symbols[i:i + chunk]
            tick = {self.ticker(s): s for s in part}
            raw = yf.download(list(tick), start=str(start), end=str(pd.Timestamp(end).date() + timedelta(days=1)),
                              auto_adjust=False, actions=False, group_by="ticker", threads=True, progress=False)
            for t, s in tick.items():
                try:
                    df = raw[t] if len(tick) > 1 else raw
                except KeyError:
                    continue
                df = self.normalize(df.dropna(how="all"))
                if not df.empty:
                    out[s] = df
        return out

    def intraday(self, symbol: str, start: date, end: date, interval: str = "5minute") -> pd.DataFrame:
        """Yahoo keeps only the last ~60 days of 5-minute candles."""
        import yfinance as yf

        yi = {"minute": "1m", "5minute": "5m", "15minute": "15m", "30minute": "30m", "60minute": "60m"}[interval]
        earliest = date.today() - timedelta(days=59)
        df = yf.Ticker(self.ticker(symbol)).history(start=str(max(pd.Timestamp(start).date(), earliest)),
                                                    end=str(pd.Timestamp(end).date() + timedelta(days=1)),
                                                    interval=yi, auto_adjust=False, actions=False)
        return self.normalize_intraday(df)
