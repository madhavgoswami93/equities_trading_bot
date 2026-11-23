"""Zerodha Kite Connect provider.

Historical candles need the paid "Connect" plan (the free "Personal" plan has
orders/portfolio only). Credentials come from the environment:

    KITE_API_KEY, KITE_API_SECRET   -- from developers.kite.trade
    KITE_ACCESS_TOKEN               -- valid for one trading day; generate with
                                       `python -m swingbot kite-login`
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .base import DataProvider, require_env, with_retry

TOKEN_FILE = Path(".secrets/kite_access_token")
MAX_DAYS_PER_CALL = 2000  # Kite limit for the "day" interval
INTRADAY_DAYS = {"minute": 60, "3minute": 100, "5minute": 100, "10minute": 100, "15minute": 200,
                 "30minute": 200, "60minute": 400}  # Kite's max days per request by interval
MIN_INTERVAL_S = 0.35  # historical API allows ~3 requests/second


def _client():
    from kiteconnect import KiteConnect  # optional dependency: pip install kiteconnect

    (api_key,) = require_env("KITE_API_KEY")
    kite = KiteConnect(api_key=api_key)
    token = os.environ.get("KITE_ACCESS_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None)
    if token:
        kite.set_access_token(token)
    return kite


def interactive_login() -> str:
    """Print the login URL, read the request_token from the redirect, store the access token."""
    kite = _client()
    (secret,) = require_env("KITE_API_SECRET")
    print("Open this URL, log in, and copy `request_token` from the redirect URL:\n ", kite.login_url())
    request_token = input("request_token: ").strip()
    token = kite.generate_session(request_token, api_secret=secret)["access_token"]
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    print(f"Access token saved to {TOKEN_FILE} (valid until ~6am tomorrow).")
    return token


class KiteProvider(DataProvider):
    name = "kite"
    benchmark_symbol = "NIFTY 50"

    def __init__(self, exchange: str = "NSE"):
        self.kite = _client()
        self.exchange = exchange
        self._tokens: dict[str, int] | None = None
        self._last = 0.0

    def token(self, symbol: str) -> int:
        if self._tokens is None:
            inst = pd.DataFrame(self.kite.instruments(self.exchange))
            self._tokens = dict(zip(inst["tradingsymbol"], inst["instrument_token"]))
        try:
            return int(self._tokens[symbol])
        except KeyError:
            raise KeyError(f"{symbol} not found in {self.exchange} instrument dump") from None

    def _candles(self, symbol: str, start, end, interval: str, max_days: int) -> pd.DataFrame | None:
        tok = self.token(symbol)
        chunks, cur = [], pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        while cur <= end:
            to = min(cur + timedelta(days=max_days - 1), end)
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            rows = with_retry(self.kite.historical_data, tok, cur, to, interval)
            self._last = time.monotonic()
            if rows:
                chunks.append(pd.DataFrame(rows).set_index("date"))
            cur = to + timedelta(days=1)
        return pd.concat(chunks) if chunks else None

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self.normalize(self._candles(symbol, start, end, "day", MAX_DAYS_PER_CALL))

    def intraday(self, symbol: str, start: date, end: date, interval: str = "5minute") -> pd.DataFrame:
        if interval not in INTRADAY_DAYS:
            raise ValueError(f"interval must be one of {', '.join(INTRADAY_DAYS)}")
        return self.normalize_intraday(self._candles(symbol, start, end, interval, INTRADAY_DAYS[interval]))
