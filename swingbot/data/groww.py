"""Groww Trade API provider (pip install growwapi).

Historical candles require Groww's paid Trade API subscription. Auth, from the
environment, in order of preference:

    GROWW_ACCESS_TOKEN                       -- a token you generated already
    GROWW_API_KEY + GROWW_TOTP_SECRET        -- TOTP flow (no daily approval)
    GROWW_API_KEY + GROWW_API_SECRET         -- key/secret flow (approve daily
                                                on the Groww Cloud API keys page)

Groww documents equity candles from 2020 onward and caps daily candles at 180
days per request on the current `get_historical_candles` endpoint, so long
histories are fetched in 180-day chunks. Use Kite (or Yahoo for research) when
you need data before 2020.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd

from .base import DataProvider, require_env

MAX_DAYS_PER_CALL = 180
MIN_INTERVAL_S = 0.2


def _client():
    from growwapi import GrowwAPI  # optional dependency

    token = os.environ.get("GROWW_ACCESS_TOKEN")
    if not token:
        (api_key,) = require_env("GROWW_API_KEY")
        if os.environ.get("GROWW_TOTP_SECRET"):
            import pyotp

            token = GrowwAPI.get_access_token(api_key=api_key, totp=pyotp.TOTP(os.environ["GROWW_TOTP_SECRET"]).now())
        else:
            (secret,) = require_env("GROWW_API_SECRET")
            token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
    return GrowwAPI(token)


class GrowwProvider(DataProvider):
    name = "groww"
    benchmark_symbol = None  # the Nifty index comes from the history provider (Yahoo)

    def __init__(self, exchange: str = "NSE"):
        self.g = _client()
        self.exchange = exchange
        self._last = 0.0

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        g = self.g
        chunks, cur = [], pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        while cur <= end:
            to = min(cur + timedelta(days=MAX_DAYS_PER_CALL - 1), end)
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            resp = g.get_historical_candles(
                exchange=g.EXCHANGE_NSE if self.exchange == "NSE" else g.EXCHANGE_BSE,
                segment=g.SEGMENT_CASH,
                groww_symbol=f"{self.exchange}-{symbol}",
                start_time=f"{cur} 09:15:00",
                end_time=f"{to} 15:30:00",
                candle_interval=g.CANDLE_INTERVAL_DAY,
            )
            self._last = time.monotonic()
            candles = (resp or {}).get("candles") or []
            if candles:
                df = pd.DataFrame([row[:6] for row in candles],
                                  columns=["date", "open", "high", "low", "close", "volume"])
                ts = df["date"]
                df["date"] = pd.to_datetime(ts, unit="s") if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts)
                chunks.append(df.set_index("date"))
            cur = to + timedelta(days=1)
        return self.normalize(pd.concat(chunks) if chunks else None)
