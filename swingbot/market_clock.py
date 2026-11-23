"""NSE session clock (IST). The bot trades on daily bars, so the only times that
matter are: has today's session finished, and which session is the latest
complete one. Exchange holidays need no list here: a holiday simply has no bar
in the data, and the live guard refuses to decide on a day without one."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
OPEN = time(9, 15)
CLOSE = time(15, 30)
BARS_FINAL = time(15, 35)  # daily candles are complete a few minutes after the close


def now_ist() -> datetime:
    return datetime.now(IST)


def last_complete_session(now: datetime | None = None) -> date:
    """The latest weekday whose session has finished. Before 15:35 IST today's
    candle is still forming, so it is excluded from downloads."""
    now = now or now_ist()
    d = now.date()
    if d.weekday() < 5 and now.time() >= BARS_FINAL:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def is_market_open(now: datetime | None = None) -> bool:
    now = now or now_ist()
    return now.weekday() < 5 and OPEN <= now.time() < CLOSE
