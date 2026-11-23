from __future__ import annotations

from dataclasses import dataclass

from .. import indicators as ind
from ..panel import Panel
from .base import Signals, Strategy, as_bool, as_float


@dataclass
class TrendPullback(Strategy):
    """Buy short-term weakness inside an established up-trend.

    Entry : close > SMA(trend) and SMA(fast) > SMA(trend) and RSI(rsi_len) < rsi_entry
    Exit  : RSI(rsi_len) > rsi_exit, or initial ATR stop, or time stop
    Rank  : strongest 6-month momentum first
    Typical hold 2-10 sessions.
    """

    name = "pullback"

    trend_len: int = 200
    fast_len: int = 50
    rsi_len: int = 3
    rsi_entry: float = 15.0
    rsi_exit: float = 70.0
    atr_len: int = 14
    atr_stop: float = 2.5
    target_r: float = 0.0
    max_hold: int = 10

    def generate(self, p: Panel) -> Signals:
        c, h, l = p.frame("close"), p.frame("high"), p.frame("low")
        trend = ind.sma(c, self.trend_len)
        fast = ind.sma(c, self.fast_len)
        r = ind.rsi(c, self.rsi_len)
        a = ind.atr(h, l, c, self.atr_len)
        entry = (c > trend) & (fast > trend) & (r < self.rsi_entry) & self.tradable(p)
        return Signals(
            entry=as_bool(entry),
            exit=as_bool(r > self.rsi_exit),
            stop_dist=as_float(self.atr_stop * a),
            score=as_float(ind.roc(c, 126)),
            target_r=self.target_r,
            min_hold=self.min_hold,
            max_hold=self.max_hold,
        )
