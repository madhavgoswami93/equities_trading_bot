from __future__ import annotations

from dataclasses import dataclass

from .. import indicators as ind
from ..panel import Panel
from .base import Signals, Strategy, as_bool, as_float


@dataclass
class VolumeBreakout(Strategy):
    """Momentum breakout from an N-day range on expanding volume.

    Entry : close > highest high of the prior `lookback` sessions,
            volume > vol_mult x 20-day average volume, close > SMA(trend_len)
    Exit  : ATR trailing stop from the highest close, initial ATR stop,
            optional R-multiple target, time stop
    Rank  : 3-month rate of change
    """

    name = "breakout"

    lookback: int = 20
    vol_mult: float = 1.5
    trend_len: int = 50
    atr_len: int = 14
    atr_stop: float = 2.0
    atr_trail: float = 3.0
    target_r: float = 0.0
    max_hold: int = 15

    def generate(self, p: Panel) -> Signals:
        c, h, l, v = p.frame("close"), p.frame("high"), p.frame("low"), p.frame("volume")
        prior_high = ind.highest(h, self.lookback).shift(1)
        a = ind.atr(h, l, c, self.atr_len)
        entry = (
            (c > prior_high)
            & (v > self.vol_mult * ind.sma(v, 20))
            & (c > ind.sma(c, self.trend_len))
            & self.tradable(p)
        )
        return Signals(
            entry=as_bool(entry),
            stop_dist=as_float(self.atr_stop * a),
            trail_dist=as_float(self.atr_trail * a) if self.atr_trail > 0 else None,
            score=as_float(ind.roc(c, 63)),
            target_r=self.target_r,
            min_hold=self.min_hold,
            max_hold=self.max_hold,
        )
