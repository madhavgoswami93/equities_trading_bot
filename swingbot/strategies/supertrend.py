from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import indicators as ind
from ..panel import Panel
from .base import Signals, Strategy, as_bool, as_float


@dataclass
class SupertrendSwing(Strategy):
    """Daily Supertrend flip (port of the old intraday kc_supertrend idea).

    Entry : Supertrend direction flips from down to up
    Exit  : direction flips down, initial stop at the Supertrend line
            (floored at `min_atr_stop` ATRs), Supertrend line as trailing stop,
            time stop
    Rank  : 3-month rate of change
    """

    name = "supertrend"

    st_len: int = 10
    st_mult: float = 3.0
    min_atr_stop: float = 1.0
    max_hold: int = 15

    def generate(self, p: Panel) -> Signals:
        c, h, l = p.frame("close"), p.frame("high"), p.frame("low")
        line, d = ind.supertrend(h, l, c, self.st_len, self.st_mult)
        a = ind.atr(h, l, c, self.st_len)
        flip_up = (d == 1) & (d.shift(1) == -1)
        dist = np.maximum(c - line, self.min_atr_stop * a)
        return Signals(
            entry=as_bool(flip_up & self.tradable(p)),
            exit=as_bool(d == -1),
            stop_dist=as_float(dist),
            # trailing: stop follows the line; distance from highest close is
            # approximated by the current close-to-line gap
            trail_dist=as_float((c - line).where(d == 1)),
            score=as_float(ind.roc(c, 63)),
            min_hold=self.min_hold,
            max_hold=self.max_hold,
        )
