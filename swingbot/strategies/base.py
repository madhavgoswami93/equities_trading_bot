"""Strategy interface.

A strategy is *pure signal generation*: given a Panel it returns arrays saying
where it wants to be long, how far away the protective stop is, and how it
wants to get out. Execution, sizing, costs and portfolio constraints all live
in the engine, so strategies stay short and cannot accidentally peek ahead
(the engine only acts on a signal at the next bar's open).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import ClassVar

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..panel import Panel


@dataclass
class Signals:
    entry: np.ndarray  # bool (T, N): go long at next open
    stop_dist: np.ndarray  # float (T, N): initial stop distance in Rs from the fill
    exit: np.ndarray | None = None  # bool (T, N): discretionary exit at next open
    score: np.ndarray | None = None  # float (T, N): ranking when slots are scarce
    trail_dist: np.ndarray | None = None  # float (T, N): trailing stop from highest close
    target_r: float = 0.0  # take profit at entry + target_r * stop_dist (0 = off)
    min_hold: int = 1  # bars before a signal exit is honoured (stops always are)
    max_hold: int = 15  # time stop in bars (15 sessions ~ 3 weeks)


@dataclass
class Strategy:
    """Base class. Subclasses add their own dataclass fields as parameters."""

    name: ClassVar[str] = "base"

    min_hold: int = 1
    max_hold: int = 15
    min_price: float = 20.0  # skip penny stocks
    min_turnover_cr: float = 5.0  # min 20-day average traded value, Rs crore
    regime_sma: int = 200  # entries only when benchmark > its SMA (0 = off)

    def generate(self, p: Panel) -> Signals:  # pragma: no cover - abstract
        raise NotImplementedError

    # -------------------------------------------------------------- helpers
    def params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def with_params(self, **kw) -> Strategy:
        return replace(self, **kw)

    def tradable(self, p: Panel) -> pd.DataFrame:
        """Universe filter common to all strategies: price floor + liquidity."""
        c = p.frame("close")
        turnover = ind.sma(c * p.frame("volume"), 20)
        return (c >= self.min_price) & (turnover >= self.min_turnover_cr * 1e7)

    def regime(self, p: Panel) -> np.ndarray:
        T = len(p.dates)
        if p.benchmark is None or not self.regime_sma:
            return np.ones(T, dtype=bool)
        b = pd.Series(p.benchmark)
        return (b > b.rolling(self.regime_sma, min_periods=self.regime_sma).mean()).to_numpy()


def as_bool(x) -> np.ndarray:
    return np.asarray(pd.DataFrame(x).fillna(False), dtype=bool)


def as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)
