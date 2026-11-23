from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from swingbot import Panel, RiskConfig
from swingbot.strategies.base import Signals, Strategy


def make_panel(close, open_=None, high=None, low=None, volume=1e7):
    close = np.asarray(close, float).reshape(len(close), -1)
    T, N = close.shape
    open_ = close.copy() if open_ is None else np.asarray(open_, float).reshape(T, N)
    high = np.maximum(open_, close) if high is None else np.asarray(high, float).reshape(T, N)
    low = np.minimum(open_, close) if low is None else np.asarray(low, float).reshape(T, N)
    dates = pd.bdate_range("2024-01-01", periods=T)
    return Panel(dates, [f"S{j}" for j in range(N)], open_, high, low, close, np.full((T, N), volume))


@dataclass
class Scripted(Strategy):
    """Replays hand-written signal arrays."""

    name = "scripted"
    entry: np.ndarray = field(default=None)
    stop: float = 10.0
    exit: np.ndarray = field(default=None)
    target_r: float = 0.0
    trail: float = 0.0
    max_hold: int = 15
    regime_sma: int = 0
    score: np.ndarray = field(default=None)

    def generate(self, p):
        T, N = p.shape
        e = np.asarray(self.entry)[:T].reshape(T, N)
        return Signals(
            entry=e,
            stop_dist=np.full((T, N), self.stop),
            exit=None if self.exit is None else np.asarray(self.exit)[:T].reshape(T, N),
            trail_dist=np.full((T, N), self.trail) if self.trail else None,
            score=None if self.score is None else np.asarray(self.score)[:T].reshape(T, N),
            target_r=self.target_r, min_hold=self.min_hold, max_hold=self.max_hold,
        )

    def settings(self):
        return {}


def onehot(T, *idx, n=1):
    a = np.zeros((T, n), bool)
    for i in idx:
        a[i] = True
    return a


@pytest.fixture
def free_risk():
    return RiskConfig(capital=100_000, risk_per_trade=0.01, max_positions=5, max_position_pct=1.0,
                      max_adv_pct=0, max_entry_gap=0, slippage_bps=0, costs="free")
