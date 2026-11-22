from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..costs import CostModel
from ..panel import Panel
from ..strategies.base import Signals, Strategy
from .kernel import EXIT_NAMES, simulate
from .result import Result


@dataclass
class EngineConfig:
    """Portfolio / execution assumptions (everything that is not the strategy)."""

    initial_cash: float = 1_000_000.0
    risk_per_trade: float = 0.01  # equity fraction lost if the initial stop is hit
    max_positions: int = 10
    max_position_pct: float = 0.20  # cap per position as a fraction of equity
    max_adv_pct: float = 0.02  # never buy more than 2% of 20-day avg traded value
    max_entry_gap: float = 0.05  # skip entry if open gaps up > 5% over signal close (0 = off)
    slippage_bps: float = 10.0  # per side, applied to every fill
    costs: CostModel = field(default_factory=CostModel.zerodha)


class Backtest:
    """Couples a Panel, a Strategy and an EngineConfig.

    >>> res = Backtest(panel, TrendPullback(), EngineConfig()).run()
    >>> print(res.summary())

    Signals are computed once on the full history (they are causal, so this is
    safe) and the simulation can then be run on any sub-window via start/end,
    which is what walk-forward analysis uses: indicators are fully warmed up
    at the start of every window.
    """

    def __init__(self, panel: Panel, strategy: Strategy, config: EngineConfig | None = None):
        self.panel = panel
        self.strategy = strategy
        self.config = config or EngineConfig()
        self._signals: Signals | None = None
        self._regime: np.ndarray | None = None
        self._adv: np.ndarray | None = None

    @property
    def signals(self) -> Signals:
        if self._signals is None:
            self._signals = self.strategy.generate(self.panel)
            self._regime = self.strategy.regime(self.panel)
        return self._signals

    def run(self, start=None, end=None) -> Result:
        p, cfg, sig = self.panel, self.config, self.signals
        T, N = p.shape
        i0 = p.index_of(start) if start is not None else 0
        i1 = p.index_of(end, side="right") if end is not None else T
        if i1 - i0 < 2:
            raise ValueError("backtest window has fewer than 2 bars")
        if self._adv is None:
            self._adv = ind.sma(p.frame("close") * p.frame("volume"), 20).to_numpy()

        w = slice(i0, i1)
        nan = np.full((T, N), np.nan)
        eq, cash, inv, npos, tr = simulate(
            p.open[w], p.high[w], p.low[w], p.close[w],
            np.ascontiguousarray(sig.entry[w]),
            np.ascontiguousarray(sig.exit[w]) if sig.exit is not None else np.zeros((i1 - i0, N), bool),
            np.ascontiguousarray(sig.score[w]) if sig.score is not None else np.zeros((i1 - i0, N)),
            np.ascontiguousarray(sig.stop_dist[w]),
            np.ascontiguousarray((sig.trail_dist if sig.trail_dist is not None else nan)[w]),
            np.ascontiguousarray(self._adv[w]),
            np.ascontiguousarray(self._regime[w]),
            float(sig.target_r), int(sig.min_hold), int(sig.max_hold),
            float(cfg.initial_cash), float(cfg.risk_per_trade), int(cfg.max_positions),
            float(cfg.max_position_pct), float(cfg.max_adv_pct), float(cfg.max_entry_gap),
            cfg.slippage_bps / 1e4, cfg.costs.as_array(),
        )
        dates = p.dates[w]
        bench = None
        if p.benchmark is not None:
            bench = pd.Series(p.benchmark[w], index=dates, name=p.benchmark_name or "benchmark")
        return Result(
            equity=pd.Series(eq, index=dates, name="equity"),
            cash=pd.Series(cash, index=dates, name="cash"),
            invested=pd.Series(inv, index=dates, name="invested"),
            positions=pd.Series(npos, index=dates, name="positions"),
            trades=self._trades_frame(tr, dates),
            strategy=self.strategy.name,
            params=self.strategy.params(),
            benchmark=bench,
        )

    def _trades_frame(self, tr: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        cols = ["symbol", "entry_date", "exit_date", "qty", "entry_price", "exit_price",
                "risk", "costs", "pnl", "exit_reason", "bars_held", "return_pct", "r_multiple"]
        if len(tr) == 0:
            return pd.DataFrame(columns=cols)
        sym = np.asarray(self.panel.symbols)[tr[:, 0].astype(int)]
        ei, xi = tr[:, 1].astype(int), tr[:, 2].astype(int)
        # trading-day holding period from the symbol's own calendar is close enough
        # to the panel calendar for liquid NSE names
        df = pd.DataFrame({
            "symbol": sym,
            "entry_date": dates[ei],
            "exit_date": dates[xi],
            "qty": tr[:, 3].astype(int),
            "entry_price": tr[:, 4],
            "exit_price": tr[:, 5],
            "risk": tr[:, 6],
            "costs": tr[:, 7],
            "pnl": tr[:, 8],
            "exit_reason": np.asarray(EXIT_NAMES)[tr[:, 9].astype(int)],
            "bars_held": xi - ei,
        })
        df["return_pct"] = df["pnl"] / (df["qty"] * df["entry_price"])
        df["r_multiple"] = df["pnl"] / df["risk"]
        return df.sort_values(["entry_date", "symbol"], kind="stable").reset_index(drop=True)


def run_backtest(panel: Panel, strategy: Strategy, config: EngineConfig | None = None, start=None, end=None) -> Result:
    return Backtest(panel, strategy, config).run(start, end)
