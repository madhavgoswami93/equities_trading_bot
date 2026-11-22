from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..config import RiskConfig, StrategySpec, specs_param_array
from ..panel import Panel
from ..strategies.base import Strategy
from . import kernel as K
from .result import Result


class Engine:
    """Signals for every strategy, stacked into (S, T, N) arrays, plus the risk
    and cost arrays the kernel needs. Built once per panel and reused by the
    backtester, the paper trader and the live trader."""

    def __init__(self, panel: Panel, strategies, risk: RiskConfig | None = None):
        if isinstance(strategies, (Strategy, StrategySpec)):
            strategies = [strategies]
        self.panel = panel
        self.specs = [StrategySpec.wrap(s) for s in strategies]
        self.risk = risk or RiskConfig()
        T, N = panel.shape
        S = len(self.specs)
        sigs = [s.strategy.generate(panel) for s in self.specs]
        self.E = np.zeros((S, T, N), bool)
        self.X = np.zeros((S, T, N), bool)
        self.SC = np.zeros((S, T, N))
        self.SD = np.full((S, T, N), np.nan)
        self.TD = np.full((S, T, N), np.nan)
        self.regime = np.ones((S, T), bool)
        for i, (spec, sg) in enumerate(zip(self.specs, sigs)):
            self.E[i] = sg.entry
            self.SD[i] = sg.stop_dist
            if sg.exit is not None:
                self.X[i] = sg.exit
            if sg.score is not None:
                self.SC[i] = sg.score
            if sg.trail_dist is not None:
                self.TD[i] = sg.trail_dist
            self.regime[i] = spec.strategy.regime(panel)
        self.signals = sigs
        self.sp = specs_param_array(self.specs, sigs)
        self.rc = self.risk.as_array()
        self.cp = self.risk.costs.as_array()
        self.adv = np.nan_to_num(ind.sma(panel.frame("close") * panel.frame("volume"), 20).to_numpy(), nan=0.0)

    @property
    def strategy_names(self) -> list[str]:
        return [s.name for s in self.specs]

    def advance(self, pos, acct, t0: int, t1: int, entries_on_last: bool):
        """Run the kernel over days [t0, t1) on the given state (mutated in place)."""
        p, n = self.panel, t1 - t0
        trades = np.empty((max(n, 1) * int(self.risk.max_positions) + p.shape[1] + 1, K.TR_COLS))
        eq, cash, inv = np.empty(n), np.empty(n), np.empty(n)
        npos = np.empty(n, np.int64)
        n_tr = K.run(t0, t1, entries_on_last, p.open, p.high, p.low, p.close, self.adv, self.regime,
                     self.E, self.X, self.SC, self.SD, self.TD, self.sp, self.rc, self.cp, pos, acct,
                     trades, 0, eq, cash, inv, npos)
        return trades[:n_tr], eq, cash, inv, npos

    def decide(self, pos, acct, t: int):
        """Only the close-of-day decision for day t (live trading: the broker did the fills)."""
        p = self.panel
        return K.on_close(t, p.close[t], self.adv[t], self.E[:, t], self.X[:, t], self.SC[:, t], self.SD[:, t],
                          self.TD[:, t], self.regime[:, t], True, pos, acct, self.sp, self.rc)

    def trades_frame(self, tr: np.ndarray) -> pd.DataFrame:
        cols = ["strategy", "symbol", "entry_date", "exit_date", "qty", "entry_price", "exit_price", "risk",
                "costs", "pnl", "exit_reason", "bars_held", "return_pct", "r_multiple"]
        if len(tr) == 0:
            return pd.DataFrame(columns=cols)
        dates = self.panel.dates
        ei, xi = tr[:, K.TR_ENTRY_I].astype(int), tr[:, K.TR_EXIT_I].astype(int)
        df = pd.DataFrame({
            "strategy": np.asarray(self.strategy_names)[tr[:, K.TR_STRAT].astype(int)],
            "symbol": np.asarray(self.panel.symbols)[tr[:, K.TR_SYMBOL].astype(int)],
            "entry_date": dates[ei],
            "exit_date": dates[xi],
            "qty": tr[:, K.TR_QTY].astype(int),
            "entry_price": tr[:, K.TR_ENTRY_PX],
            "exit_price": tr[:, K.TR_EXIT_PX],
            "risk": tr[:, K.TR_RISK],
            "costs": tr[:, K.TR_COSTS],
            "pnl": tr[:, K.TR_PNL],
            "exit_reason": np.asarray(K.EXIT_NAMES)[tr[:, K.TR_REASON].astype(int)],
            "bars_held": xi - ei,
        })
        df["return_pct"] = df["pnl"] / (df["qty"] * df["entry_price"])
        df["r_multiple"] = df["pnl"] / df["risk"]
        return df.sort_values(["entry_date", "symbol"], kind="stable").reset_index(drop=True)


class Backtest:
    """Backtest one or more strategies as one portfolio under shared risk rules.

    >>> res = Backtest(panel, [spec_a, spec_b], RiskConfig(max_positions=5)).run()

    Signals are computed once on the full history (they are causal), so any
    sub-window can be simulated with fully warmed-up indicators -- that is
    what walk-forward analysis uses.
    """

    def __init__(self, panel: Panel, strategies, risk: RiskConfig | None = None):
        self.panel = panel
        self._strategies = strategies
        self.risk = risk or RiskConfig()
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = Engine(self.panel, self._strategies, self.risk)
        return self._engine

    def run(self, start=None, end=None) -> Result:
        eng, p = self.engine, self.panel
        T = p.shape[0]
        i0 = p.index_of(start) if start is not None else 0
        i1 = p.index_of(end, side="right") if end is not None else T
        if i1 - i0 < 2:
            raise ValueError("backtest window has fewer than 2 bars")
        pos, acct = K.new_state(p.shape[1], self.risk.capital)
        # the last bar of the window places no new entries: everything is liquidated there
        if i1 == T:
            tr, eq, cash, inv, npos = eng.advance(pos, acct, i0, i1, entries_on_last=False)
        else:
            a = eng.advance(pos, acct, i0, i1 - 1, entries_on_last=True)
            b = self._last_day(pos, acct, i1 - 1)
            tr = np.vstack([a[0], b[0]])
            eq, cash, inv, npos = (np.r_[x, y] for x, y in zip(a[1:], b[1:]))
        extra = np.empty((p.shape[1], K.TR_COLS))
        n = K.liquidate(i1 - 1, pos, acct, self.risk.costs.as_array(), extra, 0)
        tr = np.vstack([tr, extra[:n]])
        eq[-1], cash[-1], inv[-1] = acct[K.CASH], acct[K.CASH], 0.0
        dates = p.dates[i0:i1]
        bench = None
        if p.benchmark is not None:
            bench = pd.Series(p.benchmark[i0:i1], index=dates, name=p.benchmark_name or "benchmark")
        return Result(
            equity=pd.Series(eq, index=dates, name="equity"),
            cash=pd.Series(cash, index=dates, name="cash"),
            invested=pd.Series(inv, index=dates, name="invested"),
            positions=pd.Series(npos, index=dates, name="positions"),
            trades=eng.trades_frame(tr),
            strategy=" + ".join(eng.strategy_names),
            params={s.name: {"quantity": str(s.quantity), **s.strategy.settings()} for s in eng.specs},
            benchmark=bench,
        )

    def _last_day(self, pos, acct, t):
        """Day t with new entries disabled (the kernel does this itself only on the
        panel's final bar, so mask the regime for a window that ends earlier)."""
        eng = self.engine
        saved = eng.regime[:, t].copy()
        eng.regime[:, t] = False
        try:
            return eng.advance(pos, acct, t, t + 1, entries_on_last=True)
        finally:
            eng.regime[:, t] = saved


def run_backtest(panel: Panel, strategies, risk: RiskConfig | None = None, start=None, end=None) -> Result:
    return Backtest(panel, strategies, risk).run(start, end)
