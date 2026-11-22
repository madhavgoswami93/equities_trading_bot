"""Parameter sweeps and walk-forward analysis.

Sweeps run in parallel processes. On Linux/macOS the panel is shared with the
workers through fork (copy-on-write), so a 500-stock x 20-year panel is not
pickled once per job.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..panel import Panel
from ..strategies.base import Strategy
from .backtest import Backtest, EngineConfig

_SHARED: dict = {}


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _run_one(args):
    params, windows = args
    panel, strategy, config = _SHARED["panel"], _SHARED["strategy"], _SHARED["config"]
    bt = Backtest(panel, strategy.with_params(**params), config)
    out = []
    for start, end in windows:
        try:
            s = bt.run(start, end).stats
        except ValueError:
            s = {}
        out.append({k: v for k, v in s.items() if not isinstance(v, dict)})
    return params, out


def _map(jobs, workers: int | None):
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1 or len(jobs) == 1 or "fork" not in mp.get_all_start_methods():
        return [_run_one(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_run_one, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


def sweep(panel: Panel, strategy: Strategy, grid: dict[str, list], config: EngineConfig | None = None,
          start=None, end=None, workers: int | None = None, sort_by: str = "sharpe") -> pd.DataFrame:
    """Backtest every combination in `grid`; returns one row per combination."""
    _SHARED.update(panel=panel, strategy=strategy, config=config or EngineConfig())
    try:
        res = _map([(p, [(start, end)]) for p in param_grid(grid)], workers)
    finally:
        _SHARED.clear()
    df = pd.DataFrame([{**p, **s[0]} for p, s in res])
    return df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True) if sort_by in df else df


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame  # one row per fold: chosen params, in-sample & out-of-sample metrics
    oos_equity: pd.Series  # stitched out-of-sample equity curve

    def summary(self) -> str:
        from .metrics import equity_stats

        s = equity_stats(self.oos_equity)
        return (f"Walk-forward: {len(self.folds)} folds, OOS {s.get('start')} -> {s.get('end')}\n"
                f"OOS CAGR {s.get('cagr', np.nan):.2%}  Sharpe {s.get('sharpe', np.nan):.2f}  "
                f"MaxDD {s.get('max_drawdown', np.nan):.2%}\n\n"
                + self.folds.to_string(index=False))


def walk_forward(panel: Panel, strategy: Strategy, grid: dict[str, list], config: EngineConfig | None = None,
                 train_years: float = 3, test_years: float = 1, metric: str = "sharpe",
                 min_trades: int = 30, workers: int | None = None) -> WalkForwardResult:
    """Rolling walk-forward: optimise on `train_years`, trade the best params on
    the following `test_years`, roll forward by `test_years`, repeat."""
    config = config or EngineConfig()
    d0, d_end = panel.dates[0], panel.dates[-1]
    windows = []
    train_start = d0 + pd.DateOffset(days=365)  # leave a year for indicator warm-up
    while True:
        train_end = train_start + pd.DateOffset(days=int(365 * train_years))
        test_end = train_end + pd.DateOffset(days=int(365 * test_years))
        if train_end >= d_end:
            break
        windows.append((train_start, train_end, train_end + pd.Timedelta(days=1), min(test_end, d_end)))
        train_start += pd.DateOffset(days=int(365 * test_years))
    if not windows:
        raise ValueError("not enough history for a single walk-forward fold")

    combos = param_grid(grid)
    _SHARED.update(panel=panel, strategy=strategy, config=config)
    try:
        # every combo is evaluated on every train window in a single parallel pass
        res = _map([(p, [(w[0], w[1]) for w in windows]) for p in combos], workers)
    finally:
        _SHARED.clear()

    rows, pieces = [], []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        cands = [(p, s[k]) for p, s in res if s[k].get("trades", 0) >= min_trades and np.isfinite(s[k].get(metric, np.nan))]
        if not cands:
            cands = [(p, s[k]) for p, s in res]
        best_p, best_s = max(cands, key=lambda x: x[1].get(metric, -np.inf) if np.isfinite(x[1].get(metric, np.nan)) else -np.inf)
        oos = Backtest(panel, strategy.with_params(**best_p), config).run(te_s, te_e)
        o = oos.stats
        rows.append({"train": f"{tr_s.date()}..{tr_e.date()}", "test": f"{te_s.date()}..{te_e.date()}", **best_p,
                     f"is_{metric}": best_s.get(metric), "oos_cagr": o.get("cagr"), "oos_sharpe": o.get("sharpe"),
                     "oos_max_dd": o.get("max_drawdown"), "oos_trades": o.get("trades")})
        pieces.append(oos.equity.pct_change().fillna(0.0))
    rets = pd.concat(pieces)
    rets = rets[~rets.index.duplicated()]
    return WalkForwardResult(pd.DataFrame(rows), config.initial_cash * (1 + rets).cumprod())
