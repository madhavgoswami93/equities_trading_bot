"""Parameter sweeps and walk-forward analysis for one strategy of the file.

Sweeps run in parallel processes. On Linux/macOS the panel is shared with the
workers through fork (copy-on-write), so a big panel is not pickled per job.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import RiskConfig, StrategySpec
from ..panel import Panel
from .backtest import Backtest

_SHARED: dict = {}


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _run_one(args):
    params, windows = args
    panel, spec, risk = _SHARED["panel"], _SHARED["spec"], _SHARED["risk"]
    bt = Backtest(panel, [spec.with_params(**params)], risk)
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


def sweep(panel: Panel, spec, grid: dict[str, list], risk: RiskConfig | None = None,
          start=None, end=None, workers: int | None = None, sort_by: str = "sharpe") -> pd.DataFrame:
    """Backtest every combination in `grid`; one row per combination."""
    _SHARED.update(panel=panel, spec=StrategySpec.wrap(spec), risk=risk or RiskConfig())
    try:
        res = _map([(p, [(start, end)]) for p in param_grid(grid)], workers)
    finally:
        _SHARED.clear()
    df = pd.DataFrame([{**p, **s[0]} for p, s in res])
    return df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True) if sort_by in df else df


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos_equity: pd.Series

    def summary(self) -> str:
        from .metrics import equity_stats

        s = equity_stats(self.oos_equity)
        return (f"Walk-forward: {len(self.folds)} folds, OOS {s.get('start')} -> {s.get('end')}\n"
                f"OOS CAGR {s.get('cagr', np.nan):.2%}  Sharpe {s.get('sharpe', np.nan):.2f}  "
                f"MaxDD {s.get('max_drawdown', np.nan):.2%}\n\n" + self.folds.to_string(index=False))


def walk_forward(panel: Panel, spec, grid: dict[str, list], risk: RiskConfig | None = None,
                 train_years: float = 3, test_years: float = 1, metric: str = "sharpe",
                 min_trades: int = 30, workers: int | None = None) -> WalkForwardResult:
    """Optimise on `train_years`, trade the best params on the next `test_years`, roll, repeat."""
    risk = risk or RiskConfig()
    spec = StrategySpec.wrap(spec)
    d_end = panel.dates[-1]
    windows = []
    train_start = panel.dates[0] + pd.DateOffset(days=365)  # a year of indicator warm-up
    while True:
        train_end = train_start + pd.DateOffset(days=int(365 * train_years))
        if train_end >= d_end:
            break
        test_end = train_end + pd.DateOffset(days=int(365 * test_years))
        windows.append((train_start, train_end, train_end + pd.Timedelta(days=1), min(test_end, d_end)))
        train_start += pd.DateOffset(days=int(365 * test_years))
    if not windows:
        raise ValueError("not enough history for a single walk-forward fold")

    _SHARED.update(panel=panel, spec=spec, risk=risk)
    try:
        res = _map([(p, [(w[0], w[1]) for w in windows]) for p in param_grid(grid)], workers)
    finally:
        _SHARED.clear()

    def score(s):
        v = s.get(metric, np.nan)
        return v if np.isfinite(v) else -np.inf

    rows, pieces = [], []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        cands = [(p, s[k]) for p, s in res if s[k].get("trades", 0) >= min_trades] or [(p, s[k]) for p, s in res]
        best_p, best_s = max(cands, key=lambda x: score(x[1]))
        o = Backtest(panel, [spec.with_params(**best_p)], risk).run(te_s, te_e)
        rows.append({"train": f"{tr_s.date()}..{tr_e.date()}", "test": f"{te_s.date()}..{te_e.date()}", **best_p,
                     f"is_{metric}": best_s.get(metric), "oos_cagr": o.stats.get("cagr"),
                     "oos_sharpe": o.stats.get("sharpe"), "oos_max_dd": o.stats.get("max_drawdown"),
                     "oos_trades": o.stats.get("trades")})
        pieces.append(o.equity.pct_change().fillna(0.0))
    rets = pd.concat(pieces)
    rets = rets[~rets.index.duplicated()]
    return WalkForwardResult(pd.DataFrame(rows), risk.capital * (1 + rets).cumprod())
