"""Synthetic market generator for tests and benchmarks (no network, deterministic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..panel import Panel


def synthetic_panel(n_symbols: int = 50, n_days: int = 2520, seed: int = 7, start: str = "2010-01-01") -> Panel:
    """Correlated GBM stocks with a shared regime-switching market factor,
    per-stock drift, intraday ranges and volumes. Not realistic enough to judge
    a strategy -- only to exercise the engine."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    regime = np.where(np.sin(np.arange(n_days) / 180.0) > -0.3, 0.0006, -0.0008)
    mkt = regime + 0.009 * rng.standard_normal(n_days)
    beta = rng.uniform(0.6, 1.4, n_symbols)
    drift = rng.normal(0.0002, 0.0003, n_symbols)
    idio = rng.uniform(0.010, 0.022, n_symbols) * rng.standard_normal((n_days, n_symbols))
    rets = mkt[:, None] * beta + drift + idio
    close = 100 * np.exp(np.cumsum(rets, axis=0)) * rng.uniform(0.5, 20, n_symbols)
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(0.004 * rng.standard_normal((n_days, n_symbols)))
    rng_hl = np.abs(0.012 * rng.standard_normal((n_days, n_symbols)))
    high = np.maximum(open_, close) * (1 + rng_hl)
    low = np.minimum(open_, close) * (1 - rng_hl)
    volume = rng.lognormal(13, 0.6, (n_days, n_symbols)) * (1 + 3 * np.abs(rets) / 0.02)
    # stagger listings so the panel has leading NaNs like a real universe
    listing = rng.integers(0, n_days // 4, n_symbols)
    listing[: n_symbols // 2] = 0
    for j, k in enumerate(listing):
        for a in (open_, high, low, close, volume):
            a[:k, j] = np.nan
    bench = 10000 * np.exp(np.cumsum(mkt))
    syms = [f"SYM{j:03d}" for j in range(n_symbols)]
    return Panel(dates, syms, open_, high, low, close, volume, benchmark=bench, benchmark_name="SYNTH_INDEX")
