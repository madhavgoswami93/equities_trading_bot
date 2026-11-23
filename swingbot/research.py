"""Strategy research: which strategies work, and on which kinds of stocks.

Two views of every strategy in the file:

1. Signal study -- every signal is taken as its own trade with an equal Rs 1 lakh, no
   slot or cash limits (liquidity cap and entry-gap rule still apply). This measures
   the raw edge per trade without portfolio effects, and gives enough trades to split
   by stock characteristics.
2. Portfolio backtest -- the same strategy under the file's real risk rules
   (max positions, 40% cap, ...), for the equity curve you would actually have had.

Every signal-study trade is tagged with what the stock looked like on the signal day,
so results can be stripped by any of these factors:

    price        close on the signal day (Rs)
    volume       20-day average shares traded
    turnover     20-day average traded value (Rs crore/day)
    cap          large / mid / small / micro / other (NSE index membership)
    industry     NSE industry (top 750 stocks; others "Unclassified")
    volatility   ATR(14) as % of price
    momentum     63-day return before the signal
    trend        close vs its 50-day average (% above/below)
    market       Nifty 50 above or below its 50-day average
    weekday      day the entry filled
    month        month the entry filled
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import Quantity, RiskConfig, StrategySpec
from .engine.backtest import Backtest
from .panel import Panel

PRICE_BINS = [0, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
TURNOVER_BINS = [0, 0.5, 1, 5, 25, 100, 500, np.inf]  # Rs crore per day
VOLUME_BINS = [0, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, np.inf]  # shares per day
CAP_ORDER = ["large", "mid", "small", "micro", "other"]

FACTORS = {
    "price": "Price (Rs)",
    "volume": "Avg daily volume (shares)",
    "turnover": "Avg daily turnover (Rs cr)",
    "cap": "Market-cap class",
    "industry": "Industry",
    "volatility": "Volatility (ATR % of price)",
    "momentum": "3-month return before signal",
    "trend": "Distance from 50-day average",
    "market": "Nifty 50 regime",
    "weekday": "Entry weekday",
    "month": "Entry month",
}


def feature_frames(p: Panel) -> dict[str, np.ndarray]:
    c, h, l, v = p.frame("close"), p.frame("high"), p.frame("low"), p.frame("volume")
    avg_vol = ind.sma(v, 20)
    out = {
        "price": c.to_numpy(),
        "volume": avg_vol.to_numpy(),
        "turnover": (ind.sma(c * v, 20) / 1e7).to_numpy(),
        "volatility": (ind.atr(h, l, c, 14) / c * 100).to_numpy(),
        "momentum": (ind.roc(c, 63) * 100).to_numpy(),
        "trend": ((c / ind.sma(c, 50) - 1) * 100).to_numpy(),
    }
    if p.benchmark is not None:
        b = pd.Series(p.benchmark)
        out["market"] = np.where(b > b.rolling(50, min_periods=50).mean(), "Nifty above 50-DMA",
                                 np.where(b.notna(), "Nifty below 50-DMA", "n/a"))
    return out


def tag_trades(trades: pd.DataFrame, p: Panel, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach signal-day features (the close before the entry fill) to each trade."""
    if trades.empty:
        return trades
    t = trades.copy()
    feats = feature_frames(p)
    col = {s: j for j, s in enumerate(p.symbols)}
    j = t["symbol"].map(col).to_numpy()
    i = np.maximum(p.dates.get_indexer(pd.DatetimeIndex(t["entry_date"])) - 1, 0)
    for k, arr in feats.items():
        t[k] = arr[i] if arr.ndim == 1 else arr[i, j]
    if meta is not None:
        for k in ("cap", "industry"):
            if k in meta:
                t[k] = t["symbol"].map(meta[k]).fillna("other" if k == "cap" else "Unclassified")
    t["weekday"] = pd.DatetimeIndex(t["entry_date"]).day_name().str[:3]
    t["month"] = pd.DatetimeIndex(t["entry_date"]).strftime("%Y-%m")
    return t


def bucketize(t: pd.DataFrame, factor: str, quantiles: bool = False) -> pd.Series:
    x = t[factor]
    if factor == "price" and not quantiles:
        return pd.cut(x, PRICE_BINS, labels=_labels(PRICE_BINS, "₹"), right=False)
    if factor == "turnover" and not quantiles:
        return pd.cut(x, TURNOVER_BINS, labels=_labels(TURNOVER_BINS, "₹", " cr"), right=False)
    if factor == "volume" and not quantiles:
        return pd.cut(x, VOLUME_BINS, labels=_labels(VOLUME_BINS, ""), right=False)
    if factor == "cap":
        return pd.Categorical(x, categories=[c for c in CAP_ORDER if c in set(x)], ordered=True)
    if pd.api.types.is_numeric_dtype(x):
        try:
            return pd.qcut(x, 5, labels=[f"Q{k} ({lo:.1f} to {hi:.1f})" for k, (lo, hi) in
                                         enumerate(_qedges(x), 1)], duplicates="drop")
        except ValueError:
            return pd.qcut(x.rank(method="first"), 5, labels=[f"Q{k}" for k in range(1, 6)])
    return x.astype(str)


def _qedges(x: pd.Series):
    e = x.quantile(np.linspace(0, 1, 6)).to_numpy()
    return list(zip(e[:-1], e[1:]))


def _labels(bins, pre, post=""):
    def f(v):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3 and not pre:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        out.append(f"{pre}{f(lo)}+{post}" if np.isinf(hi) else f"{pre}{f(lo)}-{f(hi)}{post}")
    return out


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t["return_pct"]
    wins, losses = t.loc[t.pnl > 0, "pnl"].sum(), -t.loc[t.pnl <= 0, "pnl"].sum()
    return {
        "trades": len(t),
        "win_rate": float((t.pnl > 0).mean()),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "expectancy_r": float(t["r_multiple"].replace([np.inf, -np.inf], np.nan).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "avg_hold": float(t["bars_held"].mean()),
        "pnl_per_lakh": float(t["pnl"].sum() / len(t)),
    }


def bucket_table(t: pd.DataFrame, factor: str, by_strategy: bool = False, quantiles: bool = False) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    t = t.assign(bucket=bucketize(t, factor, quantiles))
    keys = ["strategy", "bucket"] if by_strategy else ["bucket"]
    rows = [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **summarize(g)}
            for k, g in t.groupby(keys, observed=True, sort=True)]
    return pd.DataFrame(rows)


def signal_study(panel: Panel, specs: list[StrategySpec], risk: RiskConfig, start=None, end=None,
                 meta: pd.DataFrame | None = None, per_trade: float = 100_000.0) -> pd.DataFrame:
    """Every signal as an independent equal-size trade. Returns tagged trades for all strategies."""
    study_risk = replace(risk, capital=1e13, max_positions=100_000, max_position_pct=1.0,
                         max_drawdown_halt=0.0, risk_per_trade=risk.risk_per_trade)
    out = []
    for spec in specs:
        s = replace(spec, quantity=Quantity.parse(f"amount {per_trade:g}"), max_positions=0)
        res = Backtest(panel, [s], study_risk).run(start, end)
        out.append(tag_trades(res.trades, panel, meta))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def leaderboard(panel: Panel, specs: list[StrategySpec], risk: RiskConfig, study: pd.DataFrame,
                start=None, end=None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        st = summarize(study[study.strategy == spec.name]) if len(study) else {"trades": 0}
        res = Backtest(panel, [spec], risk).run(start, end).stats
        rows.append({"strategy": spec.name, **{f"signal_{k}": v for k, v in st.items()},
                     "portfolio_cagr": res.get("cagr"), "portfolio_return": res.get("total_return"),
                     "portfolio_sharpe": res.get("sharpe"), "portfolio_max_dd": res.get("max_drawdown"),
                     "portfolio_trades": res.get("trades"), "exposure": res.get("exposure")})
    return pd.DataFrame(rows).sort_values("signal_expectancy_r", ascending=False, na_position="last")
