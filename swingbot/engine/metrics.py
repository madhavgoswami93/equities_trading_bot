"""Performance statistics from an equity curve and a trade list."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch (in trading days) spent below a previous equity high."""
    under = (equity < equity.cummax()).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def equity_stats(equity: pd.Series, rf: float = 0.0) -> dict:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return {}
    ret = eq.pct_change().dropna()
    years = len(ret) / TRADING_DAYS
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    ex = ret - rf / TRADING_DAYS
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    down = ex.clip(upper=0)
    dd_dev = np.sqrt((down**2).mean()) * np.sqrt(TRADING_DAYS)
    mdd = drawdown(eq).min()
    return {
        "start": eq.index[0].date(),
        "end": eq.index[-1].date(),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(ex.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else np.nan,
        "sortino": float(ex.mean() * TRADING_DAYS / dd_dev) if dd_dev > 0 else np.nan,
        "max_drawdown": float(mdd),
        "max_dd_days": max_drawdown_duration(eq),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    r = trades["r_multiple"].replace([np.inf, -np.inf], np.nan).dropna()
    gross_loss = -losses.sum()
    return {
        "trades": n,
        "win_rate": float(len(wins) / n),
        "avg_win_pct": float(trades.loc[pnl > 0, "return_pct"].mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(trades.loc[pnl <= 0, "return_pct"].mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else np.inf,
        "expectancy_r": float(r.mean()) if len(r) else np.nan,
        # Van Tharp's System Quality Number on R-multiples (also reported by
        # backtrader's SQN analyzer, on PnL instead of R)
        "sqn": float(np.sqrt(len(r)) * r.mean() / r.std(ddof=1)) if len(r) > 1 and r.std() > 0 else np.nan,
        "avg_hold_days": float(trades["bars_held"].mean()),
        "max_consec_losses": _longest_run(pnl <= 0),
        "total_costs": float(trades["costs"].sum()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def _longest_run(mask: pd.Series) -> int:
    best = run = 0
    for m in mask.to_numpy():
        run = run + 1 if m else 0
        best = max(best, run)
    return best
