"""Vectorised indicators over (dates x symbols) DataFrames.

Every function is causal: the value at row t only uses rows <= t. Inputs and
outputs are DataFrames with identical shape, so one call computes the
indicator for the entire universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._jit import njit


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's smoothing (RMA) as used by ATR and RSI."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    pc = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))
    return tr.where(pc.notna(), high - low)


def atr(high, low, close, n: int = 14) -> pd.DataFrame:
    return wilder(true_range(high, low, close), n)


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder(-d.clip(upper=0), n)
    out = 100 - 100 / (1 + up / dn)
    return out.where(dn != 0, 100.0).where(up.notna())


def roc(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1


def highest(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).max()


def lowest(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).min()


def supertrend(high, low, close, n: int = 10, mult: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (line, direction) where direction is +1 (up-trend) / -1 / NaN."""
    a = atr(high, low, close, n).to_numpy()
    line, direction = _supertrend_kernel(high.to_numpy(), low.to_numpy(), close.to_numpy(), a, mult)
    return (pd.DataFrame(line, index=close.index, columns=close.columns),
            pd.DataFrame(direction, index=close.index, columns=close.columns))


@njit(cache=True)
def _supertrend_kernel(h, l, c, a, mult):  # pragma: no cover - jitted
    T, N = c.shape
    line = np.full((T, N), np.nan)
    direction = np.full((T, N), np.nan)
    for j in range(N):
        fu = np.nan
        fl = np.nan
        pc = np.nan
        d = 1.0
        started = False
        for t in range(T):
            if np.isnan(a[t, j]) or np.isnan(c[t, j]):
                continue
            mid = (h[t, j] + l[t, j]) / 2.0
            bu = mid + mult * a[t, j]
            bl = mid - mult * a[t, j]
            if not started:
                fu, fl, started = bu, bl, True
                pc = c[t, j]
            else:
                fu = bu if (bu < fu or pc > fu) else fu
                fl = bl if (bl > fl or pc < fl) else fl
                if d == 1.0 and c[t, j] < fl:
                    d = -1.0
                elif d == -1.0 and c[t, j] > fu:
                    d = 1.0
                pc = c[t, j]
            direction[t, j] = d
            line[t, j] = fl if d == 1.0 else fu
    return line, direction
