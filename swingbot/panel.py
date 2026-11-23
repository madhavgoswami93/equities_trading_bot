"""Panel: the whole universe as aligned (T dates x N symbols) float arrays.

Holding the data as dense 2-D arrays is what makes both signal generation
(vectorised pandas/numpy over all symbols at once) and the simulation kernel
(numba loop over contiguous memory) fast. Missing bars (not yet listed,
suspended, delisted) are NaN.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    symbols: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    benchmark: np.ndarray | None = None  # benchmark close, shape (T,)
    benchmark_name: str | None = None
    _frames: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        benchmark: pd.Series | None = None,
        benchmark_name: str | None = None,
        start=None,
        end=None,
    ) -> Panel:
        """Build from {symbol: DataFrame[open, high, low, close, volume]} indexed by date."""
        if not frames:
            raise ValueError("no data frames given")
        symbols = sorted(frames)
        idx = None
        for s in symbols:
            i = pd.DatetimeIndex(frames[s].index).normalize()
            idx = i if idx is None else idx.union(i)
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        arrays = {f: np.full((len(idx), len(symbols)), np.nan) for f in FIELDS}
        for j, s in enumerate(symbols):
            df = frames[s].copy()
            df.index = pd.DatetimeIndex(df.index).normalize()
            df = df[~df.index.duplicated(keep="last")].reindex(idx)
            for f in FIELDS:
                arrays[f][:, j] = df[f].to_numpy(dtype=np.float64)
        bench = None
        if benchmark is not None:
            b = benchmark.copy()
            b.index = pd.DatetimeIndex(b.index).normalize()
            bench = b[~b.index.duplicated(keep="last")].reindex(idx).ffill().to_numpy(np.float64)
        return cls(idx, symbols, benchmark=bench, benchmark_name=benchmark_name, **arrays)

    # ------------------------------------------------------------- accessors
    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    def frame(self, name: str) -> pd.DataFrame:
        """A field as a (dates x symbols) DataFrame -- convenient for indicators."""
        if name not in self._frames:
            self._frames[name] = pd.DataFrame(getattr(self, name), index=self.dates, columns=self.symbols, copy=False)
        return self._frames[name]

    def index_of(self, date, side: str = "left") -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date), side=side))

    def head(self, n: int) -> Panel:
        """The first n sessions -- exactly what a live run on day n-1 would have seen."""
        return Panel(
            self.dates[:n], list(self.symbols),
            *(getattr(self, f)[:n] for f in FIELDS),
            benchmark=None if self.benchmark is None else self.benchmark[:n], benchmark_name=self.benchmark_name,
        )

    def select(self, symbols: list[str]) -> Panel:
        cols = [self.symbols.index(s) for s in symbols]
        return Panel(
            self.dates, list(symbols),
            *(getattr(self, f)[:, cols] for f in FIELDS),
            benchmark=self.benchmark, benchmark_name=self.benchmark_name,
        )

    # --------------------------------------------------------------- quality
    def quality_report(self, gap_threshold: float = 0.35) -> pd.DataFrame:
        """Flag suspicious data: non-positive prices, high<low, and huge overnight
        gaps (usually an un-adjusted split/bonus -- these silently wreck swing
        backtests because stops fire on fake crashes)."""
        c = self.frame("close")
        prev = c.shift(1)
        gap = (self.frame("open") / prev - 1).abs()
        rows = []
        for j, s in enumerate(self.symbols):
            bad_px = int(np.sum(self.close[:, j] <= 0))
            bad_hl = int(np.sum(self.high[:, j] < self.low[:, j]))
            big = gap.iloc[:, j][gap.iloc[:, j] > gap_threshold]
            if bad_px or bad_hl or len(big):
                rows.append(dict(symbol=s, nonpositive=bad_px, high_lt_low=bad_hl,
                                 big_gaps=len(big), gap_dates=[d.date() for d in big.index[:5]]))
        return pd.DataFrame(rows)
