"""Book: the portfolio state on disk (paper and live), keyed by symbol.

The kernel works on dense arrays indexed by panel column; the book stores the
same information keyed by symbol and date so it survives universe changes and
restarts. `to_arrays` / `from_arrays` convert losslessly between the two.

    runs/<name>/<mode>/book.json     current state (positions, pending orders, cash)
    runs/<name>/<mode>/trades.csv    closed-trade ledger (appended)
    runs/<name>/<mode>/equity.csv    daily equity (appended)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..panel import Panel
from . import kernel as K

_EXIT = {n: i for i, n in enumerate(K.EXIT_NAMES)}


@dataclass
class Book:
    cash: float
    peak: float
    last_date: str | None = None  # last session fully processed
    halt_left: int = 0  # drawdown-halt sessions remaining
    positions: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)  # entries for the next open
    meta: dict = field(default_factory=dict)  # broker ids etc. (live), keyed by symbol

    # --------------------------------------------------------------- disk
    @classmethod
    def new(cls, capital: float) -> Book:
        return cls(cash=capital, peak=capital)

    @classmethod
    def load(cls, path: Path) -> Book:
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=float))
        tmp.replace(path)  # atomic: a crash never leaves a half-written book

    # ------------------------------------------------------------ arrays
    def to_arrays(self, panel: Panel, strategies: list[str]):
        pos, acct = K.new_state(panel.shape[1], self.cash)
        acct[K.PEAK] = self.peak
        acct[K.HALT_LEFT] = self.halt_left
        col = {s: j for j, s in enumerate(panel.symbols)}
        sidx = {s: i for i, s in enumerate(strategies)}
        for p in self.positions:
            j = _col(col, p["symbol"])
            pos[j, K.QTY] = p["qty"]
            pos[j, K.ENTRY_PX] = p["entry_price"]
            pos[j, K.STOP] = p["stop"]
            pos[j, K.TARGET] = p["target"] if p["target"] is not None else np.inf
            pos[j, K.RISK_PS] = p["risk_per_share"]
            pos[j, K.ENTRY_COST] = p["entry_cost"]
            pos[j, K.ENTRY_T] = min(panel.index_of(p["entry_date"]), panel.shape[0] - 1)
            pos[j, K.BARS] = p["bars"]
            pos[j, K.HI_CLOSE] = p["high_close"]
            pos[j, K.TRAILED] = float(p["trailed"])
            pos[j, K.STRAT] = _strat(sidx, p["strategy"])
            pos[j, K.PEXIT] = _EXIT[p["pending_exit"]] if p.get("pending_exit") else -1
            pos[j, K.LAST_CLOSE] = p["last_close"]
        for o in self.pending:
            j = _col(col, o["symbol"])
            pos[j, K.PRANK] = o["rank"]
            pos[j, K.PQTY] = o["qty"]
            pos[j, K.PLIMIT] = o["limit"] if o["limit"] is not None else np.inf
            pos[j, K.PSD] = o["stop_dist"]
            pos[j, K.PSTRAT] = _strat(sidx, o["strategy"])
        return pos, acct

    @classmethod
    def from_arrays(cls, pos, acct, panel: Panel, strategies: list[str], last_date, meta=None) -> Book:
        positions, pending = [], []
        for j, sym in enumerate(panel.symbols):
            r = pos[j]
            if r[K.QTY] > 0:
                positions.append({
                    "symbol": sym,
                    "strategy": strategies[int(r[K.STRAT])],
                    "qty": int(r[K.QTY]),
                    "entry_price": float(r[K.ENTRY_PX]),
                    "entry_date": str(panel.dates[int(r[K.ENTRY_T])].date()),
                    "stop": float(r[K.STOP]),
                    "target": float(r[K.TARGET]) if np.isfinite(r[K.TARGET]) else None,
                    "risk_per_share": float(r[K.RISK_PS]),
                    "entry_cost": float(r[K.ENTRY_COST]),
                    "bars": int(r[K.BARS]),
                    "high_close": float(r[K.HI_CLOSE]),
                    "trailed": bool(r[K.TRAILED]),
                    "pending_exit": K.EXIT_NAMES[int(r[K.PEXIT])] if r[K.PEXIT] >= 0 else None,
                    "last_close": float(r[K.LAST_CLOSE]),
                })
            if r[K.PRANK] >= 0:
                pending.append({
                    "symbol": sym,
                    "strategy": strategies[int(r[K.PSTRAT])],
                    "rank": int(r[K.PRANK]),
                    "qty": int(r[K.PQTY]),
                    "limit": float(r[K.PLIMIT]) if np.isfinite(r[K.PLIMIT]) else None,
                    "stop_dist": float(r[K.PSD]),
                })
        pending.sort(key=lambda o: o["rank"])
        held = {p["symbol"] for p in positions} | {o["symbol"] for o in pending}
        meta = {k: v for k, v in (meta or {}).items() if k in held}
        return cls(cash=float(acct[K.CASH]), peak=float(acct[K.PEAK]), halt_left=int(acct[K.HALT_LEFT]),
                   last_date=str(pd.Timestamp(last_date).date()),
                   positions=positions, pending=pending, meta=meta)

    def equity(self) -> float:
        return self.cash + sum(p["qty"] * p["last_close"] for p in self.positions)


def _col(col: dict, sym: str) -> int:
    if sym not in col:
        raise KeyError(f"{sym} is in the book but missing from the data universe -- add it back or close it manually")
    return col[sym]


def _strat(sidx: dict, name: str) -> int:
    if name not in sidx:
        raise KeyError(f"book refers to strategy {name!r} which is not enabled in the strategy file")
    return sidx[name]


def append_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
