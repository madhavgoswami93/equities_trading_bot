"""Runner: one entry point, three modes, one engine.

    backtest  history -> Result                        (fresh state, full range)
    paper     new bars since last run -> updated book  (same kernel, resumed state)
    live      broker fills -> book -> decision -> orders sent to the broker

Paper trading *is* the backtest kernel advanced one day at a time on live
data, so a strategy behaves in paper exactly as it did in the backtest (this
is enforced by tests/test_runner.py). Live differs only in that real broker
fills replace the simulated ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import data as D
from .brokers.base import Order, plan_orders
from .config import RunConfig
from .engine import kernel as K
from .engine.backtest import Backtest, Engine
from .engine.book import Book, append_csv
from .engine.result import Result
from .panel import Panel

log = logging.getLogger(__name__)


@dataclass
class DailyReport:
    mode: str
    date: str
    equity: float
    cash: float
    closed: pd.DataFrame
    book: Book
    orders: list[Order] = field(default_factory=list)
    sent: bool = False
    note: str = ""

    def __str__(self):
        lines = [f"[{self.mode}] {self.date}   equity {self.equity:,.0f}   cash {self.cash:,.0f}   "
                 f"open positions {len(self.book.positions)}"]
        if self.note:
            lines.append(self.note)
        if not self.closed.empty:
            cols = [c for c in ("strategy", "symbol", "entry_date", "exit_date", "qty", "entry_price", "exit_price",
                                "exit_reason", "pnl") if c in self.closed]
            lines += ["", "Closed:", self.closed[cols].round(2).to_string(index=False)]
        if self.book.positions:
            df = pd.DataFrame(self.book.positions)
            df["unrealised"] = (df["last_close"] - df["entry_price"]) * df["qty"]
            df["weight"] = df["qty"] * df["last_close"] / self.equity
            lines += ["", "Positions:", df[["strategy", "symbol", "qty", "entry_date", "entry_price", "last_close",
                                            "stop", "bars", "weight", "unrealised"]].round(3).to_string(index=False)]
        if self.orders:
            hdr = "Orders SENT for next session:" if self.sent else "Orders for next session (not sent):"
            lines += ["", hdr, *(f"  {o}" for o in self.orders)]
        return "\n".join(lines)


class Runner:
    def __init__(self, cfg: RunConfig, panel: Panel | None = None):
        self.cfg = cfg
        self._panel = panel

    # ---------------------------------------------------------------- data
    def update_data(self) -> None:
        dc = self.cfg.data
        if dc.provider == "synthetic":
            return
        D.load_env()
        provider = D.get_provider(dc.provider)
        D.fetch_universe(provider, D.load_universe(dc.universe), pd.Timestamp(dc.start).date(), date.today(),
                         root=dc.cache_dir)
        bp = dc.benchmark_provider
        if bp and bp != dc.provider and D.benchmark_for(bp):
            D.fetch_universe(D.get_provider(bp), [], pd.Timestamp(dc.start).date(), date.today(), root=dc.cache_dir)

    def panel(self, update: bool = False) -> Panel:
        if self._panel is not None and not update:
            return self._panel
        dc = self.cfg.data
        if dc.provider == "synthetic":
            self._panel = D.synthetic_panel(60, 2520)
            return self._panel
        if update:
            self.update_data()
        bprov = dc.benchmark_provider or dc.provider
        self._panel = D.load_panel(dc.provider, D.load_universe(dc.universe), start=dc.start,
                                   benchmark=D.benchmark_for(bprov), benchmark_provider=bprov, root=dc.cache_dir)
        q = self._panel.quality_report()
        if len(q):
            log.warning("data quality issues (likely un-adjusted corporate actions):\n%s", q.to_string(index=False))
        return self._panel

    # ------------------------------------------------------------ backtest
    def backtest(self, start=None, end=None, only: str | None = None) -> Result:
        specs = [self.cfg.strategy(only)] if only else self.cfg.active
        return Backtest(self.panel(), specs, self.cfg.risk).run(start, end)

    # --------------------------------------------------------------- paper
    def paper(self, update: bool = True, start=None, reset: bool = False) -> DailyReport:
        """Advance the paper book over every bar it has not seen yet.

        First run: starts at `start` (default: the latest bar, i.e. the first
        decision is made on the most recent close)."""
        d = self.cfg.state_dir("paper")
        bp = d / "book.json"
        if reset and d.exists():
            for f in ("book.json", "trades.csv", "equity.csv"):
                (d / f).unlink(missing_ok=True)
        panel = self.panel(update)
        eng = Engine(panel, self.cfg.active, self.cfg.risk)
        T = panel.shape[0]
        if bp.exists():
            book = Book.load(bp)
            t0 = panel.index_of(book.last_date, side="right")
        else:
            book = Book.new(self.cfg.risk.capital)
            t0 = panel.index_of(start) if start else T - 1
        if t0 >= T:
            return self._report("paper", book, pd.DataFrame(), note="already up to date -- no new bars")
        pos, acct = book.to_arrays(panel, eng.strategy_names)
        tr, eq, cash, _, npos = eng.advance(pos, acct, t0, T, entries_on_last=True)
        new_book = Book.from_arrays(pos, acct, panel, eng.strategy_names, panel.dates[-1])
        closed = eng.trades_frame(tr)
        new_book.save(bp)
        append_csv(d / "trades.csv", closed)
        append_csv(d / "equity.csv", pd.DataFrame({"date": panel.dates[t0:T].date, "equity": eq, "cash": cash,
                                                   "positions": npos}))
        return self._report("paper", new_book, closed, note=f"processed {T - t0} new session(s)")

    # ---------------------------------------------------------------- live
    def live(self, phase: str = "close", update: bool = True, send: bool = False, broker=None,
             force: bool = False) -> DailyReport:
        """phase 'open'  : sync fills and place stops on new positions (after ~09:20)
        phase 'close' : sync, decide on today's close, send tomorrow's orders (after ~15:45)"""
        from .brokers.kite import get_broker

        D.load_env()
        d = self.cfg.state_dir("live")
        bp = d / "book.json"
        book = Book.load(bp) if bp.exists() else Book.new(self.cfg.risk.capital)
        panel = self.panel(update and phase == "close")
        eng = Engine(panel, self.cfg.active, self.cfg.risk)
        today = panel.dates[-1]
        if phase == "close" and book.last_date == str(today.date()) and not force:
            return self._report("live", book, pd.DataFrame(),
                                note=f"{today.date()} already processed (use --force to redo)")
        broker = broker or get_broker(self.cfg.live.broker, self.cfg.live)
        closed = pd.DataFrame(broker.sync(book, eng, pd.Timestamp.now().normalize() if phase == "open" else today,
                                          final=phase == "close"))
        append_csv(d / "trades.csv", closed)

        if phase == "close":
            pos, acct = book.to_arrays(panel, eng.strategy_names)
            eng.decide(pos, acct, panel.shape[0] - 1)
            book = Book.from_arrays(pos, acct, panel, eng.strategy_names, today, meta=book.meta)
            orders = plan_orders(book)
            append_csv(d / "equity.csv", pd.DataFrame({"date": [today.date()], "equity": [acct[K.EQUITY]],
                                                       "cash": [acct[K.CASH]], "positions": [len(book.positions)]}))
        else:
            orders = [o for o in plan_orders(book) if o.action == "STOP"]

        really = send and not self.cfg.live.dry_run
        if really:
            last = {p["symbol"]: p["last_close"] for p in book.positions}
            broker.execute(book, orders, last)
        book.save(bp)
        note = "" if really else "DRY RUN -- nothing sent (needs --send and `dry_run = false` in [live])"
        return self._report("live", book, closed, orders=orders, sent=really, note=note)

    def status(self, mode: str = "paper") -> DailyReport:
        bp = self.cfg.state_dir(mode) / "book.json"
        if not bp.exists():
            raise FileNotFoundError(f"no {mode} book yet at {bp}")
        return self._report(mode, Book.load(bp), pd.DataFrame())

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _report(mode, book: Book, closed, orders=None, sent=False, note="") -> DailyReport:
        return DailyReport(mode, book.last_date or "-", book.equity(), book.cash, closed, book,
                           orders if orders is not None else plan_orders(book), sent, note)
