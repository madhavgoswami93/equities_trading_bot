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
from pathlib import Path
from dataclasses import dataclass, field

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
    def db(self) -> D.MarketDB:
        return D.MarketDB(self.cfg.data.db)

    def update_data(self, full: bool = False) -> list[str]:
        """Download new sessions into the local database. Returns symbols that failed."""
        dc = self.cfg.data
        if dc.provider == "synthetic":
            return []
        D.load_env()
        from .market_clock import last_complete_session

        update = D.get_provider(dc.provider)
        history = D.get_provider(dc.history_provider) if dc.history_provider and dc.history_provider != dc.provider \
            else update
        # stop at the last finished session: a candle downloaded mid-day is incomplete
        return D.sync_universe(self.db(), D.load_universe(dc.universe), update, history,
                               pd.Timestamp(dc.start).date(), last_complete_session(), full=full)

    def panel(self, update: bool = False) -> Panel:
        if self._panel is not None and not update:
            return self._panel
        dc = self.cfg.data
        if dc.provider == "synthetic":
            self._panel = D.synthetic_panel(60, 2520)
            return self._panel
        if update:
            self.update_data()
        self._panel = D.load_panel(D.load_universe(dc.universe), start=dc.start, db=self.db())
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
    def halt_file(self) -> Path:
        return self.cfg.state_dir("live") / "HALT"

    def live(self, phase: str = "close", update: bool = True, send: bool = False, broker=None,
             force: bool = False, check_clock: bool = True) -> DailyReport:
        """phase 'close' (after ~15:45 IST): sync fills, decide on today's close, send what the
                         broker accepts after hours (Kite: everything as AMO; Groww: stop updates)
        phase 'open'  (after ~09:16 IST): sync fills, send what still has to go out
                         (Kite: stops on new positions; Groww: queued buys/sells + stops)

        Safety: refuses to decide on a close that is not today's finished session, refuses to
        process the same day twice, and a HALT file blocks new entries (exits and stops still run)."""
        from .brokers.base import get_broker
        from .market_clock import now_ist, last_complete_session

        D.load_env()
        d = self.cfg.state_dir("live")
        bp = d / "book.json"
        book = Book.load(bp) if bp.exists() else Book.new(self.cfg.risk.capital)
        panel = self.panel(update and phase == "close")
        eng = Engine(panel, self.cfg.active, self.cfg.risk)
        today = panel.dates[-1]
        if phase == "close" and not force:
            if book.last_date == str(today.date()):
                return self._report("live", book, pd.DataFrame(),
                                    note=f"{today.date()} already processed (use --force to redo)")
            if check_clock and today.date() != last_complete_session():
                return self._report("live", book, pd.DataFrame(), note=(
                    f"latest bar is {today.date()}, but the latest finished session is {last_complete_session()} "
                    f"(now {now_ist():%Y-%m-%d %H:%M} IST). Not deciding on stale or partial data -- run after "
                    "15:35 IST on a trading day, check `swingbot db FILE status`, or pass --force."))
        broker = broker or get_broker(self.cfg.live.broker, self.cfg.live)
        sync_day = pd.Timestamp(now_ist().date()) if phase == "open" else today
        closed = pd.DataFrame(broker.sync(book, eng, sync_day, final=phase == "close"))
        append_csv(d / "trades.csv", closed)

        if phase == "close":
            pos, acct = book.to_arrays(panel, eng.strategy_names)
            eng.decide(pos, acct, panel.shape[0] - 1)
            book = Book.from_arrays(pos, acct, panel, eng.strategy_names, today, meta=book.meta)
            append_csv(d / "equity.csv", pd.DataFrame({"date": [today.date()], "equity": [acct[K.EQUITY]],
                                                       "cash": [acct[K.CASH]], "positions": [len(book.positions)]}))
        planned = plan_orders(book)
        notes = []
        if broker.supports_amo:
            to_send = planned if phase == "close" else [o for o in planned if o.action == "STOP"]
        else:
            to_send = [o for o in planned if o.action == "STOP"] if phase == "close" else planned
            if phase == "close" and any(o.action != "STOP" for o in planned):
                notes.append(f"{broker.name} has no after-market orders: BUY/SELL go out with "
                             "`live --phase open --send` after 09:15 IST")
        if self.halt_file().exists():
            blocked = [o for o in to_send if o.action == "BUY"]
            to_send = [o for o in to_send if o.action != "BUY"]
            notes.append(f"HALTED ({self.halt_file()}): {len(blocked)} new entries blocked; exits and stops still run")

        really = send and not self.cfg.live.dry_run
        if really and to_send:
            last = {p["symbol"]: p["last_close"] for p in book.positions}
            broker.execute(book, to_send, last)
        book.save(bp)
        book.save(d / "history" / f"book-{book.last_date}-{phase}.json")  # daily snapshots for audits / recovery
        if not really:
            notes.append("DRY RUN -- nothing sent (needs --send and `dry_run = false` in [live])")
        return self._report("live", book, closed, orders=to_send, sent=really, note="\n".join(notes))

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
