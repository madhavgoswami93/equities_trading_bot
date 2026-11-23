"""Broker interface + the order plan derived from a book.

After the close-of-day decision the book says what the portfolio *should* do
at the next open. `plan_orders` turns that into concrete orders, identically
for paper (printed) and live (sent):

    BUY   pending entry      -> limit order at close * (1 + max_entry_gap), CNC
    SELL  pending exit       -> market order at the open, CNC; cancel its stop
    STOP  every open position -> a protective stop-loss (GTT on Kite) at the
                                engine's current (possibly trailed) stop price
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..engine.book import Book

TICK = 0.05  # a multiple of every NSE equity tick size


def round_tick(price: float, down: bool = True) -> float:
    import math

    n = price / TICK
    return round((math.floor(n + 1e-9) if down else math.ceil(n - 1e-9)) * TICK, 2)


@dataclass
class Order:
    action: str  # BUY | SELL | STOP
    symbol: str
    qty: int
    price: float | None  # limit for BUY, trigger for STOP, None = market
    strategy: str
    reason: str = ""

    def __str__(self):
        px = "MKT" if self.price is None else f"{self.price:,.2f}"
        return f"{self.action:<4} {self.symbol:<12} {self.qty:>6}  @ {px:>10}  [{self.strategy}] {self.reason}"


def plan_orders(book: Book) -> list[Order]:
    orders = []
    for p in book.positions:
        if p.get("pending_exit"):
            orders.append(Order("SELL", p["symbol"], p["qty"], None, p["strategy"], f"exit: {p['pending_exit']}"))
        else:
            kind = "trailing stop" if p["trailed"] else "stop"
            orders.append(Order("STOP", p["symbol"], p["qty"], round_tick(p["stop"]), p["strategy"], kind))
    for o in book.pending:
        px = round_tick(o["limit"]) if o["limit"] is not None else None
        orders.append(Order("BUY", o["symbol"], o["qty"], px, o["strategy"],
                            f"entry, initial stop {o['stop_dist']:,.2f} below fill"))
    return orders


class Broker(ABC):
    name = "base"
    supports_amo = True  # can queue orders after the close for the next open (Kite: yes, Groww: no)

    @abstractmethod
    def sync(self, book: Book, engine, today, final: bool = True) -> list[dict]:
        """Apply real fills since the last run to the book (in place). `final`
        (the after-close run) also cancels entry orders that never filled.
        Returns closed trades as dicts for the ledger."""

    @abstractmethod
    def execute(self, book: Book, orders: list[Order], last_prices: dict[str, float]) -> None:
        """Send the orders and remember broker ids in book.meta."""


def get_broker(name: str, live_cfg) -> Broker:
    if name == "kite":
        from .kite import KiteBroker

        return KiteBroker(exchange="NSE", tag=live_cfg.tag, stop_limit_buffer=live_cfg.stop_limit_buffer)
    if name == "groww":
        from .groww import GrowwBroker

        return GrowwBroker(stop_limit_buffer=live_cfg.stop_limit_buffer)
    raise NotImplementedError(f"live broker {name!r} is not supported; use 'kite' or 'groww'")
