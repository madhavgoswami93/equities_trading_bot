"""Groww live adapter (Groww Trade API, `pip install growwapi`).

One Groww subscription covers both market data and order placement, so Groww
can be your only broker API. Daily candles only go back to ~2020 on Groww, so
keep `history_provider = "yahoo"` for the long backfill of the local database.

Groww documents no after-market orders, so the daily routine differs from Kite:
  ~15:45  swingbot live FILE --send                sync fills, decide on the close, update GTT stops;
                                                  tomorrow's BUY/SELL orders are kept in the book
  ~09:16  swingbot live FILE --phase open --send  send the queued BUY (limit) and SELL (market)
                                                  orders, place GTT stops on positions filled so far
Entries are DAY limit orders at close * (1 + max_entry_gap), the same price the
backtest uses; an entry that has not filled by the close is cancelled.

Protective stops are GTT smart orders (SELL, trigger DOWN, limit just below the
trigger), valid for a year and modified in place as the stop trails up. Every
order carries a deterministic reference id, so re-running a phase after a crash
finds the order already placed instead of sending it twice.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from ..engine.book import Book
from .base import Broker, Order, round_tick

log = logging.getLogger(__name__)
DONE = {"EXECUTED", "COMPLETED", "DELIVERY_AWAITED"}
DEAD = {"REJECTED", "FAILED", "CANCELLED"}
GTT_FIRED = {"TRIGGERED", "COMPLETED"}


def reference_id(day, side: str, symbol: str) -> str:
    """8-20 alphanumeric chars, deterministic per (day, side, symbol)."""
    sym = re.sub(r"[^A-Za-z0-9]", "", symbol).upper()
    return f"sb{pd.Timestamp(day):%y%m%d}{side[0]}{sym}"[:20]


class GrowwBroker(Broker):
    name = "groww"
    supports_amo = False

    def __init__(self, groww=None, stop_limit_buffer: float = 0.01):
        if groww is None:
            from ..data.groww import _client

            groww = _client()
        self.g = groww
        self.buffer = stop_limit_buffer

    # ------------------------------------------------------------- helpers
    def _status(self, oid: str) -> dict:
        g = self.g
        st = g.get_order_status(groww_order_id=oid, segment=g.SEGMENT_CASH) or {}
        if int(st.get("filled_quantity") or 0) > 0 and "average_fill_price" not in st:
            st = {**st, **(g.get_order_detail(groww_order_id=oid, segment=g.SEGMENT_CASH) or {})}
        return st

    def _place(self, ref: str, **kw) -> str:
        g = self.g
        try:  # idempotency: a crash after sending but before saving must not double the order
            prior = g.get_order_status_by_reference(order_reference_id=ref, segment=g.SEGMENT_CASH)
            if prior and prior.get("groww_order_id") and prior.get("order_status") not in DEAD:
                log.info("order %s already placed as %s", ref, prior["groww_order_id"])
                return prior["groww_order_id"]
        except Exception:  # noqa: BLE001 - "not found" is the normal case
            pass
        r = g.place_order(validity=g.VALIDITY_DAY, exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH,
                          product=g.PRODUCT_CNC, order_reference_id=ref, **kw)
        return r["groww_order_id"]

    def _cancel_gtt(self, gtt_id: str) -> None:
        g = self.g
        try:
            g.cancel_smart_order(segment=g.SEGMENT_CASH, smart_order_type=g.SMART_ORDER_TYPE_GTT,
                                 smart_order_id=gtt_id)
        except Exception as e:  # noqa: BLE001
            log.warning("cancel GTT %s failed: %s", gtt_id, e)

    def _today_sell_price(self, symbol: str) -> float | None:
        g = self.g
        try:
            r = g.get_order_list(segment=g.SEGMENT_CASH, page=0, page_size=25) or {}
            for o in r.get("order_list", r if isinstance(r, list) else []):
                if (o.get("trading_symbol") == symbol and o.get("transaction_type") == "SELL"
                        and o.get("order_status") in DONE and o.get("average_fill_price")):
                    return float(o["average_fill_price"])
        except Exception as e:  # noqa: BLE001
            log.warning("order list failed: %s", e)
        return None

    # ----------------------------------------------------------- fills -> book
    def sync(self, book: Book, engine, today, final: bool = True) -> list[dict]:
        g, closed = self.g, []
        today = str(pd.Timestamp(today).date())
        costs = engine.risk.costs
        target_r = {s.name: sig.target_r for s, sig in zip(engine.specs, engine.signals)}

        keep_pending = []
        for o in book.pending:
            m = book.meta.get(o["symbol"], {})
            oid = m.get("entry_order_id")
            if not oid:
                if final:
                    continue  # never sent: expires like an unfilled order
                keep_pending.append(o)  # queued for this morning's open phase
                continue
            st = self._status(oid)
            filled = int(st.get("filled_quantity") or 0)
            if filled > 0:
                px = float(st.get("average_fill_price") or st.get("price"))
                fee = costs.buy(filled * px)
                book.cash -= filled * px + fee
                tr = target_r.get(o["strategy"], 0.0)
                book.positions.append({
                    "symbol": o["symbol"], "strategy": o["strategy"], "qty": filled, "entry_price": px,
                    "entry_date": today, "stop": px - o["stop_dist"],
                    "target": px + tr * o["stop_dist"] if tr > 0 else None,
                    "risk_per_share": o["stop_dist"], "entry_cost": fee, "bars": 0, "high_close": px,
                    "trailed": False, "pending_exit": None, "last_close": px,
                })
                m.pop("entry_order_id", None)
                if st.get("order_status") not in DONE | DEAD:
                    g.cancel_order(segment=g.SEGMENT_CASH, groww_order_id=oid)  # drop the unfilled rest
                log.info("filled BUY %s %d @ %.2f", o["symbol"], filled, px)
            elif st.get("order_status") in DEAD or final:
                if st.get("order_status") not in DEAD:
                    g.cancel_order(segment=g.SEGMENT_CASH, groww_order_id=oid)
                m.pop("entry_order_id", None)
            else:
                keep_pending.append(o)
        book.pending = keep_pending

        keep = []
        for p in book.positions:
            m = book.meta.setdefault(p["symbol"], {})
            exit_px, reason = None, None
            if m.get("exit_order_id"):
                st = self._status(m["exit_order_id"])
                if st.get("order_status") in DONE:
                    exit_px, reason = float(st.get("average_fill_price")), p.get("pending_exit") or "signal"
            if exit_px is None and m.get("gtt_id"):
                gt = g.get_smart_order(segment=g.SEGMENT_CASH, smart_order_type=g.SMART_ORDER_TYPE_GTT,
                                       smart_order_id=m["gtt_id"]) or {}
                if gt.get("status") in GTT_FIRED:
                    exit_px = self._today_sell_price(p["symbol"]) or p["stop"]
                    reason = "trail" if p["trailed"] else "stop"
            if exit_px is None:
                keep.append(p)
                continue
            fee = costs.sell(p["qty"] * exit_px)
            book.cash += p["qty"] * exit_px - fee
            pnl = (exit_px - p["entry_price"]) * p["qty"] - p["entry_cost"] - fee
            closed.append({**{k: p[k] for k in ("strategy", "symbol", "entry_date", "qty", "entry_price")},
                           "exit_date": today, "exit_price": exit_px, "exit_reason": reason,
                           "costs": p["entry_cost"] + fee, "pnl": pnl,
                           "r_multiple": pnl / (p["risk_per_share"] * p["qty"])})
            if m.get("gtt_id") and reason not in ("stop", "trail"):
                self._cancel_gtt(m["gtt_id"])
            book.meta.pop(p["symbol"], None)
            log.info("closed %s @ %.2f (%s) pnl %.0f", p["symbol"], exit_px, reason, pnl)
        book.positions = keep
        self._check_holdings(book)
        return closed

    def _check_holdings(self, book: Book) -> None:
        g = self.g
        try:
            have: dict[str, float] = {}
            for h in (g.get_holdings_for_user(timeout=10) or {}).get("holdings", []):
                have[h["trading_symbol"]] = have.get(h["trading_symbol"], 0) + float(h.get("quantity", 0))
            for p in (g.get_positions_for_user(segment=g.SEGMENT_CASH) or {}).get("positions", []):
                if p.get("product") == "CNC":
                    have[p["trading_symbol"]] = have.get(p["trading_symbol"], 0) + float(p.get("quantity", 0))
        except Exception as e:  # noqa: BLE001 - reconciliation is advisory
            log.warning("could not reconcile holdings: %s", e)
            return
        for p in book.positions:
            if have.get(p["symbol"], 0) < p["qty"]:
                log.warning("RECONCILE: book holds %d %s, Groww shows %g", p["qty"], p["symbol"],
                            have.get(p["symbol"], 0))

    # ---------------------------------------------------------- book -> orders
    def execute(self, book: Book, orders: list[Order], last_prices: dict[str, float]) -> None:
        g = self.g
        day = pd.Timestamp.now(tz="Asia/Kolkata").date()
        for o in orders:
            m = book.meta.setdefault(o.symbol, {})
            if o.action == "BUY":
                if m.get("entry_order_id"):
                    continue
                m["entry_order_id"] = self._place(
                    reference_id(day, "BUY", o.symbol), trading_symbol=o.symbol, quantity=o.qty,
                    order_type=g.ORDER_TYPE_LIMIT if o.price else g.ORDER_TYPE_MARKET, price=o.price,
                    transaction_type=g.TRANSACTION_TYPE_BUY)
            elif o.action == "SELL":
                if m.get("exit_order_id"):
                    continue
                if m.get("gtt_id"):
                    self._cancel_gtt(m.pop("gtt_id"))
                    m.pop("gtt_trigger", None)
                m["exit_order_id"] = self._place(
                    reference_id(day, "SELL", o.symbol), trading_symbol=o.symbol, quantity=o.qty,
                    order_type=g.ORDER_TYPE_MARKET, transaction_type=g.TRANSACTION_TYPE_SELL)
            elif o.action == "STOP":
                if m.get("gtt_trigger") == o.price and m.get("gtt_qty") == o.qty:
                    continue
                leg = {"order_type": g.ORDER_TYPE_LIMIT, "price": f"{round_tick(o.price * (1 - self.buffer)):.2f}",
                       "transaction_type": g.TRANSACTION_TYPE_SELL}
                if m.get("gtt_id"):
                    g.modify_smart_order(smart_order_id=m["gtt_id"], smart_order_type=g.SMART_ORDER_TYPE_GTT,
                                         segment=g.SEGMENT_CASH, quantity=o.qty, trigger_price=f"{o.price:.2f}",
                                         order=leg)
                else:
                    r = g.create_smart_order(
                        smart_order_type=g.SMART_ORDER_TYPE_GTT, reference_id=reference_id(day, "STOP", o.symbol),
                        segment=g.SEGMENT_CASH, trading_symbol=o.symbol, quantity=o.qty,
                        trigger_price=f"{o.price:.2f}", trigger_direction=g.TRIGGER_DIRECTION_DOWN, order=leg,
                        product_type=g.PRODUCT_CNC, exchange=g.EXCHANGE_NSE, duration=g.VALIDITY_DAY)
                    m["gtt_id"] = r["smart_order_id"]
                m["gtt_trigger"], m["gtt_qty"] = o.price, o.qty
