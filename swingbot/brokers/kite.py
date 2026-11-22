"""Zerodha Kite live adapter.

Orders, GTTs and portfolio endpoints work on the *free* Kite Connect
"Personal" plan; market data can come from any provider in the strategy file
(e.g. free Yahoo EOD candles), so live swing trading costs nothing extra.

Daily routine (IST):
  ~09:20  swingbot live FILE --phase open   sync the AMO entry/exit fills, put GTT stops on new positions
  ~15:45  swingbot live FILE                sync, decide on today's close, send AMOs for tomorrow + stop updates

Entries are AMO limit orders at close * (1 + max_entry_gap) so they fill at
the open like the backtest; exits are AMO market orders; protective stops are
single-leg GTTs that sell with a limit `stop_limit_buffer` below the trigger.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..engine.book import Book
from .base import Broker, Order, round_tick

log = logging.getLogger(__name__)
FILLED = "COMPLETE"
DEAD = {"CANCELLED", "REJECTED"}


class KiteBroker(Broker):
    name = "kite"

    def __init__(self, kite=None, exchange: str = "NSE", tag: str = "swingbot", stop_limit_buffer: float = 0.01):
        if kite is None:
            from ..data.kite import _client

            kite = _client()
        self.kite = kite
        self.exchange = exchange
        self.tag = tag[:20]
        self.buffer = stop_limit_buffer

    # ----------------------------------------------------------- fills -> book
    def _last_state(self, order_id):
        hist = self.kite.order_history(order_id)
        return hist[-1] if hist else {}

    def sync(self, book: Book, engine, today, final: bool = True) -> list[dict]:
        k, closed = self.kite, []
        today = str(pd.Timestamp(today).date())
        costs = engine.risk.costs
        target_r = {s.name: sig.target_r for s, sig in zip(engine.specs, engine.signals)}

        # entries placed at the last close
        still_pending = []
        for o in book.pending:
            m = book.meta.get(o["symbol"], {})
            oid = m.get("entry_order_id")
            if not oid:
                continue  # never sent (dry run) -> expires like an unfilled order
            st = self._last_state(oid)
            filled = int(st.get("filled_quantity") or 0)
            if filled > 0:
                px = float(st["average_price"])
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
                log.info("filled BUY %s %d @ %.2f", o["symbol"], filled, px)
                if st.get("status") not in (FILLED, *DEAD):
                    self._cancel(oid)  # partial fill: drop the rest
            elif st.get("status") in DEAD or st.get("status") == FILLED:
                m.pop("entry_order_id", None)
            elif final:
                self._cancel(oid)  # day over and not filled (gap above the limit): drop it
                m.pop("entry_order_id", None)
            else:
                still_pending.append(o)  # morning run before the AMO executed
        book.pending = still_pending

        # exits: our market sells, or a GTT stop that triggered
        keep = []
        for p in book.positions:
            m = book.meta.setdefault(p["symbol"], {})
            exit_px, reason = None, None
            if m.get("exit_order_id"):
                st = self._last_state(m["exit_order_id"])
                if st.get("status") == FILLED:
                    exit_px, reason = float(st["average_price"]), p.get("pending_exit") or "signal"
            if exit_px is None and m.get("gtt_id"):
                g = k.get_gtt(m["gtt_id"])
                if g.get("status") == "triggered":
                    exit_px = self._gtt_fill_price(g) or p["stop"]
                    reason = "trail" if p["trailed"] else "stop"
            if exit_px is None:
                keep.append(p)
                continue
            fee = costs.sell(p["qty"] * exit_px)
            book.cash += p["qty"] * exit_px - fee
            pnl = (exit_px - p["entry_price"]) * p["qty"] - p["entry_cost"] - fee
            closed.append({**{k_: p[k_] for k_ in ("strategy", "symbol", "entry_date", "qty", "entry_price")},
                           "exit_date": today, "exit_price": exit_px, "exit_reason": reason,
                           "costs": p["entry_cost"] + fee, "pnl": pnl,
                           "r_multiple": pnl / (p["risk_per_share"] * p["qty"])})
            if m.get("gtt_id") and reason not in ("stop", "trail"):
                self._delete_gtt(m["gtt_id"])
            book.meta.pop(p["symbol"], None)
            log.info("closed %s @ %.2f (%s) pnl %.0f", p["symbol"], exit_px, reason, pnl)
        book.positions = keep
        self._check_holdings(book)
        return closed

    def _gtt_fill_price(self, g) -> float | None:
        try:
            oid = g["orders"][0]["result"]["order_result"]["order_id"]
            st = self._last_state(oid)
            return float(st["average_price"]) if st.get("status") == FILLED else None
        except (KeyError, IndexError, TypeError):
            return None

    def _check_holdings(self, book: Book) -> None:
        """Warn (never auto-fix) when the broker disagrees with the book."""
        try:
            have = {h["tradingsymbol"]: int(h["quantity"]) + int(h.get("t1_quantity", 0))
                    for h in self.kite.holdings()}
            for p in self.kite.positions().get("net", []):
                if p.get("product") == "CNC":
                    have[p["tradingsymbol"]] = have.get(p["tradingsymbol"], 0) + int(p["quantity"])
        except Exception as e:  # noqa: BLE001 - reconciliation is advisory
            log.warning("could not reconcile holdings: %s", e)
            return
        for p in book.positions:
            if have.get(p["symbol"], 0) < p["qty"]:
                log.warning("RECONCILE: book holds %d %s, broker shows %d", p["qty"], p["symbol"],
                            have.get(p["symbol"], 0))

    # ---------------------------------------------------------- book -> orders
    def execute(self, book: Book, orders: list[Order], last_prices: dict[str, float]) -> None:
        k = self.kite
        for o in orders:
            m = book.meta.setdefault(o.symbol, {})
            if o.action == "BUY":
                m["entry_order_id"] = k.place_order(
                    variety=k.VARIETY_AMO, exchange=self.exchange, tradingsymbol=o.symbol,
                    transaction_type=k.TRANSACTION_TYPE_BUY, quantity=o.qty, product=k.PRODUCT_CNC,
                    order_type=k.ORDER_TYPE_LIMIT if o.price else k.ORDER_TYPE_MARKET, price=o.price,
                    validity=k.VALIDITY_DAY, tag=self.tag)
            elif o.action == "SELL":
                if m.get("exit_order_id"):
                    continue  # already sent
                if m.get("gtt_id"):
                    self._delete_gtt(m.pop("gtt_id"))
                    m.pop("gtt_trigger", None)
                m["exit_order_id"] = k.place_order(
                    variety=k.VARIETY_AMO, exchange=self.exchange, tradingsymbol=o.symbol,
                    transaction_type=k.TRANSACTION_TYPE_SELL, quantity=o.qty, product=k.PRODUCT_CNC,
                    order_type=k.ORDER_TYPE_MARKET, validity=k.VALIDITY_DAY, tag=self.tag)
            elif o.action == "STOP":
                if m.get("gtt_trigger") == o.price and m.get("gtt_qty") == o.qty:
                    continue  # unchanged
                args = dict(
                    trigger_type=k.GTT_TYPE_SINGLE, tradingsymbol=o.symbol, exchange=self.exchange,
                    trigger_values=[o.price], last_price=last_prices[o.symbol],
                    orders=[{"exchange": self.exchange, "tradingsymbol": o.symbol, "transaction_type": "SELL",
                             "quantity": o.qty, "order_type": "LIMIT", "product": "CNC",
                             "price": round_tick(o.price * (1 - self.buffer))}])
                if m.get("gtt_id"):
                    k.modify_gtt(trigger_id=m["gtt_id"], **args)
                else:
                    m["gtt_id"] = k.place_gtt(**args)["trigger_id"]
                m["gtt_trigger"], m["gtt_qty"] = o.price, o.qty

    def _cancel(self, order_id):
        try:
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
        except Exception as e:  # noqa: BLE001
            log.warning("cancel %s failed: %s", order_id, e)

    def _delete_gtt(self, gtt_id):
        try:
            self.kite.delete_gtt(gtt_id)
        except Exception as e:  # noqa: BLE001
            log.warning("delete GTT %s failed: %s", gtt_id, e)


def get_broker(name: str, live_cfg) -> Broker:
    if name == "kite":
        return KiteBroker(exchange="NSE", tag=live_cfg.tag, stop_limit_buffer=live_cfg.stop_limit_buffer)
    raise NotImplementedError(
        f"live broker {name!r} is not implemented; use 'kite' (orders are free on the Kite Personal plan)")
