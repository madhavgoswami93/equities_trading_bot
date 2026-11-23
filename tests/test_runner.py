"""Backtest == paper (by construction) and the live Kite flow against a fake broker."""

import numpy as np
import pandas as pd
from conftest import Scripted, make_panel, onehot

from swingbot import Backtest, RiskConfig, Runner, StrategySpec, TrendPullback, VolumeBreakout
from swingbot.brokers.kite import KiteBroker
from swingbot.config import DataConfig, LiveConfig, Quantity, RunConfig
from swingbot.data import synthetic_panel


def cfg_for(tmp_path, specs, risk=None, dry_run=True):
    return RunConfig(name="t", data=DataConfig(provider="synthetic"), risk=risk or RiskConfig(),
                     live=LiveConfig(dry_run=dry_run), strategies=specs, runs_dir=str(tmp_path))


def test_paper_day_by_day_equals_backtest(tmp_path):
    """Run paper trading one session at a time on growing data (as in real life)
    and check it reproduces the backtest trade-for-trade."""
    p = synthetic_panel(40, 900, seed=21)
    specs = [StrategySpec("pull", TrendPullback(), quantity=Quantity.parse("risk 1%")),
             StrategySpec("brk", VolumeBreakout(), quantity=Quantity.parse("equity 20%"), priority=2)]
    risk = RiskConfig(max_positions=5, max_position_pct=0.40, max_drawdown_halt=0.1, halt_cooldown=10)
    cfg = cfg_for(tmp_path, specs, risk)
    start = 400
    for n in range(start + 1, len(p.dates) + 1):  # day n-1 is "today"
        rep = Runner(cfg, panel=p.head(n)).paper(update=False, start=p.dates[start] if n == start + 1 else None)
    bt = Backtest(p, specs, risk).run(start=p.dates[start])

    paper_trades = pd.read_csv(tmp_path / "t" / "paper" / "trades.csv", parse_dates=["entry_date", "exit_date"])
    last = p.dates[-1]
    key = ["entry_date", "symbol"]
    a = bt.trades[bt.trades.exit_date < last].sort_values(key).reset_index(drop=True)
    b = paper_trades[paper_trades.exit_date < last].sort_values(key).reset_index(drop=True)
    assert len(a) > 30
    pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False)
    paper_eq = pd.read_csv(tmp_path / "t" / "paper" / "equity.csv")["equity"].to_numpy()
    assert np.allclose(paper_eq[:-1], bt.equity.to_numpy()[:-1])
    assert len(rep.book.positions) <= 5


def test_paper_is_resumable_and_idempotent(tmp_path):
    p = synthetic_panel(20, 600, seed=4)
    cfg = cfg_for(tmp_path, [StrategySpec("brk", VolumeBreakout())])
    Runner(cfg, panel=p.head(500)).paper(update=False, start=p.dates[300])
    rep = Runner(cfg, panel=p.head(500)).paper(update=False)
    assert "already up to date" in rep.note
    rep = Runner(cfg, panel=p).paper(update=False)
    assert rep.date == str(p.dates[-1].date())


class FakeKite:
    VARIETY_AMO, VARIETY_REGULAR = "amo", "regular"
    TRANSACTION_TYPE_BUY, TRANSACTION_TYPE_SELL = "BUY", "SELL"
    PRODUCT_CNC, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, VALIDITY_DAY = "CNC", "LIMIT", "MARKET", "DAY"
    GTT_TYPE_SINGLE = "single"

    def __init__(self):
        self.orders_, self.gtts = {}, {}

    def place_order(self, **kw):
        oid = f"O{len(self.orders_) + 1}"
        self.orders_[oid] = {"kw": kw, "status": "OPEN", "filled_quantity": 0, "average_price": 0}
        return oid

    def order_history(self, oid):
        return [self.orders_[oid]]

    def fill(self, oid, px):
        self.orders_[oid].update(status="COMPLETE", filled_quantity=self.orders_[oid]["kw"]["quantity"], average_price=px)

    def cancel_order(self, variety, order_id):
        self.orders_[order_id]["status"] = "CANCELLED"

    def place_gtt(self, **kw):
        gid = len(self.gtts) + 1
        self.gtts[gid] = {"kw": kw, "status": "active"}
        return {"trigger_id": gid}

    def modify_gtt(self, trigger_id, **kw):
        self.gtts[trigger_id]["kw"] = kw

    def get_gtt(self, gid):
        return self.gtts[gid]

    def delete_gtt(self, gid):
        self.gtts[gid]["status"] = "deleted"

    def holdings(self):
        return []

    def orders(self):
        return [{"tradingsymbol": o["kw"]["tradingsymbol"], "transaction_type": o["kw"]["transaction_type"],
                 "tag": o["kw"].get("tag"), "status": o["status"]} for o in self.orders_.values()]

    def positions(self):
        return {"net": []}


def test_live_flow_with_fake_kite(tmp_path):
    T = 20
    p = make_panel(np.full(T, 100.0))
    risk = RiskConfig(capital=100_000, max_adv_pct=0, slippage_bps=0, costs="free", max_entry_gap=0.02)
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], risk, dry_run=False)
    kite = FakeKite()
    broker = KiteBroker(kite=kite)

    # day 5 close: entry signal -> AMO limit buy at close + 2%
    rep = Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=broker, check_clock=False)
    assert rep.sent and [o.action for o in rep.orders] == ["BUY"]
    buy = kite.orders_["O1"]["kw"]
    assert (buy["variety"], buy["order_type"], buy["price"], buy["quantity"], buy["product"]) == \
        ("amo", "LIMIT", 102.0, 100, "CNC")

    # day 6: AMO filled at 100.5 at the open; after the close a GTT stop is placed
    kite.fill("O1", 100.5)
    rep = Runner(cfg, panel=p.head(7)).live(update=False, send=True, broker=broker, check_clock=False)
    assert [(x["symbol"], x["qty"], x["entry_price"]) for x in rep.book.positions] == [("S0", 100, 100.5)]
    assert kite.gtts[1]["kw"]["trigger_values"] == [90.5]

    # running the same day twice is refused
    assert "already processed" in Runner(cfg, panel=p.head(7)).live(update=False, send=True, broker=broker, check_clock=False).note

    # day 7: stop triggered -> trade closed in the ledger at the stop price
    kite.gtts[1]["status"] = "triggered"
    rep = Runner(cfg, panel=p.head(8)).live(update=False, send=True, broker=broker, check_clock=False)
    assert rep.book.positions == []
    t = pd.read_csv(tmp_path / "t" / "live" / "trades.csv")
    assert (t.loc[0, "exit_reason"], t.loc[0, "exit_price"], t.loc[0, "qty"]) == ("stop", 90.5, 100)
    assert np.isclose(rep.book.cash, 100_000 - 1000)


def test_live_dry_run_sends_nothing(tmp_path):
    T = 20
    p = make_panel(np.full(T, 100.0))
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], dry_run=True)
    kite = FakeKite()
    rep = Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=KiteBroker(kite=kite), check_clock=False)
    assert not rep.sent and kite.orders_ == {} and "DRY RUN" in rep.note


def test_live_refuses_stale_data(tmp_path):
    T = 20
    p = make_panel(np.full(T, 100.0))  # 2024 dates: never "today"
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], dry_run=False)
    kite = FakeKite()
    rep = Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=KiteBroker(kite=kite))
    assert "Not deciding on stale or partial data" in rep.note and kite.orders_ == {}


def test_halt_blocks_new_entries_only(tmp_path):
    T = 20
    p = make_panel(np.full(T, 100.0))
    risk = RiskConfig(capital=100_000, max_adv_pct=0, slippage_bps=0, costs="free", max_entry_gap=0.02)
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], risk, dry_run=False)
    kite = FakeKite()
    r = Runner(cfg, panel=p.head(6))
    r.halt_file().parent.mkdir(parents=True, exist_ok=True)
    r.halt_file().write_text("x")
    rep = r.live(update=False, send=True, broker=KiteBroker(kite=kite), check_clock=False)
    assert kite.orders_ == {} and "HALTED" in rep.note


def test_kite_does_not_resend_after_crash(tmp_path):
    """Orders sent but the book not saved (crash): the re-run must not send them again."""
    T = 20
    p = make_panel(np.full(T, 100.0))
    risk = RiskConfig(capital=100_000, max_adv_pct=0, slippage_bps=0, costs="free", max_entry_gap=0.02)
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], risk, dry_run=False)
    kite = FakeKite()
    Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=KiteBroker(kite=kite), check_clock=False)
    (tmp_path / "t" / "live" / "book.json").unlink()  # simulate losing the saved state
    Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=KiteBroker(kite=kite), check_clock=False)
    assert len(kite.orders_) == 1


class FakeGroww:
    SEGMENT_CASH, EXCHANGE_NSE, PRODUCT_CNC, VALIDITY_DAY = "CASH", "NSE", "CNC", "DAY"
    ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET = "LIMIT", "MARKET"
    TRANSACTION_TYPE_BUY, TRANSACTION_TYPE_SELL = "BUY", "SELL"
    SMART_ORDER_TYPE_GTT, TRIGGER_DIRECTION_DOWN = "GTT", "DOWN"

    def __init__(self):
        self.orders, self.gtts = {}, {}

    def place_order(self, **kw):
        oid = f"G{len(self.orders) + 1}"
        self.orders[oid] = {"kw": kw, "order_status": "NEW", "filled_quantity": 0,
                            "order_reference_id": kw["order_reference_id"]}
        return {"groww_order_id": oid, "order_status": "NEW"}

    def get_order_status_by_reference(self, order_reference_id, segment):
        for oid, o in self.orders.items():
            if o["order_reference_id"] == order_reference_id:
                return {"groww_order_id": oid, "order_status": o["order_status"]}
        raise KeyError("not found")

    def get_order_status(self, groww_order_id, segment):
        o = self.orders[groww_order_id]
        return {k: o[k] for k in ("order_status", "filled_quantity")}

    def get_order_detail(self, groww_order_id, segment):
        return {"average_fill_price": self.orders[groww_order_id].get("average_fill_price")}

    def fill(self, oid, px):
        o = self.orders[oid]
        o.update(order_status="EXECUTED", filled_quantity=o["kw"]["quantity"], average_fill_price=px)

    def cancel_order(self, segment, groww_order_id):
        self.orders[groww_order_id]["order_status"] = "CANCELLED"

    def create_smart_order(self, **kw):
        gid = f"gtt{len(self.gtts) + 1}"
        self.gtts[gid] = {"kw": kw, "status": "ACTIVE"}
        return {"smart_order_id": gid, "status": "ACTIVE"}

    def modify_smart_order(self, smart_order_id, **kw):
        self.gtts[smart_order_id]["kw"].update(kw)

    def get_smart_order(self, segment, smart_order_type, smart_order_id):
        return self.gtts[smart_order_id]

    def cancel_smart_order(self, segment, smart_order_type, smart_order_id):
        self.gtts[smart_order_id]["status"] = "CANCELLED"

    def get_order_list(self, **kw):
        return {"order_list": []}

    def get_holdings_for_user(self, timeout=5):
        return {"holdings": []}

    def get_positions_for_user(self, segment=None):
        return {"positions": []}


def test_live_flow_with_fake_groww(tmp_path):
    """Groww has no AMO: buys are queued at the close and sent in the next open phase."""
    from swingbot.brokers.groww import GrowwBroker

    T = 20
    p = make_panel(np.full(T, 100.0))
    risk = RiskConfig(capital=100_000, max_adv_pct=0, slippage_bps=0, costs="free", max_entry_gap=0.02)
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], risk, dry_run=False)
    g = FakeGroww()
    broker = GrowwBroker(groww=g)

    # day 5 close: decision made, but nothing can be sent after hours
    rep = Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=broker, check_clock=False)
    assert g.orders == {} and "no after-market orders" in rep.note
    assert [o["symbol"] for o in rep.book.pending] == ["S0"]

    # day 6 morning: open phase sends the queued limit buy
    rep = Runner(cfg, panel=p.head(6)).live(phase="open", update=False, send=True, broker=broker, check_clock=False)
    buy = g.orders["G1"]["kw"]
    assert (buy["transaction_type"], buy["order_type"], buy["price"], buy["quantity"], buy["product"]) == \
        ("BUY", "LIMIT", 102.0, 100, "CNC")
    # re-running the open phase sends nothing new
    Runner(cfg, panel=p.head(6)).live(phase="open", update=False, send=True, broker=broker, check_clock=False)
    assert len(g.orders) == 1

    # filled; day 6 close: position booked, GTT stop placed at entry - stop distance
    g.fill("G1", 100.5)
    rep = Runner(cfg, panel=p.head(7)).live(update=False, send=True, broker=broker, check_clock=False)
    assert [(x["symbol"], x["qty"], x["entry_price"]) for x in rep.book.positions] == [("S0", 100, 100.5)]
    gtt = g.gtts["gtt1"]["kw"]
    assert (gtt["trigger_price"], gtt["trigger_direction"], gtt["order"]["transaction_type"]) == \
        ("90.50", "DOWN", "SELL")

    # day 7: the GTT fires -> trade closed at the stop
    g.gtts["gtt1"]["status"] = "TRIGGERED"
    rep = Runner(cfg, panel=p.head(8)).live(update=False, send=True, broker=broker, check_clock=False)
    assert rep.book.positions == []
    t = pd.read_csv(tmp_path / "t" / "live" / "trades.csv")
    assert (t.loc[0, "exit_reason"], t.loc[0, "exit_price"]) == ("stop", 90.5)


def test_groww_reference_ids_are_valid():
    from swingbot.brokers.groww import reference_id

    for sym in ("M&M", "BAJAJ-AUTO", "ULTRACEMCO", "X"):
        r = reference_id("2026-11-23", "BUY", sym)
        assert 8 <= len(r) <= 20 and r.isalnum()
