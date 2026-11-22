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
        self.orders, self.gtts = {}, {}

    def place_order(self, **kw):
        oid = f"O{len(self.orders) + 1}"
        self.orders[oid] = {"kw": kw, "status": "OPEN", "filled_quantity": 0, "average_price": 0}
        return oid

    def order_history(self, oid):
        return [self.orders[oid]]

    def fill(self, oid, px):
        self.orders[oid].update(status="COMPLETE", filled_quantity=self.orders[oid]["kw"]["quantity"], average_price=px)

    def cancel_order(self, variety, order_id):
        self.orders[order_id]["status"] = "CANCELLED"

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
    rep = Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=broker)
    assert rep.sent and [o.action for o in rep.orders] == ["BUY"]
    buy = kite.orders["O1"]["kw"]
    assert (buy["variety"], buy["order_type"], buy["price"], buy["quantity"], buy["product"]) == \
        ("amo", "LIMIT", 102.0, 100, "CNC")

    # day 6: AMO filled at 100.5 at the open; after the close a GTT stop is placed
    kite.fill("O1", 100.5)
    rep = Runner(cfg, panel=p.head(7)).live(update=False, send=True, broker=broker)
    assert [(x["symbol"], x["qty"], x["entry_price"]) for x in rep.book.positions] == [("S0", 100, 100.5)]
    assert kite.gtts[1]["kw"]["trigger_values"] == [90.5]

    # running the same day twice is refused
    assert "already processed" in Runner(cfg, panel=p.head(7)).live(update=False, send=True, broker=broker).note

    # day 7: stop triggered -> trade closed in the ledger at the stop price
    kite.gtts[1]["status"] = "triggered"
    rep = Runner(cfg, panel=p.head(8)).live(update=False, send=True, broker=broker)
    assert rep.book.positions == []
    t = pd.read_csv(tmp_path / "t" / "live" / "trades.csv")
    assert (t.loc[0, "exit_reason"], t.loc[0, "exit_price"], t.loc[0, "qty"]) == ("stop", 90.5, 100)
    assert np.isclose(rep.book.cash, 100_000 - 1000)


def test_live_dry_run_sends_nothing(tmp_path):
    T = 20
    p = make_panel(np.full(T, 100.0))
    cfg = cfg_for(tmp_path, [StrategySpec("s", Scripted(entry=onehot(T, 5)))], dry_run=True)
    kite = FakeKite()
    rep = Runner(cfg, panel=p.head(6)).live(update=False, send=True, broker=KiteBroker(kite=kite))
    assert not rep.sent and kite.orders == {} and "DRY RUN" in rep.note
