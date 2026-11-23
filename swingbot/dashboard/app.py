"""swingbot dashboard --  python -m swingbot dashboard strategies.toml

Tabs:
  Backtest       run the strategy file (or a subset) with risk overrides; charts + tables
  Paper          the paper book, next-session orders, equity curve and ledger; run today's step
  Live           read-only view of the live book (orders are sent from the CLI only)
  Signals        what each strategy signals on the latest close, and the orders the engine would place
  Strategy file  edit, validate and save the TOML file
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from swingbot.brokers.base import plan_orders
from swingbot.config import load_config
from swingbot.dashboard import charts as C
from swingbot.engine.backtest import Backtest, Engine
from swingbot.engine.book import Book
from swingbot.runner import Runner
from swingbot.strategies.expr import FUNCTIONS


# ------------------------------------------------------------------ setup
def _cli_file() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="strategies.toml")
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.file


st.set_page_config(page_title="swingbot", page_icon=":chart_with_upwards_trend:", layout="wide")


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - older streamlit
        return "light"


MODE = theme_mode()


def data_stamp(cfg) -> float:
    """Changes whenever the cached candles change -> invalidates the panel cache."""
    d = Path(cfg.data.cache_dir) / cfg.data.provider
    return max((f.stat().st_mtime for f in d.glob("*.parquet")), default=0.0) if d.exists() else 0.0


@st.cache_resource(show_spinner="Loading market data...")
def load_panel(file: str, file_mtime: float, stamp: float):
    return Runner(load_config(file)).panel()


@st.cache_resource(show_spinner="Running backtest...", max_entries=8)
def cached_backtest(file: str, file_mtime: float, stamp: float, names: tuple, start, end, overrides: tuple):
    cfg = load_config(file)
    panel = load_panel(file, file_mtime, stamp)
    risk = replace(cfg.risk, **dict(overrides))
    return Backtest(panel, [cfg.strategy(n) for n in names], risk).run(start, end)


def fmt_pct(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.1%}"


def fmt_num(v, d=2):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.{d}f}"


def money(v):
    """Indian compact notation so large values fit a metric tile: ₹15.30 L, ₹1.25 Cr."""
    a = abs(v)
    if a >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("swingbot")
    file = st.text_input("Strategy file", value=st.session_state.get("file", _cli_file()))
    st.session_state["file"] = file
    try:
        cfg = load_config(file)
    except Exception as e:  # noqa: BLE001 - show any config error in the UI
        st.error(f"Cannot load {file}: {e}")
        st.stop()
    fm = Path(file).stat().st_mtime
    st.caption(f"**{cfg.name}** - {len(cfg.active)} active of {len(cfg.strategies)} strategies")
    r = cfg.risk
    st.markdown(
        f"""
**Risk rules**
- Max open positions: **{r.max_positions}**
- Max per position: **{r.max_position_pct:.0%}** of equity
- Default risk per trade: **{r.risk_per_trade:.2%}**
- Capital: **{money(r.capital)}**
- Drawdown pause: **{'off' if not r.max_drawdown_halt else f'{r.max_drawdown_halt:.0%} for {r.halt_cooldown} sessions'}**
"""
    )
    st.caption(f"Data: {cfg.data.provider} - universe {cfg.data.universe}")
    if st.button("Download latest market data", width="stretch"):
        with st.spinner("Fetching candles..."):
            Runner(cfg).update_data()
        st.cache_resource.clear()
        st.success("Data updated")

try:
    panel = load_panel(file, fm, data_stamp(cfg))
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot load market data: {e}. Run `python -m swingbot fetch {file}` or use the sidebar button.")
    st.stop()

st.sidebar.caption(f"{panel.shape[1]} symbols, {panel.dates[0].date()} to **{panel.dates[-1].date()}**")

tab_bt, tab_paper, tab_live, tab_sig, tab_file, tab_arch = st.tabs(
    ["Backtest", "Paper trading", "Live", "Signals today", "Strategy file", "Architecture"])


# --------------------------------------------------------------- backtest
def render_result(res, names):
    s = res.stats
    cols = st.columns(6)
    cols[0].metric("CAGR", fmt_pct(s.get("cagr")),
                   help=f"Nifty 50 CAGR {fmt_pct(s.get('benchmark_cagr'))}" if "benchmark_cagr" in s else None)
    cols[1].metric("Sharpe", fmt_num(s.get("sharpe")))
    cols[2].metric("Max drawdown", fmt_pct(s.get("max_drawdown")))
    cols[3].metric("Trades", f"{s.get('trades', 0):,}")
    cols[4].metric("Win rate", fmt_pct(s.get("win_rate")))
    cols[5].metric("Profit factor", fmt_num(s.get("profit_factor")))
    cols = st.columns(6)
    cols[0].metric("Final equity", money(s.get("final_equity", 0)))
    cols[1].metric("Expectancy", f"{fmt_num(s.get('expectancy_r'))} R")
    cols[2].metric("Avg hold", f"{fmt_num(s.get('avg_hold_days'), 1)} days")
    cols[3].metric("Exposure", fmt_pct(s.get("exposure")))
    cols[4].metric("Max positions held", f"{int(res.positions.max()) if len(res.positions) else 0}")
    cols[5].metric("Costs paid", money(s.get("total_costs", 0)))

    st.subheader("Equity vs Nifty 50")
    st.plotly_chart(C.equity_chart(res.equity, res.benchmark, MODE), width="stretch")
    st.subheader("Drawdown")
    st.plotly_chart(C.drawdown_chart(res.equity, MODE), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Realised PnL by strategy")
        if not res.trades.empty:
            st.plotly_chart(C.strategy_pnl_chart(res.trades, names, MODE), width="stretch")
        st.dataframe(res.by_strategy().style.format({
            "win_rate": "{:.1%}", "pnl": "{:,.0f}", "avg_return_pct": "{:.2%}", "expectancy_r": "{:.2f}",
            "avg_hold": "{:.1f}", "costs": "{:,.0f}"}), width="stretch")
    with right:
        st.subheader("Exit reasons")
        if not res.trades.empty:
            st.plotly_chart(C.bar_chart(res.trades.exit_reason.value_counts(), MODE, ",d"), width="stretch")

    st.subheader("Monthly returns")
    st.plotly_chart(C.monthly_heatmap(res.monthly_returns().drop(columns="year"), MODE), width="stretch")

    st.subheader("By stock")
    st.dataframe(res.by_symbol().style.format({
        "win_rate": "{:.1%}", "pnl": "{:,.0f}", "avg_return_pct": "{:.2%}", "avg_r": "{:.2f}", "avg_hold": "{:.1f}"}),
        width="stretch", height=320)

    st.subheader("Trades")
    t = res.trades.copy()
    f1, f2, f3 = st.columns(3)
    pick_s = f1.multiselect("Strategy", sorted(t.strategy.unique()) if len(t) else [])
    pick_sym = f2.multiselect("Symbol", sorted(t.symbol.unique()) if len(t) else [])
    pick_r = f3.multiselect("Exit reason", sorted(t.exit_reason.unique()) if len(t) else [])
    if pick_s:
        t = t[t.strategy.isin(pick_s)]
    if pick_sym:
        t = t[t.symbol.isin(pick_sym)]
    if pick_r:
        t = t[t.exit_reason.isin(pick_r)]
    st.dataframe(t.sort_values("entry_date", ascending=False), width="stretch", height=360,
                 column_config={"entry_date": st.column_config.DateColumn(), "exit_date": st.column_config.DateColumn(),
                                "return_pct": st.column_config.NumberColumn(format="percent"),
                                "r_multiple": st.column_config.NumberColumn(format="%.2f"),
                                **{c: st.column_config.NumberColumn(format="%.2f")
                                   for c in ("entry_price", "exit_price")},
                                **{c: st.column_config.NumberColumn(format="%.0f") for c in ("risk", "costs", "pnl")}})
    st.download_button("Download trades CSV", res.trades.to_csv(index=False), "trades.csv", "text/csv")


with tab_bt:
    names_all = [s.name for s in cfg.strategies]
    c1, c2, c3 = st.columns([3, 1, 1])
    names = c1.multiselect("Strategies (run together as one portfolio)", names_all,
                           default=[s.name for s in cfg.active])
    start = c2.date_input("Start", value=max(panel.dates[0], pd.Timestamp("2010-01-01")).date(),
                          min_value=panel.dates[0].date(), max_value=panel.dates[-1].date())
    end = c3.date_input("End", value=panel.dates[-1].date(), min_value=panel.dates[0].date(),
                        max_value=panel.dates[-1].date())
    with st.expander("Override risk rules for this run (the file is not changed)"):
        o1, o2, o3, o4 = st.columns(4)
        ov = dict(
            max_positions=o1.number_input("Max positions", 1, 50, r.max_positions),
            max_position_pct=o2.number_input("Max % per position", 0.05, 1.0, float(r.max_position_pct), 0.05),
            risk_per_trade=o3.number_input("Risk per trade", 0.001, 0.05, float(r.risk_per_trade), 0.0025,
                                           format="%.4f"),
            capital=o4.number_input("Capital (Rs)", 10_000.0, 1e9, float(r.capital), 100_000.0),
        )
    if not names:
        st.info("Pick at least one strategy.")
    else:
        try:
            res = cached_backtest(file, fm, data_stamp(cfg), tuple(names), pd.Timestamp(start), pd.Timestamp(end),
                                  tuple(sorted(ov.items())))
            render_result(res, names)
        except Exception as e:  # noqa: BLE001
            st.error(f"Backtest failed: {e}")


# ----------------------------------------------------------- paper / live
def render_book(mode: str):
    d = cfg.state_dir(mode)
    bp = d / "book.json"
    if not bp.exists():
        st.info(f"No {mode} book yet.")
        return
    book = Book.load(bp)
    eq = book.equity()
    c = st.columns(5)
    c[0].metric("As of", book.last_date or "-")
    c[1].metric("Equity", money(eq), delta=f"{eq / cfg.risk.capital - 1:+.2%} since start")
    c[2].metric("Cash", money(book.cash))
    c[3].metric("Open positions", f"{len(book.positions)} / {cfg.risk.max_positions}")
    c[4].metric("Drawdown pause", f"{book.halt_left} sessions" if book.halt_left else "off")

    if book.positions:
        st.subheader("Positions")
        df = pd.DataFrame(book.positions)
        df["weight"] = df.qty * df.last_close / eq
        df["unrealised"] = (df.last_close - df.entry_price) * df.qty - df.entry_cost
        df["to_stop"] = df.stop / df.last_close - 1
        st.dataframe(df[["strategy", "symbol", "qty", "entry_date", "entry_price", "last_close", "stop", "to_stop",
                         "bars", "weight", "unrealised", "pending_exit"]].round(
                         {"entry_price": 2, "last_close": 2, "stop": 2, "unrealised": 0}), width="stretch", hide_index=True,
                     column_config={"weight": st.column_config.ProgressColumn(
                         "weight", format="percent", min_value=0, max_value=cfg.risk.max_position_pct),
                         "to_stop": st.column_config.NumberColumn(format="percent"),
                         "unrealised": st.column_config.NumberColumn(format="%.0f")})
    orders = plan_orders(book)
    st.subheader("Orders for the next session")
    if orders:
        st.dataframe(pd.DataFrame([o.__dict__ for o in orders]), width="stretch", hide_index=True)
    else:
        st.caption("No orders.")

    eqf = d / "equity.csv"
    if eqf.exists():
        e = pd.read_csv(eqf, parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date")
        if len(e) > 1:
            st.subheader("Equity")
            bench = None
            if panel.benchmark is not None:
                bench = pd.Series(panel.benchmark, index=panel.dates).reindex(e.index)
            st.plotly_chart(C.equity_chart(e.equity, bench, MODE, label=mode.title()), width="stretch")
    tf = d / "trades.csv"
    if tf.exists():
        st.subheader("Closed trades")
        st.dataframe(pd.read_csv(tf).iloc[::-1].round(2), width="stretch", height=300)


with tab_paper:
    st.caption("Paper trading runs the backtest engine one day at a time on live data. "
               "Schedule `python -m swingbot paper FILE` daily after 15:45 IST, or step it here.")
    c1, c2, c3 = st.columns([1, 1, 2])
    upd = c1.checkbox("Download new data first", value=False)
    replay = c2.date_input("First run: start from", value=None, help="Leave empty to start from the latest close")
    if c3.button("Run paper trading now", type="primary"):
        with st.spinner("Advancing the paper book..."):
            rep = Runner(cfg).paper(update=upd, start=replay)
        if upd:
            st.cache_resource.clear()
        st.success(f"{rep.date}: {rep.note}")
    render_book("paper")

with tab_live:
    st.caption("Read-only view. Orders are only sent from the command line: "
               "`python -m swingbot live FILE --send` (and `dry_run = false` in [live]).")
    render_book("live")


# --------------------------------------------------------------- signals
with tab_sig:
    st.caption(f"Signals on the close of **{panel.dates[-1].date()}**, after the risk rules, "
               "for the paper book if there is one, otherwise for a fresh portfolio.")
    try:
        eng = Engine(panel, cfg.active, cfg.risk)
        bp = cfg.state_dir("paper") / "book.json"
        book = Book.load(bp) if bp.exists() else Book.new(cfg.risk.capital)
        t = panel.shape[0] - 1
        raw = []
        for i, spec in enumerate(eng.specs):
            for j in np.flatnonzero(eng.E[i, t]):
                raw.append({"strategy": spec.name, "symbol": panel.symbols[j], "close": panel.close[t, j],
                            "stop_distance": eng.SD[i, t, j], "score": eng.SC[i, t, j],
                            "regime_ok": bool(eng.regime[i, t])})
        if book.last_date != str(panel.dates[t].date()):  # book not yet stepped to this close: decide now
            try:
                pos, acct = book.to_arrays(panel, eng.strategy_names)
                eng.decide(pos, acct, t)
                book = Book.from_arrays(pos, acct, panel, eng.strategy_names, panel.dates[t])
            except KeyError as e:
                st.warning(str(e))
        held ={p["symbol"]: p["strategy"] for p in book.positions}
        ordered = {o["symbol"]: o["strategy"] for o in book.pending}
        per_strat = pd.Series([p["strategy"] for p in book.positions if not p.get("pending_exit")]).value_counts()
        open_after = sum(1 for p in book.positions if not p.get("pending_exit"))

        def status(row):
            spec = cfg.strategy(row["strategy"])
            if row["symbol"] in held:
                return f"already held ({held[row['symbol']]})"
            if ordered.get(row["symbol"]) == row["strategy"]:
                return "ORDER placed"
            if row["symbol"] in ordered:
                return f"taken by {ordered[row['symbol']]} (higher priority)"
            if not row["regime_ok"]:
                return "market regime off"
            if spec.max_positions and per_strat.get(spec.name, 0) >= spec.max_positions:
                return f"strategy cap reached ({spec.max_positions})"
            if open_after + len(ordered) >= cfg.risk.max_positions:
                return f"no free slot (max {cfg.risk.max_positions})"
            return "skipped (lower rank, cash or liquidity)"

        st.subheader("Entry signals and what the risk rules did with them")
        if raw:
            sig_df = pd.DataFrame(raw)
            sig_df["status"] = sig_df.apply(status, axis=1)
            st.dataframe(sig_df.sort_values(["strategy", "score"], ascending=[True, False]).round(2),
                         width="stretch", hide_index=True)
        else:
            st.caption("No strategy signals an entry on this close.")
        st.subheader("Orders for the next session")
        orders = plan_orders(book)
        if orders:
            st.dataframe(pd.DataFrame([o.__dict__ for o in orders]), width="stretch", hide_index=True)
        else:
            st.caption("No orders.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not compute signals: {e}")


# ----------------------------------------------------------- strategy file
with tab_file:
    st.caption("Edit the strategy file. **Validate** parses it and computes every strategy's signals on the "
               "loaded data; **Save** validates first and keeps a .bak copy.")
    text = st.text_area("strategies.toml", value=Path(file).read_text(), height=520, label_visibility="collapsed")

    def validate(src: str):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tf:
            tf.write(src)
        try:
            c = load_config(tf.name)
            rows = []
            for spec in c.strategies:
                sig = spec.strategy.generate(panel)
                rows.append({"strategy": spec.name, "enabled": spec.enabled, "quantity": str(spec.quantity),
                             "priority": spec.priority, "signals (all history)": int(sig.entry.sum()),
                             "signals on last close": int(sig.entry[-1].sum())})
            return c, pd.DataFrame(rows)
        finally:
            Path(tf.name).unlink(missing_ok=True)

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Validate"):
        try:
            _, rows = validate(text)
            st.success("Valid")
            st.dataframe(rows, width="stretch", hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Invalid: {e}")
    if b2.button("Save", type="primary"):
        try:
            validate(text)
            shutil.copy(file, file + ".bak")
            Path(file).write_text(text)
            st.cache_resource.clear()
            st.success(f"Saved {file} (previous version in {file}.bak)")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Not saved: {e}")
    with st.expander("Expression reference"):
        st.markdown("**Variables:** `open high low close volume` plus anything in `params`  \n"
                    "**Logic:** `and or not`, `< <= > >= == !=`, `+ - * / **`  \n"
                    "**Functions:** " + ", ".join(f"`{f}`" for f in FUNCTIONS) + "  \n"
                    "**Quantity:** `risk` | `risk 0.5%` | `equity 20%` | `shares 50` | `amount 100000`")


# ------------------------------------------------------------ architecture
ARCH_PAGE = Path(__file__).resolve().parents[2] / "docs" / "architecture.html"
with tab_arch:
    if ARCH_PAGE.exists():
        components.html(ARCH_PAGE.read_text(), height=2900, scrolling=True)
    else:
        st.info(f"Architecture page not found at {ARCH_PAGE}.")
