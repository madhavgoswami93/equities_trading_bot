"""Command line:  python -m swingbot <command> <strategy-file> [options]

Every command reads the same strategy file, so the path from idea to live is:

    swingbot backtest strategies.toml      # research
    swingbot paper    strategies.toml      # run daily after the close on live data
    swingbot live     strategies.toml      # same decisions, sent to the broker
"""

from __future__ import annotations

import argparse
import ast
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .config import load_config
from .runner import Runner

EXAMPLE = Path(__file__).resolve().parents[1] / "strategies.example.toml"


def _value(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _grid(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = [_value(x) for x in v.split(",")]
    if not out:
        sys.exit("give at least one --grid key=v1,v2,...")
    return out


def _runner(a) -> Runner:
    cfg = load_config(a.file)
    return Runner(cfg)


# ------------------------------------------------------------------ commands
def cmd_init(a):
    dst = Path(a.file)
    if dst.exists():
        sys.exit(f"{dst} already exists")
    shutil.copy(EXAMPLE, dst)
    print(f"wrote {dst} -- edit it, then: python -m swingbot fetch {dst} && python -m swingbot backtest {dst}")


def cmd_fetch(a):
    r = _runner(a)
    logging.getLogger("swingbot.data.store").setLevel(logging.INFO)  # progress + ETA for long downloads
    if not logging.getLogger().handlers or logging.getLogger().level > logging.INFO:
        logging.getLogger().setLevel(logging.INFO)
        for h in logging.getLogger().handlers:
            h.setLevel(logging.INFO)
    intraday = True if a.intraday else (False if a.no_intraday else None)
    failed = r.update_data(full=a.full, intraday=intraday, daily=not a.intraday_only)
    db = r.db()
    cov = db.coverage()
    print(f"\n{len(cov)} symbols (daily) in {r.cfg.data.db}, latest session {cov['last'].max() if len(cov) else '-'}")
    ic = db.intraday_coverage()
    if len(ic):
        print(ic.to_string(index=False))
    if failed:
        print(f"{len(failed)} failed (renamed, delisted or no data): {', '.join(failed[:30])}"
              f"{' ...' if len(failed) > 30 else ''}  -- details: swingbot db FILE log")


def cmd_universe(a):
    from .data.universe import build_nse_universe

    u = build_nse_universe(a.out)
    print(f"wrote {a.out}: {len(u)} NSE stocks ({', '.join(f'{k} {v}' for k, v in u.cap.value_counts().items())})")
    print('use it with  [data] universe = "nse_all"  in the strategy file')


def cmd_db(a):
    r = _runner(a)
    db = r.db()
    if a.action == "status":
        cov = db.coverage()
        if cov.empty:
            print(f"{db.path} is empty -- run: python -m swingbot fetch {a.file}")
            return
        print(cov.to_string(index=False))
        stale = cov[cov["last"] < cov["last"].max()]
        print(f"\n{len(cov)} symbols, {int(cov.bars.sum()):,} bars, {db.path} "
              f"({db.path.stat().st_size / 1e6:.1f} MB)")
        if len(stale):
            print(f"behind the latest session: {', '.join(stale.symbol)}")
    elif a.action == "log":
        print(db.sql("select * from fetch_log order by logged_at desc limit 50").to_string(index=False))
    elif a.action == "sql":
        if not a.query:
            sys.exit('give a query: swingbot db FILE sql "select ..."')
        print(db.sql(a.query).to_string(index=False))
    elif a.action == "import-parquet":
        from .data import import_parquet

        print(f"imported {import_parquet(db, a.query or 'data/cache'):,} bars into {db.path}")


def cmd_backtest(a):
    r = _runner(a)
    t0 = time.perf_counter()
    res = r.backtest(a.start, a.end, only=a.strategy)
    dt = time.perf_counter() - t0
    risk = r.cfg.risk
    print(f"Risk: max {risk.max_positions} open positions, none above {risk.max_position_pct:.0%} of equity, "
          f"capital {risk.capital:,.0f}\n")
    print(res.summary())
    print(f"\n{r.panel().shape[1]} symbols x {r.panel().shape[0]} days in {dt * 1000:.0f} ms (incl. signals)")
    if a.top:
        print("\nTop symbols by PnL:\n" + res.by_symbol().head(a.top).round(3).to_string())
    out = a.out or (Path(r.cfg.runs_dir) / r.cfg.name / "backtest")
    res.save(out)
    try:
        res.plot(Path(out) / "equity.png")
    except ImportError:
        pass
    print(f"\nsaved equity.csv, trades.csv, by_strategy.csv, by_symbol.csv, stats.json -> {out}")


def _log_to_file(r, mode: str) -> None:
    """Append INFO-level logs to runs/<name>/<mode>/swingbot.log; the console keeps its level."""
    d = r.cfg.state_dir(mode)
    d.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for hd in root.handlers:
        hd.setLevel(root.level)
    fh = logging.FileHandler(d / "swingbot.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.setLevel(min(root.level, logging.INFO))


def _run_and_alert(r, mode: str, fn):
    from .data import load_env
    from .notify import telegram

    load_env()
    _log_to_file(r, mode)
    try:
        rep = fn()
    except Exception as e:
        logging.getLogger("swingbot").exception("%s run failed", mode)
        if r.cfg.alerts.telegram:
            telegram(f"swingbot {r.cfg.name} [{mode}] FAILED: {type(e).__name__}: {e}")
        raise
    print(rep)
    if r.cfg.alerts.telegram:
        telegram(f"swingbot {r.cfg.name}\n{rep}")


def cmd_paper(a):
    r = _runner(a)
    _run_and_alert(r, "paper", lambda: r.paper(update=not a.no_update, start=a.start, reset=a.reset))


def cmd_live(a):
    r = _runner(a)
    _run_and_alert(r, "live", lambda: r.live(phase=a.phase, update=not a.no_update, send=a.send, force=a.force))


def cmd_halt(a):
    r = _runner(a)
    f = r.halt_file()
    if a.off:
        f.unlink(missing_ok=True)
        print(f"resumed: new entries allowed ({f} removed)")
    else:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("new entries blocked; remove with: swingbot halt FILE --off\n")
        print("HALTED: live runs will send no new entries (exits and stops still run). Undo: --off")


def cmd_status(a):
    print(_runner(a).status(a.mode))


def cmd_sweep(a):
    from .engine.optimize import sweep

    r = _runner(a)
    spec = r.cfg.strategy(a.strategy)
    grid = _grid(a.grid)
    t0 = time.perf_counter()
    df = sweep(r.panel(), spec, grid, r.cfg.risk, a.start, a.end, workers=a.workers, sort_by=a.metric)
    print(f"{len(df)} combinations of {spec.name!r} in {time.perf_counter() - t0:.1f}s")
    cols = list(grid) + ["cagr", "sharpe", "max_drawdown", "trades", "win_rate", "profit_factor", "sqn"]
    print(df[[c for c in cols if c in df]].head(a.top).round(3).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


def cmd_walkforward(a):
    from .engine.optimize import walk_forward

    r = _runner(a)
    wf = walk_forward(r.panel(), r.cfg.strategy(a.strategy), _grid(a.grid), r.cfg.risk, a.train_years,
                      a.test_years, a.metric, workers=a.workers)
    print(wf.summary())


def cmd_dashboard(a):
    import subprocess

    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    # bound to this machine only: the dashboard can run paper steps and edit the strategy file,
    # so it must not be reachable from the Wi-Fi network unless asked for with --host 0.0.0.0
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
           "--server.address", a.host, "--server.headless", "true", "--browser.gatherUsageStats", "false",
           "--", "--file", a.file]
    print(f"dashboard on http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}  (Ctrl+C to stop)")
    sys.exit(subprocess.call(cmd))


def cmd_docs(a):
    import functools
    import http.server

    root = Path(__file__).resolve().parents[1] / "docs"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler) as srv:
        print(f"architecture page on http://127.0.0.1:{a.port}/architecture.html  (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


def cmd_research(a):
    from . import research as R
    from .data import UNIVERSE_DIR
    from .data.universe import universe_meta

    r = _runner(a)
    panel = r.panel()
    specs = [r.cfg.strategy(n) for n in a.strategy] if a.strategy else r.cfg.active
    upath = Path(r.cfg.data.universe)
    upath = upath if upath.exists() else UNIVERSE_DIR / f"{r.cfg.data.universe}.csv"
    meta = universe_meta(upath) if upath.exists() else None
    t0 = time.perf_counter()
    study = R.signal_study(panel, specs, r.cfg.risk, a.start, a.end, meta)
    board = R.leaderboard(panel, specs, r.cfg.risk, study, a.start, a.end)
    out = Path(r.cfg.runs_dir) / r.cfg.name / "research"
    out.mkdir(parents=True, exist_ok=True)
    study.to_parquet(out / "signal_trades.parquet", index=False)
    board.to_csv(out / "leaderboard.csv", index=False)
    print(f"{panel.shape[1]} stocks, {len(specs)} strategies, {len(study):,} signal trades in "
          f"{time.perf_counter() - t0:.0f}s  ({a.start or panel.dates[0].date()} -> {a.end or panel.dates[-1].date()})\n")
    cols = {"strategy": "strategy", "signal_trades": "signals", "signal_win_rate": "win%",
            "signal_avg_return": "avg ret", "signal_expectancy_r": "exp R", "signal_profit_factor": "PF",
            "signal_avg_hold": "hold", "portfolio_return": "port ret", "portfolio_sharpe": "port Sharpe",
            "portfolio_max_dd": "port maxDD"}
    show = board[list(cols)].rename(columns=cols)
    for c in ("win%", "avg ret", "port ret", "port maxDD"):
        show[c] = show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a")
    print("Leaderboard (signal study: every signal, Rs 1 lakh each | portfolio: your [risk] rules)")
    print(show.round(2).to_string(index=False))
    for f in a.by:
        if f not in study:
            print(f"\n(no factor {f!r}; choose from {', '.join(R.FACTORS)})")
            continue
        tb = R.bucket_table(study, f)
        tb.to_csv(out / f"by_{f}.csv", index=False)
        print(f"\nAll strategies by {R.FACTORS.get(f, f)}")
        print(_fmt_buckets(tb).to_string(index=False))
    print(f"\nsaved to {out}/  -- explore every factor in the dashboard's Research tab")


def _fmt_buckets(tb):
    tb = tb.copy()
    for c in ("win_rate", "avg_return", "median_return"):
        if c in tb:
            tb[c] = tb[c].map(lambda v: f"{v:.1%}")
    for c in ("expectancy_r", "profit_factor", "avg_hold"):
        if c in tb:
            tb[c] = tb[c].map(lambda v: f"{v:.2f}")
    if "pnl_per_lakh" in tb:
        tb["pnl_per_lakh"] = tb["pnl_per_lakh"].map(lambda v: f"{v:,.0f}")
    return tb


def cmd_kite_login(a):
    from .data import load_env
    from .data.kite import interactive_login

    load_env()
    interactive_login()


def cmd_bench(a):
    from .config import RiskConfig
    from .data import synthetic_panel
    from .engine.backtest import Backtest
    from .strategies import REGISTRY

    panel = synthetic_panel(a.symbols, a.days)
    strategies = [REGISTRY[n]() for n in ("pullback", "breakout", "supertrend")]
    bt = Backtest(panel, strategies, RiskConfig(max_positions=5, max_position_pct=0.40))
    t0 = time.perf_counter()
    _ = bt.engine
    t1 = time.perf_counter()
    bt.run()
    t2 = time.perf_counter()
    for _ in range(5):
        res = bt.run()
    t3 = time.perf_counter()
    T, N = panel.shape
    print(f"{N} symbols x {T} days x 3 strategies | signals {(t1 - t0) * 1000:.0f} ms | first run "
          f"{(t2 - t1) * 1000:.0f} ms | warm run {(t3 - t2) / 5 * 1000:.1f} ms | {res.stats.get('trades')} trades")


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="swingbot", description="Swing-trading engine for NSE: backtest -> paper -> live")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, func, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("file", help="strategy file (TOML)")
        p.set_defaults(func=func)
        return p

    cmd("init", cmd_init, "write a commented example strategy file")
    p = cmd("fetch", cmd_fetch, "download new candles into the local database (resumable)")
    p.add_argument("--full", action="store_true", help="re-download full daily history for every symbol")
    p.add_argument("--intraday", action="store_true", help="also download intraday candles (default 5minute)")
    p.add_argument("--no-intraday", action="store_true", help="skip intraday even if [data] intraday is set")
    p.add_argument("--intraday-only", action="store_true", help="only the intraday candles")

    p = sub.add_parser("universe", help="build the all-NSE stock list with cap class and industry")
    p.add_argument("action", choices=["build"])
    p.add_argument("--out", default="data/universes/nse_all.csv")
    p.set_defaults(func=cmd_universe)

    p = cmd("db", cmd_db, "inspect the local market database")
    p.add_argument("action", choices=["status", "log", "sql", "import-parquet"])
    p.add_argument("query", nargs="?", help="SQL for `sql`; old cache folder for `import-parquet`")

    p = cmd("backtest", cmd_backtest, "backtest all enabled strategies as one portfolio")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--strategy", help="backtest only this strategy from the file")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out")

    p = cmd("paper", cmd_paper, "paper trade on live data (run daily after the close)")
    p.add_argument("--start", help="first run only: replay paper trading from this date")
    p.add_argument("--reset", action="store_true", help="discard the paper book and start over")
    p.add_argument("--no-update", action="store_true", help="do not download new data first")

    p = cmd("live", cmd_live, "live trading via the broker in [live] (dry run unless --send)")
    p.add_argument("--phase", choices=["open", "close"], default="close")
    p.add_argument("--send", action="store_true", help="really send orders (also needs dry_run = false)")
    p.add_argument("--force", action="store_true", help="skip the safety checks (day already processed, stale data)")
    p.add_argument("--no-update", action="store_true")

    p = cmd("halt", cmd_halt, "kill switch: block new live entries (exits and stops still run)")
    p.add_argument("--off", action="store_true", help="lift the halt")

    p = cmd("status", cmd_status, "show the paper or live book")
    p.add_argument("--mode", choices=["paper", "live"], default="paper")

    for name, func in (("sweep", cmd_sweep), ("walkforward", cmd_walkforward)):
        p = cmd(name, func, f"{name} parameters of one strategy in the file")
        p.add_argument("--strategy", required=True)
        p.add_argument("-g", "--grid", action="append", help="param=v1,v2,v3 (repeatable)")
        p.add_argument("--metric", default="sharpe")
        p.add_argument("--workers", type=int)
        if name == "sweep":
            p.add_argument("--start")
            p.add_argument("--end")
            p.add_argument("--top", type=int, default=20)
            p.add_argument("--out")
        else:
            p.add_argument("--train-years", type=float, default=3)
            p.add_argument("--test-years", type=float, default=1)

    p = cmd("dashboard", cmd_dashboard, "open the web dashboard (needs the [dashboard] extra)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 exposes it to your network (not recommended)")

    p = sub.add_parser("docs", help="serve the architecture page locally")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_docs)

    p = cmd("research", cmd_research, "test every strategy on every stock; split results by stock type")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--strategy", action="append", help="only these strategies (repeatable)")
    p.add_argument("--by", nargs="*", default=["cap", "price", "turnover"],
                   help="factors to print: price volume turnover cap industry volatility momentum trend market "
                        "weekday month")

    p = sub.add_parser("kite-login", help="generate today's Kite access token")
    p.set_defaults(func=cmd_kite_login)
    p = sub.add_parser("bench", help="engine speed on synthetic data")
    p.add_argument("--symbols", type=int, default=500)
    p.add_argument("--days", type=int, default=5000)
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        a.func(a)
    except (RuntimeError, ValueError, KeyError, FileNotFoundError, NotImplementedError) as e:
        if a.verbose:
            raise
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
