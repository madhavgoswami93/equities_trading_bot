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
    r.update_data()
    print(f"data cache updated in {r.cfg.data.cache_dir}/{r.cfg.data.provider}/")


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


def cmd_paper(a):
    r = _runner(a)
    print(r.paper(update=not a.no_update, start=a.start, reset=a.reset))


def cmd_live(a):
    r = _runner(a)
    print(r.live(phase=a.phase, update=not a.no_update, send=a.send, force=a.force))


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
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
           "--server.headless", "true", "--", "--file", a.file]
    print(f"dashboard on http://localhost:{a.port}  (Ctrl+C to stop)")
    sys.exit(subprocess.call(cmd))


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
    cmd("fetch", cmd_fetch, "download / update daily candles for the file's universe")

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
    p.add_argument("--force", action="store_true", help="re-run a day that was already processed")
    p.add_argument("--no-update", action="store_true")

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
