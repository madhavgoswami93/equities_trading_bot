"""Command line interface:  python -m swingbot <command> --help"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

from . import data as D
from .costs import CostModel
from .engine import Backtest, EngineConfig, sweep, walk_forward
from .strategies import REGISTRY, get_strategy


def _value(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _params(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = _value(v.strip())
    return out


def _grid(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = [_value(x) for x in v.split(",")]
    return out


def _config(a) -> EngineConfig:
    return EngineConfig(
        initial_cash=a.cash, risk_per_trade=a.risk, max_positions=a.max_positions,
        max_position_pct=a.max_position_pct, slippage_bps=a.slippage_bps, costs=CostModel.by_name(a.costs),
    )


def _panel(a):
    if a.provider == "synthetic":
        return D.synthetic_panel(a.synthetic_symbols, a.synthetic_days)
    symbols = D.load_universe(a.universe)
    bench_provider = a.benchmark_provider or a.provider
    bench = D.benchmark_for(bench_provider)
    panel = D.load_panel(a.provider, symbols, benchmark=bench, benchmark_provider=bench_provider)
    q = panel.quality_report()
    if len(q):
        logging.warning("data quality issues (likely un-adjusted corporate actions):\n%s", q.to_string(index=False))
    return panel


# ------------------------------------------------------------------ commands
def cmd_fetch(a):
    D.load_env()
    provider = D.get_provider(a.provider)
    symbols = D.load_universe(a.universe)
    failed = D.fetch_universe(provider, symbols, pd.Timestamp(a.start).date(),
                              pd.Timestamp(a.end).date() if a.end else date.today())
    print(f"fetched {len(symbols) - len(failed)}/{len(symbols)} symbols into data/cache/{provider.name}/")
    if failed:
        print("failed:", ", ".join(failed))


def cmd_backtest(a):
    panel = _panel(a)
    strat = get_strategy(a.strategy, **_params(a.param))
    t0 = time.perf_counter()
    res = Backtest(panel, strat, _config(a)).run(a.start, a.end)
    dt = time.perf_counter() - t0
    print(res.summary())
    print(f"\n{panel.shape[1]} symbols x {panel.shape[0]} days in {dt * 1000:.0f} ms (incl. signal generation)")
    if a.top:
        print("\nTop symbols by PnL:\n", res.by_symbol().head(a.top).round(3).to_string())
    if a.out:
        print("saved to", res.save(a.out))
        try:
            res.plot(f"{a.out}/equity.png")
        except ImportError:
            pass


def cmd_sweep(a):
    panel = _panel(a)
    grid = _grid(a.grid)
    if not grid:
        sys.exit("give at least one --grid key=v1,v2,...")
    strat = get_strategy(a.strategy, **_params(a.param))
    t0 = time.perf_counter()
    df = sweep(panel, strat, grid, _config(a), a.start, a.end, workers=a.workers, sort_by=a.metric)
    print(f"{len(df)} combinations in {time.perf_counter() - t0:.1f}s")
    cols = list(grid) + ["cagr", "sharpe", "max_drawdown", "trades", "win_rate", "profit_factor", "sqn"]
    print(df[[c for c in cols if c in df]].head(a.top or 20).round(3).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


def cmd_walkforward(a):
    panel = _panel(a)
    strat = get_strategy(a.strategy, **_params(a.param))
    wf = walk_forward(panel, strat, _grid(a.grid), _config(a), a.train_years, a.test_years, a.metric,
                      workers=a.workers)
    print(wf.summary())


def cmd_scan(a):
    """Entry candidates as of the last bar in the cache -> orders for tomorrow's open."""
    panel = _panel(a)
    strat = get_strategy(a.strategy, **_params(a.param))
    sig = strat.generate(panel)
    regime_ok = bool(strat.regime(panel)[-1])
    t = len(panel.dates) - 1
    idx = np.flatnonzero(sig.entry[t])
    rows = []
    for j in idx:
        c, sd = panel.close[t, j], sig.stop_dist[t, j]
        if not (sd > 0):
            continue
        qty = int(min(a.cash * a.risk / sd, a.cash * a.max_position_pct / c))
        rows.append(dict(symbol=panel.symbols[j], close=c, stop=c - sd, risk_per_share=sd, qty=qty,
                         score=sig.score[t, j] if sig.score is not None else np.nan))
    print(f"Signals for {panel.dates[t].date()} ({strat.name}); market regime {'ON' if regime_ok else 'OFF -- no new entries'}")
    if rows:
        df = pd.DataFrame(rows).sort_values("score", ascending=False).head(a.max_positions)
        print(df.round(2).to_string(index=False))
    else:
        print("no entry signals")


def cmd_kite_login(a):
    D.load_env()
    from .data.kite import interactive_login

    interactive_login()


def cmd_bench(a):
    """Engine speed on synthetic data."""
    panel = D.synthetic_panel(a.synthetic_symbols, a.synthetic_days)
    bt = Backtest(panel, get_strategy(a.strategy), EngineConfig())
    t0 = time.perf_counter()
    _ = bt.signals
    t1 = time.perf_counter()
    bt.run()  # includes JIT compile on first ever call (cached afterwards)
    t2 = time.perf_counter()
    n = 10
    for _ in range(n):
        res = bt.run()
    t3 = time.perf_counter()
    T, N = panel.shape
    print(f"panel {N} symbols x {T} days = {T * N / 1e6:.2f}M bars")
    print(f"signals: {(t1 - t0) * 1000:.0f} ms | first run (JIT/cache load): {(t2 - t1) * 1000:.0f} ms | "
          f"warm run: {(t3 - t2) / n * 1000:.1f} ms ({T * N / ((t3 - t2) / n) / 1e6:.0f}M bars/s) | "
          f"{res.stats.get('trades')} trades")


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="swingbot", description="Swing-trading backtester for NSE equities")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_strategy=True):
        p.add_argument("--provider", default="yahoo", choices=["kite", "groww", "yahoo", "synthetic"])
        p.add_argument("--benchmark-provider", default=None, help="where to read the Nifty index from (default: --provider)")
        p.add_argument("--universe", default="nifty50", help="name in data/universes/, a CSV path, or SYM1,SYM2")
        p.add_argument("--start", default=None)
        p.add_argument("--end", default=None)
        p.add_argument("--synthetic-symbols", type=int, default=100)
        p.add_argument("--synthetic-days", type=int, default=2520)
        if needs_strategy:
            p.add_argument("-s", "--strategy", default="pullback", choices=sorted(REGISTRY))
            p.add_argument("-p", "--param", action="append", help="strategy parameter key=value (repeatable)")
            p.add_argument("--cash", type=float, default=1_000_000)
            p.add_argument("--risk", type=float, default=0.01, help="equity fraction risked per trade")
            p.add_argument("--max-positions", type=int, default=10)
            p.add_argument("--max-position-pct", type=float, default=0.20)
            p.add_argument("--slippage-bps", type=float, default=10.0)
            p.add_argument("--costs", default="zerodha", choices=["zerodha", "groww", "free"])

    p = sub.add_parser("fetch", help="download / update daily candles into the local cache")
    common(p, needs_strategy=False)
    p.set_defaults(func=cmd_fetch, start="2005-01-01")

    p = sub.add_parser("backtest", help="run one backtest")
    common(p)
    p.add_argument("--top", type=int, default=10, help="show N best symbols")
    p.add_argument("--out", default=None, help="directory for equity.csv, trades.csv, stats.json, equity.png")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("sweep", help="grid search over strategy parameters (parallel)")
    common(p)
    p.add_argument("-g", "--grid", action="append", help="key=v1,v2,v3 (repeatable)")
    p.add_argument("--metric", default="sharpe")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default=None, help="CSV path for all results")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("walkforward", help="rolling walk-forward optimisation")
    common(p)
    p.add_argument("-g", "--grid", action="append")
    p.add_argument("--metric", default="sharpe")
    p.add_argument("--train-years", type=float, default=3)
    p.add_argument("--test-years", type=float, default=1)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("scan", help="today's entry candidates with stop and size")
    common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("kite-login", help="generate today's Kite access token")
    p.set_defaults(func=cmd_kite_login)

    p = sub.add_parser("bench", help="engine speed benchmark on synthetic data")
    p.add_argument("-s", "--strategy", default="breakout", choices=sorted(REGISTRY))
    p.add_argument("--synthetic-symbols", type=int, default=500)
    p.add_argument("--synthetic-days", type=int, default=5000)
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd == "fetch" else logging.WARNING,
                        format="%(levelname)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
