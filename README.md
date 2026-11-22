# equities_trading_bot — `swingbot`

A swing-trading research and backtesting engine for NSE equities, built for
holding periods from **1 day to about 3 weeks**. It is fast and designed so that
future data cannot leak into a backtest.

* **Fast**: 500 stocks × 20 years (2.5M bars) simulates in about 45 ms after the
  first compile. A 24-combination parameter sweep on Nifty 50 takes under a second.
* **Honest fills**: a signal at the close is filled at the next open. Stops that
  gap through fill at the open, not at the stop price. When the stop and the
  target are both hit on one bar, the engine assumes the stop came first.
* **Real Indian costs**: STT, exchange, SEBI, stamp duty, GST and DP charges, with
  presets for Zerodha and Groww, plus slippage.
* **Portfolio-level**: risk-based position sizing, a maximum number of positions,
  signal ranking when there are more signals than free slots, a per-position
  cap, a liquidity cap (% of average daily value) and a Nifty regime filter.
* **Research tooling**: parallel parameter sweeps, rolling walk-forward
  optimisation, per-symbol breakdowns, monthly returns, and CSV/JSON/PNG output.
* **Data**: Zerodha Kite Connect, Groww Trade API and Yahoo Finance (free). Data is
  cached locally as Parquet and updated incrementally.

## Broker APIs (status as of Nov 2026)

| | Zerodha Kite Connect | Groww Trade API |
|---|---|---|
| Orders / portfolio | Free "Personal" plan | Paid plan |
| Historical candles | "Connect" plan, about ₹500/month per app (historical data included) | Paid plan, about ₹499/month + GST |
| Daily history depth | Many years (2000 days per request, chunked automatically) | Equities from **2020** (180 days per request on `get_historical_candles`) |
| Python SDK | `kiteconnect` | `growwapi` |
| Auth | Daily access token (`swingbot kite-login`) | TOTP or API key + secret (daily approval) |
| Notes | Static IP required for API orders since Apr 2026 | Rate limits per category (orders 10/s) |

**Recommendation:** use **Yahoo** (free) to research strategies with 15+ years of history. Use
**Kite** as the authoritative data and the broker for live trading. Groww's history
only starts in 2020, which is too short for robust swing backtests, but works as a
second data source or broker. Pricing changes often, so check
[zerodha.com/charges](https://zerodha.com/charges) and the
[Groww Trade API docs](https://groww.in/trade-api/docs) before you subscribe.

## Quick start

```bash
pip install -e ".[yahoo,plot,dev]"      # add kite / groww extras when needed

python -m swingbot fetch --provider yahoo --universe nifty50 --start 2008-01-01
python -m swingbot backtest -s pullback --start 2010-01-01 --out results/pullback
python -m swingbot backtest -s breakout -p lookback=55 -p atr_trail=4 --costs groww
python -m swingbot sweep -s pullback -g rsi_entry=5,10,15 -g max_hold=5,10,15 -g atr_stop=2,3
python -m swingbot walkforward -s pullback -g rsi_entry=5,10,20 -g max_hold=5,10,15
python -m swingbot scan -s breakout       # today's candidates with stop and quantity
python -m swingbot bench                  # engine speed on synthetic data
pytest -q
```

With Kite: copy `.env.example` to `.env`, fill in your key and secret, run
`python -m swingbot kite-login` each morning, then use `--provider kite`. With Groww,
use `--provider groww --benchmark-provider yahoo`, because Groww has no Nifty index
candles in this adapter.

### Python API

```python
from swingbot import Backtest, EngineConfig, CostModel, TrendPullback
from swingbot.data import load_panel, load_universe

panel = load_panel("yahoo", load_universe("nifty50"), benchmark="^NSEI")
cfg = EngineConfig(initial_cash=10_00_000, risk_per_trade=0.01, max_positions=8, costs=CostModel.zerodha())
res = Backtest(panel, TrendPullback(rsi_entry=5, max_hold=15), cfg).run(start="2012-01-01")
print(res.summary())
res.trades.head(); res.by_symbol(); res.monthly_returns(); res.plot()
```

## Architecture

```
swingbot/
  panel.py            Panel: universe as aligned (dates x symbols) numpy arrays + data-quality report
  indicators.py       vectorised, causal indicators (SMA/EMA/ATR/RSI/ROC/Donchian/Supertrend)
  costs.py            CostModel (Zerodha / Groww / free) + jitted order_cost
  strategies/         a strategy only produces Signals: entry, exit, stop distance, trail, score, holds
    pullback.py       buy RSI(3) dips in long-term up-trends
    breakout.py       20-day high breakout on 1.5x volume, ATR trailing stop
    supertrend.py     daily Supertrend flips (port of legacy/kc_supertrend.py)
  engine/
    kernel.py         the numba simulation loop (exact per-bar order of events is documented there)
    backtest.py       EngineConfig + Backtest (signals once, simulate any window)
    metrics.py        CAGR, Sharpe, Sortino, Calmar, max DD and duration, win rate, PF, expectancy (R), SQN
    result.py         Result: equity, trades, by_symbol, monthly_returns, summary, save, plot
    optimize.py       parallel grid sweep + rolling walk-forward
  data/               providers (kite, groww, yahoo, synthetic), Parquet store, universes
  cli.py              python -m swingbot ...
tests/                engine correctness (fills, gaps, stops, costs, look-ahead, numba == pure Python)
legacy/               the previous scripts, kept for reference
```

**Why this design is fast:** strategies compute signals for the whole universe in a
few vectorised pandas calls. The only loop that has to be a loop, the portfolio
simulation, is one numba-compiled pass over contiguous arrays with no Python
objects per trade. Signals are computed once and reused for every window, so
walk-forward runs never recompute indicators. Without numba, the same code runs
as pure Python (`SWINGBOT_NO_JIT=1`), and a test checks that both give identical
results.

**Adding a strategy:** subclass `Strategy`, add dataclass fields as parameters,
implement `generate(panel) -> Signals`, and register it in `strategies/__init__.py`.
Sweeps, walk-forward and the CLI then work with it automatically.

## Ideas taken from `fullstack-trading-app`

* A universe file of Nifty 50 constituents (`data/universes/nifty50.csv`).
* A per-stock results table (`Result.by_symbol()`, which replaces `result_df.csv`) and SQN,
  drawdown and Sharpe analytics. These are now computed from our own trade list, not
  backtrader analyzers.
* A local candle store: Parquet files instead of Postgres. There is no database server to run
  or maintain.
* Commission/sizer settings became `EngineConfig` plus a real Indian `CostModel`
  instead of a flat 0.01%.

Left out: backtrader (too slow for portfolio sweeps), hard-coded Windows paths,
SQL built with f-strings, and the intraday opening-range-breakout logic, which
does not fit a 1-day to 3-week holding engine.

## Things to know before trusting a backtest

1. **Survivorship bias.** `nifty50.csv` is a 2020 snapshot. It leaves out stocks that
   were dropped from the index, and it includes HDFC, which later merged into HDFC
   Bank. For serious work, use point-in-time index membership.
2. **Corporate actions.** The engine needs split- and bonus-adjusted prices.
   `Panel.quality_report()` (run by the CLI automatically) flags overnight gaps
   above 35%, which usually means an unadjusted split.
3. **Costs matter for swing trading in India.** Delivery STT is 0.1% on each side,
   so with slippage a round trip costs about 0.3–0.5%. On the sample run, costs
   were around 60% of starting capital over 2,000 trades. Strategies need a real
   edge per trade, not just a high win rate.
4. **Out-of-sample results are what count.** The best in-sample sweep (Sharpe 0.45) fell to
   about 0.06 Sharpe out of sample in walk-forward. Use `walkforward`, not
   `sweep`, to judge a strategy.

## Roadmap

* Live execution adapter: place CNC orders plus GTT stop-loss on Kite or Groww from `scan` output
* Point-in-time Nifty 200/500 membership
* Monte Carlo trade reshuffling and bootstrap confidence intervals
* Sector exposure caps and correlation-aware ranking
