# equities_trading_bot — `swingbot`

Swing trading for NSE equities, with holds from 1 day to about 3 weeks. **One strategy file, one engine,
three modes:**

```
strategies.toml ──► backtest ──► paper (live data, simulated fills) ──► live (real orders)
                      ▲              ▲                                     ▲
                      └──────── the same jitted engine core ───────────────┘
```

* **Strategies live in a TOML file.** Write signals and quantities as expressions such as
  `entry = "close > sma(close, 200) and rsi(close, 3) < 5"` and `quantity = "risk 1%"`. You can
  run any number of strategies together as one portfolio.
* **Risk rules apply to every strategy.** By default you hold at most **5 open positions**,
  no single position can exceed **40% of equity**, and trades are sized by risk per trade. A
  liquidity cap, an entry-gap limit, a drawdown circuit breaker, and per-strategy caps and
  priorities are also available.
* **Paper trading matches the backtest.** Paper trading runs the same engine one day at a
  time on live data. A test replays 500 sessions this way and gets the same trades as the
  backtest.
* **Live trading** uses Zerodha Kite. Entries are AMO limit orders at the next open, exits are
  AMO market orders, and stops are GTT orders. It starts in dry-run mode.
* **Speed:** 500 stocks × 20 years × 3 strategies simulates in about 30 ms after the first compile.
  Parameter sweeps run in parallel.

## Cheapest setup: ₹0 per month

Daily-bar swing trading only needs end-of-day candles, which are free:

| Need | Free option | Paid alternative |
|---|---|---|
| Research and backtests | **Yahoo Finance** (`provider = "yahoo"`): 15+ years, split-adjusted | Kite Connect (about ₹500/month) |
| Paper trading | **Yahoo** EOD after 15:45 IST | — |
| Live orders | **Zerodha Kite "Personal" API**: orders, GTT and portfolio endpoints are free | — |
| Free historical data APIs | **Angel One SmartAPI** and **Fyers API** are free for account holders | Groww (about ₹499/month, data only from 2020), Dhan data (about ₹499/month) |

**Recommended:** Yahoo for data plus Kite Personal for orders. Zerodha delivery brokerage is
₹0, and the API costs ₹0. Kite has required a static IP for API orders since April 2026. Broker
pricing changes often, so check each broker's own page before you subscribe.

## Quick start

```bash
pip install -e ".[yahoo,plot,dev]"            # Python 3.11+; add [kite] for live trading
python -m swingbot init strategies.toml       # commented template
python -m swingbot fetch strategies.toml      # download / refresh daily candles (cached as Parquet)
python -m swingbot backtest strategies.toml   # all enabled strategies as one portfolio
```

Then promote the same file through the stages:

```bash
# 1. research
python -m swingbot backtest    strategies.toml --start 2012-01-01
python -m swingbot sweep       strategies.toml --strategy trend_dip -g rsi_entry=3,5,10 -g max_hold=10,15
python -m swingbot walkforward strategies.toml --strategy trend_dip -g rsi_entry=3,5,10 -g max_hold=10,15

# 2. paper: run once a day after the close (cron 16:15 IST, Mon-Fri)
python -m swingbot paper strategies.toml                 # updates data, fills, decides tomorrow's orders
python -m swingbot paper strategies.toml --start 2026-06-01 --reset   # replay paper from a date
python -m swingbot status strategies.toml

# 3. live (Zerodha Kite): dry run until you set `dry_run = false` AND pass --send
python -m swingbot kite-login                            # each morning: today's access token
python -m swingbot live strategies.toml --phase open --send    # ~09:20: sync fills, place GTT stops
python -m swingbot live strategies.toml --send                 # ~15:45: sync, decide, send AMOs for tomorrow
```

Example cron entries (server clock in IST):

```
15 16 * * 1-5  cd ~/equities_trading_bot && python -m swingbot paper strategies.toml >> runs/paper.log 2>&1
```

## Dashboard

```bash
pip install -e ".[dashboard]"
python -m swingbot dashboard strategies.toml        # http://localhost:8501
```

| Tab | What it shows |
|---|---|
| **Backtest** | Pick strategies, choose a date range and optionally override the risk rules for this run. Shows key stats, equity vs Nifty 50, drawdown, realised PnL by strategy, exit reasons, a monthly-returns heatmap, results by stock, and a filterable trade list you can export as CSV |
| **Paper trading** | Current positions with their weight against the 40% cap, distance to stop, orders for the next session, the equity curve and closed trades. A button runs today's paper step |
| **Live** | Read-only view of the live book. Orders are only ever sent from the CLI |
| **Signals today** | Every entry signal on the latest close and what the risk rules did with it: ordered, already held, strategy cap reached, no free slot, or regime off |
| **Strategy file** | Edit the TOML in the browser. **Validate** computes every strategy's signals; **Save** validates first and keeps a `.bak` copy |

Light and dark themes are both supported.

## The strategy file

```toml
[risk]
capital = 1000000
max_positions = 5          # never more than 5 open positions, across all strategies
max_position_pct = 0.40    # no single position above 40% of equity
risk_per_trade = 0.01
max_entry_gap = 0.05       # entry is a limit at close +5%
max_drawdown_halt = 0.15   # pause new entries 20 sessions when 15% below peak
costs = "zerodha"

[[strategy]]
name = "trend_dip"
type = "rules"
entry = "close > sma(close, 200) and sma(close, 50) > sma(close, 200) and rsi(close, 3) < rsi_entry"
exit  = "rsi(close, 3) > 70"
stop  = "3 * atr(14)"           # initial stop distance (Rs) below the fill
score = "roc(close, 126)"       # ranking when signals exceed free slots
quantity = "risk 1%"            # or "equity 20%" | "shares 50" | "amount 100000"
max_hold = 15                   # time stop, about 3 weeks
priority = 1
params = { rsi_entry = 5 }      # names usable in expressions, and sweepable

[[strategy]]
name = "volume_breakout"
type = "rules"
entry = "close > prev(highest(high, 20)) and volume > 1.5 * avg_volume(20) and close > sma(close, 50)"
stop  = "2 * atr(14)"
trail = "3 * atr(14)"
quantity = "equity 20%"
priority = 2
max_positions = 2               # at most 2 of the 5 slots
```

See [`strategies.example.toml`](strategies.example.toml) for every option. Expressions are
parsed with an allow-list, so a strategy file cannot run arbitrary code.
Available functions: `sma ema rsi roc atr highest lowest prev avg_volume turnover supertrend
supertrend_dir crossover crossunder max min abs`. Built-in Python strategies (`pullback`,
`breakout`, `supertrend`) can also be used with `type = "..."`.

### How a quantity becomes an order

1. The strategy's `quantity` sets the requested share count, calculated at the signal close.
2. The risk rules cap it: no more than `max_position_pct` of equity at the worst-case fill
   price, no more than `max_adv_pct` of average daily traded value, and no more than the
   available cash (including cash from exits that day).
3. Slots: when more stocks signal than there are free slots under `max_positions`, they are
   filled by `priority`, then `score`. A symbol is never held by two strategies at once.

## Architecture

```
swingbot/
  config.py           strategy file -> RunConfig (RiskConfig, Quantity, StrategySpec, DataConfig, LiveConfig)
  runner.py           Runner.backtest / .paper / .live / .status  <- the single entry point
  engine/
    kernel.py         jitted core: on_open -> on_intraday -> on_close (risk rules + sizing), resumable state
    backtest.py       Engine (signals of all strategies stacked) + Backtest
    book.py           persistent portfolio state (JSON) for paper/live, trade & equity ledgers
    result.py         stats, by_strategy, by_symbol, monthly returns, plots
    metrics.py        CAGR, Sharpe, Sortino, Calmar, max DD, win rate, PF, expectancy (R), SQN
    optimize.py       parallel sweep + walk-forward for one strategy of the file
  strategies/
    expr.py           safe expression language
    rules.py          RuleStrategy (strategies written in the file)
    pullback.py, breakout.py, supertrend.py   built-ins
  brokers/
    base.py           plan_orders: book -> BUY / SELL / STOP orders (same for paper and live)
    kite.py           Zerodha: AMO entries/exits, GTT stops, fill sync, holdings reconciliation
  dashboard/          Streamlit app (app.py) + Plotly charts (charts.py)
  data/               yahoo, kite, groww, synthetic providers + incremental Parquet cache
  panel.py            the universe as aligned (dates x symbols) arrays + data-quality report
  costs.py            STT, exchange, SEBI, stamp, GST, DP (Zerodha / Groww presets)
runs/<name>/{backtest,paper,live}/   results, book.json, trades.csv, equity.csv
```

**Why paper and live behave like the backtest.** All portfolio state is held in two arrays,
`pos` and `acct`. The kernel advances that state one day at a time:

- **Backtest** starts from empty state and advances over the whole history.
- **Paper** loads yesterday's state from `book.json`, advances over the new bars, and saves it again.
- **Live** replaces the simulated fills with the broker's real fills, then runs the same
  `on_close` decision.

The order rules match the simulation:

- Entries are limit orders at `close × (1 + max_entry_gap)`. A larger gap-up doesn't fill, the
  same as in the backtest.
- Stops are GTT orders at the engine's stop price, which trails upward as the position gains.

## Things to know

1. **Survivorship bias.** `nifty50.csv` is a 2020 snapshot of the index. For serious work,
   use historical (point-in-time) index membership.
2. **Corporate actions.** Use split- and bonus-adjusted data. The quality report flags gaps
   larger than 35%.
3. **Costs.** Delivery STT is 0.1% on each side. A round trip including slippage costs about
   0.3–0.5%, which is a high bar for short swing trades.
4. **Out-of-sample results.** Judge strategies with `walkforward`, not by the best result from `sweep`.
5. **Live differences.** Live fills, partial fills, and GTT stops that fill at their limit price
   will differ slightly from the simulation. Book and broker holdings are reconciled every run,
   and mismatches are logged as warnings.
6. **Capital.** `capital` is the amount allocated to the bot, not your whole account balance.

## Ideas taken from `fullstack-trading-app`

The Nifty 50 universe file, per-stock result tables (`by_symbol`) and SQN, drawdown and Sharpe
analytics, and a local candle store (Parquet instead of Postgres). Its flat 0.01% commission
was replaced with the full set of Indian delivery charges.
