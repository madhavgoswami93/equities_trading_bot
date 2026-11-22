"""The simulation kernel: one numba-compiled pass over the bars.

Timeline for every trading day t (this ordering is what prevents look-ahead):

  OPEN   1. fill exits scheduled at the previous close (signal / time stop)
         2. fill entries ranked at the previous close, sized off yesterday's
            equity, at today's open (+ slippage), skipping excessive gap-ups
  INTRA  3. protective stop and profit target, against today's low/high.
            A gap through the level fills at the open, not at the level.
            If stop and target are both touched the stop wins (pessimistic).
  CLOSE  4. mark to market, ratchet trailing stops, count holding days,
            schedule time-stop / signal exits for tomorrow's open
         5. rank new entry candidates (by score) into the free slots

Signals at t only ever act at t+1 or later. Positions are long-only, one per
symbol, so per-position state lives in length-N arrays (no Python objects).
"""

from __future__ import annotations

import numpy as np

from .._jit import njit
from ..costs import order_cost

EXIT_STOP, EXIT_TRAIL, EXIT_TARGET, EXIT_SIGNAL, EXIT_TIME, EXIT_END = 0, 1, 2, 3, 4, 5
EXIT_NAMES = ("stop", "trail", "target", "signal", "time", "end")

# Columns of the float trade-record matrix returned by `simulate`.
TR_SYMBOL, TR_ENTRY_I, TR_EXIT_I, TR_QTY, TR_ENTRY_PX, TR_EXIT_PX, TR_RISK, TR_COSTS, TR_PNL, TR_REASON = range(10)
TR_COLS = 10


@njit(cache=True)
def _close(j, t, fill, reason, qty, entry_px, entry_i, risk_ps, entry_cost, trades, n_tr, cp):
    value = qty[j] * fill
    fee = order_cost(value, False, cp)
    k = n_tr
    trades[k, TR_SYMBOL] = j
    trades[k, TR_ENTRY_I] = entry_i[j]
    trades[k, TR_EXIT_I] = t
    trades[k, TR_QTY] = qty[j]
    trades[k, TR_ENTRY_PX] = entry_px[j]
    trades[k, TR_EXIT_PX] = fill
    trades[k, TR_RISK] = risk_ps[j] * qty[j]
    trades[k, TR_COSTS] = entry_cost[j] + fee
    trades[k, TR_PNL] = (fill - entry_px[j]) * qty[j] - entry_cost[j] - fee
    trades[k, TR_REASON] = reason
    qty[j] = 0.0
    return value - fee


@njit(cache=True)
def simulate(
    o, h, l, c,          # (T, N) float64 prices, NaN = no bar
    entry, exit_sig,     # (T, N) bool, evaluated at the close
    score,               # (T, N) float64, higher = preferred when slots are scarce
    stop_dist,           # (T, N) float64 initial stop distance in Rs (> 0 required to enter)
    trail_dist,          # (T, N) float64 trailing distance from highest close (NaN/0 = off)
    adv,                 # (T, N) float64 average daily traded value in Rs (NaN = no cap)
    regime,              # (T,)  bool, new entries allowed
    target_r, min_hold, max_hold,
    cash0, risk, max_pos, max_pos_frac, max_adv_frac, max_entry_gap, slip,
    cp,                  # cost array (CostModel.as_array)
):
    T, N = c.shape
    qty = np.zeros(N)
    entry_px = np.zeros(N)
    stop = np.zeros(N)
    target = np.zeros(N)
    risk_ps = np.zeros(N)
    entry_cost = np.zeros(N)
    entry_i = np.full(N, -1, np.int64)
    bars = np.zeros(N, np.int64)
    hi_close = np.zeros(N)
    trailed = np.zeros(N, np.bool_)
    last_close = np.full(N, np.nan)

    pend_exit = np.full(N, -1, np.int64)  # exit reason scheduled for next open, -1 = none
    pend_order = np.empty(N, np.int64)
    pend_sd = np.zeros(N)
    n_pend = 0

    equity = np.empty(T)
    cash_arr = np.empty(T)
    invested = np.empty(T)
    npos_arr = np.empty(T, np.int64)
    trades = np.empty((T * max_pos + N, TR_COLS))
    n_tr = 0

    cash = cash0
    prev_equity = cash0
    for t in range(T):
        # ---------------------------------------------------------- 1. exits at open
        for j in range(N):
            if pend_exit[j] >= 0 and qty[j] > 0.0:
                px = o[t, j]
                if np.isnan(px):
                    continue  # no bar today, try again tomorrow
                cash += _close(j, t, px * (1.0 - slip), pend_exit[j], qty, entry_px, entry_i,
                               risk_ps, entry_cost, trades, n_tr, cp)
                n_tr += 1
                pend_exit[j] = -1

        # --------------------------------------------------------- 2. entries at open
        for k in range(n_pend):
            j = pend_order[k]
            px = o[t, j]
            if np.isnan(px) or qty[j] > 0.0:
                continue
            if max_entry_gap > 0.0 and px > c[t - 1, j] * (1.0 + max_entry_gap):
                continue
            fill = px * (1.0 + slip)
            sd = pend_sd[j]
            q = np.floor(prev_equity * risk / sd)
            q = min(q, np.floor(prev_equity * max_pos_frac / fill))
            q = min(q, np.floor(cash / (fill * 1.003)))  # leave room for charges
            if max_adv_frac > 0.0 and not np.isnan(adv[t - 1, j]):
                q = min(q, np.floor(adv[t - 1, j] * max_adv_frac / fill))
            if q < 1.0:
                continue
            fee = order_cost(q * fill, True, cp)
            if q * fill + fee > cash:
                continue
            cash -= q * fill + fee
            qty[j] = q
            entry_px[j] = fill
            entry_cost[j] = fee
            risk_ps[j] = sd
            stop[j] = fill - sd
            target[j] = fill + target_r * sd if target_r > 0.0 else np.inf
            entry_i[j] = t
            bars[j] = 0
            hi_close[j] = fill
            trailed[j] = False
        n_pend = 0

        # ------------------------------------------------- 3. intraday stop / target
        for j in range(N):
            if qty[j] <= 0.0 or np.isnan(l[t, j]):
                continue
            op = o[t, j]
            reason = EXIT_TRAIL if trailed[j] else EXIT_STOP
            if op <= stop[j]:
                fill = op
            elif l[t, j] <= stop[j]:
                fill = stop[j]
            elif op >= target[j]:
                fill, reason = op, EXIT_TARGET
            elif h[t, j] >= target[j]:
                fill, reason = target[j], EXIT_TARGET
            else:
                continue
            cash += _close(j, t, fill * (1.0 - slip), reason, qty, entry_px, entry_i,
                           risk_ps, entry_cost, trades, n_tr, cp)
            n_tr += 1
            pend_exit[j] = -1

        # ----------------------------------------------- 4. close: mark + schedule
        mtm = 0.0
        held = 0
        for j in range(N):
            has_bar = not np.isnan(c[t, j])
            if has_bar:
                last_close[j] = c[t, j]
            if qty[j] <= 0.0:
                continue
            mtm += qty[j] * last_close[j]
            if not has_bar:
                held += 1
                continue
            bars[j] += 1
            td = trail_dist[t, j]
            hi_close[j] = max(hi_close[j], c[t, j])
            if td > 0.0:  # NaN compares False
                ns = hi_close[j] - td
                if ns > stop[j]:
                    stop[j] = ns
                    trailed[j] = True
            if bars[j] >= max_hold:
                pend_exit[j] = EXIT_TIME
            elif exit_sig[t, j] and bars[j] >= min_hold:
                pend_exit[j] = EXIT_SIGNAL
            if pend_exit[j] < 0:
                held += 1
        equity[t] = cash + mtm
        cash_arr[t] = cash
        invested[t] = mtm
        npos_arr[t] = 0
        for j in range(N):
            if qty[j] > 0.0:
                npos_arr[t] += 1

        # ----------------------------------------------- 5. rank new candidates
        slots = max_pos - held
        if t < T - 1 and slots > 0 and regime[t]:
            n_cand = 0
            cand = np.empty(N, np.int64)
            for j in range(N):
                sd = stop_dist[t, j]
                if entry[t, j] and qty[j] <= 0.0 and not np.isnan(c[t, j]) and sd > 0.0 and np.isfinite(sd):
                    cand[n_cand] = j
                    n_cand += 1
            if n_cand > 0:
                sc = np.empty(n_cand)
                for k in range(n_cand):
                    s = score[t, cand[k]]
                    sc[k] = -s if not np.isnan(s) else np.inf
                order = np.argsort(sc, kind="mergesort")  # stable -> deterministic ties
                n_pend = min(slots, n_cand)
                for k in range(n_pend):
                    j = cand[order[k]]
                    pend_order[k] = j
                    pend_sd[j] = stop_dist[t, j]
        prev_equity = equity[t]

    # ------------------------------------------------- liquidate at the last mark
    for j in range(N):
        if qty[j] > 0.0:
            cash += _close(j, T - 1, last_close[j], EXIT_END, qty, entry_px, entry_i,
                           risk_ps, entry_cost, trades, n_tr, cp)
            n_tr += 1
    if T > 0:
        equity[T - 1] = cash
        cash_arr[T - 1] = cash
        invested[T - 1] = 0.0
    return equity, cash_arr, invested, npos_arr, trades[:n_tr]
