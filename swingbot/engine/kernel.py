"""The engine core: three jitted phases per trading day, shared by every mode.

    on_open      fill exits scheduled at the previous close, then entries
    on_intraday  protective stop / profit target against the day's low/high
    on_close     mark to market, trail stops, schedule exits, apply portfolio
                 risk rules and decide tomorrow's entries *with quantities*

`run` strings the phases together over a range of days and is *resumable*:
all state lives in two arrays (`pos`, `acct`) that the caller owns. That is
what makes the three modes identical by construction:

    backtest  fresh state, run(0, T), liquidate at the end
    paper     load state from disk, run(last_seen + 1, T) on new bars, save
    live      broker fills replace on_open/on_intraday, then on_close decides
              orders that the broker adapter sends (AMO + GTT stops on Kite)

Order of events and fill rules (no look-ahead):
  * signals at close t are acted on at the open of t+1
  * entries are limit orders at close * (1 + max_entry_gap): a bigger gap-up
    simply does not fill (same as an IOC limit at the broker)
  * a stop gapped through fills at the open; stop and target touched on the
    same bar -> stop (pessimistic)
"""

from __future__ import annotations

import numpy as np

from .._jit import njit
from ..costs import order_cost

# ---------------------------------------------------------------- exit reasons
EXIT_STOP, EXIT_TRAIL, EXIT_TARGET, EXIT_SIGNAL, EXIT_TIME, EXIT_END, EXIT_HALT = 0, 1, 2, 3, 4, 5, 6
EXIT_NAMES = ("stop", "trail", "target", "signal", "time", "end", "halt")

# ------------------------------------------------------ position state columns
(QTY, ENTRY_PX, STOP, TARGET, RISK_PS, ENTRY_COST, ENTRY_T, BARS, HI_CLOSE, TRAILED, STRAT,
 PEXIT, PRANK, PQTY, PLIMIT, PSD, PSTRAT, LAST_CLOSE) = range(18)
NPOS_COLS = 18

# --------------------------------------------------------------- account state
CASH, PEAK, EQUITY, HALT_LEFT = range(4)
NACCT = 4

# -------------------------------------------------------- risk config columns
RC_CASH0, RC_RISK, RC_MAXPOS, RC_MAXFRAC, RC_ADV, RC_GAP, RC_SLIP, RC_DDHALT, RC_COOLDOWN = range(9)
NRC = 9

# --------------------------------------------------- per-strategy param columns
SP_TARGET_R, SP_MIN_HOLD, SP_MAX_HOLD, SP_QMODE, SP_QVAL, SP_PRIORITY, SP_MAXPOS = range(7)
NSP = 7
Q_RISK, Q_EQUITY, Q_SHARES, Q_AMOUNT = 0, 1, 2, 3

# ---------------------------------------------------------- trade record cols
(TR_SYMBOL, TR_STRAT, TR_ENTRY_I, TR_EXIT_I, TR_QTY, TR_ENTRY_PX, TR_EXIT_PX, TR_RISK, TR_COSTS,
 TR_PNL, TR_REASON) = range(11)
TR_COLS = 11

CASH_BUFFER = 1.003  # head-room for charges when planning cash


def new_state(n_symbols: int, cash: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((n_symbols, NPOS_COLS))
    pos[:, ENTRY_T] = -1
    pos[:, PEXIT] = -1
    pos[:, PRANK] = -1
    pos[:, LAST_CLOSE] = np.nan
    acct = np.array([cash, cash, cash, 0.0], dtype=np.float64)
    return pos, acct


@njit(cache=True)
def close_position(j, t, fill, reason, pos, acct, trades, n_tr, cp):
    q = pos[j, QTY]
    fee = order_cost(q * fill, False, cp)
    trades[n_tr, TR_SYMBOL] = j
    trades[n_tr, TR_STRAT] = pos[j, STRAT]
    trades[n_tr, TR_ENTRY_I] = pos[j, ENTRY_T]
    trades[n_tr, TR_EXIT_I] = t
    trades[n_tr, TR_QTY] = q
    trades[n_tr, TR_ENTRY_PX] = pos[j, ENTRY_PX]
    trades[n_tr, TR_EXIT_PX] = fill
    trades[n_tr, TR_RISK] = pos[j, RISK_PS] * q
    trades[n_tr, TR_COSTS] = pos[j, ENTRY_COST] + fee
    trades[n_tr, TR_PNL] = (fill - pos[j, ENTRY_PX]) * q - pos[j, ENTRY_COST] - fee
    trades[n_tr, TR_REASON] = reason
    acct[CASH] += q * fill - fee
    pos[j, QTY] = 0.0
    pos[j, PEXIT] = -1
    return n_tr + 1


@njit(cache=True)
def open_position(j, t, fill, q, sd, s, sp, pos, acct, cp):
    fee = order_cost(q * fill, True, cp)
    acct[CASH] -= q * fill + fee
    pos[j, QTY] = q
    pos[j, ENTRY_PX] = fill
    pos[j, ENTRY_COST] = fee
    pos[j, RISK_PS] = sd
    pos[j, STOP] = fill - sd
    tr = sp[s, SP_TARGET_R]
    pos[j, TARGET] = fill + tr * sd if tr > 0.0 else np.inf
    pos[j, ENTRY_T] = t
    pos[j, BARS] = 0
    pos[j, HI_CLOSE] = fill
    pos[j, TRAILED] = 0.0
    pos[j, STRAT] = s
    pos[j, PEXIT] = -1


@njit(cache=True)
def on_open(t, o, pos, acct, sp, rc, cp, trades, n_tr):
    N = pos.shape[0]
    slip = rc[RC_SLIP]
    for j in range(N):
        if pos[j, PEXIT] >= 0 and pos[j, QTY] > 0.0 and not np.isnan(o[j]):
            n_tr = close_position(j, t, o[j] * (1.0 - slip), int(pos[j, PEXIT]), pos, acct, trades, n_tr, cp)
    # entries in the rank order decided at the previous close
    n = 0
    idx = np.empty(N, np.int64)
    rk = np.empty(N)
    for j in range(N):
        if pos[j, PRANK] >= 0:
            idx[n] = j
            rk[n] = pos[j, PRANK]
            n += 1
    order = np.argsort(rk[:n], kind="mergesort")
    for k in range(n):
        j = idx[order[k]]
        px = o[j]
        if np.isnan(px) or px > pos[j, PLIMIT] or pos[j, QTY] > 0.0:
            continue
        fill = px * (1.0 + slip)
        q = pos[j, PQTY]
        if q * fill * CASH_BUFFER > acct[CASH]:
            q = np.floor(acct[CASH] / (fill * CASH_BUFFER))
        if q < 1.0:
            continue
        open_position(j, t, fill, q, pos[j, PSD], int(pos[j, PSTRAT]), sp, pos, acct, cp)
    for j in range(N):
        pos[j, PRANK] = -1
    return n_tr


@njit(cache=True)
def on_intraday(t, o, h, l, pos, acct, rc, cp, trades, n_tr):
    slip = rc[RC_SLIP]
    for j in range(pos.shape[0]):
        if pos[j, QTY] <= 0.0 or np.isnan(l[j]):
            continue
        stop, target = pos[j, STOP], pos[j, TARGET]
        reason = EXIT_TRAIL if pos[j, TRAILED] > 0 else EXIT_STOP
        if o[j] <= stop:
            fill = o[j]
        elif l[j] <= stop:
            fill = stop
        elif o[j] >= target:
            fill, reason = o[j], EXIT_TARGET
        elif h[j] >= target:
            fill, reason = target, EXIT_TARGET
        else:
            continue
        n_tr = close_position(j, t, fill * (1.0 - slip), reason, pos, acct, trades, n_tr, cp)
    return n_tr


@njit(cache=True)
def on_close(t, c, adv, entry, exit_sig, score, stop_dist, trail_dist, regime, allow_entries, pos, acct, sp, rc):
    """entry/exit_sig/score/stop_dist/trail_dist are (S, N) slices for day t; regime is (S,)."""
    S, N = entry.shape
    slip = rc[RC_SLIP]
    mtm = 0.0
    held = 0
    held_s = np.zeros(S, np.int64)
    exit_cash = 0.0
    for j in range(N):
        hb = not np.isnan(c[j])
        if hb:
            pos[j, LAST_CLOSE] = c[j]
        if pos[j, QTY] <= 0.0:
            continue
        s = int(pos[j, STRAT])
        mtm += pos[j, QTY] * pos[j, LAST_CLOSE]
        if hb:
            pos[j, BARS] += 1
            pos[j, HI_CLOSE] = max(pos[j, HI_CLOSE], c[j])
            td = trail_dist[s, j]
            if td > 0.0:
                ns = pos[j, HI_CLOSE] - td
                if ns > pos[j, STOP]:
                    pos[j, STOP] = ns
                    pos[j, TRAILED] = 1.0
            if pos[j, BARS] >= sp[s, SP_MAX_HOLD]:
                pos[j, PEXIT] = EXIT_TIME
            elif exit_sig[s, j] and pos[j, BARS] >= sp[s, SP_MIN_HOLD]:
                pos[j, PEXIT] = EXIT_SIGNAL
        if pos[j, PEXIT] < 0:
            held += 1
            held_s[s] += 1
        else:
            exit_cash += pos[j, QTY] * pos[j, LAST_CLOSE] * (1.0 - slip) / CASH_BUFFER
    equity = acct[CASH] + mtm
    acct[EQUITY] = equity
    acct[PEAK] = max(acct[PEAK], equity)

    # drawdown circuit breaker: equity too far below its peak -> no new entries
    # for `cooldown` sessions, then reset the peak and resume
    if acct[HALT_LEFT] > 0:
        acct[HALT_LEFT] -= 1
        if acct[HALT_LEFT] <= 0:
            acct[PEAK] = equity
        return equity
    if rc[RC_DDHALT] > 0.0 and equity < acct[PEAK] * (1.0 - rc[RC_DDHALT]):
        acct[HALT_LEFT] = max(rc[RC_COOLDOWN], 1.0)
        return equity
    slots = int(rc[RC_MAXPOS]) - held
    if not allow_entries or slots <= 0:
        return equity

    # ---- candidates across all strategies: priority first, then score
    cs = np.empty(S * N, np.int64)
    cj = np.empty(S * N, np.int64)
    k1 = np.empty(S * N)
    k2 = np.empty(S * N)
    n = 0
    for s in range(S):
        if not regime[s]:
            continue
        for j in range(N):
            sd = stop_dist[s, j]
            if entry[s, j] and pos[j, QTY] <= 0.0 and not np.isnan(c[j]) and sd > 0.0 and np.isfinite(sd):
                cs[n], cj[n] = s, j
                k1[n] = sp[s, SP_PRIORITY]
                sc = score[s, j]
                k2[n] = -sc if not np.isnan(sc) else np.inf
                n += 1
    if n == 0:
        return equity
    o1 = np.argsort(k2[:n], kind="mergesort")
    o2 = np.argsort(k1[:n][o1], kind="mergesort")
    order = o1[o2]

    cash_avail = acct[CASH] + exit_cash
    gap = rc[RC_GAP]
    rank = 0
    for k in range(n):
        if slots <= 0:
            break
        s, j = cs[order[k]], cj[order[k]]
        if pos[j, PRANK] >= 0:
            continue  # symbol already claimed by a higher-priority strategy
        if sp[s, SP_MAXPOS] > 0 and held_s[s] >= sp[s, SP_MAXPOS]:
            continue
        px = c[j]
        sd = stop_dist[s, j]
        limit = px * (1.0 + gap) if gap > 0.0 else np.inf
        ref = (limit if gap > 0.0 else px) * (1.0 + slip)  # worst-case fill
        mode, val = int(sp[s, SP_QMODE]), sp[s, SP_QVAL]
        if mode == Q_RISK:
            q = np.floor(equity * (val if val > 0.0 else rc[RC_RISK]) / sd)
        elif mode == Q_EQUITY:
            q = np.floor(equity * val / ref)
        elif mode == Q_SHARES:
            q = np.floor(val)
        else:
            q = np.floor(val / ref)
        q = min(q, np.floor(equity * rc[RC_MAXFRAC] / ref))  # per-position cap
        a = adv[j]
        if rc[RC_ADV] > 0.0 and a > 0.0:
            q = min(q, np.floor(a * rc[RC_ADV] / ref))  # liquidity cap
        q = min(q, np.floor(cash_avail / (ref * CASH_BUFFER)))
        if q < 1.0:
            continue
        pos[j, PRANK] = rank
        pos[j, PQTY] = q
        pos[j, PLIMIT] = limit
        pos[j, PSD] = sd
        pos[j, PSTRAT] = s
        cash_avail -= q * ref * CASH_BUFFER
        rank += 1
        slots -= 1
        held_s[s] += 1
    return equity


@njit(cache=True)
def run(t0, t1, entries_on_last, o, h, l, c, adv, regime, E, X, SC, SD, TD, sp, rc, cp, pos, acct,
        trades, n_tr, eq_out, cash_out, inv_out, npos_out):
    """Advance the state over days [t0, t1). Arrays are (T, N) / signals (S, T, N)."""
    T = c.shape[0]
    for t in range(t0, t1):
        if t > 0:
            n_tr = on_open(t, o[t], pos, acct, sp, rc, cp, trades, n_tr)
        n_tr = on_intraday(t, o[t], h[t], l[t], pos, acct, rc, cp, trades, n_tr)
        allow = entries_on_last or t < T - 1
        eq = on_close(t, c[t], adv[t], E[:, t, :], X[:, t, :], SC[:, t, :], SD[:, t, :], TD[:, t, :],
                      regime[:, t], allow, pos, acct, sp, rc)
        k = t - t0
        eq_out[k] = eq
        cash_out[k] = acct[CASH]
        inv_out[k] = eq - acct[CASH]
        npos = 0
        for j in range(pos.shape[0]):
            if pos[j, QTY] > 0.0:
                npos += 1
        npos_out[k] = npos
    return n_tr


@njit(cache=True)
def liquidate(t, pos, acct, cp, trades, n_tr):
    for j in range(pos.shape[0]):
        if pos[j, QTY] > 0.0:
            n_tr = close_position(j, t, pos[j, LAST_CLOSE], EXIT_END, pos, acct, trades, n_tr, cp)
        pos[j, PRANK] = -1
    acct[EQUITY] = acct[CASH]
    return n_tr
