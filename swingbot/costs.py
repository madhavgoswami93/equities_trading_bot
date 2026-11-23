"""Transaction cost models for Indian cash-equity *delivery* (CNC) trades.

Charges are applied on every fill inside the simulation kernel, so the cost
model is flattened into a small float array (`CostModel.as_array`) that the
numba kernel can consume.

Figures are the published schedules at the time of writing. Brokers and the
exchanges revise them, so check before relying on a backtest for real money:
  * Zerodha: https://zerodha.com/charges
  * Groww:   https://groww.in/pricing
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

from ._jit import njit


@dataclass(frozen=True)
class CostModel:
    brokerage_pct: float = 0.0  # fraction of order value
    brokerage_min: float = 0.0  # Rs per order (floor, applied when pct > 0)
    brokerage_max: float = 0.0  # Rs per order (cap, 0 = no cap)
    stt_buy: float = 0.001  # Securities Transaction Tax, delivery buy 0.1%
    stt_sell: float = 0.001  # delivery sell 0.1%
    exchange_txn: float = 0.0000307  # NSE transaction charge (0.00307%)
    sebi: float = 0.000001  # Rs 10 per crore
    stamp_buy: float = 0.00015  # stamp duty on buy side 0.015%
    gst: float = 0.18  # on brokerage + exchange txn + SEBI fees
    dp_per_sell: float = 15.34  # Zerodha DP charge per scrip per sell day: 3.5 CDSL + 9.5 + 18% GST

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float64)

    # ---- presets -------------------------------------------------------
    @classmethod
    def zerodha(cls) -> CostModel:
        """Zerodha equity delivery: zero brokerage."""
        return cls()

    @classmethod
    def groww(cls) -> CostModel:
        """Groww equity delivery: 0.1% brokerage, min Rs 5, max Rs 20 per order;
        DP Rs 3.5 depository + Rs 16.5 Groww per sell, plus 18% GST."""
        return cls(brokerage_pct=0.001, brokerage_min=5.0, brokerage_max=20.0, dp_per_sell=23.6)

    @classmethod
    def free(cls) -> CostModel:
        """No costs at all -- only useful for debugging / sanity checks."""
        return cls(stt_buy=0, stt_sell=0, exchange_txn=0, sebi=0, stamp_buy=0, gst=0, dp_per_sell=0)

    @classmethod
    def by_name(cls, name: str) -> CostModel:
        try:
            return {"zerodha": cls.zerodha, "groww": cls.groww, "free": cls.free}[name.lower()]()
        except KeyError:
            raise ValueError(f"unknown cost model {name!r}; use zerodha | groww | free") from None

    def buy(self, value: float) -> float:
        return float(order_cost(value, True, self.as_array()))

    def sell(self, value: float) -> float:
        return float(order_cost(value, False, self.as_array()))


@njit(cache=True)
def order_cost(value, is_buy, cp):  # pragma: no cover - jitted
    """Total charges (Rs) for one order of `value` rupees. `cp` = CostModel.as_array()."""
    if value <= 0.0:
        return 0.0
    brokerage = 0.0
    if cp[0] > 0.0:
        brokerage = value * cp[0]
        brokerage = max(brokerage, cp[1])
        if cp[2] > 0.0 and brokerage > cp[2]:
            brokerage = cp[2]
    txn = value * cp[5]
    sebi = value * cp[6]
    gst = cp[8] * (brokerage + txn + sebi)
    if is_buy:
        return brokerage + value * cp[3] + txn + sebi + value * cp[7] + gst
    return brokerage + value * cp[4] + txn + sebi + gst + cp[9]
