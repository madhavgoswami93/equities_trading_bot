from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class Result:
    equity: pd.Series
    cash: pd.Series
    invested: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    strategy: str
    params: dict
    benchmark: pd.Series | None = None
    _stats: dict | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- stats
    @property
    def stats(self) -> dict:
        if self._stats is None:
            s = metrics.equity_stats(self.equity)
            s["exposure"] = float((self.invested / self.equity).mean()) if len(self.equity) else 0.0
            s["avg_positions"] = float(self.positions.mean()) if len(self.positions) else 0.0
            s.update(metrics.trade_stats(self.trades))
            if self.benchmark is not None and len(self.benchmark.dropna()) > 1:
                b = metrics.equity_stats(self.benchmark.dropna())
                s["benchmark_cagr"] = b.get("cagr")
                s["benchmark_max_dd"] = b.get("max_drawdown")
            self._stats = s
        return self._stats

    def by_symbol(self) -> pd.DataFrame:
        """Per-stock breakdown (the old per-ticker result_df, from the trade list)."""
        if self.trades.empty:
            return pd.DataFrame()
        g = self.trades.groupby("symbol")
        out = pd.DataFrame({
            "trades": g.size(),
            "win_rate": g["pnl"].apply(lambda x: (x > 0).mean()),
            "pnl": g["pnl"].sum(),
            "avg_return_pct": g["return_pct"].mean(),
            "avg_r": g["r_multiple"].mean(),
            "avg_hold": g["bars_held"].mean(),
        })
        return out.sort_values("pnl", ascending=False)

    def monthly_returns(self) -> pd.DataFrame:
        m = self.equity.resample("ME").last().pct_change()
        m.iloc[0] = self.equity.resample("ME").last().iloc[0] / self.equity.iloc[0] - 1
        t = m.to_frame("r")
        t["year"], t["month"] = t.index.year, t.index.month
        table = t.pivot(index="year", columns="month", values="r")
        table["year"] = self.equity.resample("YE").last().pct_change().fillna(
            self.equity.resample("YE").last().iloc[0] / self.equity.iloc[0] - 1).to_numpy()
        return table

    # -------------------------------------------------------------- output
    def summary(self) -> str:
        s = self.stats
        pct = lambda k: f"{s[k] * 100:8.2f}%" if isinstance(s.get(k), (int, float)) and np.isfinite(s[k]) else "      n/a"
        num = lambda k, d=2: f"{s[k]:9.{d}f}" if isinstance(s.get(k), (int, float)) and np.isfinite(s[k]) else "      n/a"
        lines = [
            f"Strategy      {self.strategy}  {self.params}",
            f"Period        {s.get('start')} -> {s.get('end')}",
            f"Final equity  {s.get('final_equity', 0):,.0f}",
            f"Total return {pct('total_return')}   CAGR {pct('cagr')}   Vol {pct('volatility')}",
            f"Sharpe       {num('sharpe')}   Sortino {num('sortino')}   Calmar {num('calmar')}",
            f"Max drawdown {pct('max_drawdown')}   longest underwater {s.get('max_dd_days', 0)} days",
            f"Exposure     {pct('exposure')}   avg positions {num('avg_positions', 1)}",
            f"Trades       {s.get('trades', 0):9d}   win rate {pct('win_rate')}   PF {num('profit_factor')}",
            f"Avg win      {pct('avg_win_pct')}   avg loss {pct('avg_loss_pct')}   expectancy {num('expectancy_r')} R",
            f"SQN          {num('sqn')}   avg hold {num('avg_hold_days', 1)} days   costs {s.get('total_costs', 0):,.0f}",
            f"Exits        {s.get('exit_reasons', {})}",
        ]
        if "benchmark_cagr" in s:
            lines.append(f"Benchmark    CAGR {pct('benchmark_cagr')}   max DD {pct('benchmark_max_dd')}")
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"equity": self.equity, "cash": self.cash, "invested": self.invested,
                      "positions": self.positions}).to_csv(out / "equity.csv")
        self.trades.to_csv(out / "trades.csv", index=False)
        self.by_symbol().to_csv(out / "by_symbol.csv")
        (out / "stats.json").write_text(json.dumps(
            {"strategy": self.strategy, "params": self.params, **self.stats}, indent=2, default=str))
        return out

    def plot(self, path: str | Path | None = None):
        import matplotlib.pyplot as plt  # optional dependency

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        (self.equity / self.equity.iloc[0]).plot(ax=a1, label=self.strategy)
        if self.benchmark is not None:
            b = self.benchmark.dropna()
            (b / b.iloc[0]).plot(ax=a1, label="benchmark", alpha=0.7)
        a1.set_yscale("log")
        a1.legend()
        a1.set_title(f"{self.strategy}  CAGR {self.stats.get('cagr', 0):.1%}  MaxDD {self.stats.get('max_drawdown', 0):.1%}")
        metrics.drawdown(self.equity).plot(ax=a2, color="tab:red")
        a2.set_ylabel("drawdown")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=120)
        return fig
