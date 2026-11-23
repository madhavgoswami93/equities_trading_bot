"""Plotly figures for the dashboard.

Colors follow a validated categorical palette (fixed slot order, separate
light/dark steps), a single-hue look for single-series charts, a blue<->red
diverging scale with a neutral grey midpoint for returns, thin 2px lines,
recessive hairline grids, and hover tooltips on every chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
INK = {
    "light": dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  mid="#f0efec", red="#e34948", blue="#2a78d6"),
    "dark": dict(primary="#f0efec", secondary="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 mid="#383835", red="#e66767", blue="#3987e5"),
}
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def series_colors(names: list[str], mode: str) -> dict[str, str]:
    """Color follows the entity (its position in the strategy file), never its rank.
    More than 8 entities fold the rest into the last slot's neighbourhood via grey."""
    pal = SERIES[mode]
    return {n: (pal[i] if i < len(pal) else INK[mode]["muted"]) for i, n in enumerate(names)}


def _layout(fig: go.Figure, mode: str, height: int = 320, title: str | None = None, legend: bool = True):
    ink = INK[mode]
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if title else 8, b=8),
        title=dict(text=title, font=dict(size=14, color=ink["primary"]), x=0, xanchor="left") if title else None,
        font=dict(family=FONT, size=12, color=ink["secondary"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=ink["secondary"])),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    axis = dict(gridcolor=ink["grid"], gridwidth=1, linecolor=ink["axis"], zeroline=False,
                tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["muted"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def equity_chart(equity: pd.Series, benchmark: pd.Series | None, mode: str, label: str = "Portfolio") -> go.Figure:
    """Portfolio vs benchmark, both indexed to 1.0 at the start (one axis, common base)."""
    pal, ink = SERIES[mode], INK[mode]
    fig = go.Figure()
    eq = equity / equity.iloc[0]
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name=label, line=dict(color=pal[0], width=2),
                             hovertemplate="%{y:.3f}x<extra>" + label + "</extra>"))
    if benchmark is not None and benchmark.notna().sum() > 1:
        b = benchmark.dropna()
        b = b / b.iloc[0]
        fig.add_trace(go.Scatter(x=b.index, y=b, name="Nifty 50", line=dict(color=ink["muted"], width=2),
                                 hovertemplate="%{y:.3f}x<extra>Nifty 50</extra>"))
    ys = [tr.y for tr in fig.data]
    lo, hi = min(float(y.min()) for y in ys), max(float(y.max()) for y in ys)
    if hi / lo < 2:  # short / flat history: a log axis has no useful ticks -> linear % return
        for tr in fig.data:
            tr.y = tr.y - 1
            tr.hovertemplate = tr.hovertemplate.replace("%{y:.3f}x", "%{y:+.2%}")
        fig.update_yaxes(tickformat="+.0%", title="return since start")
    else:
        nice = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 100]
        ticks = [v for v in nice if lo / 1.1 <= v <= hi * 1.1] or [1]
        fig.update_yaxes(type="log", title="growth of 1 (log)", tickvals=ticks,
                         ticktext=[f"{v:g}x" for v in ticks])
    return _layout(fig, mode, 340, legend=benchmark is not None)


def drawdown_chart(equity: pd.Series, mode: str) -> go.Figure:
    dd = equity / equity.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                               line=dict(color=INK[mode]["red"], width=2),
                               fillcolor=_alpha(INK[mode]["red"], 0.18),
                               hovertemplate="%{y:.1%}<extra>Drawdown</extra>"))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, mode, 200, legend=False)


def strategy_pnl_chart(trades: pd.DataFrame, names: list[str], mode: str) -> go.Figure:
    """Cumulative realised PnL per strategy (by exit date)."""
    colors = series_colors(names, mode)
    fig = go.Figure()
    for n in names:
        t = trades[trades.strategy == n].sort_values("exit_date")
        if t.empty:
            continue
        fig.add_trace(go.Scatter(x=t.exit_date, y=t.pnl.cumsum(), name=n, mode="lines",
                                 line=dict(color=colors[n], width=2, shape="hv"),
                                 hovertemplate="₹%{y:,.0f}<extra>" + n + "</extra>"))
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    return _layout(fig, mode, 300, legend=len(fig.data) > 1)


def monthly_heatmap(table: pd.DataFrame, mode: str) -> go.Figure:
    """Monthly returns: diverging red (loss) <-> grey (0) <-> blue (gain), symmetric scale."""
    ink = INK[mode]
    months = [c for c in table.columns if c != "year"]
    z = table[months].to_numpy(dtype=float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 0.05
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(
        z=z, x=[labels[m - 1] for m in months], y=[str(y) for y in table.index],
        zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, ink["red"]], [0.5, ink["mid"]], [1, ink["blue"]]],
        text=np.where(np.isfinite(z), np.vectorize(lambda v: f"{v:+.1%}")(np.nan_to_num(z)), ""),
        hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        colorbar=dict(tickformat=".0%", thickness=10, outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, dtick=1)
    fig.update_xaxes(showgrid=False, side="top")
    _layout(fig, mode, max(220, 22 * len(table) + 60), legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def bar_chart(series: pd.Series, mode: str, fmt: str = ",.0f", title: str | None = None) -> go.Figure:
    """Single-series horizontal bars (one hue, sorted)."""
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.to_numpy(), y=[str(i) for i in s.index], orientation="h",
                           marker=dict(color=SERIES[mode][0], line=dict(width=0)),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    _layout(fig, mode, max(160, 26 * len(s) + 40), title=title, legend=False)
    fig.update_layout(hovermode="closest", bargap=0.35)
    return fig


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"
