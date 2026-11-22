"""swingbot -- swing-trading research and backtesting for Indian equities (NSE)."""

from .costs import CostModel
from .engine import Backtest, EngineConfig, Result, run_backtest, sweep, walk_forward
from .panel import Panel
from .strategies import REGISTRY, Strategy, SupertrendSwing, TrendPullback, VolumeBreakout, get_strategy

__version__ = "0.1.0"

__all__ = [
    "REGISTRY",
    "Backtest",
    "CostModel",
    "EngineConfig",
    "Panel",
    "Result",
    "Strategy",
    "SupertrendSwing",
    "TrendPullback",
    "VolumeBreakout",
    "get_strategy",
    "run_backtest",
    "sweep",
    "walk_forward",
]
