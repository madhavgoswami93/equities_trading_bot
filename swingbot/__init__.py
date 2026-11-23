"""swingbot -- swing trading for NSE equities: one strategy file, one engine,
three modes (backtest -> paper -> live)."""

from .config import Quantity, RiskConfig, RunConfig, StrategySpec, load_config
from .costs import CostModel
from .engine.backtest import Backtest, Engine, run_backtest
from .engine.optimize import sweep, walk_forward
from .engine.result import Result
from .panel import Panel
from .runner import Runner
from .strategies import REGISTRY, RuleStrategy, Strategy, SupertrendSwing, TrendPullback, VolumeBreakout

__version__ = "0.2.0"

__all__ = [
    "REGISTRY", "Backtest", "CostModel", "Engine", "Panel", "Quantity", "Result", "RiskConfig", "RuleStrategy",
    "RunConfig", "Runner", "Strategy", "StrategySpec", "SupertrendSwing", "TrendPullback", "VolumeBreakout",
    "load_config", "run_backtest", "sweep", "walk_forward",
]
