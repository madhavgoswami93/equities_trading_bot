from .base import Signals, Strategy
from .breakout import VolumeBreakout
from .pullback import TrendPullback
from .supertrend import SupertrendSwing

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (TrendPullback, VolumeBreakout, SupertrendSwing)
}


def get_strategy(name: str, **params) -> Strategy:
    try:
        return REGISTRY[name](**params)
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {', '.join(REGISTRY)}") from None


__all__ = ["REGISTRY", "Signals", "Strategy", "SupertrendSwing", "TrendPullback", "VolumeBreakout", "get_strategy"]
