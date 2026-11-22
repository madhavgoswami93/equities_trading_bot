from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..panel import Panel
from .base import Signals, Strategy, as_bool, as_float
from .expr import Evaluator, ExprError


@dataclass
class RuleStrategy(Strategy):
    """A strategy written entirely as expressions in the strategy file.

    entry / exit : boolean expressions evaluated at the close
    stop         : initial stop distance in Rs (e.g. "2.5 * atr(14)")
    trail        : optional trailing distance from the highest close
    score        : optional ranking when there are more signals than slots
    params       : named numbers usable inside the expressions (and sweepable)
    """

    name = "rules"

    entry: str = ""
    exit: str = ""
    stop: str = "2 * atr(14)"
    trail: str = ""
    score: str = ""
    target_r: float = 0.0
    params: dict = field(default_factory=dict)

    def generate(self, p: Panel) -> Signals:
        if not self.entry:
            raise ExprError("rules strategy needs an `entry` expression")
        ev = Evaluator(p, self.params)
        shape = p.shape

        def frame(expr, kind):
            v = ev(expr)
            if not isinstance(v, pd.DataFrame):  # a constant, e.g. stop = "25"
                v = pd.DataFrame(np.full(shape, float(v)), index=p.dates, columns=p.symbols)
            if kind == "bool" and v.dtypes.ne(bool).any():
                raise ExprError(f"{expr!r} must be a condition (true/false), not a number")
            return v

        entry = frame(self.entry, "bool") & self.tradable(p)
        return Signals(
            entry=as_bool(entry),
            exit=as_bool(frame(self.exit, "bool")) if self.exit else None,
            stop_dist=as_float(frame(self.stop, "num")),
            trail_dist=as_float(frame(self.trail, "num")) if self.trail else None,
            score=as_float(frame(self.score, "num")) if self.score else None,
            target_r=self.target_r,
            min_hold=self.min_hold,
            max_hold=self.max_hold,
        )

    def with_params(self, **kw) -> RuleStrategy:
        inner = {k: v for k, v in kw.items() if k in self.params}
        outer = {k: v for k, v in kw.items() if k not in self.params}
        return replace(self, params={**self.params, **inner}, **outer)
