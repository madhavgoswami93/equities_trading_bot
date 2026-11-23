"""A tiny, safe expression language for writing signals in the strategy file.

    entry = "close > sma(close, 200) and rsi(close, 3) < rsi_entry"
    stop  = "2.5 * atr(14)"

Expressions are parsed with Python's `ast` and only whitelisted nodes are
evaluated (no attribute access, no imports, no arbitrary calls), so a strategy
file can never execute code. Every value is a (dates x symbols) DataFrame or a
number, so one expression computes the signal for the whole universe.

Variables : open high low close volume, plus any name from the strategy's `params`
Logic     : and or not, < <= > >= == !=, + - * / ** and parentheses
Functions : see FUNCTIONS below.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..panel import Panel


class ExprError(ValueError):
    pass


def _n(x) -> int:
    if not float(x).is_integer() or x <= 0:
        raise ExprError(f"window length must be a positive integer, got {x}")
    return int(x)


def _builders(p: Panel) -> dict[str, Callable]:
    h, l, c, v = p.frame("high"), p.frame("low"), p.frame("close"), p.frame("volume")
    return {
        "sma": lambda x, n: ind.sma(x, _n(n)),
        "ema": lambda x, n: ind.ema(x, _n(n)),
        "rsi": lambda x, n=14: ind.rsi(x, _n(n)),
        "roc": lambda x, n: ind.roc(x, _n(n)),
        "atr": lambda n=14: ind.atr(h, l, c, _n(n)),
        "stdev": lambda x, n: x.rolling(_n(n), min_periods=_n(n)).std(),
        "highest": lambda x, n: ind.highest(x, _n(n)),
        "lowest": lambda x, n: ind.lowest(x, _n(n)),
        "prev": lambda x, n=1: x.shift(_n(n)),
        "avg_volume": lambda n=20: ind.sma(v, _n(n)),
        "turnover": lambda n=20: ind.sma(c * v, _n(n)),
        "supertrend": lambda n=10, m=3.0: ind.supertrend(h, l, c, _n(n), float(m))[0],
        "supertrend_dir": lambda n=10, m=3.0: ind.supertrend(h, l, c, _n(n), float(m))[1],
        "crossover": lambda a, b: (a > b) & (_shift(a) <= _shift(b)),
        "crossunder": lambda a, b: (a < b) & (_shift(a) >= _shift(b)),
        "max": lambda a, b: _elementwise(np.maximum, a, b),
        "min": lambda a, b: _elementwise(np.minimum, a, b),
        "abs": lambda a: abs(a),
    }


def _shift(x):
    return x.shift(1) if isinstance(x, (pd.DataFrame, pd.Series)) else x


def _elementwise(f, a, b):
    if isinstance(a, pd.DataFrame):
        return pd.DataFrame(f(a.to_numpy(), b.to_numpy() if isinstance(b, pd.DataFrame) else b),
                            index=a.index, columns=a.columns)
    if isinstance(b, pd.DataFrame):
        return _elementwise(f, b, a)
    return f(a, b)


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _bool(x):
    if isinstance(x, pd.DataFrame):
        return x.fillna(False).astype(bool) if x.dtypes.ne(bool).any() else x
    return bool(x)


class Evaluator:
    def __init__(self, panel: Panel, params: dict | None = None):
        self.vars = {f: panel.frame(f) for f in ("open", "high", "low", "close", "volume")}
        for k, v in (params or {}).items():
            if k in self.vars:
                raise ExprError(f"parameter name {k!r} shadows a price field")
            self.vars[k] = v
        self.funcs = _builders(panel)
        self._cache: dict = {}

    def __call__(self, text: str):
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as e:
            raise ExprError(f"cannot parse {text!r}: {e.msg}") from None
        return self._eval(tree.body, text)

    def _eval(self, node, src):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.vars:
                return self.vars[node.id]
            raise ExprError(f"unknown name {node.id!r} in {src!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](self._eval(node.left, src), self._eval(node.right, src))
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, src)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.Not):
                return ~_bool(v) if isinstance(v, pd.DataFrame) else not v
        if isinstance(node, ast.BoolOp):
            vals = [_bool(self._eval(v, src)) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, src)
            out = None
            for op, right_node in zip(node.ops, node.comparators):
                if type(op) not in _CMP:
                    raise ExprError(f"operator not allowed in {src!r}")
                right = self._eval(right_node, src)
                r = _CMP[type(op)](left, right)
                out = r if out is None else (out & r)
                left = right
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name not in self.funcs:
                raise ExprError(f"unknown function {name!r} in {src!r}; available: {', '.join(sorted(self.funcs))}")
            args = [self._eval(a, src) for a in node.args]
            key = (name, tuple(id(a) if isinstance(a, pd.DataFrame) else a for a in args))
            if key not in self._cache:  # sma(close, 200) used twice is computed once
                self._cache[key] = self.funcs[name](*args)
            return self._cache[key]
        raise ExprError(f"construct not allowed in {src!r}: {ast.dump(node)[:60]}")


FUNCTIONS = ["sma(x, n)", "ema(x, n)", "stdev(x, n)", "rsi(x, n=14)", "roc(x, n)", "atr(n=14)", "highest(x, n)",
             "lowest(x, n)",
             "prev(x, n=1)", "avg_volume(n=20)", "turnover(n=20)", "supertrend(n=10, m=3)",
             "supertrend_dir(n=10, m=3)", "crossover(a, b)", "crossunder(a, b)", "max(a, b)", "min(a, b)", "abs(x)"]
