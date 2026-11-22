"""The strategy file: one TOML file describes data, risk rules, the live broker
and any number of strategies. The same file drives backtest, paper and live.
See strategies.example.toml for a fully commented example."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from .costs import CostModel
from .engine import kernel as K
from .strategies import REGISTRY, Strategy


# ----------------------------------------------------------------------- risk
@dataclass
class RiskConfig:
    """Portfolio-wide risk rules, enforced by the engine for every strategy."""

    capital: float = 1_000_000.0
    max_positions: int = 5  # open positions at any time, all strategies together
    max_position_pct: float = 0.40  # no single position above 40% of equity
    risk_per_trade: float = 0.01  # default for `quantity = "risk"`
    max_adv_pct: float = 0.02  # never buy more than 2% of 20-day avg traded value
    max_entry_gap: float = 0.05  # entry is a limit at close +5%: bigger gap-ups do not fill
    slippage_bps: float = 10.0
    max_drawdown_halt: float = 0.0  # pause new trades when equity is this far below its peak (0 = off)
    halt_cooldown: int = 20  # sessions to pause; then the peak resets and trading resumes
    costs: CostModel = field(default_factory=CostModel.zerodha)

    def __post_init__(self):
        if isinstance(self.costs, str):
            self.costs = CostModel.by_name(self.costs)
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")

    def as_array(self) -> np.ndarray:
        rc = np.zeros(K.NRC)
        rc[K.RC_CASH0] = self.capital
        rc[K.RC_RISK] = self.risk_per_trade
        rc[K.RC_MAXPOS] = self.max_positions
        rc[K.RC_MAXFRAC] = self.max_position_pct
        rc[K.RC_ADV] = self.max_adv_pct
        rc[K.RC_GAP] = self.max_entry_gap
        rc[K.RC_SLIP] = self.slippage_bps / 1e4
        rc[K.RC_DDHALT] = self.max_drawdown_halt
        rc[K.RC_COOLDOWN] = self.halt_cooldown
        return rc


# ------------------------------------------------------------------- quantity
_QTY = re.compile(r"^\s*(risk|equity|shares|amount)\s*([0-9.,_]+)?\s*(%)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Quantity:
    """How many shares a signal buys. Always capped afterwards by the risk rules.

    "risk"          risk_per_trade of equity between entry and stop (default)
    "risk 0.5%"     risk 0.5% of equity per trade
    "equity 20%"    20% of equity in the position
    "shares 50"     fixed 50 shares
    "amount 100000" fixed Rs 1,00,000 per position
    """

    mode: int = K.Q_RISK
    value: float = 0.0  # 0 with mode risk -> RiskConfig.risk_per_trade

    @classmethod
    def parse(cls, text: str | None) -> Quantity:
        if not text:
            return cls()
        m = _QTY.match(str(text))
        if not m:
            raise ValueError(f"bad quantity {text!r}; use e.g. 'risk 1%', 'equity 20%', 'shares 50', 'amount 100000'")
        kind, num, pct = m.group(1).lower(), m.group(2), m.group(3)
        v = float(num.replace(",", "").replace("_", "")) if num else 0.0
        if pct or kind in ("risk", "equity") and v >= 1:
            v /= 100.0
        mode = {"risk": K.Q_RISK, "equity": K.Q_EQUITY, "shares": K.Q_SHARES, "amount": K.Q_AMOUNT}[kind]
        if mode != K.Q_RISK and v <= 0:
            raise ValueError(f"quantity {text!r} needs a positive number")
        return cls(mode, v)

    def __str__(self):
        name = ["risk", "equity", "shares", "amount"][self.mode]
        if self.mode in (K.Q_RISK, K.Q_EQUITY):
            return f"{name} {self.value:.2%}" if self.value else name
        return f"{name} {self.value:g}"


# ------------------------------------------------------------------- strategy
@dataclass
class StrategySpec:
    name: str
    strategy: Strategy
    quantity: Quantity = field(default_factory=Quantity)
    priority: int = 1  # lower = served first when slots are scarce
    max_positions: int = 0  # per-strategy cap inside the global one (0 = none)
    enabled: bool = True

    def with_params(self, **kw) -> StrategySpec:
        return replace(self, strategy=self.strategy.with_params(**kw))

    @classmethod
    def wrap(cls, s: Strategy | StrategySpec) -> StrategySpec:
        return s if isinstance(s, StrategySpec) else cls(s.name, s)


def specs_param_array(specs: list[StrategySpec], signals) -> np.ndarray:
    sp = np.zeros((len(specs), K.NSP))
    for i, (spec, sig) in enumerate(zip(specs, signals)):
        sp[i, K.SP_TARGET_R] = sig.target_r
        sp[i, K.SP_MIN_HOLD] = sig.min_hold
        sp[i, K.SP_MAX_HOLD] = sig.max_hold
        sp[i, K.SP_QMODE] = spec.quantity.mode
        sp[i, K.SP_QVAL] = spec.quantity.value
        sp[i, K.SP_PRIORITY] = spec.priority
        sp[i, K.SP_MAXPOS] = spec.max_positions
    return sp


# --------------------------------------------------------------- data / live
@dataclass
class DataConfig:
    provider: str = "yahoo"  # kite | groww | yahoo
    universe: str = "nifty50"
    benchmark_provider: str | None = None
    start: str = "2008-01-01"  # history to download / backtest from
    cache_dir: str = "data/cache"


@dataclass
class LiveConfig:
    broker: str = "kite"
    dry_run: bool = True  # print orders instead of sending; set false only when ready
    stop_limit_buffer: float = 0.01  # GTT stop sells with a limit 1% below the trigger
    tag: str = "swingbot"


@dataclass
class RunConfig:
    name: str
    data: DataConfig
    risk: RiskConfig
    live: LiveConfig
    strategies: list[StrategySpec]
    runs_dir: str = "runs"
    source: Path | None = None

    @property
    def active(self) -> list[StrategySpec]:
        return [s for s in self.strategies if s.enabled]

    def state_dir(self, mode: str) -> Path:
        return Path(self.runs_dir) / self.name / mode

    def strategy(self, name: str) -> StrategySpec:
        for s in self.strategies:
            if s.name == name:
                return s
        raise KeyError(f"no strategy named {name!r} in {self.source}")


_SPEC_KEYS = {"name", "type", "quantity", "priority", "max_positions", "enabled", "params"}


def _build_strategy(d: dict) -> StrategySpec:
    d = dict(d)
    kind = d.get("type", "rules")
    if kind not in REGISTRY:
        raise ValueError(f"strategy {d.get('name')!r}: unknown type {kind!r}; use one of {', '.join(REGISTRY)}")
    cls = REGISTRY[kind]
    kwargs = {k: v for k, v in d.items() if k not in _SPEC_KEYS}
    params = d.get("params", {})
    if kind == "rules":
        kwargs["params"] = params
    else:
        kwargs.update(params)
    allowed = {f.name for f in fields(cls)}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"strategy {d.get('name')!r}: unknown setting(s) {sorted(unknown)} for type {kind!r}")
    return StrategySpec(
        name=d.get("name", kind),
        strategy=cls(**kwargs),
        quantity=Quantity.parse(d.get("quantity")),
        priority=int(d.get("priority", 1)),
        max_positions=int(d.get("max_positions", 0)),
        enabled=bool(d.get("enabled", True)),
    )


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    strategies = [_build_strategy(s) for s in raw.get("strategy", [])]
    if not strategies:
        raise ValueError(f"{path}: define at least one [[strategy]]")
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: strategy names must be unique")
    return RunConfig(
        name=raw.get("name", path.stem),
        data=DataConfig(**raw.get("data", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        live=LiveConfig(**raw.get("live", {})),
        strategies=strategies,
        runs_dir=raw.get("runs_dir", "runs"),
        source=path,
    )
