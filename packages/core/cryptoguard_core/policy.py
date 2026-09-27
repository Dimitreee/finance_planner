"""The Policy and the simulator: what the advisor decides, and what acting on it costs.

The Policy is fixed and human-written rather than learned, so the Explanation can name exactly the
factors that produced the Action instead of reconstructing them afterwards. `rebalance` is a pure
function of the position it is handed, which is what lets the backtest and the live job share one
implementation: the backtest folds it over history, a live day applies it once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Action = Literal["BUY", "HOLD", "REDUCE"]
Rule = Literal["enter_btc", "exit_to_usdt", "hysteresis_hold"]

_BPS = 1e-4
_PERCENT = 1e-2


@dataclass(frozen=True, slots=True)
class Position:
    """The Virtual Portfolio's two legs. It holds no real funds."""

    btc: float
    usdt: float

    def value(self, price: float) -> float:
        return self.btc * price + self.usdt

    def exposure(self, price: float) -> float:
        total = self.value(price)
        return 0.0 if total == 0.0 else self.btc * price / total


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    to_btc_at: float
    to_usdt_at: float


@dataclass(frozen=True, slots=True)
class CostScenario:
    """Fee and slippage per side. A sensitivity band, never a claim about historical fees."""

    name: str
    fee_pct: float
    slippage_bps: float

    @property
    def fee_rate(self) -> float:
        return self.fee_pct * _PERCENT

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps * _BPS


# The three points of the sensitivity band, per ADR-0009. They live beside CostScenario so that the
# contract loader can check the contract against them and the two cannot drift.
COST_SCENARIOS: dict[str, CostScenario] = {
    "optimistic": CostScenario(name="optimistic", fee_pct=0.075, slippage_bps=1),
    "base": CostScenario(name="base", fee_pct=0.10, slippage_bps=5),
    "pessimistic": CostScenario(name="pessimistic", fee_pct=0.20, slippage_bps=20),
}
HEADLINE_SCENARIO = "base"


@dataclass(frozen=True, slots=True)
class RuleTrace:
    """Everything the Policy looked at, and nothing else."""

    probability: float
    to_btc_at: float
    to_usdt_at: float
    current_exposure: float
    target_exposure: float
    rule: Rule


@dataclass(frozen=True, slots=True)
class RebalanceResult:
    action: Action
    notional: float
    fee: float
    execution_price: float
    position_after: Position


def decide_target_exposure(
    probability: float, current_exposure: float, policy: PolicyConfig
) -> RuleTrace:
    """Map a Forecast to a Target Exposure. Configuration, not a learned rule."""
    target: float
    rule: Rule
    if probability >= policy.to_btc_at:
        target, rule = 1.0, "enter_btc"
    elif probability <= policy.to_usdt_at:
        target, rule = 0.0, "exit_to_usdt"
    else:
        target, rule = current_exposure, "hysteresis_hold"
    return RuleTrace(
        probability=probability,
        to_btc_at=policy.to_btc_at,
        to_usdt_at=policy.to_usdt_at,
        current_exposure=current_exposure,
        target_exposure=target,
        rule=rule,
    )


def rebalance(
    position: Position, target_exposure: float, price: float, costs: CostScenario
) -> RebalanceResult:
    """Move the position to the target, booking fees on the traded notional only.

    Slippage always moves against us: a purchase fills above the Decision Price, a sale below it.
    The fee is paid out of the cash leg on a purchase and out of the proceeds on a sale, so neither
    leg can go negative.
    """
    current = position.exposure(price)
    if target_exposure > current:
        execution_price = price * (1 + costs.slippage_rate)
        notional = position.usdt / (1 + costs.fee_rate)
        fee = position.usdt - notional
        return RebalanceResult(
            action="BUY",
            notional=notional,
            fee=fee,
            execution_price=execution_price,
            position_after=Position(btc=position.btc + notional / execution_price, usdt=0.0),
        )
    if target_exposure < current:
        execution_price = price * (1 - costs.slippage_rate)
        notional = position.btc * execution_price
        fee = notional * costs.fee_rate
        return RebalanceResult(
            action="REDUCE",
            notional=notional,
            fee=fee,
            execution_price=execution_price,
            position_after=Position(btc=0.0, usdt=position.usdt + notional - fee),
        )
    return RebalanceResult(
        action="HOLD", notional=0.0, fee=0.0, execution_price=price, position_after=position
    )


def explain(trace: RuleTrace, action: Action) -> str:
    """Short text naming only the factors that entered the Forecast or the Policy."""
    probability = f"{trace.probability:.2f}"
    held = "BTC" if trace.current_exposure >= 0.5 else "USDT"
    if trace.rule == "enter_btc":
        reason = f"that is at or above the {trace.to_btc_at:.2f} threshold for moving into BTC"
    elif trace.rule == "exit_to_usdt":
        reason = f"that is at or below the {trace.to_usdt_at:.2f} threshold for moving into USDT"
    else:
        reason = (
            f"that sits between the {trace.to_usdt_at:.2f} and {trace.to_btc_at:.2f} thresholds, "
            "so the current position is kept"
        )
    return (
        f"The model puts the probability of a higher price tomorrow at {probability}, "
        f"and {reason}. "
        f"The portfolio was holding {held}, so the action is {action}."
    )
