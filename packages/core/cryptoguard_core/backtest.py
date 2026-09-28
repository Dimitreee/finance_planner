"""The research entry point: a window, an arm and a Cost Scenario in, deterministic metrics out.

It folds the *same* `decide_target_exposure` and `rebalance` the daily job applies once, which is
what makes the backtest a measurement of the product rather than of a second implementation that
merely resembles it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

from cryptoguard_core.dataset import FEATURES_BY_ARM, DecisionDayRow
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.model import ModelRelease
from cryptoguard_core.policy import (
    COST_SCENARIOS,
    HEADLINE_SCENARIO,
    CostScenario,
    PolicyConfig,
    Position,
    decide_target_exposure,
    rebalance,
)

BASE_SCENARIO = COST_SCENARIOS[HEADLINE_SCENARIO]

# An arm names a feature set, so the label on a result has to be checked against the rows it
# was folded over. An unchecked label is worse than none: two runs over the same rows could be
# filed as Arm A and Arm C and be indistinguishable in a report.
#
# Derived from the dataset's per-arm lists rather than restated here, and keyed by the upper-case
# name a published result carries. A second hand-written map is how Arm B and Arm C came to exist
# in the builder while staying unknown to the backtest.
ARM_FEATURES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {arm.upper(): features for arm, features in FEATURES_BY_ARM.items()}
)
DEFAULT_ARM = "A"


@dataclass(frozen=True, slots=True)
class BacktestResult:
    arm: str
    cost_scenario: str
    days: int
    trades: int
    total_fees: float
    turnover: float
    time_invested: float
    final_value_usdt: float
    net_return: float
    max_drawdown: float
    cash_return: float
    buy_and_hold_return: float
    # The portfolio's value after each Decision Day, and the same for holding from day one. Both
    # are kept because the Bootstrap Interval resamples the daily path, and a bootstrap over an
    # aggregate has nothing to resample. The first day's return is measured against
    # `start_value_usdt`.
    start_value_usdt: float
    value_series: tuple[float, ...]
    buy_and_hold_series: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CostSensitivity:
    """Every scenario, and whether the answer survives the span between the extremes."""

    headline: BacktestResult
    by_scenario: Mapping[str, BacktestResult]
    cost_sensitive: bool

    @property
    def beats_buy_and_hold(self) -> bool | None:
        """True, False, or None when the costs decide it.

        Deliberately three-valued. A boolean named for one side of the comparison would report
        `True` for "the sign is stable" even when the stable sign is a loss — which is exactly the
        kind of number that gets quoted out of a table and read the other way round.
        """
        if self.cost_sensitive:
            return None
        return self.headline.net_return > self.headline.buy_and_hold_return


def _max_drawdown(values: Sequence[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, 1 - value / peak)
    return worst


def run_backtest(
    rows: Sequence[DecisionDayRow],
    model: ModelRelease,
    policy: PolicyConfig,
    costs: CostScenario,
    *,
    initial: Position,
    arm: str = DEFAULT_ARM,
    probabilities: Mapping[date, float] | None = None,
) -> BacktestResult:
    """`probabilities` replays recorded out-of-sample predictions instead of calling the model.

    A walk-forward produces one prediction per day from the fold that could not see it. Re-running
    the model over the same days would score it in-sample, which is a different and flattering
    question.
    """
    if not rows:
        raise FatalDefect("missing_decision_day_row", "a backtest needs at least one Decision Day")

    expected_features = ARM_FEATURES.get(arm)
    if expected_features is None:
        raise FatalDefect("unknown_arm", f"arm {arm!r} is not one of {sorted(ARM_FEATURES)}")
    missing = [name for name in expected_features if name not in rows[0].features]
    if missing:
        raise FatalDefect(
            "arm_feature_mismatch",
            f"arm {arm!r} needs {sorted(missing)}, which these rows do not carry",
        )
    # Containing the arm's features is not enough now that the arms are cumulative: Arm C's rows
    # carry every one of Arm A's, so a subset test alone would score them and file them as Arm A.
    # What is refused is the ambiguity — rows that also satisfy a wider arm are not this arm's.
    ambiguous = sorted(
        other
        for other, features in ARM_FEATURES.items()
        if other != arm
        and len(features) > len(expected_features)
        and all(name in rows[0].features for name in features)
    )
    if ambiguous:
        raise FatalDefect(
            "arm_feature_mismatch",
            f"these rows also carry every feature of arm(s) {ambiguous}, so filing them as "
            f"arm {arm!r} would label a wider feature set as a narrower one",
        )
    for earlier, later in zip(rows, rows[1:], strict=False):
        if later.day <= earlier.day:
            raise FatalDefect(
                "rows_out_of_order",
                f"Decision Days must strictly increase, but {later.day} follows {earlier.day}",
            )

    start_value = initial.value(rows[0].decision_price)
    if start_value <= 0:
        raise FatalDefect("degenerate_window", "a backtest needs a non-zero starting portfolio")

    position = initial
    values: list[float] = []
    fees = 0.0
    traded_notional = 0.0
    trades = 0
    invested_days = 0

    for row in rows:
        if probabilities is None:
            probability = model.predict(row.features)
        elif row.day in probabilities:
            probability = probabilities[row.day]
        else:
            raise FatalDefect(
                "missing_decision_day_row",
                f"no recorded prediction for {row.day}; a replay may not fall back to the model",
            )
        trace = decide_target_exposure(
            probability,
            current_exposure=position.exposure(row.decision_price),
            policy=policy,
        )
        result = rebalance(position, trace.target_exposure, row.decision_price, costs)
        position = result.position_after
        fees += result.fee
        traded_notional += result.notional
        trades += result.action != "HOLD"
        invested_days += position.btc > 0
        values.append(position.value(row.decision_price))

    final_value = values[-1]

    # Both baselines start from the same capital on the same first day. Cash simply holds it.
    held = rebalance(initial, 1.0, rows[0].decision_price, costs).position_after
    buy_and_hold_values = [held.value(row.decision_price) for row in rows]
    buy_and_hold_value = buy_and_hold_values[-1]

    return BacktestResult(
        arm=arm,
        cost_scenario=costs.name,
        days=len(rows),
        trades=trades,
        total_fees=fees,
        turnover=traded_notional / start_value,
        time_invested=invested_days / len(rows),
        final_value_usdt=final_value,
        net_return=final_value / start_value - 1,
        # The starting capital is the first peak: a run whose worst point is day one would
        # otherwise be reported with no drawdown at all.
        max_drawdown=_max_drawdown([start_value, *values]),
        cash_return=0.0,
        buy_and_hold_return=buy_and_hold_value / start_value - 1,
        start_value_usdt=start_value,
        value_series=tuple(values),
        buy_and_hold_series=tuple(buy_and_hold_values),
    )


@dataclass(frozen=True, slots=True)
class DailyReturns:
    """One Decision Day's log return for the strategy and for holding, which travel together."""

    strategy: float
    buy_and_hold: float


def daily_log_returns(result: BacktestResult) -> tuple[DailyReturns, ...]:
    """One (strategy, buy-and-hold) pair of log returns per Decision Day, in day order.

    Log returns rather than simple ones because the Bootstrap Interval resamples them and then
    compounds: a sum of logs is the product of the ratios, so a resample of the same days in
    another order reproduces the same total. A test pins the decomposition to the published
    aggregates, which is what stops the interval covering a quantity next to the one reported.
    """
    strategy = _log_steps(result.start_value_usdt, result.value_series, "the portfolio")
    held = _log_steps(result.start_value_usdt, result.buy_and_hold_series, "buy-and-hold")
    return tuple(
        DailyReturns(strategy=one, buy_and_hold=other)
        for one, other in zip(strategy, held, strict=True)
    )


def _log_steps(start: float, series: Sequence[float], what: str) -> list[float]:
    values = [start, *series]
    for value in values:
        if value <= 0:
            raise FatalDefect(
                "degenerate_window",
                f"{what} reaches {value} and a log return of it does not exist",
            )
    return [math.log(later / earlier) for earlier, later in zip(values, values[1:], strict=False)]


def excess_return(days: Sequence[DailyReturns]) -> float:
    """Net cumulative return after costs minus buy-and-hold's, over the days given.

    A difference rather than two numbers, because that is the quantity whose interval can span
    zero — and an interval spanning zero is Inconclusive, not a small edge (ADR-0017). It selects
    nothing.
    """
    strategy = math.exp(math.fsum(day.strategy for day in days)) - 1
    held = math.exp(math.fsum(day.buy_and_hold for day in days)) - 1
    return strategy - held


def run_cost_sensitivity(
    rows: Sequence[DecisionDayRow],
    model: ModelRelease,
    policy: PolicyConfig,
    *,
    initial: Position,
    arm: str = DEFAULT_ARM,
    probabilities: Mapping[date, float] | None = None,
) -> CostSensitivity:
    """Run every scenario. The base one is the headline; the others say whether it survives."""
    results = {
        name: run_backtest(
            rows, model, policy, scenario, initial=initial, arm=arm, probabilities=probabilities
        )
        for name, scenario in COST_SCENARIOS.items()
    }
    beats_buy_and_hold = {
        result.net_return > result.buy_and_hold_return for result in results.values()
    }
    return CostSensitivity(
        headline=results[HEADLINE_SCENARIO],
        by_scenario=MappingProxyType(results),
        cost_sensitive=len(beats_buy_and_hold) > 1,
    )
