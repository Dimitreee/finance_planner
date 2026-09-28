"""The per-day Replay series: assembled from three Cost Scenarios, and what may not differ.

Costs change what the portfolio is *worth*. They do not change what the Policy *decided*: the
Action comes from the probability and whether the portfolio was in or out, and exposure is 0 or 1
exactly, so a fee cannot move it. That makes the decision facts scenario-invariant, and this
module enforces it at assembly rather than hoping a reader of the switcher never notices a flipped
Action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest
from conftest import SNAPSHOT
from cryptoguard_core.backtest import (
    BacktestResult,
    CostSensitivity,
    ReplayDay,
    run_cost_sensitivity,
    series_from,
)
from cryptoguard_core.contract import load_contract
from cryptoguard_core.dataset import ARM_A_FEATURES, DecisionDayRow, build_decision_day_rows
from cryptoguard_core.evaluate import sensitivity_from_walk, walk_arm
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import COST_SCENARIOS, HEADLINE_SCENARIO, PolicyConfig, Position
from cryptoguard_core.replay import ReplaySeries

FIRST_DAY = date(2022, 1, 1)


def rows(count: int = 40) -> tuple[DecisionDayRow, ...]:
    return tuple(
        DecisionDayRow(
            day=FIRST_DAY + timedelta(days=index),
            feature_cutoff_ms=0,
            anchor_price=100.0 + index,
            decision_price=100.0 + index,
            label=1,
            label_end_ms=0,
            bars_in_day=24,
            features=dict.fromkeys(ARM_A_FEATURES, 0.0),
            news_lag_hours=None,
        )
        for index in range(count)
    )


def probabilities(days: Sequence[DecisionDayRow]) -> Mapping[date, float]:
    return {row.day: (0.9 if (index // 5) % 2 == 0 else 0.1) for index, row in enumerate(days)}


def sensitivity(days: Sequence[DecisionDayRow] | None = None) -> CostSensitivity:
    days = days or rows()
    return run_cost_sensitivity(
        days,
        PreviousDirectionBaseline(),
        PolicyConfig(to_btc_at=0.55, to_usdt_at=0.45),
        initial=Position(usdt=1000.0, btc=0.0),
        arm="A",
        probabilities=probabilities(days),
    )


def series() -> ReplaySeries:
    return series_from(sensitivity())


def test_the_decision_facts_are_kept_once_because_costs_cannot_change_them() -> None:
    """One row per day, not one per scenario: a fee cannot change what the Policy decided."""
    built = series()
    assert [entry.day for entry in built.days] == [row.day for row in rows()]
    assert {entry.action for entry in built.days} <= {"BUY", "HOLD", "REDUCE"}
    expected = probabilities(rows())
    for entry in built.days:
        assert entry.probability == expected[entry.day]


def test_every_scenario_carries_a_value_for_every_day() -> None:
    built = series()
    assert set(built.by_scenario) == set(COST_SCENARIOS)
    for scenario, values in built.by_scenario.items():
        assert [entry.day for entry in values] == [entry.day for entry in built.days], scenario


def test_the_three_curves_start_from_identical_capital() -> None:
    """Three curves from different starting money would compare nothing."""
    built = series()
    assert built.start_value_usdt == 1000.0


def test_the_scenarios_differ_in_money_and_agree_on_the_decisions() -> None:
    """The point of the switcher: the curves move, the Action history does not."""
    built = series()
    finals = {name: values[-1].value_usdt for name, values in built.by_scenario.items()}
    assert finals["optimistic"] > finals["base"] > finals["pessimistic"]


def test_a_scenario_that_disagreed_about_an_action_is_refused() -> None:
    """Enforced, not hoped for. Build a disagreement by hand and watch the assembly refuse."""
    built = sensitivity()
    tampered = dict(built.by_scenario)
    victim = tampered["pessimistic"]
    flipped = [*victim.series]
    flipped[3] = replace(flipped[3], action="BUY" if flipped[3].action != "BUY" else "REDUCE")
    tampered["pessimistic"] = _replace_series(victim, tuple(flipped))
    with pytest.raises(FatalDefect) as caught:
        series_from(
            CostSensitivity(
                headline=built.headline,
                by_scenario=tampered,
                cost_sensitive=built.cost_sensitive,
            )
        )
    assert caught.value.kind == "replay_series_disagrees"


def _replace_series(result: BacktestResult, series_rows: tuple[ReplayDay, ...]) -> BacktestResult:
    return replace(result, series=series_rows)


def test_the_series_stops_at_the_last_day_it_was_given() -> None:
    """ADR-0018: the caller builds rows to the Development Period's end; nothing extends them."""
    built = series()
    assert built.first_day == FIRST_DAY
    assert built.last_day == rows()[-1].day
    assert built.last_day == max(entry.day for entry in built.days)


def test_the_headline_scenario_is_named_rather_than_inferred() -> None:
    built = series()
    assert built.headline_scenario == HEADLINE_SCENARIO
    assert built.headline_scenario in built.by_scenario


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_no_final_holdout_day_reaches_the_series(tmp_path: Path) -> None:
    """ADR-0018 at the far end of the pipeline: the series a reader can explore stops in 2024.

    The rows are built the way the evaluate job builds them — to the Development Period's last day —
    and the assertion is on the series itself rather than the builder, because the series is what
    gets published and served. One holdout *price* is still read, legitimately: the Label of the
    last development day is the first holdout Decision Price by definition of the protocol. No
    holdout day is scored, published, or drawable.
    """
    contract = load_contract()
    development_last = contract.values["splits"]["development_last_day"]
    holdout_first = contract.values["splits"]["final_holdout_first_day"]
    bars = load_archive_backfill(SNAPSHOT).bars
    rows = build_decision_day_rows(bars, last_day=development_last)
    walk = walk_arm(rows, contract, arm="a")
    built = series_from(sensitivity_from_walk(rows, walk, contract, arm="A"))

    assert built.last_day == max(entry.day for entry in built.days)
    assert built.last_day < holdout_first
    assert not [entry.day for entry in built.days if entry.day >= holdout_first]
    for scenario, values in built.by_scenario.items():
        assert not [v.day for v in values if v.day >= holdout_first], scenario
