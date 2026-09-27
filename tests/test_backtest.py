"""The backtest seam: a window, an arm and a Cost Scenario in; deterministic metrics out.

The small hand-computable scenarios live here rather than in unit tests of `rebalance`, because this
is the highest seam above that function — and because the point is not that `rebalance` behaves, but
that a sequence of days adds up to the numbers a report would quote.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import psycopg
import pytest
from conftest import SNAPSHOT
from cryptoguard_core.backtest import BASE_SCENARIO, run_backtest, run_cost_sensitivity
from cryptoguard_core.contract import load_contract
from cryptoguard_core.dataset import (
    ARM_A_FEATURES,
    DAY_MS,
    DecisionDayRow,
    build_decision_day_rows,
)
from cryptoguard_core.decide import (
    genesis_from,
    headline_costs_from,
    policy_from,
    run_decision_job,
)
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import COST_SCENARIOS, PolicyConfig, Position
from cryptoguard_core.store import RunStore


@pytest.fixture
def job_store() -> RunStore:
    dsn = os.environ.get("CRYPTOGUARD_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set CRYPTOGUARD_TEST_DATABASE_URL to compare the backtest against the job")
    try:
        with psycopg.connect(dsn, connect_timeout=3) as connection:
            connection.execute("DROP TABLE IF EXISTS published_runs, job_runs")
            connection.commit()
    except psycopg.OperationalError as error:  # pragma: no cover - environment dependent
        pytest.skip(f"database not reachable: {error}")
    ready = RunStore(dsn)
    ready.migrate()
    return ready


POLICY = PolicyConfig(to_btc_at=0.55, to_usdt_at=0.45)
GENESIS = Position(btc=0.0, usdt=1000.0)
FIRST_DAY = date(2021, 2, 1)
FIRST_CUTOFF_MS = 1_612_137_600_000  # 2021-02-01T00:00:00Z


@dataclass(frozen=True, slots=True)
class ScriptedModel:
    """A model whose probabilities are given, so the arithmetic under test is the simulator's."""

    probabilities: Sequence[float]
    version: str = "scripted-1.0"

    def predict(self, features: Mapping[str, float]) -> float:
        return self.probabilities[int(features["index"])]


def rows_for(prices: Sequence[float]) -> tuple[DecisionDayRow, ...]:
    return tuple(
        DecisionDayRow(
            day=FIRST_DAY + timedelta(days=index),
            feature_cutoff_ms=FIRST_CUTOFF_MS + index * DAY_MS,
            anchor_price=price,
            decision_price=price,
            label=None,
            label_end_ms=FIRST_CUTOFF_MS + (index + 1) * DAY_MS,
            bars_in_day=24,
            features={"index": float(index), **{name: 0.01 for name in ARM_A_FEATURES}},
        )
        for index, price in enumerate(prices)
    )


def test_the_three_scenarios_are_the_ones_the_contract_fixes() -> None:
    assert {s.name: (s.fee_pct, s.slippage_bps) for s in COST_SCENARIOS.values()} == {
        "optimistic": (0.075, 1),
        "base": (0.10, 5),
        "pessimistic": (0.20, 20),
    }
    assert BASE_SCENARIO.name == "base"


def test_buying_from_zero_charges_the_fee_on_the_traded_notional_only() -> None:
    result = run_backtest(
        rows_for([100.0]), ScriptedModel([0.95]), POLICY, BASE_SCENARIO, initial=GENESIS
    )
    assert result.trades == 1
    # 1000 = notional + fee, fee = 0.1% of notional, so notional = 1000 / 1.001
    assert result.total_fees == pytest.approx(1000 - 1000 / 1.001)
    assert result.turnover == pytest.approx((1000 / 1.001) / 1000)
    # Bought at 100 * 1.0005 and valued at 100: the slippage and the fee are the whole loss.
    assert result.final_value_usdt == pytest.approx((1000 / 1.001) / 100.05 * 100)


def test_holding_books_no_fee_on_any_day_after_the_first() -> None:
    once = run_backtest(
        rows_for([100.0]), ScriptedModel([0.95]), POLICY, BASE_SCENARIO, initial=GENESIS
    )
    held = run_backtest(
        rows_for([100.0, 100.0, 100.0]),
        ScriptedModel([0.95, 0.95, 0.95]),
        POLICY,
        BASE_SCENARIO,
        initial=GENESIS,
    )
    assert held.trades == 1
    assert held.total_fees == pytest.approx(once.total_fees)
    assert held.final_value_usdt == pytest.approx(once.final_value_usdt)


def test_buying_from_a_partial_position_trades_only_the_cash_leg() -> None:
    partial = Position(btc=5.0, usdt=500.0)
    result = run_backtest(
        rows_for([100.0]), ScriptedModel([0.95]), POLICY, BASE_SCENARIO, initial=partial
    )
    assert result.trades == 1
    assert result.total_fees == pytest.approx(500 - 500 / 1.001)
    assert result.turnover == pytest.approx((500 / 1.001) / partial.value(100.0))


def test_reducing_to_zero_sells_the_whole_position() -> None:
    result = run_backtest(
        rows_for([100.0]),
        ScriptedModel([0.05]),
        POLICY,
        BASE_SCENARIO,
        initial=Position(btc=10.0, usdt=0.0),
    )
    assert result.trades == 1
    proceeds = 10.0 * 100.0 * 0.9995
    assert result.final_value_usdt == pytest.approx(proceeds * 0.999)


def test_cash_never_goes_negative_across_a_long_alternating_run() -> None:
    prices = [100.0, 90.0, 120.0, 80.0, 130.0, 70.0]
    result = run_backtest(
        rows_for(prices),
        ScriptedModel([0.95, 0.05, 0.95, 0.05, 0.95, 0.05]),
        POLICY,
        COST_SCENARIOS["pessimistic"],
        initial=GENESIS,
    )
    assert result.trades == 6
    assert all(value >= 0 for value in result.value_series)
    assert result.final_value_usdt >= 0


def test_metrics_are_reported_against_cash_and_buy_and_hold_from_the_same_start() -> None:
    prices = [100.0, 110.0, 90.0, 120.0]
    result = run_backtest(
        rows_for(prices),
        ScriptedModel([0.95, 0.95, 0.95, 0.95]),
        POLICY,
        BASE_SCENARIO,
        initial=GENESIS,
    )
    assert result.days == 4
    assert result.cash_return == 0.0
    # Buying once and holding is exactly what this scripted model does, so the two must agree.
    assert result.buy_and_hold_return == pytest.approx(result.net_return)
    assert result.time_invested == 1.0
    # Peak 1098.35 at 110, trough 898.65 at 90.
    assert result.max_drawdown == pytest.approx(1 - 90 / 110, abs=1e-9)


def test_the_run_is_deterministic() -> None:
    prices = [100.0, 110.0, 90.0]
    arguments = (rows_for(prices), ScriptedModel([0.95, 0.05, 0.95]), POLICY, BASE_SCENARIO)
    first = run_backtest(*arguments, initial=GENESIS)
    second = run_backtest(*arguments, initial=GENESIS)
    assert first == second


def test_every_scenario_is_reported_and_the_base_is_the_headline() -> None:
    prices = [100.0, 110.0, 90.0, 120.0]
    comparison = run_cost_sensitivity(
        rows_for(prices), ScriptedModel([0.95, 0.05, 0.95, 0.05]), POLICY, initial=GENESIS
    )
    assert set(comparison.by_scenario) == {"optimistic", "base", "pessimistic"}
    assert comparison.headline.cost_scenario == "base"
    optimistic = comparison.by_scenario["optimistic"].net_return
    pessimistic = comparison.by_scenario["pessimistic"].net_return
    assert optimistic > pessimistic, "heavier costs cannot help"


def test_a_sign_change_between_the_extremes_forbids_claiming_superiority() -> None:
    """Costs decide the answer here, so the honest report is that the answer depends on them.

    A 0.4% move per cycle sits between the optimistic round trip (0.17%) and the pessimistic one
    (0.80%), so the cheap scenario beats buy-and-hold and the expensive one does not. The numbers
    are asserted outright rather than recomputed from the production rule, which would only compare
    that rule to itself.
    """
    cycles = 8
    prices = [price for _ in range(cycles) for price in (100.0, 100.4)]
    comparison = run_cost_sensitivity(
        rows_for(prices), ScriptedModel([0.95, 0.05] * cycles), POLICY, initial=GENESIS
    )

    beats = {
        name: scenario.net_return > scenario.buy_and_hold_return
        for name, scenario in comparison.by_scenario.items()
    }
    assert beats == {"optimistic": True, "base": True, "pessimistic": False}
    assert comparison.cost_sensitive is True
    assert comparison.beats_buy_and_hold is None


def test_a_stable_losing_sign_is_not_reported_as_a_claimable_win() -> None:
    """The verdict names which side won, so a stable loss cannot read as a stable victory."""
    prices = [100.0, 101.0, 102.0, 103.0]
    comparison = run_cost_sensitivity(
        rows_for(prices),
        ScriptedModel([0.05, 0.05, 0.05, 0.05]),  # never in the market while the price rises
        POLICY,
        initial=GENESIS,
    )
    assert comparison.cost_sensitive is False
    assert comparison.beats_buy_and_hold is False


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_the_baseline_trades_often_enough_to_exercise_costs() -> None:
    """A majority baseline would never trade; this one is chosen because it does."""
    bars = load_archive_backfill(SNAPSHOT).bars
    rows = build_decision_day_rows(bars, date(2021, 2, 1), date(2021, 4, 30))
    result = run_backtest(rows, PreviousDirectionBaseline(), POLICY, BASE_SCENARIO, initial=GENESIS)
    assert result.days == 89
    assert result.trades > 10, "the point of this baseline is that it moves the portfolio"
    assert result.total_fees > 0
    assert 0.0 < result.time_invested < 1.0


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_the_backtest_and_the_job_decide_a_day_identically(job_store: RunStore) -> None:
    """The same functions, not two implementations that resemble each other.

    The job publishes one Decision Day against a database; the backtest folds the same day. If the
    two ever diverged, the backtest would stop being a measurement of the product.
    """
    contract = load_contract()
    bars = load_archive_backfill(SNAPSHOT).bars
    day = date(2021, 2, 2)
    rows = build_decision_day_rows(bars, day, day, require_label=False)

    outcome = run_decision_job(
        store=job_store,
        contract=contract,
        model=PreviousDirectionBaseline(),
        rows=rows,
        day=day,
        bundle_dir=Path(tempfile.mkdtemp()),
    )
    result = run_backtest(
        rows,
        PreviousDirectionBaseline(),
        policy_from(contract),
        headline_costs_from(contract),
        initial=genesis_from(contract),
    )

    assert outcome.run.action == ("BUY" if result.trades else "HOLD")
    assert result.final_value_usdt == pytest.approx(
        outcome.run.position_after.value(rows[0].decision_price)
    )


def test_a_first_day_loss_is_not_reported_as_no_drawdown() -> None:
    """The starting capital is the first peak; seeding the series after day one hides the fall."""
    result = run_backtest(
        rows_for([100.0]),
        ScriptedModel([0.05]),
        POLICY,
        BASE_SCENARIO,
        initial=Position(btc=10.0, usdt=0.0),
    )
    assert result.net_return < 0
    assert result.max_drawdown == pytest.approx(-result.net_return)


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(FatalDefect) as caught:
        run_backtest(
            rows_for([100.0]),
            ScriptedModel([0.95]),
            POLICY,
            BASE_SCENARIO,
            initial=GENESIS,
            arm="Z",
        )
    assert caught.value.kind == "unknown_arm"


def test_rows_that_do_not_carry_the_arms_features_are_refused() -> None:
    thin = DecisionDayRow(
        day=FIRST_DAY,
        feature_cutoff_ms=FIRST_CUTOFF_MS,
        anchor_price=100.0,
        decision_price=100.0,
        label=None,
        label_end_ms=FIRST_CUTOFF_MS + DAY_MS,
        bars_in_day=24,
        features={"index": 0.0},
    )
    with pytest.raises(FatalDefect) as caught:
        run_backtest((thin,), ScriptedModel([0.95]), POLICY, BASE_SCENARIO, initial=GENESIS)
    assert caught.value.kind == "arm_feature_mismatch"


def test_rows_out_of_order_are_refused() -> None:
    ordered = rows_for([100.0, 110.0])
    with pytest.raises(FatalDefect) as caught:
        run_backtest(
            (ordered[1], ordered[0]),
            ScriptedModel([0.95, 0.95]),
            POLICY,
            BASE_SCENARIO,
            initial=GENESIS,
        )
    assert caught.value.kind == "rows_out_of_order"
