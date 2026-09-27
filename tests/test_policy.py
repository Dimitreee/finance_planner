"""The Policy and the simulator: what the advisor decides, and what that costs.

The Policy is a fixed human-written rule, not a learned one, so the Explanation can name exactly the
factors that produced the Action instead of reconstructing them.
"""

from __future__ import annotations

import pytest
from cryptoguard_core.policy import (
    CostScenario,
    PolicyConfig,
    Position,
    decide_target_exposure,
    explain,
    rebalance,
)

BASE = CostScenario(name="base", fee_pct=0.10, slippage_bps=5)
POLICY = PolicyConfig(to_btc_at=0.55, to_usdt_at=0.45)
GENESIS = Position(btc=0.0, usdt=1000.0)


def test_genesis_is_one_thousand_usdt_and_no_btc() -> None:
    assert GENESIS.btc == 0.0
    assert GENESIS.usdt == 1000.0
    assert GENESIS.exposure(price=100.0) == 0.0


def test_a_high_probability_targets_btc() -> None:
    trace = decide_target_exposure(0.61, current_exposure=0.0, policy=POLICY)
    assert trace.target_exposure == 1.0
    assert trace.rule == "enter_btc"


def test_a_low_probability_targets_usdt() -> None:
    trace = decide_target_exposure(0.31, current_exposure=1.0, policy=POLICY)
    assert trace.target_exposure == 0.0
    assert trace.rule == "exit_to_usdt"


def test_between_the_thresholds_the_current_position_is_kept() -> None:
    """Hysteresis: the neutral zone keeps whatever position was passed in, not a fixed one."""
    for exposure in (0.0, 1.0):
        trace = decide_target_exposure(0.50, current_exposure=exposure, policy=POLICY)
        assert trace.target_exposure == exposure
        assert trace.rule == "hysteresis_hold"


def test_the_thresholds_are_inclusive_as_written() -> None:
    assert decide_target_exposure(0.55, 0.0, POLICY).rule == "enter_btc"
    assert decide_target_exposure(0.45, 1.0, POLICY).rule == "exit_to_usdt"


def test_buying_spends_the_cash_and_books_the_fee_on_the_traded_notional() -> None:
    result = rebalance(GENESIS, target_exposure=1.0, price=100.0, costs=BASE)
    assert result.action == "BUY"
    assert result.position_after.usdt == 0.0
    assert result.execution_price == pytest.approx(100.0 * 1.0005)  # slippage against us
    assert result.fee == pytest.approx(result.notional * 0.001)
    assert result.notional + result.fee == pytest.approx(1000.0)
    assert result.position_after.btc == pytest.approx(result.notional / result.execution_price)


def test_holding_books_no_fee_and_moves_nothing() -> None:
    result = rebalance(GENESIS, target_exposure=0.0, price=100.0, costs=BASE)
    assert result.action == "HOLD"
    assert result.fee == 0.0
    assert result.notional == 0.0
    assert result.position_after == GENESIS


def test_reducing_sells_the_whole_position_and_pays_from_the_proceeds() -> None:
    held = Position(btc=2.0, usdt=0.0)
    result = rebalance(held, target_exposure=0.0, price=120.0, costs=BASE)
    assert result.action == "REDUCE"
    assert result.position_after.btc == 0.0
    assert result.execution_price == pytest.approx(120.0 * 0.9995)  # slippage against us
    assert result.notional == pytest.approx(2.0 * result.execution_price)
    assert result.position_after.usdt == pytest.approx(result.notional - result.fee)


def test_cash_never_goes_negative_across_a_round_trip() -> None:
    bought = rebalance(GENESIS, 1.0, price=100.0, costs=BASE)
    sold = rebalance(bought.position_after, 0.0, price=90.0, costs=BASE)
    for position in (bought.position_after, sold.position_after):
        assert position.usdt >= 0.0
        assert position.btc >= 0.0
    assert sold.position_after.usdt < 1000.0, "a fall plus two fees cannot leave us whole"


def test_a_repeated_target_is_a_hold_not_a_second_purchase() -> None:
    bought = rebalance(GENESIS, 1.0, price=100.0, costs=BASE)
    again = rebalance(bought.position_after, 1.0, price=100.0, costs=BASE)
    assert again.action == "HOLD"
    assert again.fee == 0.0
    assert again.position_after == bought.position_after


def test_the_explanation_names_only_what_was_used() -> None:
    trace = decide_target_exposure(0.61, current_exposure=0.0, policy=POLICY)
    text = explain(trace, action="BUY")
    assert "0.61" in text
    assert "0.55" in text
    for absent in ("news", "sentiment", "volume", "headline"):
        assert absent not in text.lower(), "a price-only run may not mention news"
