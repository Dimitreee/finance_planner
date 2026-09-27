"""Freshness and the shapes the page reads. Two axes, deliberately independent.

`freshness` says whether today's decision exists. `model_mode` says which model produced whichever
decision is being shown. A fresh price-only run and a stale fused run are different things, and one
word for both would guarantee confusion in the code and in what the user is told.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from cryptoguard_core.policy import Position
from cryptoguard_core.serving import advice_view, build_paper_track, freshness_of
from cryptoguard_core.store import PublishedRun

TODAY = date(2026, 9, 28)


def run_for(day: date, *, action: str = "HOLD", mode: str = "price-only") -> PublishedRun:
    cutoff = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return PublishedRun(
        asset="BTCUSDT",
        decision_day=day,
        feature_cutoff=cutoff,
        decision_time=cutoff.replace(hour=1),
        decision_price=100.0,
        probability=0.95,
        target_exposure=1.0,
        action=action,  # type: ignore[arg-type]
        position_before=Position(btc=0.0, usdt=1000.0),
        position_after=Position(btc=2.0, usdt=50.0),
        trade_notional=None,
        trade_fee=None,
        trade_price=None,
        explanation="because the probability crossed the threshold",
        model_version="previous-direction-1.0",
        policy_version="abc123",
        contract_digest="deadbeef",
        model_mode=mode,
        run_bundle_path="/tmp/bundle.json",
    )


def test_no_published_run_at_all_is_unavailable() -> None:
    assert freshness_of(None, TODAY) == "unavailable"


def test_a_run_for_today_is_current() -> None:
    assert freshness_of(run_for(TODAY), TODAY) == "current"


def test_a_run_from_an_earlier_day_is_stale() -> None:
    assert freshness_of(run_for(date(2026, 9, 27)), TODAY) == "stale"


def test_a_missed_day_is_stale_rather_than_unavailable() -> None:
    """No trade was booked, so the position simply persisted: yesterday's decision still stands."""
    missed_today = run_for(date(2026, 9, 27))
    assert freshness_of(missed_today, TODAY) == "stale"


def test_a_backfilled_future_day_is_not_stale() -> None:
    """A later day can only come from a deliberate backfill; calling it stale would be wrong."""
    assert freshness_of(run_for(date(2026, 9, 29)), TODAY) == "current"


def test_the_view_carries_the_date_of_the_decision_being_shown() -> None:
    view = advice_view(run_for(date(2026, 9, 27)), TODAY)
    assert view.freshness == "stale"
    assert view.decision_day == date(2026, 9, 27)
    assert view.explanation is not None


def test_the_view_describes_no_day_later_than_the_newest_run() -> None:
    view = advice_view(run_for(date(2026, 9, 27)), TODAY)
    assert view.decision_day is not None and view.decision_day <= date(2026, 9, 27)


def test_freshness_and_model_mode_are_independent() -> None:
    stale_fused = advice_view(run_for(date(2026, 9, 1), mode="fused"), TODAY)
    fresh_price_only = advice_view(run_for(TODAY, mode="price-only"), TODAY)
    assert (stale_fused.freshness, stale_fused.model_mode) == ("stale", "fused")
    assert (fresh_price_only.freshness, fresh_price_only.model_mode) == ("current", "price-only")


def test_an_unavailable_view_offers_no_advice_fields() -> None:
    view = advice_view(None, TODAY)
    assert view.freshness == "unavailable"
    assert view.decision_day is None
    assert view.action is None
    assert view.explanation is None


def test_the_view_values_the_portfolio_at_the_decision_price() -> None:
    view = advice_view(run_for(TODAY), TODAY)
    assert view.portfolio_value_usdt == 2.0 * 100.0 + 50.0


def test_the_paper_track_carries_its_genesis_day() -> None:
    days = [date(2026, 9, 26), date(2026, 9, 27), date(2026, 9, 28)]
    track = build_paper_track([(day, "HOLD", 1.0, 10.0, 100.0) for day in days])
    assert track.genesis_day == date(2026, 9, 26)
    assert [point.day for point in track.points] == days
    assert track.points[0].value_usdt == 110.0
    assert track.points[0].price == 100.0


def test_an_empty_paper_track_has_no_genesis() -> None:
    track = build_paper_track([])
    assert track.genesis_day is None
    assert track.points == ()
