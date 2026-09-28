"""Turning archived headlines into two numbers per Decision Day, and what may not reach them.

The windows are taken over *availability instants* — a stated time plus the Assumed Availability
Lag — never over stated times. At the Primary Lag of 24 hours a window over stated times would
collapse to a single instant and the count would be identically zero, so the distinction is
load-bearing rather than pedantic.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.news import NewsItem
from cryptoguard_core.news_features import (
    ARM_B_EXTRA_FEATURES,
    DAY_MS,
    HeadlineAvailability,
    headline_availability,
    headline_count_features,
    require_news_span,
)

CUTOFF = datetime(2021, 2, 1, tzinfo=UTC)
CUTOFF_MS = int(CUTOFF.timestamp() * 1000)
HOUR = timedelta(hours=1)


def item(
    stated_at: datetime, *, currencies: tuple[str, ...] = ("BTC",), item_id: str = "1"
) -> NewsItem:
    """A News Item carrying only what this module reads: a stated time and its currencies."""
    naive = stated_at.replace(tzinfo=None)
    return NewsItem(
        item_id=item_id,
        title=f"headline {item_id}",
        description=None,
        source_domain=None,
        source_url=None,
        stated_at=naive,
        stated_at_raw=naive.strftime("%Y-%m-%d %H:%M:%S"),
        stated_timezone_suffix=None,
        currencies=currencies,
    )


def availability_of(*items: NewsItem, lag_hours: int) -> HeadlineAvailability:
    """An availability set whose archive claims completeness up to the cutoff under test."""
    return headline_availability(items, lag_hours=lag_hours, covers_through_ms=CUTOFF_MS)


def counts(*items: NewsItem, lag_hours: int) -> dict[str, float]:
    return headline_count_features(availability_of(*items, lag_hours=lag_hours), CUTOFF_MS)


def test_the_arm_b_feature_names_are_exactly_the_two_the_contract_froze() -> None:
    assert ARM_B_EXTRA_FEATURES == (
        "news_count_24h_log1p",
        "news_count_24h_over_7d_mean",
    )


@pytest.mark.parametrize("lag_hours", [1, 6, 24])
def test_an_item_available_exactly_at_the_cutoff_is_admissible(lag_hours: int) -> None:
    """The boundary is inclusive at the cutoff: `stated_at + lag <= Feature Cutoff`."""
    on_time = item(CUTOFF - lag_hours * HOUR)
    assert counts(on_time, lag_hours=lag_hours)["news_count_24h_log1p"] == math.log1p(1)


@pytest.mark.parametrize("lag_hours", [1, 6, 24])
def test_an_item_available_one_second_after_the_cutoff_is_not(lag_hours: int) -> None:
    late = item(CUTOFF - lag_hours * HOUR + timedelta(seconds=1))
    assert counts(late, lag_hours=lag_hours)["news_count_24h_log1p"] == 0.0


@pytest.mark.parametrize("lag_hours", [1, 6, 24])
def test_a_days_count_is_the_items_stated_in_the_24_hours_ending_at_cutoff_minus_lag(
    lag_hours: int,
) -> None:
    """What replaces the monotonicity claim: the window slides, it does not shrink."""
    shifted_end = CUTOFF - lag_hours * HOUR
    inside = [
        item(shifted_end, item_id="at-the-edge"),
        item(shifted_end - timedelta(hours=12), item_id="middle"),
        item(shifted_end - timedelta(hours=24) + timedelta(seconds=1), item_id="just-inside"),
    ]
    outside = [
        item(shifted_end - timedelta(hours=24), item_id="just-outside-old"),
        item(shifted_end + timedelta(seconds=1), item_id="not-yet-available"),
    ]
    features = counts(*inside, *outside, lag_hours=lag_hours)
    assert features["news_count_24h_log1p"] == math.log1p(len(inside))


def test_no_item_can_influence_a_day_before_its_availability_instant() -> None:
    """The property the lag exists to enforce, stated directly rather than through monotonicity."""
    stated = CUTOFF - 2 * HOUR
    available_in_time = availability_of(item(stated), lag_hours=1)
    still_withheld = availability_of(item(stated), lag_hours=6)
    assert headline_count_features(available_in_time, CUTOFF_MS)["news_count_24h_log1p"] > 0.0
    assert headline_count_features(still_withheld, CUTOFF_MS)["news_count_24h_log1p"] == 0.0


def test_only_an_exact_btc_token_counts() -> None:
    """A substring test pulls in WBTC, HBTC, BTCST and BTCUSD: 3 793 records by the audit."""
    stated = CUTOFF - 2 * HOUR
    near_misses = [
        item(stated, currencies=("WBTC",), item_id="wbtc"),
        item(stated, currencies=("HBTC",), item_id="hbtc"),
        item(stated, currencies=("BTCST",), item_id="btcst"),
        item(stated, currencies=("BTCUSD",), item_id="btcusd"),
        item(stated, currencies=(), item_id="none"),
    ]
    exact = item(stated, currencies=("ETH", "BTC"), item_id="exact")
    features = counts(*near_misses, exact, lag_hours=1)
    assert features["news_count_24h_log1p"] == math.log1p(1)


def test_a_quiet_day_is_a_measured_zero_and_the_ratio_is_zero_too() -> None:
    """Zero headlines is a measurement. It is never imputed, and it never divides by zero."""
    features = counts(lag_hours=24)
    assert features["news_count_24h_log1p"] == 0.0
    assert features["news_count_24h_over_7d_mean"] == 0.0


def test_a_quiet_day_inside_a_busy_week_also_reads_as_zero() -> None:
    """The ratio means "nothing today" whatever the week looked like, so both cases agree."""
    week = [
        item(CUTOFF - timedelta(days=days, hours=1), item_id=f"day-{days}") for days in range(2, 7)
    ]
    features = counts(*week, lag_hours=1)
    assert features["news_count_24h_log1p"] == 0.0
    assert features["news_count_24h_over_7d_mean"] == 0.0


def test_the_ratio_compares_today_against_a_baseline_that_excludes_today() -> None:
    """Seven days ending 24 hours before the cutoff, plus a one-headline prior. See ADR-0020."""
    today = [item(CUTOFF - 2 * HOUR, item_id=f"today-{n}") for n in range(4)]
    earlier = [item(CUTOFF - timedelta(days=3, hours=2), item_id=f"older-{n}") for n in range(3)]
    features = counts(*today, *earlier, lag_hours=1)
    assert features["news_count_24h_log1p"] == math.log1p(4)
    # The baseline holds 3 headlines, so the mean daily count is (3 + 1) / 7 and today is 7x it.
    assert features["news_count_24h_over_7d_mean"] == pytest.approx(7.0)


def test_a_spike_is_not_capped_by_the_length_of_the_baseline_window() -> None:
    """A baseline containing today would bound the ratio at 7 and flatten every spike together."""
    spike = [item(CUTOFF - 2 * HOUR, item_id=f"spike-{n}") for n in range(40)]
    bigger = [item(CUTOFF - 2 * HOUR, item_id=f"spike-{n}") for n in range(400)]
    assert counts(*spike, lag_hours=1)["news_count_24h_over_7d_mean"] == pytest.approx(280.0)
    assert counts(*bigger, lag_hours=1)["news_count_24h_over_7d_mean"] == pytest.approx(2800.0)


def test_the_baseline_window_excludes_its_far_edge() -> None:
    """The baseline spans `(cutoff - 8d, cutoff - 24h]`, so an item a day older is outside it."""
    today = item(CUTOFF - 2 * HOUR, item_id="today")
    outside = item(CUTOFF - timedelta(days=8, hours=1), item_id="too-old")
    inside = item(CUTOFF - timedelta(days=7), item_id="in-baseline")
    assert counts(today, outside, lag_hours=1)["news_count_24h_over_7d_mean"] == pytest.approx(7.0)
    assert counts(today, inside, lag_hours=1)["news_count_24h_over_7d_mean"] == pytest.approx(3.5)


def test_availability_instants_are_sorted_and_carry_their_lag() -> None:
    out_of_order = [
        item(CUTOFF - 2 * HOUR, item_id="second"),
        item(CUTOFF - 5 * HOUR, item_id="first"),
    ]
    availability = availability_of(*out_of_order, lag_hours=6)
    assert isinstance(availability, HeadlineAvailability)
    assert availability.lag_hours == 6
    assert list(availability.instants_ms) == sorted(availability.instants_ms)


def test_a_lag_outside_the_pre_registered_grid_is_refused() -> None:
    """The grid is frozen. A lag chosen at the call site is not a pre-registered lag."""
    with pytest.raises(FatalDefect) as caught:
        availability_of(item(CUTOFF - 2 * HOUR), lag_hours=3)
    assert caught.value.kind == "news_lag_not_pre_registered"


# --- What the archive covers, and what may not be inferred from silence ---------------------------


def test_unsorted_instants_are_refused_rather_than_silently_miscounted() -> None:
    """`count_in` bisects. Unsorted input returns wrong counts, and can return negative ones."""
    with pytest.raises(FatalDefect) as caught:
        HeadlineAvailability(
            asset="BTC",
            lag_hours=1,
            instants_ms=(CUTOFF_MS, CUTOFF_MS - HOUR_MS),
            covers_through_ms=CUTOFF_MS,
        )
    assert caught.value.kind == "news_instants_unsorted"


def test_a_window_reaching_past_the_archives_coverage_is_refused() -> None:
    """Otherwise every day past the archive's end reads as a genuinely silent 24 hours."""
    items = [item(CUTOFF - timedelta(days=days), item_id=f"day-{days}") for days in range(1, 20)]
    availability = headline_availability(items, lag_hours=1, covers_through_ms=CUTOFF_MS - DAY_MS)
    with pytest.raises(FatalDefect) as caught:
        require_news_span(
            availability, first_cutoff_ms=CUTOFF_MS - 9 * DAY_MS, last_cutoff_ms=CUTOFF_MS
        )
    assert caught.value.kind == "news_coverage_underrun"


def test_a_span_inside_the_archive_is_accepted() -> None:
    items = [item(CUTOFF - timedelta(days=days), item_id=f"day-{days}") for days in range(1, 12)]
    availability = headline_availability(items, lag_hours=1, covers_through_ms=CUTOFF_MS)
    require_news_span(availability, first_cutoff_ms=CUTOFF_MS - DAY_MS, last_cutoff_ms=CUTOFF_MS)


def test_the_warm_up_message_states_instants_a_reader_can_use() -> None:
    """An operator reading a job log needs an ISO instant, not epoch milliseconds."""
    availability = headline_availability(
        [item(CUTOFF - HOUR)], lag_hours=1, covers_through_ms=CUTOFF_MS
    )
    with pytest.raises(FatalDefect) as caught:
        require_news_span(availability, first_cutoff_ms=CUTOFF_MS, last_cutoff_ms=CUTOFF_MS)
    assert caught.value.kind == "news_warm_up_underrun"
    assert "2021-01-24T00:00:00+00:00" in str(caught.value)
