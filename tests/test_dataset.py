"""One row per Decision Day: what each field is anchored on, and what may not reach it.

The arithmetic tests run on synthetic hourly bars so the expected values are stated rather than
recomputed by the code under test. The shape of the real Research Window is asserted separately
against the verified snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import cast

import pytest
from conftest import SNAPSHOT, hourly_series
from cryptoguard_core.dataset import (
    ARM_A_FEATURES,
    RESEARCH_WINDOW_FIRST_DAY,
    RESEARCH_WINDOW_LAST_DAY,
    DecisionDayRow,
    build_decision_day_rows,
)
from cryptoguard_core.ingest import HOUR_MS, Bar, FatalDefect, load_archive_backfill

# Enough history for a 30-day Warm-Up Buffer before 2021-02-01, plus days to resolve every Label.
SERIES_START = datetime(2021, 1, 2, tzinfo=UTC)
SERIES_HOURS = 34 * 24  # through 2021-02-04T23:00Z
FIRST_DAY = date(2021, 2, 1)
LAST_DAY = date(2021, 2, 3)


Quote = Callable[[int], tuple[float, float, float]]


def build(*, quote: Quote | None = None, skip: Sequence[int] = ()) -> tuple[DecisionDayRow, ...]:
    bars = (
        hourly_series(SERIES_START, SERIES_HOURS, skip=skip)
        if quote is None
        else hourly_series(SERIES_START, SERIES_HOURS, quote=quote, skip=skip)
    )
    return build_decision_day_rows(bars, FIRST_DAY, LAST_DAY)


def test_research_window_constants_match_the_frozen_protocol() -> None:
    assert RESEARCH_WINDOW_FIRST_DAY == date(2021, 2, 1)
    assert RESEARCH_WINDOW_LAST_DAY == date(2025, 11, 30)
    assert len(ARM_A_FEATURES) == 7


def test_one_row_per_day_in_the_requested_window() -> None:
    rows = build()
    assert [row.day for row in rows] == [date(2021, 2, 1), date(2021, 2, 2), date(2021, 2, 3)]


def test_anchor_price_is_the_close_of_the_last_bar_before_the_cutoff() -> None:
    """The 23:00 bar of the previous day closes exactly at the 00:00 Feature Cutoff."""
    rows = build(quote=lambda i: (100.0 + i, 200.0 + i, 1.0))
    first = rows[0]
    hours_before_cutoff = int(
        (datetime(2021, 2, 1, tzinfo=UTC) - SERIES_START).total_seconds() // 3600
    )
    assert first.anchor_price == 200.0 + (hours_before_cutoff - 1)


def test_decision_price_is_the_open_of_the_01_00_bar_and_differs_from_the_anchor() -> None:
    rows = build(quote=lambda i: (100.0 + i, 200.0 + i, 1.0))
    first = rows[0]
    hours_to_0100 = int(
        (datetime(2021, 2, 1, 1, tzinfo=UTC) - SERIES_START).total_seconds() // 3600
    )
    assert first.decision_price == 100.0 + hours_to_0100
    assert first.decision_price != first.anchor_price


def test_label_is_the_direction_of_the_next_decision_price() -> None:
    rising = build(quote=lambda i: (100.0 + i, 100.0 + i, 1.0))
    assert [row.label for row in rising] == [1, 1, 1]

    falling = build(quote=lambda i: (1000.0 - i, 1000.0 - i, 1.0))
    assert [row.label for row in falling] == [0, 0, 0]

    # The default series repeats each day exactly, so consecutive Decision Prices are equal.
    assert [row.label for row in build()] == [0, 0, 0], "an unchanged price is not a rise"


def test_label_end_is_the_next_decision_days_01_00() -> None:
    rows = build()
    expected = int(datetime(2021, 2, 2, 1, tzinfo=UTC).timestamp() * 1000)
    assert rows[0].label_end_ms == expected


def test_one_day_log_return_is_taken_on_the_anchor_series() -> None:
    rows = build(quote=lambda i: (100.0 + i, 200.0 + i, 1.0))
    today, yesterday = rows[1].anchor_price, rows[0].anchor_price
    assert rows[1].features["r_1"] == pytest.approx(math.log(today / yesterday))


def test_realised_volatility_grows_with_hourly_movement() -> None:
    calm = build(quote=lambda i: (100.0, 100.0 + (i % 2) * 0.1, 1.0))
    wild = build(quote=lambda i: (100.0, 100.0 + (i % 2) * 0.4, 1.0))
    assert wild[0].features["rv_7"] > calm[0].features["rv_7"] > 0.0


def test_volatility_ratio_is_one_when_both_windows_move_alike() -> None:
    """Not exactly one: the windows hold 167 and 719 returns, so their sample spreads differ
    slightly on an alternating series. Within a percent is the honest expectation."""
    rows = build(quote=lambda i: (100.0, 100.0 + (i % 2) * 0.1, 1.0))
    assert rows[0].features["rv_7_over_30"] == pytest.approx(1.0, rel=0.01)


def test_a_completely_flat_thirty_days_is_fatal() -> None:
    """Thirty days of BTC without a single price move is broken data, not a quiet market."""
    flat = hourly_series(SERIES_START, SERIES_HOURS, quote=lambda i: (100.0, 100.0, 1.0))
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows(flat, FIRST_DAY, LAST_DAY)
    assert caught.value.kind == "degenerate_window"


def test_features_ignore_every_bar_at_or_after_the_feature_cutoff() -> None:
    """The 00:00 and 01:00 bars of a Decision Day exist, but the model must not have seen them."""
    cutoff_index = int((datetime(2021, 2, 1, tzinfo=UTC) - SERIES_START).total_seconds() // 3600)

    def base(i: int) -> tuple[float, float, float]:
        return (100.0, 100.0 + (i % 2) * 0.1, 1.0)

    def poisoned(i: int) -> tuple[float, float, float]:
        if i in (cutoff_index, cutoff_index + 1):  # 00:00 and 01:00 of the first Decision Day
            return (1e6, 1e6, 1e9)
        return base(i)

    clean_row = build(quote=base)[0]
    poisoned_row = build(quote=poisoned)[0]

    assert poisoned_row.features == clean_row.features
    assert poisoned_row.anchor_price == clean_row.anchor_price
    assert poisoned_row.decision_price != clean_row.decision_price  # the 01:00 open did move


def test_incomplete_day_aggregates_use_the_bars_that_exist() -> None:
    """Volume is a per-bar mean, so a missing hour does not read as a quiet hour."""
    missing = [i for i in range(24 * 25, 24 * 26)][:6]  # six hours dropped from one buffer day
    rows = build(skip=missing)
    assert rows[0].features["volume_7_over_30"] == pytest.approx(1.0)


def test_incomplete_decision_day_still_produces_a_row_carrying_its_bar_count() -> None:
    """Two hours of the Decision Day itself are absent; the day is still decided."""
    rows = build(skip=[725, 726])  # 05:00 and 06:00 of 2021-02-01
    assert rows[0].day == FIRST_DAY
    assert rows[0].bars_in_day == 22
    assert rows[0].decision_price > 0


def test_warm_up_underrun_is_fatal() -> None:
    """A trailing window may not reach earlier than the Warm-Up Buffer."""
    late = hourly_series(datetime(2021, 1, 20, tzinfo=UTC), 20 * 24)
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows(late, FIRST_DAY, LAST_DAY)
    assert caught.value.kind == "warm_up_underrun"


def test_missing_label_bar_is_fatal() -> None:
    """The last row's Label needs the next Decision Day's 01:00 bar."""
    short = hourly_series(SERIES_START, 31 * 24)  # ends 2021-02-01T23:00, before the next 01:00
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows(short, FIRST_DAY, FIRST_DAY)
    assert caught.value.kind == "missing_decision_day_bar"


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_real_snapshot_yields_the_frozen_research_window() -> None:
    bars = load_archive_backfill(SNAPSHOT).bars
    rows = build_decision_day_rows(bars, RESEARCH_WINDOW_FIRST_DAY, RESEARCH_WINDOW_LAST_DAY)

    assert len(rows) == 1_764
    assert rows[0].day == date(2021, 2, 1)
    assert rows[-1].day == date(2025, 11, 30)
    assert all(set(row.features) == set(ARM_A_FEATURES) for row in rows)
    assert all(row.label in (0, 1) for row in rows)

    short_days = [row for row in rows if row.bars_in_day < 24]
    assert [row.day for row in short_days] == [
        date(2021, 2, 11),
        date(2021, 3, 6),
        date(2021, 4, 20),
        date(2021, 4, 25),
        date(2021, 8, 13),
        date(2021, 9, 29),
        date(2023, 3, 24),
    ]

    first_cutoff = int(datetime(2021, 2, 1, tzinfo=UTC).timestamp() * 1000)
    assert rows[0].feature_cutoff_ms == first_cutoff
    assert rows[0].label_end_ms == first_cutoff + 24 * HOUR_MS + HOUR_MS


def _shift(bars: list[Bar], index: int, minutes: int) -> list[Bar]:
    moved = bars[index]._replace(open_time_ms=bars[index].open_time_ms + minutes * 60_000)
    return [*bars[:index], moved, *bars[index + 1 :]]


RISING = cast("Quote", lambda i: (100.0 + i, 200.0 + i, 1.0))
CUTOFF_INDEX = 720  # 2021-02-01T00:00Z, thirty days after the series starts


def test_anchor_ignores_a_bar_that_is_still_open_at_the_cutoff() -> None:
    """A bar opening at 23:47 closes at 00:47, after the cutoff: its close is not yet observable."""
    bars = hourly_series(SERIES_START, SERIES_HOURS, quote=RISING)
    rows = build_decision_day_rows(_shift(bars, CUTOFF_INDEX - 1, 47), FIRST_DAY, FIRST_DAY)
    assert rows[0].anchor_price == bars[CUTOFF_INDEX - 2].close


def test_a_seven_day_window_without_movement_is_fatal() -> None:
    """The degeneracy rule applies to both windows, not only the thirty-day one."""

    def flat_recently(i: int) -> tuple[float, float, float]:
        return (100.0, 100.0 + i, 1.0) if i < CUTOFF_INDEX - 7 * 24 else (100.0, 100.0, 1.0)

    bars = hourly_series(SERIES_START, SERIES_HOURS, quote=flat_recently)
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows(bars, FIRST_DAY, FIRST_DAY)
    assert caught.value.kind == "degenerate_window"
    assert "7-day" in str(caught.value)


def test_an_anchor_older_than_a_day_is_fatal() -> None:
    """A whole day without a bar leaves no Feature Anchor Price; r_1 would silently be zero."""
    bars = hourly_series(
        SERIES_START, SERIES_HOURS, quote=RISING, skip=range(CUTOFF_INDEX - 30, CUTOFF_INDEX)
    )
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows(bars, FIRST_DAY, FIRST_DAY)
    assert caught.value.kind == "stale_anchor"


def test_unaligned_decision_price_bar_is_reported_as_unaligned() -> None:
    """The same distinction ingest makes: present-but-misaligned is not absent."""
    bars = hourly_series(SERIES_START, SERIES_HOURS, quote=RISING)
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows(_shift(bars, CUTOFF_INDEX + 1, 13), FIRST_DAY, FIRST_DAY)
    assert caught.value.kind == "unaligned_decision_day_bar"


def test_duplicate_bars_are_rejected_by_the_builder() -> None:
    """The builder takes any bar sequence, including the Recent Tail: it validates its input."""
    bars = hourly_series(SERIES_START, SERIES_HOURS, quote=RISING)
    with pytest.raises(FatalDefect) as caught:
        build_decision_day_rows([*bars, bars[CUTOFF_INDEX + 1]], FIRST_DAY, FIRST_DAY)
    assert caught.value.kind == "duplicate_open_time"
