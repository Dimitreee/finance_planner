"""The moving-block bootstrap: how blocks are drawn, and what an interval is allowed to say.

The method was chosen because Decision Days are not independent draws — a trending series resampled
day by day looks steadier than it is. That reason is demonstrated against an iid bootstrap on the
same autocorrelated series rather than asserted in a comment.

Nothing here spends a Trial. The bootstrap estimates uncertainty around results already produced and
selects nothing (ADR-0017).
"""

from __future__ import annotations

import math
import random
import statistics
from datetime import date, timedelta

import pytest
from cryptoguard_core.backtest import (
    BASE_SCENARIO,
    BacktestResult,
    daily_log_returns,
    excess_return,
    run_backtest,
)
from cryptoguard_core.bootstrap import (
    INCONCLUSIVE,
    SEED,
    SEPARATED,
    STABILITY_SEEDS,
    block_indices,
    iid_bootstrap,
    moving_block_bootstrap,
)
from cryptoguard_core.dataset import ARM_A_FEATURES, DecisionDayRow
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import PolicyConfig, Position


def indices_at(seed: int, *, length: int = 50, block_days: int = 7) -> list[int]:
    return block_indices(length, block_days, random.Random(seed))


def test_a_resample_is_exactly_as_long_as_the_series_it_came_from() -> None:
    """A shorter resample would estimate the uncertainty of a shorter experiment."""
    for length in (1, 7, 50, 1096):
        assert len(indices_at(1, length=length, block_days=min(20, length))) == length


def test_the_draw_is_reproducible_from_the_seed_and_moves_when_it_changes() -> None:
    assert indices_at(4) == indices_at(4)
    assert indices_at(4) != indices_at(5)


def test_every_index_stays_inside_the_series() -> None:
    """The wrap is modular, so an index past the end comes back to the front, never out of range."""
    drawn = indices_at(3, length=50, block_days=20)
    assert min(drawn) >= 0
    assert max(drawn) < 50


def test_a_block_as_long_as_the_series_makes_every_resample_a_rotation() -> None:
    """The sharpest consequence of wrapping rather than truncating, pinned so the rule cannot drift.

    Truncating would instead return the series from a random start, padded by nothing — shorter, and
    with the days near the end appearing less often than the days near the beginning.
    """
    length = 12
    for seed in range(6):
        drawn = block_indices(length, length, random.Random(seed))
        assert sorted(drawn) == list(range(length))
        start = drawn[0]
        assert drawn == [(start + offset) % length for offset in range(length)]


def test_the_last_block_is_truncated_so_the_length_is_exact() -> None:
    """Blocks of seven cannot tile fifty days, so the final one is cut rather than overshooting."""
    drawn = block_indices(50, 7, random.Random(1))
    assert len(drawn) == 50


def test_a_block_longer_than_the_series_is_refused() -> None:
    with pytest.raises(FatalDefect) as caught:
        block_indices(10, 11, random.Random(1))
    assert caught.value.kind == "bootstrap_block_invalid"


def test_a_block_of_zero_days_is_refused() -> None:
    with pytest.raises(FatalDefect) as caught:
        block_indices(10, 0, random.Random(1))
    assert caught.value.kind == "bootstrap_block_invalid"


# --- The interval, and why blocks -----------------------------------------------------------------


def autocorrelated(length: int, *, phi: float, seed: int) -> list[float]:
    """An AR(1) series: each day carries most of the day before — the case blocks exist for."""
    rng = random.Random(seed)
    value = 0.0
    out: list[float] = []
    for _ in range(length):
        value = phi * value + rng.gauss(0.0, 1.0)
        out.append(value)
    return out


def test_a_constant_series_has_no_uncertainty_to_report() -> None:
    """The simplest check that the plumbing is not inventing spread."""
    interval = moving_block_bootstrap(
        [3.0] * 40, statistics.fmean, block_days=5, resamples=50, seed=1, confidence=0.95
    )
    assert interval.estimate == pytest.approx(3.0)
    assert interval.lower == pytest.approx(3.0)
    assert interval.upper == pytest.approx(3.0)
    assert interval.width == pytest.approx(0.0)


def test_the_interval_is_reproducible_from_the_seed_and_moves_when_it_changes() -> None:
    series = autocorrelated(200, phi=0.8, seed=7)
    first = moving_block_bootstrap(series, statistics.fmean, block_days=20, resamples=200, seed=11)
    again = moving_block_bootstrap(series, statistics.fmean, block_days=20, resamples=200, seed=11)
    other = moving_block_bootstrap(series, statistics.fmean, block_days=20, resamples=200, seed=12)

    assert (first.lower, first.upper) == (again.lower, again.upper)
    assert (first.lower, first.upper) != (other.lower, other.upper)
    assert first.estimate == other.estimate  # the point estimate is the data, not the draw


def test_blocks_give_a_wider_interval_than_iid_on_an_autocorrelated_series() -> None:
    """The whole reason the method was chosen, measured rather than asserted.

    An AR(1) series with phi = 0.9 has a variance inflation factor of (1+phi)/(1-phi) = 19 for its
    mean, so resampling it one day at a time reports a far steadier series than it is. Measured on
    this series: the block interval spans 1.218 against the iid interval's 0.399, a ratio of 3.05.
    Note the direction of the harm — the iid interval here is [-0.924, -0.524] and excludes zero,
    while the honest one is [-1.341, -0.123] and is far closer to it.
    """
    series = autocorrelated(400, phi=0.9, seed=3)
    blocks = moving_block_bootstrap(series, statistics.fmean, block_days=20, resamples=1500, seed=5)
    independent = iid_bootstrap(series, statistics.fmean, resamples=1500, seed=5)
    assert blocks.width > 1.5 * independent.width


def test_an_interval_spanning_zero_is_inconclusive_and_says_so() -> None:
    """ADR-0017: spanning zero is Inconclusive, never "a small effect"."""
    series = autocorrelated(300, phi=0.7, seed=9)  # mean zero by construction
    interval = moving_block_bootstrap(
        series, statistics.fmean, block_days=20, resamples=400, seed=2
    )
    assert interval.spans_zero
    assert interval.verdict == INCONCLUSIVE


def test_an_interval_clear_of_zero_is_not_inconclusive() -> None:
    shifted = [value + 20.0 for value in autocorrelated(300, phi=0.7, seed=9)]
    interval = moving_block_bootstrap(
        shifted, statistics.fmean, block_days=20, resamples=400, seed=2
    )
    assert not interval.spans_zero
    assert interval.verdict == SEPARATED


# --- The two quantities that get intervals --------------------------------------------------------


def backtest_over(days: int) -> BacktestResult:
    """A real backtest over a synthetic rising series, so the aggregates are non-trivial."""
    rows = tuple(
        DecisionDayRow(
            day=date(2022, 1, 1) + timedelta(days=index),
            feature_cutoff_ms=0,
            anchor_price=100.0 + index,
            decision_price=100.0 + index,
            label=1,
            label_end_ms=0,
            bars_in_day=24,
            features=dict.fromkeys(ARM_A_FEATURES, 0.0),
            news_lag_hours=None,
        )
        for index in range(days)
    )
    return run_backtest(
        rows,
        PreviousDirectionBaseline(),
        PolicyConfig(to_btc_at=0.55, to_usdt_at=0.45),
        BASE_SCENARIO,
        initial=Position(usdt=1000.0, btc=0.0),
        arm="A",
        probabilities={row.day: 0.9 if index % 7 else 0.1 for index, row in enumerate(rows)},
    )


def test_the_daily_return_series_compounds_back_to_the_published_aggregates() -> None:
    """The interval has to cover the number that is published, not a near relative of it.

    This is what makes the return bootstrap honest: the per-day decomposition it resamples
    multiplies back to exactly the `net_return` and `buy_and_hold_return` the Replay reports.
    """
    result = backtest_over(90)
    days = daily_log_returns(result)
    assert len(days) == result.days

    strategy = math.exp(math.fsum(day.strategy for day in days)) - 1
    held = math.exp(math.fsum(day.buy_and_hold for day in days)) - 1
    assert strategy == pytest.approx(result.net_return, abs=1e-12)
    assert held == pytest.approx(result.buy_and_hold_return, abs=1e-12)


def test_the_return_statistic_is_the_gap_to_buy_and_hold() -> None:
    """The quantity with the interval is the difference, so spanning zero means Inconclusive."""
    result = backtest_over(90)
    assert excess_return(daily_log_returns(result)) == pytest.approx(
        result.net_return - result.buy_and_hold_return, abs=1e-12
    )


def test_the_verdict_does_not_depend_on_which_seed_was_drawn() -> None:
    """The seed was chosen after the point estimates existed, so its influence is measured.

    What must not move is the conclusion. A separated series stays separated and a zero-mean one
    stays Inconclusive across every seed tried; only the interval's edges wander, and by less than
    the distance to zero.
    """
    separated = [value + 20.0 for value in autocorrelated(300, phi=0.7, seed=9)]
    zero_mean = autocorrelated(300, phi=0.7, seed=9)
    for seed in (SEED, *STABILITY_SEEDS):
        assert (
            moving_block_bootstrap(
                separated, statistics.fmean, block_days=20, resamples=300, seed=seed
            ).verdict
            == SEPARATED
        )
        assert (
            moving_block_bootstrap(
                zero_mean, statistics.fmean, block_days=20, resamples=300, seed=seed
            ).verdict
            == INCONCLUSIVE
        )
