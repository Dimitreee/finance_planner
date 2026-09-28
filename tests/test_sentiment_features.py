"""Arm C's two features: what the Sentiment Score aggregates to, and what absence means.

Nothing here loads the Sentiment Extractor. The scores are supplied directly so the expected
aggregates are stated rather than recomputed by the code under test; the extractor itself is
exercised in `test_sentiment_extractor.py`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news_features import (
    ARM_C_EXTRA_FEATURES,
    HeadlineAvailability,
    headline_count_features,
    sentiment_features,
)

CUTOFF = datetime(2021, 2, 1, tzinfo=UTC)
CUTOFF_MS = int(CUTOFF.timestamp() * 1000)
HOUR = timedelta(hours=1)


def signal(*pairs: tuple[datetime, float], lag_hours: int = 1) -> HeadlineAvailability:
    """An availability set carrying a Sentiment Score beside each instant, sorted by instant."""
    shifted = sorted((moment + lag_hours * HOUR, score) for moment, score in pairs)
    return HeadlineAvailability(
        asset="BTC",
        lag_hours=lag_hours,
        instants_ms=tuple(int(m.timestamp() * 1000) for m, _ in shifted),
        covers_through_ms=CUTOFF_MS,
        scores=tuple(score for _, score in shifted),
    )


def test_the_arm_c_feature_names_are_exactly_the_two_the_contract_froze() -> None:
    assert ARM_C_EXTRA_FEATURES == ("sentiment_mean_24h", "sentiment_mean_7d")


def test_the_means_are_unweighted_over_their_availability_windows() -> None:
    features = sentiment_features(
        signal(
            (CUTOFF - 2 * HOUR, 0.8),
            (CUTOFF - 3 * HOUR, -0.4),
            (CUTOFF - timedelta(days=4), 0.2),
        ),
        CUTOFF_MS,
    )
    assert features["sentiment_mean_24h"] == pytest.approx((0.8 - 0.4) / 2)
    assert features["sentiment_mean_7d"] == pytest.approx((0.8 - 0.4 + 0.2) / 3)


def test_the_seven_day_window_includes_today_because_a_level_cannot_saturate() -> None:
    """Unlike the count ratio, a mean is not divided by a window that contains it (ADR-0021)."""
    features = sentiment_features(signal((CUTOFF - 2 * HOUR, 0.5)), CUTOFF_MS)
    assert features["sentiment_mean_24h"] == pytest.approx(0.5)
    assert features["sentiment_mean_7d"] == pytest.approx(0.5)


def test_an_absent_window_is_not_a_zero_but_a_gap() -> None:
    """Absence of news is not neutral sentiment. NaN is what the in-fold median imputer consumes."""
    features = sentiment_features(signal((CUTOFF - timedelta(days=3), 0.6)), CUTOFF_MS)
    assert math.isnan(features["sentiment_mean_24h"])
    assert features["sentiment_mean_7d"] == pytest.approx(0.6)


def test_both_windows_can_be_absent_together() -> None:
    features = sentiment_features(signal((CUTOFF - timedelta(days=30), 0.6)), CUTOFF_MS)
    assert math.isnan(features["sentiment_mean_24h"])
    assert math.isnan(features["sentiment_mean_7d"])


def test_a_headline_not_yet_available_contributes_to_neither_window() -> None:
    features = sentiment_features(
        signal((CUTOFF, 0.9), (CUTOFF - 2 * HOUR, -0.2), lag_hours=1), CUTOFF_MS
    )
    # The 0.9 headline states at the cutoff, so at lag 1 it is available an hour after it.
    assert features["sentiment_mean_24h"] == pytest.approx(-0.2)


def test_scores_must_come_one_per_instant() -> None:
    with pytest.raises(FatalDefect) as caught:
        HeadlineAvailability(
            asset="BTC",
            lag_hours=1,
            instants_ms=(CUTOFF_MS - 1000, CUTOFF_MS),
            covers_through_ms=CUTOFF_MS,
            scores=(0.5,),
        )
    assert caught.value.kind == "news_scores_misaligned"


def test_asking_for_sentiment_without_scores_is_refused_not_guessed() -> None:
    without = HeadlineAvailability(
        asset="BTC", lag_hours=1, instants_ms=(CUTOFF_MS,), covers_through_ms=CUTOFF_MS
    )
    with pytest.raises(FatalDefect) as caught:
        sentiment_features(without, CUTOFF_MS)
    assert caught.value.kind == "news_scores_absent"


def test_absent_sentiment_and_a_zero_headline_count_are_the_same_day() -> None:
    """The two arms must agree on which days are silent, or they read different corpora.

    Arm B reports a measured zero and Arm C reports a gap, which is the intended difference. What
    may not differ is *when*: a day Arm B counts as empty is exactly a day Arm C cannot average.
    The report's cross-check between the two rests on this, so it is pinned here rather than
    inferred from both being written against the same window constants.
    """
    quiet = signal((CUTOFF - timedelta(days=3), 0.8))
    assert headline_count_features(quiet, CUTOFF_MS)["news_count_24h_log1p"] == 0.0
    assert math.isnan(sentiment_features(quiet, CUTOFF_MS)["sentiment_mean_24h"])

    busy = signal((CUTOFF - timedelta(days=3), 0.8), (CUTOFF - 2 * HOUR, 0.4))
    assert headline_count_features(busy, CUTOFF_MS)["news_count_24h_log1p"] > 0.0
    assert sentiment_features(busy, CUTOFF_MS)["sentiment_mean_24h"] == pytest.approx(0.4)
