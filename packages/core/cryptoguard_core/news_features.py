"""Headline counts per Decision Day: the Arm B features, and the lag that gates them.

Both windows are taken over **availability instants** — a stated publication time read as UTC plus
the Assumed Availability Lag — and never over stated times. The distinction decides whether the
features exist at all: a 24-hour window over stated times, intersected with "available by the
cutoff", collapses to a single instant when the lag is 24 hours, so the count would be identically
zero at the Primary Lag. ADR-0020 records that reading and the baseline window below.

Read the recent window as the 24 hours of *visibility* ending at the Feature Cutoff. At lag L that
is
the same set as the headlines stated in the 24 hours ending at `cutoff - L`, which is what a system
running that day would have had in front of it. A larger lag therefore slides the window back
through
the calendar rather than shrinking it, and a day's count is not monotone in the lag.

Coverage is declared, not inferred. Whether the archive reaches a Decision Day cannot be read off
the
items themselves: a silent week and an archive that stops look identical from the inside. So an
availability set carries the span the archive claims, and a build outside that span is refused.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.news import NewsItem, mentions_asset

DAY_MS = 24 * HOUR_MS
DEFAULT_ASSET = "BTC"

# Frozen by ADR-0005 and `config/experiment.yaml`. A lag chosen at a call site is not a
# pre-registered lag, so the grid lives here as a constant the contract is validated against.
LAG_GRID_HOURS = (1, 6, 24)

COUNT_WINDOW_DAYS = 1
BASELINE_WINDOW_DAYS = 7
# One headline's worth of prior, so a silent baseline cannot divide by zero. See ADR-0020: a clamp
# would have to pick a ceiling, and this does not.
BASELINE_PRIOR = 1

ARM_B_EXTRA_FEATURES = (
    "news_count_24h_log1p",
    "news_count_24h_over_7d_mean",
)
# Named here so the contract can be pinned to it before Arm C is built; ticket 02 supplies the
# values.
ARM_C_EXTRA_FEATURES = (
    "sentiment_mean_24h",
    "sentiment_mean_7d",
)
# Any arm carrying one of these reads news, and its dataset must record which lag it read them
# under.
NEWS_FEATURE_NAMES = frozenset(ARM_B_EXTRA_FEATURES + ARM_C_EXTRA_FEATURES)


def _iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


@dataclass(frozen=True, slots=True)
class HeadlineAvailability:
    """When each attributed headline became usable, sorted, under one Assumed Availability Lag.

    Holding instants rather than items is deliberate: nothing downstream of here may reach a
    headline's text, its source or its stated time, so a feature cannot quietly depend on one.

    `covers_through_ms` is the instant up to which the archive claims to be complete, carried rather
    than guessed, because absence of headlines and absence of archive are indistinguishable here.
    """

    asset: str
    lag_hours: int
    instants_ms: tuple[int, ...]
    covers_through_ms: int
    # One Sentiment Score per instant, in the same order, or None when no extractor has been run.
    # Arm B needs only the instants; Arm C needs both, and asking for a mean without them is
    # refused rather than filled with a zero, because absence of news is not neutral sentiment.
    scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # Derivable, so checked rather than trusted: `count_in` bisects, and an unsorted sequence
        # would return counts that are wrong rather than an error, including negative ones.
        if any(
            later < earlier
            for earlier, later in zip(self.instants_ms, self.instants_ms[1:], strict=False)
        ):
            raise FatalDefect(
                "news_instants_unsorted",
                "availability instants must be sorted; count_in bisects them",
            )

        if self.scores is not None and len(self.scores) != len(self.instants_ms):
            raise FatalDefect(
                "news_scores_misaligned",
                f"{len(self.scores)} Sentiment Scores for {len(self.instants_ms)} instants; "
                "they are parallel arrays and a silent mismatch would mis-attribute every score",
            )

    @property
    def earliest_ms(self) -> int | None:
        return self.instants_ms[0] if self.instants_ms else None

    def _slice(self, *, after_ms: int, through_ms: int) -> tuple[int, int]:
        return (
            bisect.bisect_right(self.instants_ms, after_ms),
            bisect.bisect_right(self.instants_ms, through_ms),
        )

    def mean_score_in(self, *, after_ms: int, through_ms: int) -> float:
        """The unweighted mean Sentiment Score in the window, or NaN when the window is empty.

        NaN rather than zero: the in-fold median imputer consumes NaN, and zero would assert that
        the window's headlines were read as neutral when there were no headlines to read.
        """
        if self.scores is None:
            raise FatalDefect(
                "news_scores_absent",
                "this availability set carries no Sentiment Scores, so no mean exists; "
                "build it through the extractor rather than defaulting the value",
            )
        start, end = self._slice(after_ms=after_ms, through_ms=through_ms)
        if end == start:
            return math.nan
        return math.fsum(self.scores[start:end]) / (end - start)

    def count_in(self, *, after_ms: int, through_ms: int) -> int:
        """Items available in `(after_ms, through_ms]`: open at the far edge, closed at the cutoff.

        Closed at the cutoff because an item that became available exactly at 00:00 was available at
        00:00. Open at the far edge so that adjacent windows partition the timeline rather than
        sharing an instant.
        """
        start, end = self._slice(after_ms=after_ms, through_ms=through_ms)
        return end - start


def headline_availability(
    items: Iterable[NewsItem],
    *,
    lag_hours: int,
    covers_through_ms: int,
    asset: str = DEFAULT_ASSET,
) -> HeadlineAvailability:
    """Availability instants for the items attributed to `asset`, under one lag from the grid.

    The stated time is read as UTC. That reading is the declared assumption the archive cannot
    confirm, and the lag is what stands in for the first-seen time it does not carry.
    """
    if lag_hours not in LAG_GRID_HOURS:
        raise FatalDefect(
            "news_lag_not_pre_registered",
            f"lag {lag_hours}h is not in the pre-registered lag grid {list(LAG_GRID_HOURS)}; "
            "a lag chosen at a call site is not a pre-registered lag",
        )

    shift_ms = lag_hours * HOUR_MS
    instants = sorted(
        int(item.stated_at.replace(tzinfo=UTC).timestamp() * 1000) + shift_ms
        for item in items
        if mentions_asset(item, asset)
    )
    return HeadlineAvailability(
        asset=asset,
        lag_hours=lag_hours,
        instants_ms=tuple(instants),
        covers_through_ms=covers_through_ms,
    )


def require_news_span(
    availability: HeadlineAvailability, *, first_cutoff_ms: int, last_cutoff_ms: int
) -> None:
    """Refuse a build whose windows reach outside what the archive holds, at either end.

    Both ends are the same defect. A seven-day baseline that quietly began inside the archive would
    report a thin start as a quiet market; a recent window past the archive's end would report every
    day as silent, which is how an Arm B build for today would behave against an archive stopping in
    2025.
    """
    earliest_needed = first_cutoff_ms - BASELINE_WINDOW_DAYS * DAY_MS - COUNT_WINDOW_DAYS * DAY_MS
    earliest = availability.earliest_ms
    if earliest is None or earliest > earliest_needed:
        raise FatalDefect(
            "news_warm_up_underrun",
            f"the baseline window needs {availability.asset} headlines available by "
            f"{_iso(earliest_needed)}, but the earliest is "
            f"{_iso(earliest) if earliest is not None else 'none at all'}; a thin archive start "
            "would be reported as a quiet market",
        )
    if availability.covers_through_ms < last_cutoff_ms:
        raise FatalDefect(
            "news_coverage_underrun",
            f"the archive claims completeness only to {_iso(availability.covers_through_ms)}, "
            f"before the last Feature Cutoff {_iso(last_cutoff_ms)}; every day past the end would "
            "read as a silent 24 hours",
        )


def headline_count_features(availability: HeadlineAvailability, cutoff_ms: int) -> dict[str, float]:
    """The two Arm B features for the Decision Day whose Feature Cutoff is `cutoff_ms`."""
    recent = availability.count_in(
        after_ms=cutoff_ms - COUNT_WINDOW_DAYS * DAY_MS, through_ms=cutoff_ms
    )
    if recent == 0:
        # A measured zero, never imputed. The ratio is zero for the same reason: it says "nothing
        # today", which is true whether the baseline was busy or equally silent.
        return {"news_count_24h_log1p": 0.0, "news_count_24h_over_7d_mean": 0.0}

    # The baseline excludes today, so the ratio is unbounded above and can express the size of a
    # spike. A baseline window containing today would cap the ratio at the window length (ADR-0020).
    baseline_end = cutoff_ms - COUNT_WINDOW_DAYS * DAY_MS
    baseline = availability.count_in(
        after_ms=baseline_end - BASELINE_WINDOW_DAYS * DAY_MS, through_ms=baseline_end
    )
    mean_daily = (baseline + BASELINE_PRIOR) / BASELINE_WINDOW_DAYS
    return {
        "news_count_24h_log1p": math.log1p(recent),
        "news_count_24h_over_7d_mean": recent / mean_daily,
    }


def sentiment_features(availability: HeadlineAvailability, cutoff_ms: int) -> dict[str, float]:
    """Arm C's two features: the mean Sentiment Score over 24 hours and over 7 days.

    Both windows end at the Feature Cutoff and the seven-day window contains the recent one. A mean
    is a level, not a ratio, so nesting cannot saturate it — the reason Arm B's baseline excludes
    today and this one does not (ADR-0021).
    """
    return {
        "sentiment_mean_24h": availability.mean_score_in(
            after_ms=cutoff_ms - COUNT_WINDOW_DAYS * DAY_MS, through_ms=cutoff_ms
        ),
        "sentiment_mean_7d": availability.mean_score_in(
            after_ms=cutoff_ms - BASELINE_WINDOW_DAYS * DAY_MS, through_ms=cutoff_ms
        ),
    }
