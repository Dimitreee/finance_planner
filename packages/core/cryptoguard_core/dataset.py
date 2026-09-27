"""One row per Decision Day: the object every model, backtest and published decision derives from.

Two prices, never conflated. The **Feature Anchor Price** is the `close` of the last Closed Bar
before the Feature Cutoff — the most recent price a model may see. The **Decision Price** is the
`open` of the 01:00 UTC bar, one hour later, and anchors both the Label and the simulated fill. The
hour between them is the operational gap in which a run is computed and published, and nothing may
be added to close it.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from cryptoguard_core.ingest import HOUR_MS, Bar, FatalDefect

DAY_MS = 24 * HOUR_MS
BARS_PER_DAY = 24

# Frozen by ADR-0004 and ADR-0006. January 2021 is spent as the Warm-Up Buffer.
RESEARCH_WINDOW_FIRST_DAY = date(2021, 2, 1)
RESEARCH_WINDOW_LAST_DAY = date(2025, 11, 30)
WARM_UP_DAYS = 30

ARM_A_FEATURES = (
    "r_1",
    "r_3",
    "r_7",
    "r_14",
    "rv_7",
    "rv_7_over_30",
    "volume_7_over_30",
)

_RETURN_LAGS = (1, 3, 7, 14)
_SHORT_WINDOW_DAYS = 7
_LONG_WINDOW_DAYS = 30


@dataclass(frozen=True, slots=True)
class DecisionDayRow:
    """One Decision Day. `bars_in_day` is metadata about data quality, never a feature."""

    day: date
    feature_cutoff_ms: int
    anchor_price: float
    decision_price: float
    label: int
    label_end_ms: int
    bars_in_day: int
    features: Mapping[str, float]


def _iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


def _midnight_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


class _Series:
    """The bar series, indexed for the two lookups the features need."""

    def __init__(self, bars: Sequence[Bar]) -> None:
        self._bars = sorted(bars, key=lambda bar: bar.open_time_ms)
        self._times = [bar.open_time_ms for bar in self._bars]
        duplicates = [time for time, count in Counter(self._times).items() if count > 1]
        if duplicates:
            # This function takes any bar sequence, including the Recent Tail, so it validates its
            # own input rather than trusting that ingest already did.
            raise FatalDefect(
                "duplicate_open_time",
                f"{len(duplicates)} duplicate open_time values, first at {_iso(min(duplicates))}",
            )
        self._by_time = {bar.open_time_ms: bar for bar in self._bars}
        self._hours = {time // HOUR_MS for time in self._times}

    @property
    def first_time_ms(self) -> int | None:
        return self._times[0] if self._times else None

    def require_bar(self, open_time_ms: int, context: str) -> Bar:
        """The bar opening exactly then, distinguishing absent from present-but-misaligned."""
        bar = self._by_time.get(open_time_ms)
        if bar is not None:
            return bar
        if open_time_ms // HOUR_MS in self._hours:
            raise FatalDefect(
                "unaligned_decision_day_bar",
                f"{context}: a bar exists in that hour but is not aligned to it",
            )
        raise FatalDefect("missing_decision_day_bar", f"{context}: the bar is absent")

    def anchor_closed_by(self, cutoff_ms: int) -> float:
        """The close of the last bar that had already closed at the cutoff.

        A bar opening at 23:47 is still open at midnight, so its close is not observable yet.
        `close_time` is never trusted, so "closed" is expressed through `open_time` alone.
        """
        index = bisect.bisect_right(self._times, cutoff_ms - HOUR_MS) - 1
        if index < 0:
            raise FatalDefect("warm_up_underrun", f"no bar has closed by {_iso(cutoff_ms)}")
        age_ms = cutoff_ms - self._times[index]
        if age_ms > DAY_MS:
            raise FatalDefect(
                "stale_anchor",
                f"the newest bar closed by {_iso(cutoff_ms)} opened {age_ms // HOUR_MS} hours "
                "earlier; that day has no Feature Anchor Price",
            )
        return self._bars[index].close

    def window(self, cutoff_ms: int, days: int) -> list[Bar]:
        """Bars opening in [cutoff - days, cutoff): Closed Bars only, cutoff excluded."""
        start = bisect.bisect_left(self._times, cutoff_ms - days * DAY_MS)
        end = bisect.bisect_left(self._times, cutoff_ms)
        return self._bars[start:end]

    def count_in_day(self, cutoff_ms: int) -> int:
        start = bisect.bisect_left(self._times, cutoff_ms)
        end = bisect.bisect_left(self._times, cutoff_ms + DAY_MS)
        return end - start


def _hourly_log_returns(bars: Sequence[Bar]) -> list[float]:
    """Close-to-close returns between bars that are genuinely adjacent.

    A pair separated by a gap is skipped rather than treated as one hour: a two-hour move recorded
    as hourly would understate nothing and overstate volatility, and neither is measured here.
    """
    returns: list[float] = []
    for previous, current in zip(bars, bars[1:], strict=False):
        if current.open_time_ms - previous.open_time_ms != HOUR_MS:
            continue
        returns.append(math.log(current.close / previous.close))
    return returns


def _realised_volatility(bars: Sequence[Bar]) -> float:
    returns = _hourly_log_returns(bars)
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns)


def _mean_volume(bars: Sequence[Bar]) -> float:
    """A per-bar mean, so a missing hour does not read as a quiet hour."""
    if not bars:
        return 0.0
    return statistics.fmean(bar.volume for bar in bars)


def _window_stats(series: _Series, cutoff_ms: int, days: int) -> tuple[float, float]:
    """Realised volatility and mean volume over a trailing window, or a Fatal Defect.

    A window with no price movement or no volume is broken data, not a quiet market, and the rule
    applies to every window rather than only the longest.
    """
    bars = series.window(cutoff_ms, days)
    volatility = _realised_volatility(bars)
    volume = _mean_volume(bars)
    if volatility == 0.0 or volume == 0.0:
        raise FatalDefect(
            "degenerate_window",
            f"the {days}-day window before {_iso(cutoff_ms)} has no price movement or no volume; "
            "that is broken data, not a quiet market",
        )
    return volatility, volume


def _features(series: _Series, cutoff_ms: int, anchor: float) -> dict[str, float]:
    features: dict[str, float] = {}
    for lag in _RETURN_LAGS:
        past = series.anchor_closed_by(cutoff_ms - lag * DAY_MS)
        features[f"r_{lag}"] = math.log(anchor / past)

    short_volatility, short_volume = _window_stats(series, cutoff_ms, _SHORT_WINDOW_DAYS)
    long_volatility, long_volume = _window_stats(series, cutoff_ms, _LONG_WINDOW_DAYS)

    features["rv_7"] = short_volatility
    features["rv_7_over_30"] = short_volatility / long_volatility
    features["volume_7_over_30"] = short_volume / long_volume
    return features


def build_decision_day_rows(
    bars: Sequence[Bar],
    first_day: date = RESEARCH_WINDOW_FIRST_DAY,
    last_day: date = RESEARCH_WINDOW_LAST_DAY,
) -> tuple[DecisionDayRow, ...]:
    """Build one row per Decision Day in [first_day, last_day].

    Trailing features read the Warm-Up Buffer, the 30 days before `first_day`; the buffer is never
    scored, fitted on or reported. The build fails if the bars do not reach back that far, because a
    window that quietly starts late would report a different market than the one it names.
    """
    series = _Series(bars)
    first_cutoff = _midnight_ms(first_day)
    earliest_needed = first_cutoff - WARM_UP_DAYS * DAY_MS
    first_available = series.first_time_ms
    if first_available is None or first_available > earliest_needed:
        raise FatalDefect(
            "warm_up_underrun",
            f"the Warm-Up Buffer needs bars from "
            f"{datetime.fromtimestamp(earliest_needed / 1000, tz=UTC).isoformat()}, "
            f"but the series starts later",
        )

    rows: list[DecisionDayRow] = []
    day = first_day
    while day <= last_day:
        cutoff = _midnight_ms(day)
        next_cutoff = _midnight_ms(day + timedelta(days=1))
        decision_bar = series.require_bar(
            cutoff + HOUR_MS,
            f"the 01:00 UTC bar of {day.isoformat()}, which carries its Decision Price",
        )
        next_decision_bar = series.require_bar(
            next_cutoff + HOUR_MS,
            f"the 01:00 UTC bar of {(day + timedelta(days=1)).isoformat()}, "
            f"which resolves the Label of {day.isoformat()}",
        )

        anchor = series.anchor_closed_by(cutoff)
        rows.append(
            DecisionDayRow(
                day=day,
                feature_cutoff_ms=cutoff,
                anchor_price=anchor,
                decision_price=decision_bar.open,
                label=int(next_decision_bar.open > decision_bar.open),
                label_end_ms=next_cutoff + HOUR_MS,
                bars_in_day=series.count_in_day(cutoff),
                features=_features(series, cutoff, anchor),
            )
        )
        day += timedelta(days=1)

    return tuple(rows)
