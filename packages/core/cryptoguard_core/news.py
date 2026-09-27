"""Reading the archived news records, and measuring what their timestamps can support.

The archive states a publication time with no timezone and records no first-seen time, so nothing
here converts a record into an instant. Attaching a zone at parse time would turn an assumption into
a silent fact; the assumption is declared once, in the Experiment Contract, as an Assumed
Availability Lag, and every evaluation built on this archive is named Retrospective With Assumed
Latency.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from cryptoguard_core.ingest import FatalDefect

# Default local location of the extracted archive; it is not tracked in git.
NEWS_CSV = Path(__file__).resolve().parents[3] / (
    "data/raw/news/news_currencies_source_joinedResult.csv"
)

_NULL = "NULL"
_STATED_FORMAT = "%Y-%m-%d %H:%M:%S"
_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")

# A window of this many consecutive hours is where the daily peak is looked for.
PEAK_WINDOW_HOURS = 8
# Crypto news clusters in US and EU business hours. Read in UTC, that peak starts in this band; a
# peak starting outside it means the stated times are offset by an amount we have not established.
_UTC_PEAK_BAND = range(10, 17)

UTC_CONSISTENT_LAG_HOURS = 6
AMBIGUOUS_LAG_HOURS = 24


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One archived headline, holding exactly what the file says and nothing more."""

    item_id: str
    title: str
    description: str | None
    source_domain: str | None
    source_url: str | None
    stated_at: datetime  # tz-naive on purpose: the file states no zone
    stated_at_raw: str
    stated_timezone_suffix: str | None
    currencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TimestampDiagnostic:
    """What the stated times look like, measured without reference to prices or Labels."""

    records: int
    hour_of_day: Mapping[int, int]
    with_timezone_suffix: int
    zero_minutes: int
    zero_seconds: int
    earliest: datetime
    latest: datetime


def _or_none(value: str | None) -> str | None:
    if value is None or value == "" or value == _NULL:
        return None
    return value


def _parse_stated_at(raw: str, item_id: str) -> tuple[datetime, str | None]:
    match = _TZ_SUFFIX.search(raw)
    suffix = match.group(0) if match else None
    body = raw[: match.start()].strip() if match else raw
    try:
        return datetime.strptime(body, _STATED_FORMAT), suffix
    except ValueError as error:
        raise FatalDefect(
            "news_timestamp_format",
            f"record {item_id}: newsDatetime {raw!r} is not '{_STATED_FORMAT}': {error}",
        ) from error


def read_news_items(path: Path) -> Iterator[NewsItem]:
    """Stream the archive. Titles may span physical lines, so records outnumber neither way."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row["newsDatetime"] or ""
            stated_at, suffix = _parse_stated_at(raw, row["id"])
            currencies = _or_none(row["currencies"])
            yield NewsItem(
                item_id=row["id"],
                title=row["title"],
                description=_or_none(row["description"]),
                source_domain=_or_none(row["sourceDomain"]),
                source_url=_or_none(row["sourceUrl"]),
                stated_at=stated_at,
                stated_at_raw=raw,
                stated_timezone_suffix=suffix,
                currencies=tuple(currencies.split(",")) if currencies else (),
            )


def mentions_asset(item: NewsItem, asset: str) -> bool:
    """Exact token match. A substring test would pull in WBTC, HBTC, BTCST and BTCUSD."""
    return asset in item.currencies


def diagnose_timestamps(items: Iterable[NewsItem]) -> TimestampDiagnostic:
    hours: Counter[int] = Counter()
    suffixes = zero_minutes = zero_seconds = records = 0
    earliest: datetime | None = None
    latest: datetime | None = None

    for item in items:
        records += 1
        hours[item.stated_at.hour] += 1
        suffixes += item.stated_timezone_suffix is not None
        zero_minutes += item.stated_at.minute == 0
        zero_seconds += item.stated_at.second == 0
        if earliest is None or item.stated_at < earliest:
            earliest = item.stated_at
        if latest is None or item.stated_at > latest:
            latest = item.stated_at

    if earliest is None or latest is None:
        # A caller mistake, not a data defect: naming it one would send a reader hunting for
        # corrupt rows that do not exist.
        raise ValueError("diagnose_timestamps received no records")

    return TimestampDiagnostic(
        records=records,
        hour_of_day=dict(hours),
        with_timezone_suffix=suffixes,
        zero_minutes=zero_minutes,
        zero_seconds=zero_seconds,
        earliest=earliest,
        latest=latest,
    )


def busiest_window_start(diagnostic: TimestampDiagnostic, hours: int = PEAK_WINDOW_HOURS) -> int:
    """The starting hour of the `hours` consecutive hours holding the most records."""
    counts = [diagnostic.hour_of_day.get(hour, 0) for hour in range(24)]
    return max(
        range(24),
        key=lambda start: sum(counts[(start + offset) % 24] for offset in range(hours)),
    )


def choose_primary_lag(diagnostic: TimestampDiagnostic) -> int:
    """The Primary Lag, in hours, decided from the stated times alone.

    This touches neither prices nor Labels, so the choice cannot be informed by the outcome — which
    is what makes it legitimate pre-registration rather than a decision made after seeing results.
    """
    if diagnostic.with_timezone_suffix:
        # `_parse_stated_at` strips an offset and keeps the naive body, so such a record enters the
        # histogram as though it shared everyone else's frame. A mixed-frame histogram cannot
        # support this decision, and quietly returning the conservative lag would hide that.
        raise FatalDefect(
            "mixed_timezone_offsets",
            f"{diagnostic.with_timezone_suffix} of {diagnostic.records} records carry a timezone "
            "offset while the rest do not; the hour-of-day histogram mixes frames and cannot "
            "decide the Primary Lag",
        )
    return (
        UTC_CONSISTENT_LAG_HOURS
        if busiest_window_start(diagnostic) in _UTC_PEAK_BAND
        else AMBIGUOUS_LAG_HOURS
    )


def daily_counts_by_stated_date(items: Iterable[NewsItem]) -> Mapping[date, int]:
    """Records per stated calendar date, before any lag or zone correction is applied."""
    return Counter(item.stated_at.date() for item in items)
