"""Turning the Archive Backfill into a validated series of Closed Bars.

Two severities, per ADR-0002. A **Fatal Defect** makes a decision impossible or untrustworthy and
stops the run with nothing published. A **Data Quality Flag** records an imperfection and lets
processing continue, because the alternative — dropping days — cannot be carried out in live
operation, where today's advice is still owed and the 01:00 bar still exists.

`open_time` is the only time key used in arithmetic. One bar in the verified archive closes 759
seconds *before* it opens, so `close_time` is validated and never computed with.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, NamedTuple

HOUR_MS = 3_600_000
BARS_PER_DAY = 24
COLUMNS = 12

# Binance spot timestamps are milliseconds through 2024-12 and microseconds from 2025-01. The width
# is declared by the data, never inferred from magnitude: an unknown width stops the run.
_WIDTH_TO_MS_DIVISOR = {13: 1, 16: 1_000}

# A normal hourly bar closes one millisecond before the next opens.
_MIN_DURATION_MS = HOUR_MS - 1
_MAX_DURATION_MS = HOUR_MS

FatalKind = Literal[
    "empty_snapshot",
    "checksum_missing",
    "checksum_mismatch",
    "archive_member",
    "column_count",
    "timestamp_width",
    "numeric_field",
    "duplicate_open_time",
    "ohlc_invariant",
    "non_positive_price",
    "missing_decision_day_bar",
    "unaligned_decision_day_bar",
    "warm_up_underrun",
    "stale_anchor",
    "degenerate_window",
    "news_timestamp_format",
    "mixed_timezone_offsets",
    "contract_inconsistent",
    "contract_mismatch",
    "source_digest_mismatch",
    "missing_decision_day_row",
    "unknown_arm",
    "arm_requires_news",
    "arm_takes_no_news",
    "news_lag_not_pre_registered",
    "news_warm_up_underrun",
    "news_coverage_underrun",
    "news_instants_unsorted",
    "news_scores_misaligned",
    "news_scores_absent",
    "arm_requires_sentiment",
    "arm_takes_no_sentiment",
    "sentiment_cache_corrupt",
    "sentiment_extractor_unavailable",
    "sentiment_scores_incomplete",
    "annotation_sample_too_small",
    "annotation_revision_mismatch",
    "annotation_labels_invalid",
    "annotation_scores_missing",
    "annotation_nothing_compared",
    "annotation_draw_exists",
    "annotation_agreement_exists",
    "arm_feature_mismatch",
    "arm_lag_mismatch",
    "paired_days_mismatch",
    "paired_label_mismatch",
    "bootstrap_block_invalid",
    "bootstrap_series_empty",
    "bootstrap_parameter_invalid",
    "bootstrap_verdict_unstable",
    "final_holdout_spent",
    "final_holdout_not_frozen",
    "final_holdout_marker_corrupt",
    "sensitivity_arm_mismatch",
    "replay_series_empty",
    "replay_series_disagrees",
    "rows_out_of_order",
    "insufficient_training_rows",
    "unknown_interval",
    "rest_response_truncated",
    "trial_budget_exhausted",
    "unknown_trial",
    "unknown_release",
]
FlagKind = Literal["unaligned_open_time", "bar_duration", "short_day"]


class FatalDefect(Exception):
    """A defect that makes a Decision Day's decision impossible or untrustworthy."""

    def __init__(self, kind: FatalKind, message: str) -> None:
        super().__init__(message)
        self.kind: FatalKind = kind


@dataclass(frozen=True, slots=True)
class DataQualityFlag:
    """A recorded imperfection that does not stop the run, and is never a model feature."""

    kind: FlagKind
    at: str  # ISO-8601: an instant for a bar, a date for a whole day
    detail: str


class Bar(NamedTuple):
    """One hourly Closed Bar, times normalised to milliseconds."""

    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    source: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    bars: tuple[Bar, ...]
    flags: tuple[DataQualityFlag, ...]
    archives_read: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_milliseconds(raw: str, source: str) -> int:
    divisor = _WIDTH_TO_MS_DIVISOR.get(len(raw))
    if divisor is None:
        raise FatalDefect(
            "timestamp_width",
            f"{source}: unexpected timestamp width {len(raw)} in {raw!r}; "
            "expected 13 (ms) or 16 (us)",
        )
    return int(raw) // divisor


def _instant(open_time_ms: int) -> str:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=UTC).isoformat()


def bar_from_row(row: Sequence[str], source: str) -> Bar:
    """One Binance kline row to a Bar, with every Fatal Defect check applied.

    Shared by the archive reader and the REST reader so that a bar arriving over HTTP is validated
    exactly as one arriving from a checksum-verified file.
    """
    if len(row) != COLUMNS:
        raise FatalDefect(
            "column_count", f"{source}: row has {len(row)} columns, expected {COLUMNS}"
        )

    open_time_ms = _to_milliseconds(row[0], source)
    try:
        open_, high, low, close = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        volume = float(row[5])
        close_time_ms = _to_milliseconds(row[6], source)
    except ValueError as error:
        raise FatalDefect(
            "numeric_field",
            f"{source}: non-numeric field in the row opening at {_instant(open_time_ms)}: {error}",
        ) from error

    if min(open_, high, low, close) <= 0:
        raise FatalDefect(
            "non_positive_price",
            f"{source}: non-positive price at {_instant(open_time_ms)} "
            f"(o={open_} h={high} l={low} c={close}); zero volume is real data, "
            "zero price is not",
        )
    if not (low <= min(open_, close) and max(open_, close) <= high and low <= high):
        raise FatalDefect(
            "ohlc_invariant",
            f"{source}: OHLC invariant violated at {_instant(open_time_ms)} "
            f"(o={open_} h={high} l={low} c={close})",
        )
    return Bar(
        open_time_ms=open_time_ms,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time_ms=close_time_ms,
        source=source,
    )


def _parse_archive(archive: Path) -> list[Bar]:
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise FatalDefect(
                "archive_member",
                f"{archive.name}: expected exactly one .csv member, found {len(members)}",
            )
        text = zf.read(members[0]).decode()

    bars: list[Bar] = []
    for row in csv.reader(io.StringIO(text)):
        if not row:  # a trailing newline is a formatting artifact, not a row
            continue
        bars.append(bar_from_row(row, archive.name))
    return bars


def _flag_bars(bars: Sequence[Bar]) -> list[DataQualityFlag]:
    flags: list[DataQualityFlag] = []
    for bar in bars:
        if bar.open_time_ms % HOUR_MS != 0:
            flags.append(
                DataQualityFlag(
                    "unaligned_open_time",
                    _instant(bar.open_time_ms),
                    f"open_time is {bar.open_time_ms % HOUR_MS} ms past the hour",
                )
            )
        duration = bar.close_time_ms - bar.open_time_ms
        if not _MIN_DURATION_MS <= duration <= _MAX_DURATION_MS:
            flags.append(
                DataQualityFlag(
                    "bar_duration",
                    _instant(bar.open_time_ms),
                    f"close_time - open_time is {duration} ms, expected "
                    f"{_MIN_DURATION_MS}..{_MAX_DURATION_MS}",
                )
            )
    return flags


def _flag_short_days(bars: Sequence[Bar]) -> list[DataQualityFlag]:
    per_day: Counter[date] = Counter(
        datetime.fromtimestamp(bar.open_time_ms / 1000, tz=UTC).date() for bar in bars
    )
    return [
        DataQualityFlag("short_day", day.isoformat(), f"{count} bars, expected {BARS_PER_DAY}")
        for day, count in sorted(per_day.items())
        if count < BARS_PER_DAY
    ]


def load_archive_backfill(snapshot_dir: Path) -> IngestResult:
    """Read every monthly archive in the snapshot into one validated series of Closed Bars.

    Each archive is checked against its published `.CHECKSUM` first: a matching checksum proves file
    integrity, not content correctness, so every other check still runs afterwards.
    """
    bars: list[Bar] = []
    archives_read = 0

    for archive in sorted(snapshot_dir.glob("*.zip")):
        archives_read += 1
        checksum_file = archive.with_suffix(".zip.CHECKSUM")
        published = checksum_file.read_text().split() if checksum_file.is_file() else []
        if not published:
            raise FatalDefect(
                "checksum_missing",
                f"{archive.name}: no published checksum at {checksum_file.name}",
            )
        expected = published[0].strip()
        actual = _sha256(archive)
        if actual != expected:
            raise FatalDefect(
                "checksum_mismatch",
                f"{archive.name}: sha256 {actual} does not match published {expected}",
            )
        bars.extend(_parse_archive(archive))

    if archives_read == 0:
        raise FatalDefect("empty_snapshot", f"no archives found in {snapshot_dir}")

    bars.sort(key=lambda bar: bar.open_time_ms)

    seen: Counter[int] = Counter(bar.open_time_ms for bar in bars)
    duplicates = [time for time, count in seen.items() if count > 1]
    if duplicates:
        raise FatalDefect(
            "duplicate_open_time",
            f"{len(duplicates)} duplicate open_time values, first at {_instant(min(duplicates))}",
        )

    flags = _flag_bars(bars) + _flag_short_days(bars)
    flags.sort(key=lambda flag: (flag.at, flag.kind))

    return IngestResult(
        bars=tuple(bars),
        flags=tuple(flags),
        archives_read=archives_read,
    )


def assert_decision_days_complete(bars: Sequence[Bar], first_day: date, last_day: date) -> None:
    """Every Decision Day needs its 00:00 bar (Feature Cutoff) and 01:00 bar (Decision Price)."""
    present = {bar.open_time_ms for bar in bars}
    hours_covered = {bar.open_time_ms // HOUR_MS for bar in bars}
    day = first_day
    while day <= last_day:
        midnight = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        for offset, label in ((0, "00:00"), (HOUR_MS, "01:00")):
            wanted = midnight + offset
            if wanted in present:
                continue
            if wanted // HOUR_MS in hours_covered:
                # The bar exists but opened off the hour, as 43 bars of February 2018 did. Using it
                # would silently shift the Feature Cutoff, so the day still cannot be decided.
                raise FatalDefect(
                    "unaligned_decision_day_bar",
                    f"{day.isoformat()}: the {label} UTC bar is not aligned to the hour",
                )
            raise FatalDefect(
                "missing_decision_day_bar",
                f"{day.isoformat()} is missing its {label} UTC bar",
            )
        day += timedelta(days=1)
