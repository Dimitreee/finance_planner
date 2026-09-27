"""Archive Backfill ingest: what stops a run and what is merely recorded.

Fixtures are slices of the real Binance archives, so the defects exercised here are the ones the
data actually contains. Two checks are the exception and are called out where they appear: duplicate
`open_time` and OHLC invariant violations measured **zero** occurrences across all 79 117 rows
of the verified snapshot, so they can only be driven by synthesised rows.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from conftest import SNAPSHOT, SnapshotBuilder, flag_kinds, synthetic_rows
from cryptoguard_core.ingest import (
    FatalDefect,
    assert_decision_days_complete,
    load_archive_backfill,
)

HOUR_MS = 3_600_000
JUNE_FIRST = 1_717_200_000_000  # 2024-06-01T00:00:00Z


def test_reads_a_clean_archive(snapshot: SnapshotBuilder) -> None:
    result = load_archive_backfill(snapshot("clean_day_2024-06-01.csv"))
    assert len(result.bars) == 24
    assert result.flags == ()
    assert result.archives_read == 1


def test_checksum_mismatch_is_fatal(snapshot: SnapshotBuilder) -> None:
    directory = snapshot("clean_day_2024-06-01.csv", checksum="0" * 64)
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(directory)
    assert caught.value.kind == "checksum_mismatch"


def test_millisecond_and_microsecond_archives_normalise_to_the_same_scale(
    snapshot: SnapshotBuilder,
) -> None:
    """Spot timestamps switched from ms to us at 2025-01: both widths read, neither guessed."""
    result = load_archive_backfill(snapshot("ms_width_2024-12.csv", "us_width_2025-01.csv"))
    times = [bar.open_time_ms for bar in result.bars]
    assert times == sorted(times)
    assert times[0] == 1_733_011_200_000  # 2024-12-01T00:00Z, from the 13-digit archive
    assert times[6] == 1_735_689_600_000  # 2025-01-01T00:00Z, from the 16-digit archive
    assert all(t % HOUR_MS == 0 for t in times)


def test_unexpected_timestamp_width_is_fatal(snapshot: SnapshotBuilder) -> None:
    """An unknown width stops the run rather than having a multiplier inferred from magnitude."""
    rows = synthetic_rows((1_717_200_000, 1.0, 2.0, 0.5, 1.5))  # 10 digits: neither ms nor us
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(snapshot(rows=rows))
    assert caught.value.kind == "timestamp_width"
    assert "10" in str(caught.value)


def test_unaligned_open_time_is_flagged_not_fatal(snapshot: SnapshotBuilder) -> None:
    """43 such bars exist in an archive whose checksum matches: they must not be rejected."""
    result = load_archive_backfill(snapshot("unaligned_2018-02.csv"))
    assert len(result.bars) == 4
    assert flag_kinds(result.flags).count("unaligned_open_time") == 4
    assert result.bars[0].open_time_ms % HOUR_MS != 0  # not silently rounded to the hour


def test_short_bar_duration_is_flagged(snapshot: SnapshotBuilder) -> None:
    result = load_archive_backfill(snapshot("short_duration_2021-04.csv"))
    durations = [flag for flag in result.flags if flag.kind == "bar_duration"]
    assert len(durations) == 1
    assert "58146" in durations[0].detail


def test_negative_duration_is_flagged_and_bars_survive(snapshot: SnapshotBuilder) -> None:
    """close_time precedes open_time here: hence close_time is validated, never computed with."""
    result = load_archive_backfill(snapshot("negative_duration_2020-12.csv"))
    assert len(result.bars) == 3
    assert flag_kinds(result.flags).count("bar_duration") == 1


def test_day_short_of_24_bars_is_flagged(snapshot: SnapshotBuilder) -> None:
    result = load_archive_backfill(snapshot("short_day_2021-02-11.csv"))
    short = [flag for flag in result.flags if flag.kind == "short_day"]
    assert len(short) == 1
    assert short[0].at == "2021-02-11"
    assert "23" in short[0].detail


def test_duplicate_open_time_is_fatal(snapshot: SnapshotBuilder) -> None:
    """Synthesised: the verified snapshot contains zero duplicate open_time values."""
    rows = synthetic_rows(
        (JUNE_FIRST, 1.0, 2.0, 0.5, 1.5),
        (JUNE_FIRST, 1.0, 2.0, 0.5, 1.5),
    )
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(snapshot(rows=rows))
    assert caught.value.kind == "duplicate_open_time"


def test_ohlc_invariant_violation_is_fatal(snapshot: SnapshotBuilder) -> None:
    """Synthesised: the verified snapshot contains zero OHLC invariant violations."""
    rows = synthetic_rows((JUNE_FIRST, 1.0, 0.9, 0.5, 1.5))  # high below both open and close
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(snapshot(rows=rows))
    assert caught.value.kind == "ohlc_invariant"


def test_decision_day_missing_its_execution_bar_is_fatal(snapshot: SnapshotBuilder) -> None:
    """A Decision Day needs the 00:00 bar for the Feature Cutoff and the 01:00 bar for the price."""
    rows = synthetic_rows(
        (JUNE_FIRST, 1.0, 2.0, 0.5, 1.5),
        (JUNE_FIRST + 2 * HOUR_MS, 1.0, 2.0, 0.5, 1.5),  # 02:00 present, 01:00 absent
    )
    result = load_archive_backfill(snapshot(rows=rows))
    with pytest.raises(FatalDefect) as caught:
        assert_decision_days_complete(result.bars, date(2024, 6, 1), date(2024, 6, 1))
    assert caught.value.kind == "missing_decision_day_bar"
    assert "01:00" in str(caught.value)


def test_complete_decision_day_passes(snapshot: SnapshotBuilder) -> None:
    result = load_archive_backfill(snapshot("clean_day_2024-06-01.csv"))
    assert_decision_days_complete(result.bars, date(2024, 6, 1), date(2024, 6, 1))


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_full_snapshot_reproduces_the_audit() -> None:
    """The production path and the standalone audit must agree on the same files."""
    result = load_archive_backfill(Path(SNAPSHOT))

    assert result.archives_read == 109
    assert len(result.bars) == 79_117

    kinds = flag_kinds(result.flags)
    assert kinds.count("unaligned_open_time") == 43
    assert kinds.count("bar_duration") == 14

    window_start, window_end = date(2021, 1, 1), date(2026, 8, 31)
    in_window = [
        flag
        for flag in result.flags
        if window_start <= date.fromisoformat(flag.at[:10]) <= window_end
    ]
    in_window_kinds = flag_kinds(in_window)
    assert in_window_kinds.count("short_day") == 7
    assert in_window_kinds.count("bar_duration") == 5

    assert_decision_days_complete(result.bars, window_start, window_end)


def test_row_with_wrong_column_count_is_fatal(snapshot: SnapshotBuilder) -> None:
    """Synthesised: every row of the verified snapshot has exactly twelve columns."""
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(snapshot(rows=[[JUNE_FIRST, 1.0, 2.0]]))
    assert caught.value.kind == "column_count"


def test_unaligned_day_is_reported_as_unaligned_rather_than_missing(
    snapshot: SnapshotBuilder,
) -> None:
    """Every bar of 2018-02-10 exists, 28 minutes past the hour. Saying "missing" would mislead."""
    result = load_archive_backfill(snapshot("unaligned_day_2018-02-10.csv"))
    assert len(result.bars) == 24
    with pytest.raises(FatalDefect) as caught:
        assert_decision_days_complete(result.bars, date(2018, 2, 10), date(2018, 2, 10))
    assert caught.value.kind == "unaligned_decision_day_bar"
    assert "not aligned" in str(caught.value)


def test_empty_snapshot_directory_is_fatal(tmp_path: Path) -> None:
    """A mistyped or unpopulated path must not read as a clean, empty series."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(empty)
    assert caught.value.kind == "empty_snapshot"


def test_missing_checksum_file_is_fatal(snapshot: SnapshotBuilder) -> None:
    """Every defect leaves this module as a FatalDefect, not as a bare OSError."""
    directory = snapshot("clean_day_2024-06-01.csv")
    next(directory.glob("*.CHECKSUM")).unlink()
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(directory)
    assert caught.value.kind == "checksum_missing"


def test_blank_lines_are_ignored(raw_archive: Callable[[dict[str, str]], Path]) -> None:
    """A trailing newline is a formatting artifact, not a row: it must not kill a backfill."""
    row = synthetic_rows((JUNE_FIRST, 1.0, 2.0, 0.5, 1.5))[0]
    text = ",".join(str(cell) for cell in row) + "\n\n\n"
    assert len(load_archive_backfill(raw_archive({"bars.csv": text})).bars) == 1


def test_archive_without_exactly_one_csv_member_is_fatal(
    raw_archive: Callable[[dict[str, str]], Path],
) -> None:
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(raw_archive({}))
    assert caught.value.kind == "archive_member"

    two = {"a.csv": "1,2\n", "b.csv": "3,4\n"}
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(raw_archive(two))
    assert caught.value.kind == "archive_member"


def test_non_numeric_price_is_fatal(snapshot: SnapshotBuilder) -> None:
    """Synthesised: the verified snapshot has zero empty or non-numeric cells."""
    row = [JUNE_FIRST, "n/a", 2.0, 0.5, 1.5, 1.0, JUNE_FIRST + 3_599_999, 1.0, 1, 1.0, 1.0, 0]
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(snapshot(rows=[row]))
    assert caught.value.kind == "numeric_field"


def test_non_positive_price_is_fatal(snapshot: SnapshotBuilder) -> None:
    """An all-zero row satisfies the OHLC invariant, so positivity is checked separately.

    Synthesised: the verified snapshot has zero such rows. Zero *volume* is different — four real
    bars carry it — so positivity is required of prices only.
    """
    with pytest.raises(FatalDefect) as caught:
        load_archive_backfill(snapshot(rows=synthetic_rows((JUNE_FIRST, 0.0, 0.0, 0.0, 0.0))))
    assert caught.value.kind == "non_positive_price"
