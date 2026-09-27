"""Reading the news archive, and what its timestamps do and do not tell us.

The archive carries a stated publication time with no timezone and no first-seen time. Nothing here
converts that to an instant: the conversion is a declared assumption made later, and the
parser's job is to hand over exactly what the file says.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news import (
    NEWS_CSV,
    TimestampDiagnostic,
    busiest_window_start,
    choose_primary_lag,
    daily_counts_by_stated_date,
    diagnose_timestamps,
    mentions_asset,
    read_news_items,
)

FIXTURE = Path(__file__).parent / "fixtures" / "news" / "sample.csv"


def items() -> list[object]:
    return list(read_news_items(FIXTURE))


def test_reads_every_record_including_a_title_containing_a_newline() -> None:
    """Physical lines and records differ: some titles span lines inside their quotes."""
    assert len(items()) == 9
    assert any("\n" in item.title for item in read_news_items(FIXTURE))


def test_null_is_absence_rather_than_the_string_null() -> None:
    parsed = list(read_news_items(FIXTURE))
    assert any(item.description is None for item in parsed)
    assert any(item.source_url is None for item in parsed)
    assert any(item.currencies == () for item in parsed)
    assert not any(item.description == "NULL" for item in parsed)


def test_asset_match_is_an_exact_token() -> None:
    """Substring matching would pull in WBTC, HBTC, BTCST and BTCUSD: different assets."""
    parsed = list(read_news_items(FIXTURE))
    assert sum(mentions_asset(item, "BTC") for item in parsed) == 3
    assert any("WBTC" in item.currencies for item in parsed)
    assert any("BTCST" in item.currencies for item in parsed)


def test_stated_time_is_naive_and_is_not_declared_utc() -> None:
    """The file states no zone. Attaching one here would turn an assumption into a silent fact."""
    for item in read_news_items(FIXTURE):
        assert item.stated_at.tzinfo is None


def test_diagnostic_counts_what_the_file_shows() -> None:
    diagnostic = diagnose_timestamps(read_news_items(FIXTURE))
    assert diagnostic.records == 9
    assert sum(diagnostic.hour_of_day.values()) == 9
    assert diagnostic.with_timezone_suffix == 0
    assert diagnostic.earliest <= diagnostic.latest


def _histogram(
    peak_start: int, hours: int = 8, *, with_timezone_suffix: int = 0
) -> TimestampDiagnostic:
    counts = {hour: 1 for hour in range(24)}
    for offset in range(hours):
        counts[(peak_start + offset) % 24] = 100
    return TimestampDiagnostic(
        records=sum(counts.values()),
        hour_of_day=counts,
        with_timezone_suffix=with_timezone_suffix,
        zero_minutes=0,
        zero_seconds=0,
        earliest=datetime(2021, 1, 1),
        latest=datetime(2025, 11, 30),
    )


def test_busiest_window_is_found_wherever_it_sits() -> None:
    for start in (0, 7, 13, 20):
        assert busiest_window_start(_histogram(start)) == start


def test_primary_lag_is_six_hours_when_the_peak_sits_in_utc_business_hours() -> None:
    """Crypto news clusters in US and EU business hours; in UTC that is the early afternoon."""
    assert choose_primary_lag(_histogram(13)) == 6


def test_primary_lag_is_twenty_four_hours_when_the_peak_sits_elsewhere() -> None:
    """An unexplained offset stays unexplained: the conservative lag covers any zone error."""
    assert choose_primary_lag(_histogram(3)) == 24
    assert choose_primary_lag(_histogram(20)) == 24


def test_daily_counts_use_the_stated_date_and_are_pre_lag() -> None:
    counts = daily_counts_by_stated_date(read_news_items(FIXTURE))
    assert sum(counts.values()) == 9
    assert all(isinstance(day, date) for day in counts)


@pytest.mark.skipif(not NEWS_CSV.exists(), reason="news archive is not in this checkout")
def test_real_archive_reproduces_the_recorded_counts() -> None:
    parsed = list(read_news_items(NEWS_CSV))
    assert len(parsed) == 248_464

    btc = [item for item in parsed if mentions_asset(item, "BTC")]
    assert len(btc) == 31_162

    diagnostic = diagnose_timestamps(iter(btc))
    assert diagnostic.with_timezone_suffix == 0
    assert diagnostic.earliest.date() == date(2017, 9, 29)
    assert diagnostic.latest.date() == date(2025, 12, 3)


def test_an_unparseable_stated_time_is_fatal(tmp_path: Path) -> None:
    """Synthesised: all 248 464 real records match '%Y-%m-%d %H:%M:%S' exactly."""
    header = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    broken = tmp_path / "broken.csv"
    broken.write_text(
        header + '\n"1","t",NULL,"2","d",NULL,"yesterday","u",'
        '"0","0","0","0","0","0","0","0","0","BTC"\n',
        encoding="utf-8",
    )
    with pytest.raises(FatalDefect) as caught:
        list(read_news_items(broken))
    assert caught.value.kind == "news_timestamp_format"


def test_a_mixed_timezone_histogram_cannot_decide_the_lag() -> None:
    """A stripped offset enters the histogram as if it shared everyone else's frame."""
    with pytest.raises(FatalDefect) as caught:
        choose_primary_lag(_histogram(13, with_timezone_suffix=1))
    assert caught.value.kind == "mixed_timezone_offsets"


def test_diagnosing_no_records_is_a_caller_mistake_not_a_data_defect() -> None:
    with pytest.raises(ValueError, match="no records"):
        diagnose_timestamps(iter([]))
