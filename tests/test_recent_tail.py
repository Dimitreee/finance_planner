"""The Recent Tail: bars arriving over HTTP rather than from a checksum-verified file.

Two things separate this path from the archive. Its last bar is usually still open, and it carries
no published checksum — so it is validated by exactly the same row checks, and the unclosed bar is
dropped by a rule rather than by a judgement about whether it looks finished.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.recent_tail import closed_bars, fetch_rest_klines, parse_rest_klines

FIXTURE = Path(__file__).parent / "fixtures" / "binance" / "rest_klines.json"
CAPTURE = json.loads(FIXTURE.read_text(encoding="utf-8"))
ROWS = CAPTURE["rows"]
CAPTURED_AT_MS = CAPTURE["captured_at_ms"]


def test_nothing_in_this_file_reaches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here runs on the captured fixture or refuses before a request is made."""
    calls: list[str] = []
    monkeypatch.setattr(
        "cryptoguard_core.recent_tail.requests.get",
        lambda *args, **kwargs: calls.append("called"),
    )
    with pytest.raises(FatalDefect):
        fetch_rest_klines("BTCUSDT", "4h")
    with pytest.raises(FatalDefect):
        fetch_rest_klines("BTCUSDT", "1h", start_ms=0, end_ms=5000 * HOUR_MS, limit=10)
    assert calls == []


def test_the_payload_parses_through_the_same_row_validator() -> None:
    bars = parse_rest_klines(ROWS, source="rest")
    assert len(bars) == len(ROWS)
    assert all(bar.source == "rest" for bar in bars)
    assert [bar.open_time_ms for bar in bars] == [row[0] for row in ROWS]


def test_rest_timestamps_are_milliseconds_as_observed() -> None:
    """Recorded, not assumed: the 2025+ monthly archives are microseconds, this endpoint is not."""
    assert all(len(str(row[0])) == 13 for row in ROWS)


def test_the_last_bar_of_a_live_response_is_still_open_and_is_dropped() -> None:
    bars = parse_rest_klines(ROWS, source="rest")
    kept = closed_bars(bars, now_ms=CAPTURED_AT_MS, bar_ms=HOUR_MS)
    assert len(kept) == len(bars) - 1
    assert kept[-1].open_time_ms + HOUR_MS <= CAPTURED_AT_MS


def test_a_bar_is_closed_exactly_when_its_hour_has_elapsed() -> None:
    """By rule, not by a heuristic about whether the bar looks finished."""
    bars = parse_rest_klines(ROWS, source="rest")
    last = bars[-1]
    assert closed_bars(bars, now_ms=last.open_time_ms + HOUR_MS - 1, bar_ms=HOUR_MS)[-1] is not last
    assert closed_bars(bars, now_ms=last.open_time_ms + HOUR_MS, bar_ms=HOUR_MS)[-1] is last


def test_an_interval_without_a_known_bar_width_is_refused() -> None:
    """A wrong width is silent: a still-forming four-hour bar would pass the one-hour test."""
    with pytest.raises(FatalDefect) as caught:
        fetch_rest_klines("BTCUSDT", "4h")
    assert caught.value.kind == "unknown_interval"


def test_a_duplicated_hour_in_one_response_is_fatal() -> None:
    """The archive treats a duplicate open_time as fatal; arriving over HTTP changes nothing."""
    with pytest.raises(FatalDefect) as caught:
        parse_rest_klines([ROWS[0], ROWS[0]], source="rest")
    assert caught.value.kind == "duplicate_open_time"


def test_rows_out_of_order_in_one_response_are_fatal() -> None:
    with pytest.raises(FatalDefect) as caught:
        parse_rest_klines([ROWS[1], ROWS[0]], source="rest")
    assert caught.value.kind == "rows_out_of_order"


def test_a_window_wider_than_the_cap_is_refused_rather_than_truncated() -> None:
    """Refused before the request is made: an obviously doomed call should not be sent."""
    first = ROWS[0][0]
    with pytest.raises(FatalDefect) as caught:
        fetch_rest_klines(
            "BTCUSDT", "1h", start_ms=first, end_ms=first + 5000 * HOUR_MS, limit=len(ROWS)
        )
    assert caught.value.kind == "rest_response_truncated"


def test_a_malformed_row_is_refused_by_the_shared_validator(tmp_path: Path) -> None:
    broken = [[*ROWS[0][:4], "not-a-number", *ROWS[0][5:]]]
    with pytest.raises(FatalDefect) as caught:
        parse_rest_klines(broken, source="rest")
    assert caught.value.kind == "numeric_field"


def test_an_unexpected_timestamp_width_is_refused() -> None:
    """The archive switched units once already; a silent multiplier would hide the next switch."""
    broken = [[123456, *ROWS[0][1:]]]
    with pytest.raises(FatalDefect) as caught:
        parse_rest_klines(broken, source="rest")
    assert caught.value.kind == "timestamp_width"
