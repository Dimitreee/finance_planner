"""Closed Bars later than the last published monthly archive, read from the Spot REST endpoint.

This path never feeds a training or backtest dataset: it has no published checksum, and the same
request returns a different window tomorrow, so a dataset built from it could not be reproduced. It
supplies the features of a live Decision Day and nothing else.

Rows pass the same per-row checks as an archive row, and the same series-level checks the archive
loader applies: sorted by `open_time`, with a duplicate treated as a Fatal Defect. The one
difference is the last bar, which in a live response is still open; it is dropped because its hour
has not elapsed, not because it looks unfinished.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import requests

from cryptoguard_core.ingest import HOUR_MS, Bar, FatalDefect, bar_from_row

REST_BASE = "https://data-api.binance.vision"
MAX_ROWS_PER_REQUEST = 1000
REQUEST_TIMEOUT_SECONDS = 30

# Only the intervals this project actually decides on. An unknown one would have to bring its own
# bar width, and a wrong width is silent: a still-forming bar would pass for a Closed Bar.
BAR_MILLISECONDS = {"1h": HOUR_MS}


def parse_rest_klines(rows: Sequence[Sequence[Any]], *, source: str) -> tuple[Bar, ...]:
    """Validate a kline payload as an archive is validated, per row and across the series."""
    bars = tuple(bar_from_row([str(cell) for cell in row], source) for row in rows)
    duplicates = [
        time for time, count in Counter(bar.open_time_ms for bar in bars).items() if count > 1
    ]
    if duplicates:
        raise FatalDefect(
            "duplicate_open_time",
            f"{source}: {len(duplicates)} duplicate open_time values in one response",
        )
    times = [bar.open_time_ms for bar in bars]
    if times != sorted(times):
        raise FatalDefect("rows_out_of_order", f"{source}: bars did not arrive in time order")
    return bars


def closed_bars(bars: Sequence[Bar], *, now_ms: int, bar_ms: int) -> tuple[Bar, ...]:
    """Only bars whose own width has fully elapsed.

    `bar_ms` is required rather than defaulted: a default would silently admit a still-forming bar
    the moment anyone read an interval other than the one the default assumed.
    """
    return tuple(bar for bar in bars if bar.open_time_ms + bar_ms <= now_ms)


def fetch_rest_klines(
    symbol: str,
    interval: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = MAX_ROWS_PER_REQUEST,
    base: str = REST_BASE,
) -> tuple[Bar, ...]:
    """One request. The caller decides what to do with the last, possibly open, bar."""
    bar_ms = BAR_MILLISECONDS.get(interval)
    if bar_ms is None:
        raise FatalDefect(
            "unknown_interval",
            f"interval {interval!r} is not one of {sorted(BAR_MILLISECONDS)}",
        )

    if start_ms is not None and end_ms is not None:
        expected = (end_ms - start_ms) // bar_ms + 1
        if expected > limit:
            # Checked before the request, not after: the endpoint caps a response and says nothing
            # about it, so a wider window comes back quietly short rather than as an error.
            raise FatalDefect(
                "rest_response_truncated",
                f"asked for about {expected} bars but the endpoint returns at most {limit}; "
                "request a narrower window",
            )

    params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    response = requests.get(f"{base}/api/v3/klines", params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_rest_klines(response.json(), source=f"rest:{symbol}:{interval}")
