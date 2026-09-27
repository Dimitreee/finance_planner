"""Are the three market sources interchangeable on Closed Bars?

The monthly archives stop at the last completed month, so anything newer has to come from the daily
archives or the Spot REST endpoint. Until this is measured, the Recent Tail rests on an assumption.

Reports only what is measured. Run from the repository root:

    ./scripts/fetch_binance_daily.sh 2026-08
    uv run python scripts/audit/audit_market_seam.py
"""

from __future__ import annotations

import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from cryptoguard_core.ingest import HOUR_MS, Bar, load_archive_backfill
from cryptoguard_core.recent_tail import closed_bars, fetch_rest_klines

MONTH = "2026-08"
SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")
DAILY = SNAPSHOT / "daily" / MONTH


def month_bounds(label: str) -> tuple[int, int]:
    year, month = (int(part) for part in label.split("-"))
    first = datetime(year, month, 1, tzinfo=UTC)
    last = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
    return int(first.timestamp() * 1000), int(last.timestamp() * 1000) - 1


def comparable(bar: Bar) -> tuple[int, float, float, float, float, float]:
    """Everything a source is expected to agree on. `source` and `close_time` are excluded:
    the first is the file it came from, and the second is never used in arithmetic."""
    return (bar.open_time_ms, bar.open, bar.high, bar.low, bar.close, bar.volume)


def compare(left: dict[int, Bar], right: dict[int, Bar], left_name: str, right_name: str) -> int:
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    differing = [
        time
        for time in sorted(set(left) & set(right))
        if comparable(left[time]) != comparable(right[time])
    ]
    print(f"\n{left_name} vs {right_name}")
    print(f"  bars only in {left_name:<18}: {len(only_left)}")
    print(f"  bars only in {right_name:<18}: {len(only_right)}")
    print(f"  bars present in both          : {len(set(left) & set(right))}")
    print(f"  bars differing in any field   : {len(differing)}")
    for time in differing[:3]:
        print(f"    {datetime.fromtimestamp(time / 1000, tz=UTC)}")
        print(f"      {left_name}: {comparable(left[time])}")
        print(f"      {right_name}: {comparable(right[time])}")
    return len(only_left) + len(only_right) + len(differing)


def main() -> int:
    first_ms, last_ms = month_bounds(MONTH)

    monthly_all = load_archive_backfill(SNAPSHOT)
    monthly = {
        bar.open_time_ms: bar for bar in monthly_all.bars if first_ms <= bar.open_time_ms <= last_ms
    }
    daily_result = load_archive_backfill(DAILY)
    daily = {bar.open_time_ms: bar for bar in daily_result.bars}

    print(f"month under test            : {MONTH}")
    print(f"monthly archive bars        : {len(monthly)}")
    print(f"daily archives read         : {daily_result.archives_read}, bars {len(daily)}")
    print(f"daily Data Quality Flags    : {len(daily_result.flags)}")

    print("\ntimestamp width as observed, counted over every row:")
    widths: dict[str, Counter[int]] = {"monthly": Counter(), "daily": Counter()}
    for name, directory in (("monthly", SNAPSHOT), ("daily", DAILY)):
        for archive in sorted(directory.glob(f"*{MONTH}*.zip")):
            with zipfile.ZipFile(archive) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                text = zf.read(member).decode()
            # Every row, not the first: an archive that switched units partway is exactly the
            # event the ingest design says must not be missed, and a first-row sample would
            # report it as one clean width.
            for line in text.splitlines():
                if line:
                    widths[name][len(line.split(",")[0])] += 1
    for name, counts in widths.items():
        print(f"  {name:<8}: {dict(counts)} digits over {sum(counts.values())} rows")

    rest = {
        bar.open_time_ms: bar
        for bar in fetch_rest_klines("BTCUSDT", "1h", start_ms=first_ms, end_ms=last_ms, limit=1000)
    }
    rest_widths = Counter(len(str(time)) for time in rest)
    print(f"  rest    : {dict(rest_widths)} digits over {sum(rest_widths.values())} rows")

    mismatches = compare(monthly, daily, "monthly", "daily")
    mismatches += compare(monthly, rest, "monthly", "rest")

    live = fetch_rest_klines("BTCUSDT", "1h", limit=3)
    if not live:
        print("\nthe live response was empty; nothing to say about the unclosed bar", flush=True)
        return 1
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    kept = closed_bars(live, now_ms=now_ms, bar_ms=HOUR_MS)
    newest = live[-1]
    print("\nlive response, the unclosed bar:")
    opened = datetime.fromtimestamp(newest.open_time_ms / 1000, tz=UTC)
    print(f"  newest bar opens at         : {opened}")
    print(
        f"  its hour ends at            : "
        f"{datetime.fromtimestamp((newest.open_time_ms + HOUR_MS) / 1000, tz=UTC)}"
    )
    print(f"  now                         : {datetime.fromtimestamp(now_ms / 1000, tz=UTC)}")
    print(f"  bars returned / kept        : {len(live)} / {len(kept)}")

    print("\nconclusion:")
    if mismatches == 0:
        print("  The three sources agree on every Closed Bar of the month tested, so the Recent")
        print("  Tail may mix daily archives and REST. This licenses the seam; it does not make")
        print("  REST reproducible, which is why it still never feeds a dataset.")
    else:
        print(f"  {mismatches} disagreements. The Recent Tail must be restricted to a single")
        print("  source until they are explained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
