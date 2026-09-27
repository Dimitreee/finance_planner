"""Audit of the CryptoPanic news archive: integrity, counts, and what the timestamps support.

Reports only what is measured. Run from the repository root:

    uv run python scripts/audit/audit_news_archive.py
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from cryptoguard_core.news import (
    NEWS_CSV,
    busiest_window_start,
    choose_primary_lag,
    daily_counts_by_stated_date,
    diagnose_timestamps,
    mentions_asset,
    read_news_items,
)

ARCHIVE = NEWS_CSV.with_suffix(".rar")
# The manifest is the single source of truth for what the bytes should be; repeating the digests
# here would let the two disagree silently after a re-fetch.
MANIFEST = NEWS_CSV.resolve().parents[3] / "data/manifests/news_cryptopanic.sha256"
WINDOW_FIRST = date(2021, 2, 1)
WINDOW_LAST = date(2025, 11, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def published_digests() -> dict[str, str]:
    """Filename -> digest, as recorded in the manifest."""
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
    return entries


def main() -> int:
    published = published_digests()
    for path in (ARCHIVE, NEWS_CSV):
        if not path.is_file():
            # Deleting the archive once it is extracted is ordinary; every measurement below needs
            # only the CSV, so the audit reports the absence and carries on.
            print(f"{path.name:44} absent")
            continue
        actual = sha256(path)
        expected = published.get(path.name)
        status = "MATCH" if actual == expected else "MISMATCH"
        print(f"{path.name:44} sha256 {status}: {actual}")

    if not NEWS_CSV.is_file():
        print(f"\n{NEWS_CSV} is absent; run scripts/fetch_news_archive.sh first")
        return 1

    items = list(read_news_items(NEWS_CSV))
    btc = [item for item in items if mentions_asset(item, "BTC")]
    print(f"\nrecords                 : {len(items)}")
    print(f"exact BTC token         : {len(btc)}")

    other_btc_tokens: Counter[str] = Counter()
    for item in items:
        for token in item.currencies:
            if "BTC" in token and token != "BTC":
                other_btc_tokens[token] += 1
    print(
        f"other tokens with 'BTC' : {sum(other_btc_tokens.values())} "
        f"across {len(other_btc_tokens)} tokens, e.g. {dict(other_btc_tokens.most_common(5))}"
    )

    diagnostic = diagnose_timestamps(iter(btc))
    print(f"\nBTC stated range        : {diagnostic.earliest} .. {diagnostic.latest}")
    print(f"with a timezone suffix  : {diagnostic.with_timezone_suffix}")
    print(
        f"minute == 00            : {diagnostic.zero_minutes} "
        f"({diagnostic.zero_minutes / diagnostic.records:.2%}, "
        f"{1 / 60:.2%} expected if unrounded)"
    )
    print(
        f"second == 00            : {diagnostic.zero_seconds} "
        f"({diagnostic.zero_seconds / diagnostic.records:.2%}, "
        f"{1 / 60:.2%} expected if unrounded)"
    )

    print("\nBTC records by stated hour of day (UTC assumed nowhere, this is the raw field):")
    for hour in range(24):
        count = diagnostic.hour_of_day.get(hour, 0)
        bar = "#" * round(60 * count / max(diagnostic.hour_of_day.values()))
        print(f"  {hour:02d}  {count:6}  {bar}")

    start = busiest_window_start(diagnostic)
    lag = choose_primary_lag(diagnostic)
    print(f"\nbusiest 8 consecutive hours start at : {start:02d}:00")
    print(f"Primary Lag chosen                   : {lag} hours")

    per_day = daily_counts_by_stated_date(iter(btc))
    span = (WINDOW_LAST - WINDOW_FIRST).days + 1
    days_with = sum(
        1 for offset in range(span) if per_day.get(WINDOW_FIRST + timedelta(days=offset), 0) > 0
    )
    in_window = sum(count for day, count in per_day.items() if WINDOW_FIRST <= day <= WINDOW_LAST)
    domains = {
        item.source_domain
        for item in btc
        if item.source_domain and WINDOW_FIRST <= item.stated_at.date() <= WINDOW_LAST
    }
    print(f"\nResearch Window {WINDOW_FIRST} .. {WINDOW_LAST} ({span} days)")
    print(f"  BTC records in window : {in_window}")
    print(f"  days with news        : {days_with} / {span}")
    print(f"  days without news     : {span - days_with}")
    print(f"  distinct domains      : {len(domains)}")

    missing_description = sum(1 for item in btc if item.description is None)
    missing_source_url = sum(1 for item in btc if item.source_url is None)
    print(f"\nBTC records missing description : {missing_description} / {len(btc)}")
    print(f"BTC records missing sourceUrl   : {missing_source_url} / {len(btc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
