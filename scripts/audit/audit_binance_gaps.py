"""Where exactly are the gaps and the misaligned bars, relative to the research window."""

import csv
import io
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

RAW = Path("/Users/barrylarge/Study/finalproject/data/raw/binance/klines/BTCUSDT/1h")
HOUR_MS = 3_600_000

rows = []
for zpath in sorted(RAW.glob("*.zip")):
    with zipfile.ZipFile(zpath) as zf:
        (name,) = zf.namelist()
        for rec in csv.reader(io.StringIO(zf.read(name).decode())):
            if not rec or rec[0] == "open_time":
                continue
            mult = 1 if len(rec[0]) == 13 else 1000
            rows.append((int(rec[0]) // mult, int(rec[6]) // mult, zpath.name))
rows.sort()


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


# bars whose open_time is not aligned to the hour boundary
misaligned = [(ot, ct, src) for ot, ct, src in rows if ot % HOUR_MS != 0]
print(f"bars with open_time not on the hour: {len(misaligned)}")
for ot, ct, src in misaligned:
    print(f"    {iso(ot)}  close {iso(ct)}  ({src})")

print()
odd = [(ot, ct, src) for ot, ct, src in rows if ct - ot not in (HOUR_MS - 1, HOUR_MS)]
print(f"bars with unusual duration: {len(odd)}")
for ot, ct, src in odd:
    print(f"    {iso(ot)} -> {iso(ct)}  duration {(ct - ot) / 1000:>10.3f}s  ({src})")

print()
for label, cutoff in (("2021-01-01", 1609459200000), ("2024-01-01", 1704067200000)):
    sub = [r for r in rows if r[0] >= cutoff]
    gaps = []
    for i in range(1, len(sub)):
        d = sub[i][0] - sub[i - 1][0]
        if d != HOUR_MS:
            gaps.append((sub[i - 1][0], sub[i][0], d / HOUR_MS - 1))
    missing = sum(int(g[2]) for g in gaps)
    print(f"from {label}: {len(sub)} bars, {len(gaps)} step breaks, {missing} missing hours")
    for prev, cur, m in gaps:
        print(f"    {iso(prev)} -> {iso(cur)}  missing {m:g}h")

# daily coverage in the research window: how many UTC days have all 24 bars
print()
per_day = Counter()
for ot, _, _ in rows:
    if ot >= 1609459200000:
        per_day[datetime.fromtimestamp(ot / 1000, tz=UTC).date()] += 1
short = {d: n for d, n in per_day.items() if n != 24}
print(f"UTC days since 2021-01-01 with data: {len(per_day)}")
print(f"  days with all 24 bars : {len(per_day) - len(short)}")
print(f"  days short of 24 bars : {len(short)} -> {sorted(short.items())}")
