"""One-off audit of the full BTCUSDT 1h Binance monthly archive set.

Reports only what is measured. Production ingest will be rewritten under test.
"""

import csv
import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

RAW = Path("/Users/barrylarge/Study/finalproject/data/raw/binance/klines/BTCUSDT/1h")
HOUR_MS = 3_600_000

COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_unit(open_time_raw):
    """Binance switched spot timestamps from ms to us on 2025-01-01."""
    digits = len(open_time_raw)
    if digits == 13:
        return "ms", 1
    if digits == 16:
        return "us", 1000
    raise ValueError(f"unexpected timestamp width {digits}: {open_time_raw!r}")


report = {
    "files": 0,
    "checksum_ok": 0,
    "checksum_bad": [],
    "header_rows": [],
    "rows": 0,
    "bad_width": [],
    "empty_cells": [],
    "units": {},
}
rows = []  # (open_time_ms, close_time_ms, o, h, l, c, source)

for zpath in sorted(RAW.glob("*.zip")):
    report["files"] += 1
    cpath = zpath.with_suffix(".zip.CHECKSUM")
    expected = cpath.read_text().split()[0].strip()
    actual = sha256(zpath)
    if actual == expected:
        report["checksum_ok"] += 1
    else:
        report["checksum_bad"].append((zpath.name, expected, actual))
        continue

    with zipfile.ZipFile(zpath) as zf:
        (name,) = zf.namelist()
        text = zf.read(name).decode("utf-8")

    reader = csv.reader(io.StringIO(text))
    for line_no, rec in enumerate(reader, start=1):
        if not rec:
            continue
        if rec[0] == "open_time":  # some later archives ship a header
            report["header_rows"].append(zpath.name)
            continue
        if len(rec) != 12:
            report["bad_width"].append((zpath.name, line_no, len(rec)))
            continue
        if any(cell.strip() == "" for cell in rec):
            report["empty_cells"].append((zpath.name, line_no))
        unit, mult = detect_unit(rec[0])
        report["units"].setdefault(unit, []).append(zpath.name)
        ot = int(rec[0]) // mult
        ct = int(rec[6]) // mult
        rows.append(
            (ot, ct, float(rec[1]), float(rec[2]), float(rec[3]), float(rec[4]), zpath.name)
        )
        report["rows"] += 1

rows.sort(key=lambda r: r[0])

# --- integrity over the concatenated series ---
dupes = []
gaps = []
ohlc_bad = []
seen = {}
for i, (ot, _ct, o, high, low, c, src) in enumerate(rows):
    if ot in seen:
        dupes.append((ot, seen[ot], src))
    seen[ot] = src
    if not (low <= min(o, c) and max(o, c) <= high and low <= high):
        ohlc_bad.append((ot, src, o, high, low, c))
    if i:
        prev = rows[i - 1][0]
        if ot != prev + HOUR_MS and ot != prev:
            gaps.append((prev, ot, (ot - prev) // HOUR_MS - 1, src))


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


print(f"archives            : {report['files']}")
print(f"checksum verified   : {report['checksum_ok']}")
print(f"checksum FAILED     : {len(report['checksum_bad'])} {report['checksum_bad'][:3]}")
header_files = sorted(set(report["header_rows"]))
header_tail = "..." if len(header_files) > 4 else ""
print(
    f"header rows skipped : {len(report['header_rows'])} files -> {header_files[:4]}{header_tail}"
)
print(f"data rows           : {report['rows']}")
print(f"rows with !=12 cols : {len(report['bad_width'])} {report['bad_width'][:3]}")
print(f"rows with empty cell: {len(report['empty_cells'])} {report['empty_cells'][:3]}")
for unit, files in sorted(report["units"].items()):
    fs = sorted(set(files))
    print(f"timestamp unit {unit:>2}   : {len(fs)} files, {fs[0]} .. {fs[-1]}")
print()
print(f"range               : {iso(rows[0][0])}  ..  {iso(rows[-1][0])}")
print(f"duplicate open_time : {len(dupes)} {dupes[:3]}")
print(f"OHLC violations     : {len(ohlc_bad)} {ohlc_bad[:3]}")
print(f"hourly step breaks  : {len(gaps)}")
for prev, cur, missing, src in gaps[:15]:
    print(f"    gap {iso(prev)} -> {iso(cur)}  missing {missing}h  ({src})")
if len(gaps) > 15:
    print(f"    ... {len(gaps) - 15} more")
print()
expected_hours = (rows[-1][0] - rows[0][0]) // HOUR_MS + 1
print(f"expected hours span : {expected_hours}")
print(f"actual unique hours : {len(seen)}")
print(f"missing hours total : {expected_hours - len(seen)}")
print()
# close_time convention check
odd_close = sum(1 for ot, ct, *_ in rows if ct - ot not in (HOUR_MS - 1, HOUR_MS))
print(f"close_time - open_time not in (3599999, 3600000): {odd_close}")

# reproduce the two hashes recorded in the handoff
for name in ("BTCUSDT-1h-2024-12.zip", "BTCUSDT-1h-2025-01.zip"):
    p = RAW / name
    if p.exists():
        print(f"{name} sha256 = {sha256(p)}")
