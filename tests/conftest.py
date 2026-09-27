"""Helpers for building Archive Backfill snapshots out of real archive slices."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path

import pytest
from cryptoguard_core.ingest import HOUR_MS, Bar, DataQualityFlag

FIXTURES = Path(__file__).parent / "fixtures" / "binance"
SNAPSHOT = Path(__file__).resolve().parents[1] / "data/raw/binance/klines/BTCUSDT/1h"

SnapshotBuilder = Callable[..., Path]


def _write_archive(
    directory: Path, stem: str, csv_text: str, *, checksum: str | None = None
) -> None:
    archive = directory / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.csv", csv_text)
    digest = checksum if checksum is not None else hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.CHECKSUM").write_text(f"{digest}  {archive.name}\n")


@pytest.fixture
def raw_archive(tmp_path: Path) -> Callable[[dict[str, str]], Path]:
    """A snapshot holding one archive with exactly the members given (possibly none, or several)."""

    def build(members: dict[str, str]) -> Path:
        directory = tmp_path / "raw"
        directory.mkdir(exist_ok=True)
        archive = directory / "BTCUSDT-1h-9999-01.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, text in members.items():
                zf.writestr(name, text)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_suffix(".zip.CHECKSUM").write_text(f"{digest}  {archive.name}\n")
        return directory

    return build


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotBuilder:
    """Build a snapshot directory from named fixture slices and/or synthesised rows."""

    def build(
        *fixture_names: str,
        rows: Sequence[Sequence[object]] | None = None,
        stem: str = "BTCUSDT-1h-9999-01",
        checksum: str | None = None,
    ) -> Path:
        directory = tmp_path / "snapshot"
        directory.mkdir(exist_ok=True)
        for name in fixture_names:
            text = (FIXTURES / name).read_text()
            _write_archive(directory, f"BTCUSDT-1h-{Path(name).stem}", text, checksum=checksum)
        if rows is not None:
            text = "\n".join(",".join(str(cell) for cell in row) for row in rows) + "\n"
            _write_archive(directory, stem, text, checksum=checksum)
        return directory

    return build


def synthetic_rows(*bars: tuple[int, float, float, float, float]) -> list[list[object]]:
    """Binance's twelve-column row shape, from (open_time_ms, open, high, low, close)."""
    out: list[list[object]] = []
    for open_time, open_, high, low, close in bars:
        out.append(
            [open_time, open_, high, low, close, 1.0, open_time + 3_599_999, 1.0, 1, 1.0, 1.0, 0]
        )
    return out


def flag_kinds(flags: Iterable[DataQualityFlag]) -> list[str]:
    return [flag.kind for flag in flags]


def hourly_series(
    start: datetime,
    hours: int,
    *,
    quote: Callable[[int], tuple[float, float, float]] = lambda i: (
        100.0 + (i % 24) * 0.1,
        100.0 + (i % 24) * 0.1 + 0.05,
        1.0,
    ),
    skip: Sequence[int] = (),
) -> list[Bar]:
    """A continuous hourly series; `quote(i)` gives (open, close, volume) for the i-th hour.

    The default price depends only on the hour of day, so it moves within a day — real
    volatility for the trailing windows — while every day's 01:00 open and 23:00 close repeat
    exactly. That makes an unchanged Decision Price, and therefore a Label tie, the default case.
    """
    first_ms = int(start.timestamp() * 1000)
    bars: list[Bar] = []
    for index in range(hours):
        if index in skip:
            continue
        open_, close, volume = quote(index)
        open_time = first_ms + index * HOUR_MS
        bars.append(
            Bar(
                open_time_ms=open_time,
                open=open_,
                high=max(open_, close),
                low=min(open_, close),
                close=close,
                volume=volume,
                close_time_ms=open_time + HOUR_MS - 1,
                source="synthetic",
            )
        )
    return bars
