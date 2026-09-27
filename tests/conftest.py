"""Helpers for building Archive Backfill snapshots out of real archive slices."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pytest
from cryptoguard_core.ingest import DataQualityFlag

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
