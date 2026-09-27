"""Model releases: writing one, promoting one, and rolling back.

Writing a release is not promoting it. A better backtest number promotes nothing by itself; a
release becomes the one that produces Published Runs only when someone says so. Rollback moves the
pointer and leaves every past Published Run exactly as it was.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.training import FoldPreprocessing

CURRENT_POINTER = "CURRENT"


@dataclass(frozen=True, slots=True)
class LogisticRelease:
    """A fitted Arm A model, with the in-fold preprocessing it was fitted alongside."""

    version: str
    preprocessing: FoldPreprocessing
    coefficients: tuple[float, ...]
    intercept: float
    contract_digest: str
    arm: str

    def predict(self, features: Mapping[str, float]) -> float:
        """No clipping: that belongs to the previous-direction baseline alone."""
        transformed = np.array(self.preprocessing.transform(features))
        logit = float(np.dot(transformed, np.array(self.coefficients)) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.version}.json"
        path.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "arm": self.arm,
                    "contract_digest": self.contract_digest,
                    "intercept": self.intercept,
                    "coefficients": list(self.coefficients),
                    "preprocessing": {
                        "features": list(self.preprocessing.features),
                        "medians": list(self.preprocessing.medians),
                        "means": list(self.preprocessing.means),
                        "stds": list(self.preprocessing.stds),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path


def load_release(directory: Path, version: str) -> LogisticRelease:
    path = directory / f"{version}.json"
    if not path.is_file():
        raise FatalDefect("unknown_release", f"no release {version!r} in {directory}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    pre = raw["preprocessing"]
    return LogisticRelease(
        version=raw["version"],
        preprocessing=FoldPreprocessing(
            features=tuple(pre["features"]),
            medians=tuple(pre["medians"]),
            means=tuple(pre["means"]),
            stds=tuple(pre["stds"]),
        ),
        coefficients=tuple(raw["coefficients"]),
        intercept=raw["intercept"],
        contract_digest=raw["contract_digest"],
        arm=raw["arm"],
    )


def promote(directory: Path, version: str) -> None:
    """Make this release the one that produces Published Runs. A deliberate act, on its own."""
    load_release(directory, version)  # refuses if it was never written
    (directory / CURRENT_POINTER).write_text(f"{version}\n", encoding="utf-8")


def rollback_to(directory: Path, version: str) -> None:
    """Switch the pointer back. Past Published Runs are never rewritten."""
    promote(directory, version)


def current_release(directory: Path) -> LogisticRelease | None:
    pointer = directory / CURRENT_POINTER
    if not pointer.is_file():
        return None
    return load_release(directory, pointer.read_text(encoding="utf-8").strip())
