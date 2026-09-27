"""The Trial budget: how many results were looked at, and which of them replaced another.

A Trial is an evaluation whose result was looked at and could have influenced a choice. A rerun
after a fixed bug is recorded as a correction and does not consume a Trial, because a budget that
punishes fixing a bug would encourage leaving it in.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptoguard_core.ingest import FatalDefect


@dataclass(frozen=True, slots=True)
class Trial:
    number: int
    arm: str
    lag_hours: int | None
    contract_digest: str
    score: float
    recorded_at: str
    corrects: int | None


class TrialLog:
    """A JSON-backed register. Appending is the only operation; nothing is ever edited."""

    def __init__(self, path: Path, *, budget: int) -> None:
        self._path = path
        self._budget = budget

    def entries(self) -> tuple[Trial, ...]:
        if not self._path.is_file():
            return ()
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return tuple(Trial(**entry) for entry in raw)

    def spent(self) -> int:
        """Corrections replace an earlier result rather than adding to the count."""
        return sum(1 for entry in self.entries() if entry.corrects is None)

    def remaining(self) -> int:
        return self._budget - self.spent()

    def next_number(self) -> int:
        """Monotonic across corrections too, so a version derived from it cannot collide."""
        return len(self.entries()) + 1

    def record(
        self,
        *,
        arm: str,
        lag_hours: int | None,
        contract_digest: str,
        score: float,
        corrects: int | None = None,
    ) -> Trial:
        existing = self.entries()
        if corrects is not None and corrects not in {entry.number for entry in existing}:
            raise FatalDefect("unknown_trial", f"trial {corrects} does not exist to be corrected")
        if corrects is None and self.remaining() <= 0:
            raise FatalDefect(
                "trial_budget_exhausted",
                f"the Trial budget of {self._budget} is spent; raising it is a recorded decision, "
                "not an accident",
            )
        trial = Trial(
            number=len(existing) + 1,
            arm=arm,
            lag_hours=lag_hours,
            contract_digest=contract_digest,
            score=score,
            recorded_at=datetime.now(tz=UTC).isoformat(),
            corrects=corrects,
        )
        self._write([*existing, trial])
        return trial

    def _write(self, entries: Sequence[Trial]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(entry) for entry in entries], indent=2) + "\n", encoding="utf-8"
        )
