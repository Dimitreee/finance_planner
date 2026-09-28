"""The Primary Comparison: one arm against another, paired day by day.

Paired rather than two averages side by side. Both arms score the same 1 096 Development Period
days from the same folds, so the difference on each day is a measurement, and the day-to-day noise
the two arms share cancels instead of being counted twice. Two arms can also share a mean log loss
and disagree on most days; an average cannot tell those apart and a paired series can.

The sign convention is stated once and never inverted: a **negative** mean difference means the
treatment arm carries the lower loss, because the difference is treatment minus baseline.

Nothing here selects or promotes anything. The pre-registered comparison is Arm C at the Primary Lag
against Arm A (`evaluation.primary_comparison` in the contract); every other pairing this module can
compute is description, and ADR-0007 keeps the Headline Metric out of the selection path entirely.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, NoReturn

from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.metrics import daily_log_loss
from cryptoguard_core.training import Prediction

PairingDefect = Literal["paired_days_mismatch", "paired_label_mismatch"]


@dataclass(frozen=True, slots=True)
class DailyDifference:
    day: date
    treatment_loss: float
    baseline_loss: float

    @property
    def difference(self) -> float:
        return self.treatment_loss - self.baseline_loss


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A paired daily log-loss difference, with both arms' own losses over the paired days."""

    treatment_name: str
    baseline_name: str
    days: int
    mean_difference: float
    treatment_log_loss: float
    baseline_log_loss: float
    daily: tuple[DailyDifference, ...]

    @property
    def treatment_predicts_better(self) -> bool:
        """Descriptive only. A point estimate on either side of zero selects nothing (ADR-0007)."""
        return self.mean_difference < 0


def _by_day(predictions: Sequence[Prediction]) -> Mapping[date, Prediction]:
    return {prediction.day: prediction for prediction in predictions}


def _refuse_pairing(kind: PairingDefect, detail: str) -> NoReturn:
    raise FatalDefect(kind, detail)


def paired_log_loss_difference(
    treatment: Sequence[Prediction],
    baseline: Sequence[Prediction],
    *,
    treatment_name: str,
    baseline_name: str,
    expect_days: int,
) -> PairedComparison:
    """Pair the two arms on the days both scored, and refuse if that is not `expect_days` days.

    The expected count is passed in rather than inferred because it is pre-registered: the
    Development Period scores 1 096 days, and a join that silently returned fewer would answer a
    different question with the same name.
    """
    treatment_by_day = _by_day(treatment)
    baseline_by_day = _by_day(baseline)
    shared = sorted(set(treatment_by_day) & set(baseline_by_day))
    if len(shared) != expect_days:
        _refuse_pairing(
            "paired_days_mismatch",
            f"{treatment_name} scored {len(treatment_by_day)} days and {baseline_name} scored "
            f"{len(baseline_by_day)}; they share {len(shared)}, but {expect_days} were expected",
        )

    entries: list[DailyDifference] = []
    for day in shared:
        treated = treatment_by_day[day]
        baselined = baseline_by_day[day]
        if treated.label != baselined.label:
            _refuse_pairing(
                "paired_label_mismatch",
                f"on {day.isoformat()} {treatment_name} scored label {treated.label} and "
                f"{baseline_name} scored {baselined.label}; the Label comes from the price and "
                "cannot depend on the arm",
            )
        entries.append(
            DailyDifference(
                day=day,
                treatment_loss=daily_log_loss(treated.label, treated.probability),
                baseline_loss=daily_log_loss(baselined.label, baselined.probability),
            )
        )

    count = len(entries)
    return PairedComparison(
        treatment_name=treatment_name,
        baseline_name=baseline_name,
        days=count,
        mean_difference=math.fsum(entry.difference for entry in entries) / count,
        treatment_log_loss=math.fsum(entry.treatment_loss for entry in entries) / count,
        baseline_log_loss=math.fsum(entry.baseline_loss for entry in entries) / count,
        daily=tuple(entries),
    )
