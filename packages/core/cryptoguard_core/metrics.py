"""Scoring a set of probabilities against what happened.

Log loss is the Selection Metric because the Policy consumes a probability and compares it against
fixed thresholds: accuracy is invariant to any monotone transform preserving the crossing at 0.5, so
two models with identical accuracy can fire the 0.55 threshold on 5% of days and on 60% of days.

The clipping here belongs to the *metric*, not to any model: it stops one confident miss from
swallowing a run's score. No model's output is clipped on its way to the Policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

METRIC_EPSILON = 1e-15

# The loss of always answering "one half": the reference every forecast in this project is read
# against. Derived rather than written as 0.6931, because the rounded form appears in a report and a
# rounded reference subtracted from an unrounded score puts a fifth-decimal error into every delta.
NO_SKILL_LOG_LOSS = math.log(2)


def _paired(labels: Sequence[int], probabilities: Sequence[float]) -> list[tuple[int, float]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be the same length")
    if not labels:
        raise ValueError("no predictions to score")
    return list(zip(labels, probabilities, strict=True))


def daily_log_loss(label: int, probability: float) -> float:
    """One observation's contribution to the log loss, with the metric's clipping applied.

    Exposed because the Primary Comparison pairs the two arms day by day, and a second copy of this
    arithmetic beside it would be free to drift from the metric it claims to decompose.
    """
    clipped = min(max(probability, METRIC_EPSILON), 1 - METRIC_EPSILON)
    return -math.log(clipped) if label == 1 else -math.log(1 - clipped)


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """The mean of the per-observation losses — the same function the paired comparison uses.

    Summed in sequence rather than with `math.fsum`, which would be more accurate and would move the
    last bits of a number already recorded against a Trial. `a + (-b)` is bit-identical to `a - b`,
    so routing through `daily_log_loss` leaves every published score exactly where it was.
    """
    pairs = _paired(labels, probabilities)
    total = 0.0
    for label, probability in pairs:
        total += daily_log_loss(label, probability)
    return total / len(pairs)


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    pairs = _paired(labels, probabilities)
    return sum((probability - label) ** 2 for label, probability in pairs) / len(pairs)


def accuracy(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    pairs = _paired(labels, probabilities)
    return sum((probability > 0.5) == bool(label) for label, probability in pairs) / len(pairs)


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    lower: float
    upper: float
    count: int
    mean_prediction: float | None
    observed_rate: float | None


def calibration_curve(
    labels: Sequence[int], probabilities: Sequence[float], bins: int = 10
) -> tuple[CalibrationPoint, ...]:
    """Predicted probability against the rate that actually occurred, in equal-width bins."""
    pairs = _paired(labels, probabilities)
    width = 1.0 / bins
    points: list[CalibrationPoint] = []
    for index in range(bins):
        lower = index * width
        upper = (index + 1) * width
        inside = [
            (label, probability)
            for label, probability in pairs
            if (lower <= probability < upper) or (index == bins - 1 and probability == 1.0)
        ]
        count = len(inside)
        points.append(
            CalibrationPoint(
                lower=lower,
                upper=upper,
                count=count,
                mean_prediction=(sum(p for _, p in inside) / count) if count else None,
                observed_rate=(sum(label for label, _ in inside) / count) if count else None,
            )
        )
    return tuple(points)
