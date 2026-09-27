"""Fitting Arm A under the frozen protocol: what a fold may see, and what is fitted inside it.

The admissibility rule is the part a date split cannot express. A row may enter a fit only once its
outcome is knowable, so the day immediately before a Refit Cutoff is excluded even though its
features are old enough — its Label resolves an hour after the cutoff. That exclusion is a
consequence of the rule, not a configured buffer.

Everything statistical is fitted inside the fold: the imputation medians, the scaler, and the
regularisation strength. Fitting any of them once over the whole window would carry the future into
the past through preprocessing parameters, which no date-based test would detect.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression

from cryptoguard_core.dataset import ARM_A_FEATURES, DecisionDayRow
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.metrics import (
    CalibrationPoint,
    accuracy,
    brier_score,
    calibration_curve,
    log_loss,
)

DEFAULT_INNER_VALIDATION_DAYS = 90
MIN_TRAINING_ROWS_MARGIN = 10
SELECTION_METRIC = "log_loss"
DEFAULT_REGULARISATION = (0.01, 0.1, 1.0)
FIRST_TEST_QUARTER = "2022Q1"
LAST_TEST_QUARTER = "2024Q4"


def _midnight_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class Fold:
    test_first_day: date
    test_last_day: date

    @property
    def refit_cutoff_day(self) -> date:
        """The fit happens at the start of the test quarter and may see nothing later."""
        return self.test_first_day

    @property
    def test_days(self) -> int:
        return (self.test_last_day - self.test_first_day).days + 1


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    first = date(year, 3 * (quarter - 1) + 1, 1)
    last = date(year + 1, 1, 1) if quarter == 4 else date(year, 3 * quarter + 1, 1)
    return first, last - timedelta(days=1)


def quarterly_folds(
    first: str = FIRST_TEST_QUARTER, last: str = LAST_TEST_QUARTER
) -> tuple[Fold, ...]:
    """One fold per test quarter, expanding: the training window grows, the test quarter moves.

    The defaults exist for tests. A real run takes the quarters from the Experiment Contract, so the
    digest recorded against a result certifies the protocol that was actually walked.
    """
    start = (int(first[:4]), int(first[-1]))
    end = (int(last[:4]), int(last[-1]))
    folds: list[Fold] = []
    year, quarter = start
    while (year, quarter) <= end:
        folds.append(Fold(*_quarter_bounds(year, quarter)))
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    return tuple(folds)


def admissible_rows(rows: Sequence[DecisionDayRow], cutoff_ms: int) -> tuple[DecisionDayRow, ...]:
    """Rows whose outcome was already knowable at the cutoff."""
    return tuple(row for row in rows if row.label is not None and row.label_end_ms <= cutoff_ms)


@dataclass(frozen=True, slots=True)
class FoldPreprocessing:
    """Imputation medians and scaler statistics, fitted on one fold's training rows only."""

    features: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def transform(self, values: Mapping[str, float]) -> list[float]:
        out: list[float] = []
        for index, name in enumerate(self.features):
            raw = values.get(name)
            usable = self.medians[index] if raw is None or raw != raw else raw
            spread = self.stds[index] or 1.0
            out.append((usable - self.means[index]) / spread)
        return out


def _require_both_classes(rows: Sequence[DecisionDayRow], what: str) -> None:
    """A single-class split makes the solver raise a bare ValueError past the refusal channel."""
    if len({row.label for row in rows}) < 2:
        raise FatalDefect(
            "insufficient_training_rows", f"{what} holds only one outcome, so nothing can be fitted"
        )


def _fit_preprocessing(
    rows: Sequence[DecisionDayRow], features: Sequence[str]
) -> FoldPreprocessing:
    medians: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for name in features:
        present = [
            row.features[name]
            for row in rows
            if name in row.features and row.features[name] == row.features[name]
        ]
        if not present:
            raise FatalDefect(
                "insufficient_training_rows", f"no usable values of {name!r} in this fold"
            )
        medians.append(statistics.median(present))
        means.append(statistics.fmean(present))
        stds.append(statistics.pstdev(present))
    return FoldPreprocessing(
        features=tuple(features), medians=tuple(medians), means=tuple(means), stds=tuple(stds)
    )


@dataclass(frozen=True, slots=True)
class FittedFold:
    fold: Fold
    preprocessing: FoldPreprocessing
    regularisation: float
    training_days: int
    inner_validation_days: int
    coefficients: tuple[float, ...]
    intercept: float

    def predict(self, values: Mapping[str, float]) -> float:
        transformed = np.array(self.preprocessing.transform(values))
        logit = float(np.dot(transformed, np.array(self.coefficients)) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))


def _design(
    rows: Sequence[DecisionDayRow], preprocessing: FoldPreprocessing
) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([preprocessing.transform(row.features) for row in rows])
    labels = np.array([row.label for row in rows])
    return features, labels


def _fit_logistic(
    rows: Sequence[DecisionDayRow], preprocessing: FoldPreprocessing, regularisation: float
) -> LogisticRegression:
    features, labels = _design(rows, preprocessing)
    # `l1_ratio=0` is the current spelling of an L2 penalty; `penalty="l2"` is deprecated.
    model = LogisticRegression(C=regularisation, l1_ratio=0, solver="lbfgs", max_iter=1000)
    model.fit(features, labels)
    return model


def fit_fold(
    rows: Sequence[DecisionDayRow],
    fold: Fold,
    *,
    regularisation: Sequence[float] = DEFAULT_REGULARISATION,
    features: Sequence[str] = ARM_A_FEATURES,
    inner_validation_days: int = DEFAULT_INNER_VALIDATION_DAYS,
) -> FittedFold:
    """Fit one fold, choosing the regularisation on an inner validation inside it."""
    cutoff_ms = _midnight_ms(fold.refit_cutoff_day)
    admitted = admissible_rows(rows, cutoff_ms)
    minimum = inner_validation_days + MIN_TRAINING_ROWS_MARGIN
    if len(admitted) < minimum:
        raise FatalDefect(
            "insufficient_training_rows",
            f"fold starting {fold.test_first_day} admits {len(admitted)} rows, "
            f"fewer than the {minimum} required",
        )

    inner_cutoff_day = fold.refit_cutoff_day - timedelta(days=inner_validation_days)
    inner_cutoff_ms = _midnight_ms(inner_cutoff_day)
    inner_training = admissible_rows(admitted, inner_cutoff_ms)
    inner_validation = tuple(row for row in admitted if row.day >= inner_cutoff_day)
    if not inner_training or not inner_validation:
        raise FatalDefect(
            "insufficient_training_rows",
            f"fold starting {fold.test_first_day} cannot form an inner validation split",
        )

    _require_both_classes(inner_training, "the inner training split")
    inner_preprocessing = _fit_preprocessing(inner_training, features)
    best: tuple[float, float] | None = None
    for candidate in regularisation:
        model = _fit_logistic(inner_training, inner_preprocessing, candidate)
        probabilities = [
            float(model.predict_proba([inner_preprocessing.transform(row.features)])[0][1])
            for row in inner_validation
        ]
        labels = [row.label for row in inner_validation if row.label is not None]
        score = log_loss(labels, probabilities)
        if best is None or score < best[1]:
            best = (candidate, score)
    if best is None:
        raise FatalDefect("insufficient_training_rows", "no regularisation candidates were given")

    chosen = best[0]
    _require_both_classes(admitted, "the fold's training rows")
    preprocessing = _fit_preprocessing(admitted, features)
    final = _fit_logistic(admitted, preprocessing, chosen)
    return FittedFold(
        fold=fold,
        preprocessing=preprocessing,
        regularisation=chosen,
        training_days=len(admitted),
        inner_validation_days=len(inner_validation),
        coefficients=tuple(float(value) for value in final.coef_[0]),
        intercept=float(final.intercept_[0]),
    )


@dataclass(frozen=True, slots=True)
class Prediction:
    day: date
    probability: float
    label: int


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    predictions: tuple[Prediction, ...]
    fits: tuple[FittedFold, ...]
    selection_metric: str
    log_loss: float
    brier: float
    accuracy: float
    calibration: tuple[CalibrationPoint, ...]


def walk_forward(
    rows: Sequence[DecisionDayRow],
    folds: Sequence[Fold],
    *,
    regularisation: Sequence[float] = DEFAULT_REGULARISATION,
    features: Sequence[str] = ARM_A_FEATURES,
    inner_validation_days: int = DEFAULT_INNER_VALIDATION_DAYS,
) -> WalkForwardResult:
    """Fit each fold and score the quarter that follows it. No probability is clipped."""
    if not folds:
        raise FatalDefect("insufficient_training_rows", "a walk-forward needs at least one fold")
    predictions: list[Prediction] = []
    fits: list[FittedFold] = []
    for fold in folds:
        fitted = fit_fold(
            rows,
            fold,
            regularisation=regularisation,
            features=features,
            inner_validation_days=inner_validation_days,
        )
        fits.append(fitted)
        for row in rows:
            if fold.test_first_day <= row.day <= fold.test_last_day and row.label is not None:
                predictions.append(
                    Prediction(
                        day=row.day, probability=fitted.predict(row.features), label=row.label
                    )
                )

    if not predictions:
        raise FatalDefect(
            "insufficient_training_rows",
            "no scored days: the folds and the rows given do not overlap",
        )
    labels = [prediction.label for prediction in predictions]
    probabilities = [prediction.probability for prediction in predictions]
    return WalkForwardResult(
        predictions=tuple(predictions),
        fits=tuple(fits),
        selection_metric=SELECTION_METRIC,
        log_loss=log_loss(labels, probabilities),
        brier=brier_score(labels, probabilities),
        accuracy=accuracy(labels, probabilities),
        calibration=calibration_curve(labels, probabilities),
    )
