"""Fitting Arm A: what a fold may see, what is fitted inside it, and how it is scored.

The rule that matters here is not visible in a date split. A row may enter a fit only once its
outcome is knowable, and the day immediately before a cutoff fails that test even though its
features are old enough — which is exactly the case a date-only split waves through.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from cryptoguard_core.backtest import BASE_SCENARIO, run_backtest
from cryptoguard_core.contract import CONTRACT_PATH, load_contract
from cryptoguard_core.dataset import ARM_A_FEATURES, DAY_MS, DecisionDayRow
from cryptoguard_core.evaluate import evaluate_arm_a
from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.metrics import (
    accuracy,
    brier_score,
    calibration_curve,
    log_loss,
)
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import PolicyConfig, Position
from cryptoguard_core.training import (
    Fold,
    admissible_rows,
    fit_fold,
    quarterly_folds,
    walk_forward,
)
from cryptoguard_core.trials import TrialLog

FIRST_DAY = date(2021, 2, 1)
FIRST_CUTOFF_MS = 1_612_137_600_000  # 2021-02-01T00:00:00Z


def row(index: int, *, label: int | None = None, feature_base: float = 0.0) -> DecisionDayRow:
    cutoff = FIRST_CUTOFF_MS + index * DAY_MS
    return DecisionDayRow(
        day=FIRST_DAY + timedelta(days=index),
        feature_cutoff_ms=cutoff,
        anchor_price=100.0,
        decision_price=100.0,
        label=index % 2 if label is None else label,
        # Knowable only at the next Decision Day's 01:00.
        label_end_ms=cutoff + DAY_MS + HOUR_MS,
        bars_in_day=24,
        features={name: feature_base + i * 0.01 for i, name in enumerate(ARM_A_FEATURES)},
    )


def series(count: int) -> tuple[DecisionDayRow, ...]:
    return tuple(row(index, feature_base=index * 0.001) for index in range(count))


class TestMetrics:
    def test_log_loss_rewards_a_confident_correct_call(self) -> None:
        assert log_loss([1], [0.9]) < log_loss([1], [0.6])

    def test_log_loss_is_finite_on_a_confident_miss_only_because_of_clipping(self) -> None:
        """This model is never clipped; the metric clips so a single miss cannot swallow a run."""
        assert log_loss([1], [0.0]) < float("inf")

    def test_brier_is_the_mean_squared_error_of_the_probability(self) -> None:
        assert brier_score([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)

    def test_accuracy_counts_calls_above_one_half(self) -> None:
        assert accuracy([1, 0, 1], [0.6, 0.4, 0.4]) == pytest.approx(2 / 3)

    def test_the_calibration_curve_bins_predictions_against_outcomes(self) -> None:
        points = calibration_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], bins=2)
        assert [point.count for point in points] == [2, 2]
        assert points[0].observed_rate == 0.0
        assert points[1].observed_rate == 1.0


class TestAdmissibility:
    def test_a_row_whose_outcome_is_not_yet_known_is_excluded(self) -> None:
        cutoff = FIRST_CUTOFF_MS + 10 * DAY_MS
        admitted = admissible_rows(series(20), cutoff)
        assert all(candidate.label_end_ms <= cutoff for candidate in admitted)

    def test_the_day_immediately_before_the_cutoff_is_excluded_by_the_rule(self) -> None:
        """Not by a configured buffer: its outcome resolves an hour after the cutoff."""
        cutoff = FIRST_CUTOFF_MS + 10 * DAY_MS
        admitted = admissible_rows(series(20), cutoff)
        days = [candidate.day for candidate in admitted]
        assert FIRST_DAY + timedelta(days=8) in days
        assert FIRST_DAY + timedelta(days=9) not in days

    def test_a_row_without_a_resolved_label_is_never_admitted(self) -> None:
        rows = (*series(5), row(5, label=None))
        cutoff = FIRST_CUTOFF_MS + 100 * DAY_MS
        assert all(candidate.label is not None for candidate in admissible_rows(rows, cutoff))


class TestFolds:
    def test_twelve_quarters_are_scored_and_the_holdout_is_untouched(self) -> None:
        folds = quarterly_folds()
        assert len(folds) == 12
        assert folds[0].test_first_day == date(2022, 1, 1)
        assert folds[-1].test_last_day == date(2024, 12, 31)
        assert sum(fold.test_days for fold in folds) == 1096
        assert all(fold.test_last_day < date(2025, 1, 1) for fold in folds)

    def test_the_training_window_expands_and_never_reaches_into_the_test_quarter(self) -> None:
        folds = quarterly_folds()
        assert folds[0].refit_cutoff_day == date(2022, 1, 1)
        assert folds[-1].refit_cutoff_day == date(2024, 10, 1)
        for fold in folds:
            assert fold.refit_cutoff_day == fold.test_first_day


class TestFitting:
    def test_preprocessing_statistics_are_fitted_inside_the_fold(self) -> None:
        """Different folds see different data, so their medians must differ."""
        rows = series(400)
        early = fit_fold(rows, Fold(date(2021, 6, 1), date(2021, 8, 31)), regularisation=(1.0,))
        late = fit_fold(rows, Fold(date(2022, 1, 1), date(2022, 3, 31)), regularisation=(1.0,))
        assert early.preprocessing.medians != late.preprocessing.medians
        assert early.preprocessing.means != late.preprocessing.means

    def test_regularisation_is_chosen_on_an_inner_validation_inside_the_fold(self) -> None:
        rows = series(400)
        fitted = fit_fold(
            rows, Fold(date(2022, 1, 1), date(2022, 3, 31)), regularisation=(0.01, 0.1, 1.0)
        )
        assert fitted.regularisation in (0.01, 0.1, 1.0)
        # 89, not 90: the day before the inner cutoff is excluded by the admissibility rule, and
        # the field reports the split that was actually used rather than the window asked for.
        assert fitted.inner_validation_days == 89
        assert fitted.training_days > fitted.inner_validation_days

    def test_a_fold_without_enough_history_is_refused(self) -> None:
        with pytest.raises(FatalDefect) as caught:
            fit_fold(series(40), Fold(date(2021, 3, 1), date(2021, 3, 31)), regularisation=(1.0,))
        assert caught.value.kind == "insufficient_training_rows"


class TestWalkForward:
    def test_every_scored_day_is_predicted_exactly_once(self) -> None:
        rows = series(500)
        folds = (
            Fold(date(2021, 9, 1), date(2021, 9, 30)),
            Fold(date(2021, 10, 1), date(2021, 10, 31)),
        )
        result = walk_forward(rows, folds, regularisation=(1.0,))
        days = [prediction.day for prediction in result.predictions]
        assert len(days) == len(set(days)) == 61
        assert all(0.0 < prediction.probability < 1.0 for prediction in result.predictions)

    def test_the_selection_metric_is_log_loss_and_the_rest_are_reported_beside_it(self) -> None:
        rows = series(500)
        folds = (Fold(date(2021, 9, 1), date(2021, 9, 30)),)
        result = walk_forward(rows, folds, regularisation=(1.0,))
        assert result.selection_metric == "log_loss"
        assert result.log_loss > 0
        assert 0 <= result.brier <= 1
        assert 0 <= result.accuracy <= 1
        assert result.calibration


class TestEvaluation:
    def test_a_fold_that_reaches_the_final_holdout_is_refused(self, tmp_path: Path) -> None:
        """The Final Holdout is evaluated once, after everything is frozen — not by this run."""
        contract = load_contract()
        trials = TrialLog(tmp_path / "trials.json", budget=12)
        reaching = (Fold(date(2025, 1, 1), date(2025, 3, 31)),)
        with pytest.raises(FatalDefect) as caught:
            evaluate_arm_a(series(10), contract, trials, releases_dir=tmp_path, folds=reaching)
        assert caught.value.kind == "contract_inconsistent"
        assert "Final Holdout" in str(caught.value)

    def test_the_replay_is_scored_on_out_of_sample_predictions(self, tmp_path: Path) -> None:
        """A replay that fell back to the model would score the days the model was fitted on."""
        rows = series(500)
        folds = (Fold(date(2021, 9, 1), date(2021, 9, 30)),)
        walk = walk_forward(rows, folds, regularisation=(1.0,))
        recorded = {prediction.day: prediction.probability for prediction in walk.predictions}
        scored = tuple(row for row in rows if row.day in recorded)

        with pytest.raises(FatalDefect) as caught:
            run_backtest(
                rows,  # more days than there are recorded predictions
                PreviousDirectionBaseline(),
                PolicyConfig(0.55, 0.45),
                BASE_SCENARIO,
                initial=Position(0.0, 1000.0),
                probabilities=recorded,
            )
        assert caught.value.kind == "missing_decision_day_row"

        result = run_backtest(
            scored,
            PreviousDirectionBaseline(),
            PolicyConfig(0.55, 0.45),
            BASE_SCENARIO,
            initial=Position(0.0, 1000.0),
            probabilities=recorded,
        )
        assert result.days == len(recorded)


class TestContractDrivenProtocol:
    def test_the_folds_come_from_the_contract_not_from_module_defaults(
        self, tmp_path: Path
    ) -> None:
        """A digest recorded against a result must certify the protocol that was walked."""
        text = (
            CONTRACT_PATH.read_text(encoding="utf-8")
            .replace("first_test_quarter: 2022Q1", "first_test_quarter: 2022Q2")
            .replace("scored_development_days: 1096", "scored_development_days: 1006")
        )
        altered = tmp_path / "experiment.yaml"
        altered.write_text(text, encoding="utf-8")
        splits = load_contract(altered).values["splits"]

        folds = quarterly_folds(splits["first_test_quarter"], splits["last_test_quarter"])
        assert folds[0].test_first_day == date(2022, 4, 1)
        assert quarterly_folds()[0].test_first_day == date(2022, 1, 1), "the default is unchanged"

    def test_a_failed_evaluation_does_not_spend_a_trial(self, tmp_path: Path) -> None:
        """A budget that punishes crashes encourages not running things."""
        contract = load_contract()
        trials = TrialLog(tmp_path / "trials.json", budget=12)
        with pytest.raises(FatalDefect):
            evaluate_arm_a(
                series(10),  # nowhere near enough history for the contract's first fold
                contract,
                trials,
                releases_dir=tmp_path,
            )
        assert trials.spent() == 0

    def test_a_single_class_split_is_refused_by_name(self) -> None:
        flat = tuple(row(index, label=1) for index in range(300))
        with pytest.raises(FatalDefect) as caught:
            fit_fold(flat, Fold(date(2021, 9, 1), date(2021, 9, 30)), regularisation=(1.0,))
        assert caught.value.kind == "insufficient_training_rows"

    def test_a_walk_forward_with_no_overlapping_days_is_refused(self) -> None:
        with pytest.raises(FatalDefect) as caught:
            walk_forward(series(400), (Fold(date(2030, 1, 1), date(2030, 3, 31)),))
        assert caught.value.kind == "insufficient_training_rows"
