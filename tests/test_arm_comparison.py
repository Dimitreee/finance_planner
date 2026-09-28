"""Arms B and C across the lag grid, and the Primary Comparison.

The protocol does not vary with the arm: the same folds, the same admissibility rule, the same
in-fold preprocessing and the same regularisation grid fit all three. What varies is the feature
matrix, which is the whole point — a comparison between arms is only about features.

Nothing here promotes an arm. Promotion is a deliberate act (ADR-0007), and the Headline Metric
selects nothing.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from conftest import SNAPSHOT
from cryptoguard_core.backtest import BASE_SCENARIO, run_backtest
from cryptoguard_core.comparison import paired_log_loss_difference
from cryptoguard_core.contract import load_contract
from cryptoguard_core.dataset import (
    ARM_A_FEATURES,
    DAY_MS,
    FEATURES_BY_ARM,
    DecisionDayRow,
    build_decision_day_rows,
)
from cryptoguard_core.evaluate import development_last_day, evaluate_arm, walk_arm
from cryptoguard_core.ingest import HOUR_MS, FatalDefect, load_archive_backfill
from cryptoguard_core.metrics import NO_SKILL_LOG_LOSS, daily_log_loss, log_loss
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import PolicyConfig, Position
from cryptoguard_core.training import Fold, Prediction
from cryptoguard_core.trials import TrialLog

FIRST_DAY = date(2021, 2, 1)
FIRST_CUTOFF_MS = 1_612_137_600_000  # 2021-02-01T00:00:00Z


def row(index: int, *, arm: str = "a", lag_hours: int | None = None) -> DecisionDayRow:
    """One synthetic Decision Day carrying exactly the named arm's features."""
    cutoff = FIRST_CUTOFF_MS + index * DAY_MS
    names = FEATURES_BY_ARM[arm]
    return DecisionDayRow(
        day=FIRST_DAY + timedelta(days=index),
        feature_cutoff_ms=cutoff,
        anchor_price=100.0 + index * 0.01,
        decision_price=100.0 + index * 0.01,
        label=index % 2,
        label_end_ms=cutoff + DAY_MS + HOUR_MS,
        bars_in_day=24,
        features={name: index * 0.001 + position * 0.01 for position, name in enumerate(names)},
        news_lag_hours=lag_hours,
    )


def series(
    count: int, *, arm: str = "a", lag_hours: int | None = None
) -> tuple[DecisionDayRow, ...]:
    return tuple(row(index, arm=arm, lag_hours=lag_hours) for index in range(count))


def fresh_trials(tmp_path: Path) -> TrialLog:
    return TrialLog(tmp_path / "trials.json", budget=12)


FOLDS = (Fold(date(2022, 1, 1), date(2022, 3, 31)),)


def test_a_trial_records_the_arm_and_the_lag_it_was_fitted_under(tmp_path: Path) -> None:
    """A Trial that does not name its arm and lag cannot be checked against a budget."""
    trials = fresh_trials(tmp_path)
    evaluate_arm(
        series(400, arm="b", lag_hours=6),
        load_contract(),
        trials,
        arm="b",
        lag_hours=6,
        releases_dir=tmp_path / "releases",
        folds=FOLDS,
    )
    recorded = trials.entries()[-1]
    assert recorded.arm == "B"
    assert recorded.lag_hours == 6
    assert trials.spent() == 1


def test_arm_a_records_no_lag_because_it_reads_no_news(tmp_path: Path) -> None:
    trials = fresh_trials(tmp_path)
    evaluate_arm(
        series(400),
        load_contract(),
        trials,
        arm="a",
        releases_dir=tmp_path / "releases",
        folds=FOLDS,
    )
    recorded = trials.entries()[-1]
    assert recorded.arm == "A"
    assert recorded.lag_hours is None


def test_rows_built_for_another_arm_are_refused_rather_than_relabelled(tmp_path: Path) -> None:
    """Fitting Arm A's rows under Arm C's name would record a Trial that did not happen."""
    with pytest.raises(FatalDefect) as caught:
        evaluate_arm(
            series(400),
            load_contract(),
            fresh_trials(tmp_path),
            arm="c",
            lag_hours=24,
            releases_dir=tmp_path / "releases",
            folds=FOLDS,
        )
    assert caught.value.kind == "arm_feature_mismatch"
    assert tuple(ARM_A_FEATURES) != FEATURES_BY_ARM["c"]


def test_rows_built_under_another_lag_are_refused(tmp_path: Path) -> None:
    """The lag is what the Trial claims to have tested, so it comes from the rows."""
    with pytest.raises(FatalDefect) as caught:
        evaluate_arm(
            series(400, arm="b", lag_hours=1),
            load_contract(),
            fresh_trials(tmp_path),
            arm="b",
            lag_hours=24,
            releases_dir=tmp_path / "releases",
            folds=FOLDS,
        )
    assert caught.value.kind == "arm_lag_mismatch"


def test_a_lag_outside_the_pre_registered_grid_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FatalDefect) as caught:
        evaluate_arm(
            series(400, arm="b", lag_hours=3),
            load_contract(),
            fresh_trials(tmp_path),
            arm="b",
            lag_hours=3,
            releases_dir=tmp_path / "releases",
            folds=FOLDS,
        )
    assert caught.value.kind == "news_lag_not_pre_registered"


# --- Arm A must not move --------------------------------------------------------------------------

# Read off the recorded Trial in artifacts/trials.json, not recomputed here: the anchor for a
# regression has to come from the run being protected, or it protects nothing.
ARM_A_LOG_LOSS = 0.6943589668354004
ARM_A_BRIER = 0.25059577870170197
ARM_A_SCORED_DAYS = 1096


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_arm_a_reproduces_its_recorded_numbers_through_the_generalised_entry_point(
    tmp_path: Path,
) -> None:
    """If these move, the refactor changed the fit and the arm comparison is invalid."""
    bars = load_archive_backfill(SNAPSHOT).bars
    rows = build_decision_day_rows(bars, arm="a")
    evaluation = evaluate_arm(
        rows,
        load_contract(),
        fresh_trials(tmp_path),
        arm="a",
        releases_dir=tmp_path / "releases",
    )
    assert evaluation.scored_days == ARM_A_SCORED_DAYS
    assert evaluation.log_loss == pytest.approx(ARM_A_LOG_LOSS, abs=1e-12)
    assert evaluation.brier == pytest.approx(ARM_A_BRIER, abs=1e-12)
    assert len(evaluation.predictions) == ARM_A_SCORED_DAYS
    # The finding this project rests on, restated where a refactor would break it.
    assert evaluation.log_loss > NO_SKILL_LOG_LOSS


def test_the_development_period_a_build_may_touch_stops_before_the_holdout() -> None:
    """The date range the loader is allowed to touch, checked where it is decided.

    Found in review: the folds stopping at 2024-12-31 was the only guard, so a build running to the
    end of the Research Window put Final Holdout dates in the same process as the comparison.
    """
    contract = load_contract()
    assert development_last_day(contract) < contract.values["splits"]["final_holdout_first_day"]


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_a_build_stopped_at_the_development_period_still_scores_every_day(tmp_path: Path) -> None:
    """Narrowing what is loaded must remove reach without moving a number.

    The shorter build carries 334 fewer rows and not one of them was ever fitted or scored, so the
    walk returns the identical 1 096 predictions — which is what makes the restriction free.
    """
    contract = load_contract()
    bars = load_archive_backfill(SNAPSHOT).bars
    whole_window = build_decision_day_rows(bars, arm="a")
    development = build_decision_day_rows(bars, arm="a", last_day=development_last_day(contract))

    assert max(row.day for row in development) == development_last_day(contract)
    assert max(row.day for row in whole_window) > development_last_day(contract)
    assert walk_arm(development, contract, arm="a").predictions == (
        walk_arm(whole_window, contract, arm="a").predictions
    )


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_no_fold_the_evaluation_walks_reaches_the_final_holdout(tmp_path: Path) -> None:
    """ADR-0018: the holdout is spent once, by ticket 06, and not by anything here."""
    contract = load_contract()
    holdout_first = contract.values["splits"]["final_holdout_first_day"]
    rows = build_decision_day_rows(load_archive_backfill(SNAPSHOT).bars, arm="a")
    evaluation = evaluate_arm(
        rows, contract, fresh_trials(tmp_path), arm="a", releases_dir=tmp_path / "releases"
    )
    assert max(prediction.day for prediction in evaluation.predictions) < holdout_first
    for replay in evaluation.replays:
        assert replay.window_last_day < holdout_first


# --- The Primary Comparison -----------------------------------------------------------------------


def prediction(day: date, probability: float, label: int) -> Prediction:
    return Prediction(day=day, probability=probability, label=label)


def test_the_daily_losses_average_to_the_metric_they_are_taken_from() -> None:
    """Ties the per-day primitive to the Selection Metric: if they drift, one of them is wrong."""
    labels = [1, 0, 1, 0]
    probabilities = [0.8, 0.3, 0.4, 0.9]
    daily = [daily_log_loss(label, p) for label, p in zip(labels, probabilities, strict=True)]
    assert sum(daily) / len(daily) == pytest.approx(log_loss(labels, probabilities))


def test_the_paired_difference_is_negative_when_the_treatment_predicts_better() -> None:
    """Sign convention stated by test: negative means the treatment arm has the lower loss."""
    days = [date(2022, 1, 1) + timedelta(days=index) for index in range(3)]
    treatment = tuple(prediction(day, 0.9, 1) for day in days)
    baseline = tuple(prediction(day, 0.6, 1) for day in days)
    comparison = paired_log_loss_difference(
        treatment, baseline, treatment_name="C@24", baseline_name="A", expect_days=3
    )
    assert comparison.days == 3
    assert comparison.mean_difference < 0
    assert comparison.mean_difference == pytest.approx(
        daily_log_loss(1, 0.9) - daily_log_loss(1, 0.6)
    )
    assert comparison.treatment_log_loss == pytest.approx(daily_log_loss(1, 0.9))
    assert comparison.baseline_log_loss == pytest.approx(daily_log_loss(1, 0.6))


def test_only_days_scored_by_both_arms_are_paired() -> None:
    """An inner join, because a day one arm could not score is not evidence about the other."""
    days = [date(2022, 1, 1) + timedelta(days=index) for index in range(4)]
    treatment = tuple(prediction(day, 0.7, 1) for day in days[:3])
    baseline = tuple(prediction(day, 0.6, 1) for day in days[1:])
    comparison = paired_log_loss_difference(
        treatment, baseline, treatment_name="C@24", baseline_name="A", expect_days=2
    )
    assert comparison.days == 2
    assert [entry.day for entry in comparison.daily] == days[1:3]


def test_a_join_that_does_not_reach_the_expected_length_is_refused() -> None:
    """The count is pre-registered at 1 096; a quietly shorter join is a different experiment."""
    days = [date(2022, 1, 1) + timedelta(days=index) for index in range(3)]
    treatment = tuple(prediction(day, 0.7, 1) for day in days[:2])
    baseline = tuple(prediction(day, 0.6, 1) for day in days)
    with pytest.raises(FatalDefect) as caught:
        paired_log_loss_difference(
            treatment, baseline, treatment_name="C@24", baseline_name="A", expect_days=3
        )
    assert caught.value.kind == "paired_days_mismatch"


def test_two_arms_disagreeing_on_a_day_s_label_is_refused() -> None:
    """The Label comes from the price, so it cannot depend on the arm."""
    day = date(2022, 1, 1)
    with pytest.raises(FatalDefect) as caught:
        paired_log_loss_difference(
            (prediction(day, 0.7, 1),),
            (prediction(day, 0.6, 0),),
            treatment_name="C@24",
            baseline_name="A",
            expect_days=1,
        )
    assert caught.value.kind == "paired_label_mismatch"


# --- The Headline Metric is computed and selects nothing ------------------------------------------


def test_a_trial_records_the_selection_metric_and_never_the_headline_metric(
    tmp_path: Path,
) -> None:
    """ADR-0007. If the budget recorded returns, the Headline Metric would be the selection path."""
    trials = fresh_trials(tmp_path)
    evaluation = evaluate_arm(
        series(400, arm="b", lag_hours=24),
        load_contract(),
        trials,
        arm="b",
        lag_hours=24,
        releases_dir=tmp_path / "releases",
        folds=FOLDS,
    )
    recorded = trials.entries()[-1]
    assert recorded.score == evaluation.log_loss
    assert recorded.score != evaluation.headline_return


def test_every_arm_carries_a_headline_metric_under_the_pre_registered_scenario(
    tmp_path: Path,
) -> None:
    """Computed for all of them, so a reader can see the disagreement between the two metrics."""
    for arm, lag in (("a", None), ("b", 24), ("c", 24)):
        evaluation = evaluate_arm(
            series(400, arm=arm, lag_hours=lag),
            load_contract(),
            fresh_trials(tmp_path / arm),
            arm=arm,
            lag_hours=lag,
            releases_dir=tmp_path / arm / "releases",
            folds=FOLDS,
        )
        headline = [replay for replay in evaluation.replays if replay.is_headline]
        assert len(headline) == 1
        assert evaluation.headline_return == headline[0].net_return
        assert len(evaluation.replays) == 3


def test_a_wider_arm_cannot_be_scored_under_a_narrower_arm_s_name() -> None:
    """Arm C's rows carry every one of Arm A's features, so containing them cannot be the test.

    Found in review: the backtest checked that the rows held the claimed arm's features and nothing
    more, which every Arm C row satisfies for Arm A. A Replay Result filed under the wrong arm is
    indistinguishable in a report from one filed correctly.
    """
    rows = series(40, arm="c", lag_hours=24)
    with pytest.raises(FatalDefect) as caught:
        run_backtest(
            rows,
            PreviousDirectionBaseline(),
            PolicyConfig(to_btc_at=0.55, to_usdt_at=0.45),
            BASE_SCENARIO,
            initial=Position(usdt=1000.0, btc=0.0),
            arm="A",
            probabilities={row.day: 0.5 for row in rows},
        )
    assert caught.value.kind == "arm_feature_mismatch"
    assert "wider feature set" in str(caught.value)
