"""The research run: fit one Experiment Arm across the folds, record the Trial, publish the Replay.

Two deliberate boundaries. The Replay Result is scored on the walk-forward predictions, each made by
a fold that could not see the day it scores — re-running the final model over the same days would be
an in-sample number and a flattering one. And the Final Holdout is not touched: the scored window
ends where the Development Period ends.

Nothing here varies with the arm except the feature matrix. The folds, the admissibility rule, the
in-fold preprocessing, the regularisation grid and the inner validation are the same for all three,
because a comparison between arms that also changed the protocol would measure the two together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cryptoguard_core.backtest import run_cost_sensitivity
from cryptoguard_core.contract import ExperimentContract
from cryptoguard_core.dataset import ARMS_NEEDING_NEWS, FEATURES_BY_ARM, DecisionDayRow
from cryptoguard_core.decide import DEFAULT_ASSET, genesis_from, policy_from
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import HEADLINE_SCENARIO
from cryptoguard_core.release import LogisticRelease
from cryptoguard_core.store import PublishedReplay
from cryptoguard_core.training import (
    Fold,
    Prediction,
    WalkForwardResult,
    quarterly_folds,
    walk_forward,
)
from cryptoguard_core.trials import TrialLog


@dataclass(frozen=True, slots=True)
class Evaluation:
    release: LogisticRelease
    replays: tuple[PublishedReplay, ...]
    log_loss: float
    brier: float
    accuracy: float
    scored_days: int
    trial_number: int
    # No defaults on these three. A result able to default its own arm label, or to come back with
    # an empty prediction series, lets every guard in this module pass over a mislabelled run.
    arm: str
    lag_hours: int | None
    # Per scored day, in day order: what the Primary Comparison pairs on. Kept because a paired
    # difference needs the days, not the averages — two arms can share a mean and disagree daily.
    predictions: tuple[Prediction, ...]

    @property
    def headline_return(self) -> float:
        """The Headline Metric under the pre-registered Cost Scenario. It selects nothing."""
        for replay in self.replays:
            if replay.is_headline:
                return replay.net_return
        raise FatalDefect(
            "contract_inconsistent",
            f"no replay was published under the headline Cost Scenario {HEADLINE_SCENARIO!r}",
        )


def _require_rows_match(rows: Sequence[DecisionDayRow], arm: str, lag_hours: int | None) -> None:
    """The rows must already be the arm's rows. Their features and lag are the claim being checked.

    Taking the arm and the lag from the caller alone would let a Trial be recorded against a feature
    matrix it never saw — Arm A's seven features filed as an Arm C result, or a one-hour dataset
    filed under the Primary Lag. Both would leave every guard in the fit passing.
    """
    expected = FEATURES_BY_ARM.get(arm)
    if expected is None:
        raise FatalDefect("unknown_arm", f"arm {arm!r} is not one of {sorted(FEATURES_BY_ARM)}")
    for row in rows:
        if tuple(row.features) != expected:
            raise FatalDefect(
                "arm_feature_mismatch",
                f"arm {arm!r} expects {list(expected)} but {row.day.isoformat()} carries "
                f"{list(row.features)}",
            )
        if row.news_lag_hours != lag_hours:
            raise FatalDefect(
                "arm_lag_mismatch",
                f"the fit claims lag {lag_hours!r} but {row.day.isoformat()} was built under "
                f"{row.news_lag_hours!r}; the lag is what the Trial says it tested",
            )
    if (arm in ARMS_NEEDING_NEWS) != (lag_hours is not None):
        raise FatalDefect(
            "arm_lag_mismatch",
            f"arm {arm!r} reads news: {arm in ARMS_NEEDING_NEWS}, but the lag given is "
            f"{lag_hours!r}; a price-only arm has no lag and a news arm cannot lack one",
        )


def _holdout_first_day(contract: ExperimentContract) -> date:
    first = contract.values["splits"]["final_holdout_first_day"]
    if not isinstance(first, date):
        raise FatalDefect("contract_inconsistent", "splits.final_holdout_first_day must be a date")
    return first


def development_last_day(contract: ExperimentContract) -> date:
    """The last day a research build may include. Anything later belongs to the Final Holdout.

    Here rather than at the call site because the folds are not the only thing that can reach the
    holdout: a build that runs to the end of the Research Window leaves holdout dates in the same
    process as the comparison, one edit away from being scored. The folds stop earlier anyway, so
    narrowing the build moves no number — and it removes the reach (ADR-0018).
    """
    last = contract.values["splits"]["development_last_day"]
    if not isinstance(last, date):
        raise FatalDefect("contract_inconsistent", "splits.development_last_day must be a date")
    holdout_first = _holdout_first_day(contract)
    if last >= holdout_first:
        raise FatalDefect(
            "contract_inconsistent",
            f"the Development Period ends {last}, which is not before the Final Holdout's first "
            f"day {holdout_first}",
        )
    return last


def folds_from(
    contract: ExperimentContract, folds: Sequence[Fold] | None = None
) -> tuple[Fold, ...]:
    """The folds to walk, taken from the contract, with the Final Holdout guard applied.

    From the contract rather than from module defaults: a digest recorded against a result has to
    certify the protocol that was actually walked. The guard lives here rather than at each call
    site so that no path — a fit, a comparison, a diagnostic — reaches the holdout by forgetting it.
    """
    splits = contract.values["splits"]
    holdout_first = _holdout_first_day(contract)
    chosen = (
        tuple(folds)
        if folds is not None
        else quarterly_folds(splits["first_test_quarter"], splits["last_test_quarter"])
    )
    for fold in chosen:
        if fold.test_last_day >= holdout_first:
            raise FatalDefect(
                "contract_inconsistent",
                f"fold ending {fold.test_last_day} reaches the Final Holdout, which starts "
                f"{holdout_first}",
            )
    return chosen


def walk_arm(
    rows: Sequence[DecisionDayRow],
    contract: ExperimentContract,
    *,
    arm: str = "a",
    lag_hours: int | None = None,
    folds: Sequence[Fold] | None = None,
) -> WalkForwardResult:
    """Walk one arm across the folds without spending a Trial.

    Separate from `evaluate_arm` because a Trial is a *look at a new result*, and re-deriving a
    result already on the record is not one. The Primary Comparison needs Arm A's per-day series
    again, and paying a second Trial for the same number would misreport the budget.
    """
    _require_rows_match(rows, arm, lag_hours)
    grid = tuple(contract.values["news"]["lag_grid_hours"])
    if lag_hours is not None and lag_hours not in grid:
        raise FatalDefect(
            "news_lag_not_pre_registered",
            f"lag {lag_hours}h is not in the contract's grid {list(grid)}; a lag chosen at a call "
            "site is not a pre-registered lag",
        )
    return walk_forward(
        rows,
        folds_from(contract, folds),
        features=FEATURES_BY_ARM[arm],
        regularisation=tuple(contract.values["model"]["regularisation_grid"]),
        inner_validation_days=contract.values["splits"]["inner_validation_days"],
    )


def replays_from_walk(
    rows: Sequence[DecisionDayRow],
    walk: WalkForwardResult,
    contract: ExperimentContract,
    *,
    arm: str,
    model_version: str,
    asset: str = DEFAULT_ASSET,
) -> tuple[PublishedReplay, ...]:
    """One Replay Result per Cost Scenario, scored on the walk-forward predictions.

    Shared by the Trial-spending fit and by a re-derivation that spends none, so the Headline Metric
    of an arm already on the record is computed by the same path as a new arm's rather than by a
    second copy of these arguments.
    """
    scored_days = {prediction.day for prediction in walk.predictions}
    scored_rows = tuple(row for row in rows if row.day in scored_days)
    probabilities = {prediction.day: prediction.probability for prediction in walk.predictions}
    sensitivity = run_cost_sensitivity(
        scored_rows,
        PreviousDirectionBaseline(),  # unused: the recorded predictions take precedence
        policy_from(contract),
        initial=genesis_from(contract),
        arm=arm,
        probabilities=probabilities,
    )
    return tuple(
        PublishedReplay(
            asset=asset,
            arm=arm,
            model_version=model_version,
            cost_scenario=name,
            is_headline=name == HEADLINE_SCENARIO,
            window_first_day=scored_rows[0].day,
            window_last_day=scored_rows[-1].day,
            days=result.days,
            trades=result.trades,
            net_return=result.net_return,
            buy_and_hold_return=result.buy_and_hold_return,
            cash_return=result.cash_return,
            max_drawdown=result.max_drawdown,
            turnover=result.turnover,
            time_invested=result.time_invested,
            total_fees=result.total_fees,
            selection_metric=walk.selection_metric,
            selection_score=walk.log_loss,
            contract_digest=contract.digest,
        )
        for name, result in sensitivity.by_scenario.items()
    )


def evaluate_arm(
    rows: Sequence[DecisionDayRow],
    contract: ExperimentContract,
    trials: TrialLog,
    *,
    arm: str = "a",
    lag_hours: int | None = None,
    releases_dir: Path,
    asset: str = DEFAULT_ASSET,
    folds: Sequence[Fold] | None = None,
    corrects: int | None = None,
) -> Evaluation:
    """Fit one arm across the folds, spend one Trial, and return its Replay Result."""
    recorded_arm = arm.upper()
    walk = walk_arm(rows, contract, arm=arm, lag_hours=lag_hours, folds=folds)

    # The release that would be promoted is the last fold's fit: the most history any fold saw
    # without reaching into the Final Holdout.
    last_fit = walk.fits[-1]
    release = LogisticRelease(
        # The Trial number, not the spend: a correction does not increase the spend, so two
        # corrections would otherwise produce the same version and overwrite each other.
        version=f"arm-{arm}-logistic-{trials.next_number()}.0",
        preprocessing=last_fit.preprocessing,
        coefficients=last_fit.coefficients,
        intercept=last_fit.intercept,
        contract_digest=contract.digest,
        arm=recorded_arm,
    )
    replays = replays_from_walk(
        rows, walk, contract, arm=recorded_arm, model_version=release.version, asset=asset
    )
    scored_rows = tuple(
        row for row in rows if row.day in {prediction.day for prediction in walk.predictions}
    )

    # Spent only now. A run that produced nothing should not cost a Trial: a budget that punishes
    # crashes encourages not running things.
    trial = trials.record(
        arm=recorded_arm,
        lag_hours=lag_hours,
        contract_digest=contract.digest,
        score=walk.log_loss,
        corrects=corrects,
    )
    release.save(releases_dir)

    return Evaluation(
        release=release,
        replays=replays,
        log_loss=walk.log_loss,
        brier=walk.brier,
        accuracy=walk.accuracy,
        scored_days=len(scored_rows),
        trial_number=trial.number,
        arm=recorded_arm,
        lag_hours=lag_hours,
        predictions=walk.predictions,
    )
