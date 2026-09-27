"""The research run: fit Arm A across the folds, record the Trial, and publish the Replay Result.

Two deliberate boundaries. The Replay Result is scored on the walk-forward predictions, each made by
a fold that could not see the day it scores — re-running the final model over the same days would be
an in-sample number and a flattering one. And the Final Holdout is not touched: the scored window
ends where the Development Period ends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cryptoguard_core.backtest import run_cost_sensitivity
from cryptoguard_core.contract import ExperimentContract
from cryptoguard_core.dataset import ARM_A_FEATURES, DecisionDayRow
from cryptoguard_core.decide import DEFAULT_ASSET, genesis_from, policy_from
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import HEADLINE_SCENARIO
from cryptoguard_core.release import LogisticRelease
from cryptoguard_core.store import PublishedReplay
from cryptoguard_core.training import Fold, quarterly_folds, walk_forward
from cryptoguard_core.trials import TrialLog

ARM = "A"


@dataclass(frozen=True, slots=True)
class Evaluation:
    release: LogisticRelease
    replays: tuple[PublishedReplay, ...]
    log_loss: float
    brier: float
    accuracy: float
    scored_days: int
    trial_number: int


def _holdout_first_day(contract: ExperimentContract) -> date:
    first = contract.values["splits"]["final_holdout_first_day"]
    if not isinstance(first, date):
        raise FatalDefect("contract_inconsistent", "splits.final_holdout_first_day must be a date")
    return first


def evaluate_arm_a(
    rows: Sequence[DecisionDayRow],
    contract: ExperimentContract,
    trials: TrialLog,
    *,
    releases_dir: Path,
    asset: str = DEFAULT_ASSET,
    folds: Sequence[Fold] | None = None,
    corrects: int | None = None,
) -> Evaluation:
    splits = contract.values["splits"]
    holdout_first = _holdout_first_day(contract)
    # From the contract, not from module defaults: a digest recorded against a result has to
    # certify the protocol that was actually walked.
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

    walk = walk_forward(
        rows,
        chosen,
        features=ARM_A_FEATURES,
        regularisation=tuple(contract.values["model"]["regularisation_grid"]),
        inner_validation_days=splits["inner_validation_days"],
    )

    # The release that would be promoted is the last fold's fit: the most history any fold saw
    # without reaching into the Final Holdout.
    last_fit = walk.fits[-1]
    release = LogisticRelease(
        # The Trial number, not the spend: a correction does not increase the spend, so two
        # corrections would otherwise produce the same version and overwrite each other.
        version=f"arm-a-logistic-{trials.next_number()}.0",
        preprocessing=last_fit.preprocessing,
        coefficients=last_fit.coefficients,
        intercept=last_fit.intercept,
        contract_digest=contract.digest,
        arm=ARM,
    )
    scored_days = {prediction.day for prediction in walk.predictions}
    scored_rows = tuple(row for row in rows if row.day in scored_days)
    probabilities = {prediction.day: prediction.probability for prediction in walk.predictions}
    sensitivity = run_cost_sensitivity(
        scored_rows,
        PreviousDirectionBaseline(),  # unused: the recorded predictions take precedence
        policy_from(contract),
        initial=genesis_from(contract),
        arm=ARM,
        probabilities=probabilities,
    )

    replays = tuple(
        PublishedReplay(
            asset=asset,
            arm=ARM,
            model_version=release.version,
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

    # Spent only now. A run that produced nothing should not cost a Trial: a budget that punishes
    # crashes encourages not running things.
    trial = trials.record(
        arm=ARM,
        lag_hours=None,
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
    )
