"""The Final Holdout: one frozen configuration, one evaluation, one number, and no second chance.

This is the only measurement in the project that cannot be repeated, so the rule is a mechanism
rather than a discipline. The refusal is checked **before** anything is computed: a second
invocation must not be able to produce a number and then decline to print it, because a number
that exists has already been seen.

**What is frozen, and when.** Arm, lag, Policy thresholds, regularisation procedure and Cost
Scenario are written into the Experiment Contract before the run, and that contract's digest is
recorded in the marker. A configuration chosen after seeing the holdout is not a holdout result.

**It spends no Trial.** A Trial is an evaluation whose result could have influenced a choice. Here
the configuration is frozen beforehand and ADR-0018 forbids the number influencing anything
afterwards: a defect found later is disclosed, or the claim withdrawn, never re-run. The reasoning
is recorded in the marker rather than left to a reader to reconstruct.

**The holdout fold does not pass through `evaluate.folds_from`.** That guard exists so no
development path can reach 2025, and it refuses exactly the fold this module needs. Bypassing it
is deliberate and lives here, in the one module named for the thing it may touch, rather than
being arranged by relaxing the guard everything else depends on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from cryptoguard_core.backtest import BacktestResult
from cryptoguard_core.contract import ExperimentContract, contract_value
from cryptoguard_core.dataset import FEATURES_BY_ARM, DecisionDayRow
from cryptoguard_core.evaluate import sensitivity_from_walk
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.policy import HEADLINE_SCENARIO
from cryptoguard_core.training import Fold, WalkForwardResult, walk_forward

MARKER_PATH = Path(__file__).resolve().parents[3] / "artifacts/final_holdout.json"


@dataclass(frozen=True, slots=True)
class FrozenConfiguration:
    """Every choice that was fixed before the single evaluation ran."""

    arm: str
    lag_hours: int | None
    cost_scenario: str
    to_btc_at: float
    to_usdt_at: float
    regularisation_grid: tuple[float, ...]
    regularisation_selected: str
    contract_digest: str


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """The one row. Never averaged with a development result and never spliced into a Replay."""

    first_day: date
    last_day: date
    scored_days: int
    log_loss: float
    brier: float
    accuracy: float
    net_return: float
    buy_and_hold_return: float
    trades: int
    max_drawdown: float
    turnover: float
    time_invested: float


@dataclass(frozen=True, slots=True)
class HoldoutSpend:
    """The marker: what was spent, on what configuration, and why it is not a Trial."""

    spent_at: str
    evaluations_allowed: int
    configuration: FrozenConfiguration
    result: HoldoutResult
    spends_a_trial: bool
    trial_reasoning: str


def read_spend(path: Path = MARKER_PATH) -> HoldoutSpend | None:
    """The recorded spend, or None if the holdout is untouched."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        configuration = raw["configuration"]
        result = raw["result"]
        return _spend_from(raw, configuration, result)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        # The guard's own read path. A truncated or hand-edited marker handing a traceback to
        # whoever runs the command is the one failure that must not look like an ordinary crash:
        # the question "is the holdout spent?" would be left unanswered by a stack trace.
        raise FatalDefect(
            "final_holdout_marker_corrupt",
            f"{path} exists but cannot be read as a spend record ({error!r}). Until it is readable "
            "the holdout must be treated as spent, because an unreadable marker is not evidence "
            "that it is not",
        ) from error


def _spend_from(
    raw: Mapping[str, Any], configuration: Mapping[str, Any], result: Mapping[str, Any]
) -> HoldoutSpend:
    return HoldoutSpend(
        spent_at=raw["spent_at"],
        evaluations_allowed=raw["evaluations_allowed"],
        configuration=FrozenConfiguration(
            arm=configuration["arm"],
            lag_hours=configuration["lag_hours"],
            cost_scenario=configuration["cost_scenario"],
            to_btc_at=configuration["to_btc_at"],
            to_usdt_at=configuration["to_usdt_at"],
            regularisation_grid=tuple(configuration["regularisation_grid"]),
            regularisation_selected=configuration["regularisation_selected"],
            contract_digest=configuration["contract_digest"],
        ),
        result=HoldoutResult(
            first_day=date.fromisoformat(result["first_day"]),
            last_day=date.fromisoformat(result["last_day"]),
            scored_days=result["scored_days"],
            log_loss=result["log_loss"],
            brier=result["brier"],
            accuracy=result["accuracy"],
            net_return=result["net_return"],
            buy_and_hold_return=result["buy_and_hold_return"],
            trades=result["trades"],
            max_drawdown=result["max_drawdown"],
            turnover=result["turnover"],
            time_invested=result["time_invested"],
        ),
        spends_a_trial=raw["spends_a_trial"],
        trial_reasoning=raw["trial_reasoning"],
    )


def refuse_if_spent(path: Path = MARKER_PATH) -> None:
    """Refuse if the one permitted evaluation has already happened.

    Called before the data is even loaded. The temptation this guards against is not dishonesty but
    convenience — a rerun "just to check" after a fix — and by then the number has been seen.
    """
    spent = read_spend(path)
    if spent is None:
        return
    raise FatalDefect(
        "final_holdout_spent",
        f"the Final Holdout was evaluated on {spent.spent_at} under contract "
        f"{spent.configuration.contract_digest[:12]} and only "
        f"{spent.evaluations_allowed} evaluation is permitted (ADR-0018). A defect found later is "
        f"disclosed or the claim is withdrawn; it does not license a second run. The marker is "
        f"{path}",
    )


def write_spend(path: Path, spend: HoldoutSpend) -> None:
    """Record the spend. Refuses to overwrite: the marker is the only evidence it is gone."""
    refuse_if_spent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(spend)
    payload["configuration"]["regularisation_grid"] = list(spend.configuration.regularisation_grid)
    payload["result"]["first_day"] = spend.result.first_day.isoformat()
    payload["result"]["last_day"] = spend.result.last_day.isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Frozen and mirrored in `config/experiment.yaml`; the contract refuses to load if the two disagree.
HOLDOUT_ARM = "a"
HOLDOUT_LAG_HOURS: int | None = None
HOLDOUT_COST_SCENARIO = HEADLINE_SCENARIO
HOLDOUT_SPENDS_A_TRIAL = False
# The day the configuration above was written into the contract. Pinned like the rest of it: a date
# nothing checks is a date that can be edited afterwards to look earlier than the run.
HOLDOUT_FROZEN_ON = date(2026, 9, 28)
# Checked rather than merely recorded: a run taking this from the contract could be told it has
# two evaluations, and a marker stating two would be quoting itself as the authority.
EVALUATIONS_ALLOWED = 1

TRIAL_REASONING = (
    "Confirmatory, not selective: the arm, the lag, the Policy thresholds, the regularisation "
    "procedure and the Cost Scenario were all frozen in the Experiment Contract before this ran, "
    "and ADR-0018 forbids the number influencing anything afterwards — a defect found later is "
    "disclosed or the claim is withdrawn, never re-run. A Trial is an evaluation whose result "
    "could have changed a choice; this one has none left to change, so it spends none."
)


def holdout_fold(contract: ExperimentContract) -> Fold:
    """The single fold: score the Final Holdout, having trained on everything admissible before it.

    Built here and not through `evaluate.folds_from`, which refuses any fold reaching 2025 — the
    guard that keeps every development path out of this period. The boundary is the admissibility
    rule, not a date filter: a Decision Day whose Label resolves after the holdout opens is excluded
    because its outcome was not knowable at the Refit Cutoff, which is a day stricter than a date
    comparison.
    """
    first = contract_value(contract, "splits", "final_holdout_first_day")
    last = contract_value(contract, "splits", "final_holdout_last_day")
    if not isinstance(first, date) or not isinstance(last, date):
        raise FatalDefect(
            "contract_inconsistent", "the Final Holdout's first and last day must both be dates"
        )
    return Fold(test_first_day=first, test_last_day=last)


def frozen_configuration(contract: ExperimentContract) -> FrozenConfiguration:
    """The frozen choices, read from the contract and refused if they disagree with the code."""
    stated = contract.values.get("final_holdout")
    if stated is None:
        raise FatalDefect(
            "final_holdout_not_frozen",
            "the contract has no final_holdout section, so nothing is frozen and there is no "
            "configuration to evaluate. This is the one section `contract._check` does not "
            "validate, which is why the refusal lives here",
        )
    pinned = {
        "arm": HOLDOUT_ARM,
        "lag_hours": HOLDOUT_LAG_HOURS,
        "cost_scenario": HOLDOUT_COST_SCENARIO,
        "spends_a_trial": HOLDOUT_SPENDS_A_TRIAL,
        "frozen_on": HOLDOUT_FROZEN_ON,
    }
    for key, expected in pinned.items():
        if stated.get(key) != expected:
            raise FatalDefect(
                "final_holdout_not_frozen",
                f"final_holdout.{key} is {stated.get(key)!r} but the code states {expected!r}; "
                "configuration has to be frozen before the run, and these two are what freeze it",
            )
    allowed = contract_value(contract, "splits", "final_holdout_evaluations_allowed")
    if allowed != EVALUATIONS_ALLOWED:
        raise FatalDefect(
            "final_holdout_not_frozen",
            f"splits.final_holdout_evaluations_allowed is {allowed!r} but the code permits "
            f"{EVALUATIONS_ALLOWED}; the number of permitted evaluations is not a value a run may "
            "take from the file it is about to write",
        )
    return FrozenConfiguration(
        arm=HOLDOUT_ARM.upper(),
        lag_hours=HOLDOUT_LAG_HOURS,
        cost_scenario=HOLDOUT_COST_SCENARIO,
        to_btc_at=float(contract_value(contract, "policy", "to_btc_at")),
        to_usdt_at=float(contract_value(contract, "policy", "to_usdt_at")),
        regularisation_grid=tuple(
            float(value) for value in contract_value(contract, "model", "regularisation_grid")
        ),
        regularisation_selected=str(contract_value(contract, "model", "regularisation_selected")),
        contract_digest=contract.digest,
    )


def rederive_holdout(
    rows: Sequence[DecisionDayRow], contract: ExperimentContract
) -> tuple[WalkForwardResult, BacktestResult]:
    """Compute the frozen configuration's per-day series. Writes nothing and records nothing.

    Deterministic: the same rows and the same frozen configuration give the same numbers, which is
    why the acceptance test can exercise the success path at all. Separating this from the spend is
    what lets a *description* of the recorded result — a Bootstrap Interval around it — be computed
    without a second evaluation, on the same reasoning that let Arm A be re-walked for the Primary
    Comparison without a second Trial. A caller describing the recorded number must check that what
    it re-derived equals what the marker holds; re-derivation returning something else describes a
    different result, not the recorded one.
    """
    configuration = frozen_configuration(contract)
    walk = walk_forward(
        rows,
        (holdout_fold(contract),),
        features=FEATURES_BY_ARM[HOLDOUT_ARM],
        regularisation=configuration.regularisation_grid,
        inner_validation_days=contract_value(contract, "splits", "inner_validation_days"),
    )
    # Through `sensitivity_from_walk`, which is the one place a Replay Result is produced. A second
    # copy of its arguments here could drift from the simulation every other number came from, and
    # the frozen Cost Scenario is picked out of the three it runs rather than passed in twice.
    sensitivity = sensitivity_from_walk(rows, walk, contract, arm=configuration.arm)
    return walk, sensitivity.by_scenario[configuration.cost_scenario]


def evaluate_final_holdout(
    rows: Sequence[DecisionDayRow],
    contract: ExperimentContract,
    *,
    marker_path: Path = MARKER_PATH,
) -> HoldoutSpend:
    """Spend the one permitted evaluation, and record that it is gone.

    The refusal is first, before the configuration is read and before anything is fitted. The marker
    is written last, from the same values that are returned, so a marker cannot describe a run that
    did not finish.
    """
    refuse_if_spent(marker_path)
    configuration = frozen_configuration(contract)
    fold = holdout_fold(contract)
    splits = contract.values["splits"]
    walk, backtest = rederive_holdout(rows, contract)

    spend = HoldoutSpend(
        spent_at=datetime.now(tz=UTC).isoformat(),
        evaluations_allowed=int(splits["final_holdout_evaluations_allowed"]),
        configuration=configuration,
        result=HoldoutResult(
            first_day=fold.test_first_day,
            last_day=fold.test_last_day,
            scored_days=len(walk.predictions),
            log_loss=walk.log_loss,
            brier=walk.brier,
            accuracy=walk.accuracy,
            net_return=backtest.net_return,
            buy_and_hold_return=backtest.buy_and_hold_return,
            trades=backtest.trades,
            max_drawdown=backtest.max_drawdown,
            turnover=backtest.turnover,
            time_invested=backtest.time_invested,
        ),
        spends_a_trial=HOLDOUT_SPENDS_A_TRIAL,
        trial_reasoning=TRIAL_REASONING,
    )
    write_spend(marker_path, spend)
    return spend
