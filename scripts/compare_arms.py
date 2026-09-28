"""Fit arms B and C across the pre-registered lag grid and answer the Primary Comparison.

**This script spends Trials — one per (arm, lag), six in all.** It refuses to spend a Trial on a
pair the log already holds, and it refuses to start unless the remaining budget covers every pair it
is about to fit, so a run that stops halfway cannot leave the budget somewhere nobody planned.

Arm A is walked again for its per-day series and spends nothing: a Trial is a look at a new result,
and Arm A's result is already on the record as Trial 1. Its log loss is checked against the recorded
one before anything is compared, because a baseline that moved would invalidate every difference.

Nothing here promotes an arm. The Headline Metric is computed for all of them and selects nothing
(ADR-0007); promotion stays a deliberate act.

Run from the repository root:

    uv run python scripts/compare_arms.py --dry-run   # the plan and the budget; fits nothing
    uv run python scripts/compare_arms.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cryptoguard_core.comparison import PairedComparison, paired_log_loss_difference
from cryptoguard_core.contract import ExperimentContract, load_contract
from cryptoguard_core.dataset import DecisionDayRow, build_decision_day_rows
from cryptoguard_core.evaluate import (
    Evaluation,
    development_last_day,
    evaluate_arm,
    replays_from_walk,
    walk_arm,
)
from cryptoguard_core.ingest import Bar, FatalDefect, load_archive_backfill
from cryptoguard_core.metrics import NO_SKILL_LOG_LOSS
from cryptoguard_core.news import ARCHIVE_COVERS_THROUGH_MS, NEWS_CSV, NewsItem, read_news_items
from cryptoguard_core.news_features import HeadlineAvailability, headline_availability
from cryptoguard_core.sentiment import READINGS_CACHE, ReadingCache, news_signal
from cryptoguard_core.store import PublishedReplay
from cryptoguard_core.training import Prediction, WalkForwardResult
from cryptoguard_core.trials import TrialLog

SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")
RELEASES_DIR = Path("artifacts/releases")
TRIALS_PATH = Path("artifacts/trials.json")
RESULT_PATH = Path("artifacts/arm_comparison.json")

BASELINE_ARM = "a"
NEWS_ARMS = ("b", "c")


def _headline(replays: Sequence[PublishedReplay]) -> PublishedReplay:
    for replay in replays:
        if replay.is_headline:
            return replay
    raise FatalDefect(
        "contract_inconsistent", "no replay was published under the headline Cost Scenario"
    )


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm at one lag: what it scored, what it earned, and which Trial paid for it."""

    arm: str
    lag_hours: int | None
    trial_number: int | None
    log_loss: float
    brier: float
    accuracy: float
    scored_days: int
    headline_return: float
    buy_and_hold_return: float
    trades: int
    predictions: tuple[Prediction, ...]

    @property
    def label(self) -> str:
        if self.lag_hours is None:
            return self.arm.upper()
        return f"{self.arm.upper()}@{self.lag_hours}"

    @classmethod
    def from_walk(
        cls,
        arm: str,
        lag_hours: int | None,
        walk: WalkForwardResult,
        replays: Sequence[PublishedReplay],
        *,
        trial_number: int | None,
    ) -> ArmResult:
        headline = _headline(replays)
        return cls(
            arm=arm,
            lag_hours=lag_hours,
            trial_number=trial_number,
            log_loss=walk.log_loss,
            brier=walk.brier,
            accuracy=walk.accuracy,
            scored_days=len(walk.predictions),
            headline_return=headline.net_return,
            buy_and_hold_return=headline.buy_and_hold_return,
            trades=headline.trades,
            predictions=walk.predictions,
        )

    @classmethod
    def from_evaluation(cls, arm: str, lag_hours: int | None, evaluation: Evaluation) -> ArmResult:
        headline = _headline(evaluation.replays)
        return cls(
            arm=arm,
            lag_hours=lag_hours,
            trial_number=evaluation.trial_number,
            log_loss=evaluation.log_loss,
            brier=evaluation.brier,
            accuracy=evaluation.accuracy,
            scored_days=evaluation.scored_days,
            headline_return=evaluation.headline_return,
            buy_and_hold_return=headline.buy_and_hold_return,
            trades=headline.trades,
            predictions=evaluation.predictions,
        )


def recorded_arm_a_score(trials: TrialLog) -> float:
    """The log loss Arm A is on the record with. The comparison's baseline must reproduce it."""
    for entry in reversed(trials.entries()):
        if entry.arm == BASELINE_ARM.upper():
            return entry.score
    raise FatalDefect(
        "unknown_trial",
        "no Arm A Trial is recorded, so there is no baseline to compare against; run "
        "`make evaluate` first",
    )


def already_fitted(trials: TrialLog, arm: str, lag_hours: int) -> int | None:
    """The Trial number that already holds this (arm, lag), if any. Corrections do not count."""
    for entry in trials.entries():
        if entry.corrects is None and entry.arm == arm.upper() and entry.lag_hours == lag_hours:
            return entry.number
    return None


def news_for(arm: str, lag_hours: int, items: Sequence[NewsItem]) -> HeadlineAvailability:
    """Headline availability under one lag, carrying Sentiment Scores only where an arm reads them.

    Arm B is handed instants alone. Handing it scores would build a column it does not declare, and
    `build_decision_day_rows` refuses exactly that.
    """
    if arm == "c":
        return news_signal(
            items,
            lag_hours=lag_hours,
            covers_through_ms=ARCHIVE_COVERS_THROUGH_MS,
            cache=ReadingCache(READINGS_CACHE),
        )
    return headline_availability(
        items, lag_hours=lag_hours, covers_through_ms=ARCHIVE_COVERS_THROUGH_MS
    )


def build_rows(
    arm: str,
    lag_hours: int | None,
    bars: Sequence[Bar],
    items: Sequence[NewsItem],
    *,
    last_day: date,
) -> tuple[DecisionDayRow, ...]:
    """Rows for one arm, stopping at `last_day` — the Development Period's end, never later.

    The folds already stop at 2024-12-31, so rows past it were never fitted or scored. Building them
    anyway left Final Holdout dates in the same process as the comparison, one edit away from being
    scored. Verified before the restriction was added: the walk over the shorter build returns the
    identical 1 096 predictions and the identical log loss, so this narrows what is touched without
    moving a single number (ADR-0018).
    """
    if lag_hours is None:
        return build_decision_day_rows(bars, arm=arm, last_day=last_day)
    return build_decision_day_rows(
        bars, arm=arm, last_day=last_day, news=news_for(arm, lag_hours, items)
    )


def contract_value(contract: ExperimentContract, section: str, key: str) -> object:
    """One contract value, or a named refusal. A `KeyError` traceback is not a refusal."""
    try:
        return contract.values[section][key]
    except KeyError:
        raise FatalDefect(
            "contract_inconsistent",
            f"the contract has no {section}.{key}; this script reads it and cannot proceed without "
            "it",
        ) from None


def promoted_version(releases_dir: Path) -> str:
    """The version the CURRENT pointer names, or a named refusal saying how to create it."""
    pointer = releases_dir / "CURRENT"
    if not pointer.is_file():
        raise FatalDefect(
            "unknown_release",
            f"{pointer} does not exist, so Arm A's Replay cannot name the model it came from; run "
            "`make evaluate` first",
        )
    return pointer.read_text(encoding="utf-8").strip()


def primary_pair(contract: ExperimentContract) -> tuple[str, int]:
    """The arms the contract pre-registers as the Primary Comparison, and the Primary Lag."""
    stated = str(contract_value(contract, "evaluation", "primary_comparison"))
    if stated != "arm_c_vs_arm_a":
        raise FatalDefect(
            "contract_inconsistent",
            f"evaluation.primary_comparison is {stated!r}; this script answers arm_c_vs_arm_a",
        )
    return "c", int(str(contract_value(contract, "news", "primary_lag_hours")))


def print_table(results: Sequence[ArmResult]) -> None:
    print(
        f"\narm  lag  trial   log loss  vs {NO_SKILL_LOG_LOSS:.6f}     brier  accuracy  "
        "headline  trades"
    )
    for result in results:
        trial = "-" if result.trial_number is None else str(result.trial_number)
        lag = "-" if result.lag_hours is None else str(result.lag_hours)
        print(
            f"{result.arm.upper():<4} {lag:>3}  {trial:>5}   {result.log_loss:.6f}  "
            f"{result.log_loss - NO_SKILL_LOG_LOSS:+.6f}  {result.brier:.6f}  "
            f"{result.accuracy:.4f}  {100 * result.headline_return:>7.1f}%  {result.trades:>6}"
        )
    print("\nThe Headline Metric above selects nothing (ADR-0007). No arm is promoted here.")


def write_result(
    path: Path,
    *,
    contract: ExperimentContract,
    expect_days: int,
    primary_label: str,
    baseline_label: str,
    results: Sequence[ArmResult],
    comparisons: dict[str, PairedComparison],
) -> None:
    payload = {
        "contract_digest": contract.digest,
        "no_skill_log_loss": NO_SKILL_LOG_LOSS,
        "scored_development_days": expect_days,
        "primary_comparison": f"{primary_label} vs {baseline_label}",
        "arms": [
            {
                "arm": result.arm.upper(),
                "lag_hours": result.lag_hours,
                "trial_number": result.trial_number,
                "log_loss": result.log_loss,
                "brier": result.brier,
                "accuracy": result.accuracy,
                "scored_days": result.scored_days,
                "headline_return": result.headline_return,
                "buy_and_hold_return": result.buy_and_hold_return,
                "trades": result.trades,
            }
            for result in results
        ],
        "paired_differences": [
            {
                "treatment": comparison.treatment_name,
                "baseline": comparison.baseline_name,
                "days": comparison.days,
                "mean_difference": comparison.mean_difference,
                "treatment_log_loss": comparison.treatment_log_loss,
                "baseline_log_loss": comparison.baseline_log_loss,
                "is_primary": label == primary_label,
            }
            for label, comparison in comparisons.items()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the budget, fit nothing"
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--trials", type=Path, default=TRIALS_PATH)
    parser.add_argument("--releases", type=Path, default=RELEASES_DIR)
    parser.add_argument("--out", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args()

    contract = load_contract()
    trials = TrialLog(arguments.trials, budget=int(contract.values["trials"]["budget"]))
    grid = tuple(int(lag) for lag in contract.values["news"]["lag_grid_hours"])
    pairs = [(arm, lag) for arm in NEWS_ARMS for lag in grid]

    print(f"contract digest:         {contract.digest[:12]}")
    print(f"lag grid:                {list(grid)}")
    print(f"Trials spent / budget:   {trials.spent()} / {trials.spent() + trials.remaining()}")
    for arm, lag in pairs:
        number = already_fitted(trials, arm, lag)
        if number is not None:
            print(f"  {arm.upper()}@{lag} is already Trial {number}; it will not be refitted")
    outstanding = [(arm, lag) for arm, lag in pairs if already_fitted(trials, arm, lag) is None]
    planned = [f"{arm.upper()}@{lag}" for arm, lag in outstanding]
    print(f"to fit now:              {planned}")

    if len(outstanding) > trials.remaining():
        raise FatalDefect(
            "trial_budget_exhausted",
            f"{len(outstanding)} fits are outstanding and {trials.remaining()} Trials remain; "
            "raising the budget is a recorded decision, not an accident",
        )
    if outstanding and len(outstanding) != len(pairs):
        # A run that fits only some of the grid cannot produce the comparison, and overwriting the
        # stored result with a subset would publish a partial answer under the full answer's name.
        # The earlier run's artefact is the complete one and is left alone.
        raise FatalDefect(
            "trial_budget_exhausted",
            f"{len(pairs) - len(outstanding)} of {len(pairs)} pairs are already spent, so this run "
            "can only fit part of the grid and cannot write a complete comparison. The artefact "
            "from the run that spent them stands; re-deriving it would mean spending Trials again",
        )
    if arguments.dry_run:
        print("dry run: nothing was fitted and no Trial was spent")
        return
    if not outstanding:
        print("every pair is already spent; nothing to do")
        return

    last_day = development_last_day(contract)
    print(f"rows built through:      {last_day} (the Final Holdout is not built)")
    bars = load_archive_backfill(arguments.snapshot).bars
    items = list(read_news_items(NEWS_CSV))

    # The baseline, re-derived and checked against the record. No Trial: this number is Trial 1.
    baseline_rows = build_rows(BASELINE_ARM, None, bars, items, last_day=last_day)
    baseline_walk = walk_arm(baseline_rows, contract, arm=BASELINE_ARM)
    on_record = recorded_arm_a_score(trials)
    if abs(baseline_walk.log_loss - on_record) > 1e-12:
        raise FatalDefect(
            "contract_mismatch",
            f"Arm A now scores {baseline_walk.log_loss!r} but is on the record at {on_record!r}; "
            "the baseline moved, so every difference computed against it would be meaningless",
        )
    promoted = promoted_version(arguments.releases)
    baseline = ArmResult.from_walk(
        BASELINE_ARM,
        None,
        baseline_walk,
        replays_from_walk(
            baseline_rows,
            baseline_walk,
            contract,
            arm=BASELINE_ARM.upper(),
            model_version=promoted,
        ),
        trial_number=None,
    )
    print(
        f"\nArm A reproduced:        log loss {baseline.log_loss:.6f} over "
        f"{baseline.scored_days} days, matching the record; no Trial spent"
    )

    results = [baseline]
    for arm, lag in pairs:
        if already_fitted(trials, arm, lag) is not None:
            continue
        rows = build_rows(arm, lag, bars, items, last_day=last_day)
        evaluation = evaluate_arm(
            rows, contract, trials, arm=arm, lag_hours=lag, releases_dir=arguments.releases
        )
        results.append(ArmResult.from_evaluation(arm, lag, evaluation))
        print(
            f"Trial {evaluation.trial_number}: {arm.upper()}@{lag:<3} "
            f"log loss {evaluation.log_loss:.6f}  headline return "
            f"{100 * evaluation.headline_return:+.1f}%"
        )

    expect_days = int(str(contract_value(contract, "splits", "scored_development_days")))
    treatment_arm, primary_lag = primary_pair(contract)
    primary_label = f"{treatment_arm.upper()}@{primary_lag}"
    comparisons = {
        result.label: paired_log_loss_difference(
            result.predictions,
            baseline.predictions,
            treatment_name=result.label,
            baseline_name=baseline.label,
            expect_days=expect_days,
        )
        for result in results
        if result.lag_hours is not None
    }

    print(f"\nPrimary Comparison (pre-registered): {primary_label} against {baseline.label}")
    primary = comparisons.get(primary_label)
    if primary is None:
        print("  not computed here: its Trial was spent in an earlier run")
    else:
        print(f"  paired days:           {primary.days}")
        print(f"  mean daily difference: {primary.mean_difference:+.6f}")
        print(f"  {primary_label} log loss:  {primary.treatment_log_loss:.6f}")
        print(f"  {baseline.label} log loss:     {primary.baseline_log_loss:.6f}")
        better = "lower" if primary.treatment_predicts_better else "no lower"
        print(f"  the treatment arm's loss is {better} than the baseline's")

    print_table(results)
    write_result(
        arguments.out,
        contract=contract,
        expect_days=expect_days,
        primary_label=primary_label,
        baseline_label=baseline.label,
        results=results,
        comparisons=comparisons,
    )
    print(f"\nwritten: {arguments.out}")


if __name__ == "__main__":
    main()
