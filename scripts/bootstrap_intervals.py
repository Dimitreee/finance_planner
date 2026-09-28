"""Bootstrap Intervals for the Primary Comparison and for the Headline Metric.

**No Trial is spent.** Both arms are re-walked to recover their per-day series, which re-derives
results already on the record; the bootstrap then estimates uncertainty around them and ranks
nothing (ADR-0017).

The intervals have deliberately different standing, and every row says which it carries:

- the paired daily log-loss difference between Arm C at the Primary Lag and Arm A is **evidence**,
  and may inform a conclusion about forecasting skill (ADR-0017);
- net cumulative return after costs against buy-and-hold is **description**. It exists so a return
  quoted alone cannot be read as repeatable, and nothing here ranks by it (ADR-0007).

The return interval is reported under **all three Cost Scenarios**, because ADR-0009 says all three
are always reported and the span between the extremes is the point of having them.

An interval spanning zero is reported as Inconclusive — never as a small effect.

Run from the repository root:

    uv run python scripts/bootstrap_intervals.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cryptoguard_core.backtest import daily_log_returns, excess_return
from cryptoguard_core.bootstrap import (
    BLOCK_DAYS,
    CONFIDENCE,
    RESAMPLES,
    SEED,
    SENSITIVITY_BLOCK_DAYS,
    STABILITY_SEEDS,
    BootstrapInterval,
    Verdict,
    moving_block_bootstrap,
)
from cryptoguard_core.comparison import paired_log_loss_difference
from cryptoguard_core.contract import contract_value, load_contract
from cryptoguard_core.evaluate import (
    arm_rows,
    development_last_day,
    sensitivity_from_walk,
    walk_arm,
)
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.news import NEWS_CSV, read_news_items
from cryptoguard_core.policy import HEADLINE_SCENARIO

SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")
RELEASES_DIR = Path("artifacts/releases")
RESULT_PATH = Path("artifacts/bootstrap_intervals.json")
BASELINE_ARM = "a"

Standing = Literal["evidence", "description"]
EVIDENCE: Standing = "evidence"
DESCRIPTION: Standing = "description"

EVIDENCE_CAPTION = (
    "evidence: this interval may inform a conclusion about forecasting skill (ADR-0017)"
)
DESCRIPTION_CAPTION = "description: this interval informs nothing and selects nothing (ADR-0007)"


@dataclass(frozen=True, slots=True)
class Row:
    """One block length's interval for one quantity, and its verdict at every stability seed.

    The verdicts are per block length, not per quantity. Checking only the reported length was a
    real defect found in review: at 40-day blocks one of the four stability seeds puts the lower
    bound at -2.6e-06 and turns Separated into Inconclusive, so a check reading only 20-day blocks
    reported the conclusion as seed-invariant when it is not.
    """

    block_days: int
    role: Literal["reported", "sensitivity"]
    interval: BootstrapInterval
    verdicts_by_seed: dict[int, Verdict]

    @property
    def verdict_stable(self) -> bool:
        return set(self.verdicts_by_seed.values()) == {self.interval.verdict}


@dataclass(frozen=True, slots=True)
class Reported:
    """One quantity's intervals, its standing, and the verdict at the other seeds."""

    name: str
    standing: Standing
    caption: str
    units: Literal["natural", "percent"]
    cost_scenario: str | None
    reported: Row
    sensitivity: tuple[Row, ...]

    @property
    def rows(self) -> tuple[Row, ...]:
        return (self.reported, *self.sensitivity)

    @property
    def unstable_blocks(self) -> tuple[int, ...]:
        """Block lengths whose verdict depends on which seed was drawn."""
        return tuple(row.block_days for row in self.rows if not row.verdict_stable)


def promoted_arm(releases_dir: Path) -> str:
    """The arm the CURRENT pointer actually names, refusing if it is not the one this script covers.

    ADR-0017 asks for the return interval of "the promoted arm". A hard-coded "A" would keep saying
    Arm A after a promotion, so the promoted arm is read off the pointer and checked.
    """
    pointer = releases_dir / "CURRENT"
    if not pointer.is_file():
        raise FatalDefect(
            "unknown_release",
            f"{pointer} does not exist, so no arm is promoted and there is no Headline Metric to "
            "describe; run `make evaluate` first",
        )
    version = pointer.read_text(encoding="utf-8").strip()
    arm = version.removeprefix("arm-").split("-", 1)[0].upper()
    if arm != BASELINE_ARM.upper():
        raise FatalDefect(
            "unknown_release",
            f"the promoted release {version!r} is arm {arm}, but this script bootstraps the return "
            f"of arm {BASELINE_ARM.upper()}; the two must be the same arm",
        )
    return arm


def render(interval: BootstrapInterval, units: str) -> str:
    scale = 100.0 if units == "percent" else 1.0
    suffix = "%" if units == "percent" else ""
    return (
        f"{scale * interval.estimate:+.4f}{suffix}  "
        f"[{scale * interval.lower:+.4f}{suffix}, {scale * interval.upper:+.4f}{suffix}]  "
        f"{interval.verdict}"
    )


def measure[Observation](
    name: str,
    standing: Standing,
    caption: str,
    units: Literal["natural", "percent"],
    series: Sequence[Observation],
    statistic: Callable[[Sequence[Observation]], float],
    *,
    cost_scenario: str | None = None,
    resamples: int,
) -> Reported:
    """The reported interval, the two sensitivity rows, and the verdict at the other seeds."""
    print(name)
    print(f"  {caption}")
    rows: list[Row] = []
    for block_days in (BLOCK_DAYS, *SENSITIVITY_BLOCK_DAYS):
        interval = moving_block_bootstrap(
            series, statistic, block_days=block_days, resamples=resamples, seed=SEED
        )
        role: Literal["reported", "sensitivity"] = (
            "reported" if block_days == BLOCK_DAYS else "sensitivity"
        )
        verdicts = {
            seed: moving_block_bootstrap(
                series, statistic, block_days=block_days, resamples=resamples, seed=seed
            ).verdict
            for seed in STABILITY_SEEDS
        }
        row = Row(block_days=block_days, role=role, interval=interval, verdicts_by_seed=verdicts)
        state = "seed-invariant" if row.verdict_stable else f"SEED-DEPENDENT {verdicts}"
        print(f"  {block_days:>3}-day blocks ({role:<11}) {render(interval, units)}  {state}")
        rows.append(row)
    print()
    return Reported(
        name=name,
        standing=standing,
        caption=caption,
        units=units,
        cost_scenario=cost_scenario,
        reported=rows[0],
        sensitivity=tuple(rows[1:]),
    )


def as_payload(entry: Reported) -> dict[str, object]:
    return {
        "name": entry.name,
        "standing": entry.standing,
        "caption": entry.caption,
        "units": entry.units,
        "cost_scenario": entry.cost_scenario,
        "intervals": [
            {
                "block_days": row.block_days,
                "role": row.role,
                "seed": row.interval.seed,
                "estimate": row.interval.estimate,
                "lower": row.interval.lower,
                "upper": row.interval.upper,
                "verdict": row.interval.verdict,
                "verdict_stable_across_seeds": row.verdict_stable,
                "verdicts_by_stability_seed": {
                    str(seed): verdict for seed, verdict in row.verdicts_by_seed.items()
                },
            }
            for row in entry.rows
        ],
        "reported_verdict_stable_across_seeds": entry.reported.verdict_stable,
        "seed_dependent_block_days": list(entry.unstable_blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--out", type=Path, default=RESULT_PATH)
    parser.add_argument("--releases", type=Path, default=RELEASES_DIR)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    arguments = parser.parse_args()

    contract = load_contract()
    last_day = development_last_day(contract)
    primary_lag = int(contract_value(contract, "news", "primary_lag_hours"))
    expect_days = int(contract_value(contract, "splits", "scored_development_days"))
    promoted = promoted_arm(arguments.releases)
    bars = load_archive_backfill(arguments.snapshot).bars
    items = list(read_news_items(NEWS_CSV))

    print(f"contract digest:  {contract.digest[:12]}")
    print(f"resamples:        {arguments.resamples}   seed: {SEED}   confidence: {CONFIDENCE}")
    print(f"rows built through: {last_day}   promoted arm: {promoted}\n")

    baseline_rows = arm_rows(BASELINE_ARM, None, bars, items, last_day=last_day)
    baseline_walk = walk_arm(baseline_rows, contract, arm=BASELINE_ARM)
    treatment_rows = arm_rows("c", primary_lag, bars, items, last_day=last_day)
    treatment_walk = walk_arm(treatment_rows, contract, arm="c", lag_hours=primary_lag)
    paired = paired_log_loss_difference(
        treatment_walk.predictions,
        baseline_walk.predictions,
        treatment_name=f"C@{primary_lag}",
        baseline_name="A",
        expect_days=expect_days,
    )
    # The same simulation the published Replay Result came from, not a second copy of its arguments.
    sensitivity = sensitivity_from_walk(
        baseline_rows, baseline_walk, contract, arm=BASELINE_ARM.upper()
    )

    reported = [
        measure(
            f"paired daily log-loss difference, C@{primary_lag} minus A",
            EVIDENCE,
            EVIDENCE_CAPTION,
            "natural",
            [entry.difference for entry in paired.daily],
            statistics.fmean,
            resamples=arguments.resamples,
        )
    ]
    for scenario, result in sensitivity.by_scenario.items():
        headline = " (headline)" if scenario == HEADLINE_SCENARIO else ""
        reported.append(
            measure(
                f"net cumulative return after costs minus buy-and-hold, arm {promoted}, "
                f"{scenario} costs{headline}",
                DESCRIPTION,
                DESCRIPTION_CAPTION,
                "percent",
                daily_log_returns(result),
                excess_return,
                cost_scenario=scenario,
                resamples=arguments.resamples,
            )
        )

    print(f"cost-dependent by ADR-0009: {sensitivity.cost_sensitive}")
    print(f"beats buy-and-hold (None = the costs decide): {sensitivity.beats_buy_and_hold}\n")

    # Reported and disclosed, never suppressed. A sensitivity row whose verdict moves with the seed
    # is a finding about how close to zero that row sits, and hiding the artefact would hide it.
    for entry in reported:
        if entry.unstable_blocks:
            print(
                f"NOTE: {entry.name}: the verdict is seed-dependent at "
                f"{list(entry.unstable_blocks)}-day blocks, so that row is not reportable as "
                "either Separated or Inconclusive without saying so."
            )

    # The reported row is the one a conclusion may rest on. If *its* verdict moves with the seed,
    # there is no conclusion to report and the run refuses rather than picking the flattering seed.
    unreportable = [entry.name for entry in reported if not entry.reported.verdict_stable]
    if unreportable:
        raise FatalDefect(
            "bootstrap_verdict_unstable",
            f"at the reported block length the verdict changed with the seed for {unreportable}; "
            "no conclusion may rest on a row that depends on which seed was drawn",
        )

    payload = {
        "contract_digest": contract.digest,
        "resamples": arguments.resamples,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "reported_block_days": BLOCK_DAYS,
        "days": expect_days,
        "headline_cost_scenario": HEADLINE_SCENARIO,
        "cost_sensitive": sensitivity.cost_sensitive,
        "beats_buy_and_hold": sensitivity.beats_buy_and_hold,
        "promoted_arm": promoted,
        "quantities": [as_payload(entry) for entry in reported],
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {arguments.out}")
    print(
        "No Trial was spent: the bootstrap estimates uncertainty around results already produced."
    )


if __name__ == "__main__":
    main()
