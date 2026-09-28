"""Bootstrap Intervals around the Final Holdout's recorded result. Describes; spends nothing.

**This is not a second evaluation and it must prove that it is not.** The frozen configuration's
per-day series is deterministic, so it is re-derived here and then **checked against the marker**:
if the re-derived log loss and net return are not identical to the recorded ones, this is
describing some other result and the run refuses. That check is what separates a description of
the recorded number from a fresh measurement of the same period — the same reasoning that let Arm
A be re-walked for the Primary Comparison without paying a second Trial.

It writes no marker, changes no configuration, and selects nothing.

**Both intervals are description.** ADR-0017 grants evidence standing to the Primary Comparison's
interval alone. The holdout is not that comparison, so neither of these may inform a conclusion;
they exist so the recorded numbers cannot be read as precise.

Run from the repository root:

    uv run python scripts/holdout_interval.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from cryptoguard_core.backtest import daily_log_returns, excess_return
from cryptoguard_core.bootstrap import (
    BLOCK_DAYS,
    CONFIDENCE,
    RESAMPLES,
    SEED,
    SENSITIVITY_BLOCK_DAYS,
    STABILITY_SEEDS,
    BootstrapInterval,
    moving_block_bootstrap,
)
from cryptoguard_core.contract import load_contract
from cryptoguard_core.dataset import build_decision_day_rows
from cryptoguard_core.holdout import MARKER_PATH, holdout_fold, read_spend, rederive_holdout
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.metrics import NO_SKILL_LOG_LOSS, daily_log_loss

SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")
RESULT_PATH = Path("artifacts/holdout_intervals.json")
DESCRIPTION = "description"


def measure[Observation](
    name: str,
    units: str,
    series: Sequence[Observation],
    statistic: Callable[[Sequence[Observation]], float],
    *,
    resamples: int,
) -> dict[str, object]:
    """The reported interval, the sensitivity rows, and the verdict at every stability seed."""
    print(name)
    print("  description: this interval informs nothing and selects nothing (ADR-0017)")
    rows: list[dict[str, object]] = []
    for block_days in (BLOCK_DAYS, *SENSITIVITY_BLOCK_DAYS):
        interval = moving_block_bootstrap(
            series, statistic, block_days=block_days, resamples=resamples, seed=SEED
        )
        verdicts = {
            seed: moving_block_bootstrap(
                series, statistic, block_days=block_days, resamples=resamples, seed=seed
            ).verdict
            for seed in STABILITY_SEEDS
        }
        stable = set(verdicts.values()) == {interval.verdict}
        role = "reported" if block_days == BLOCK_DAYS else "sensitivity"
        print(
            f"  {block_days:>3}-day blocks ({role:<11}) {render(interval, units)}  "
            f"{'seed-invariant' if stable else f'SEED-DEPENDENT {verdicts}'}"
        )
        rows.append(
            {
                "block_days": block_days,
                "role": role,
                "seed": interval.seed,
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "verdict": interval.verdict,
                "verdict_stable_across_seeds": stable,
                "verdicts_by_stability_seed": {str(k): v for k, v in verdicts.items()},
            }
        )
    print()
    return {"name": name, "standing": DESCRIPTION, "units": units, "intervals": rows}


def render(interval: BootstrapInterval, units: str) -> str:
    scale = 100.0 if units == "percent" else 1.0
    suffix = "%" if units == "percent" else ""
    return (
        f"{scale * interval.estimate:+.6f}{suffix}  "
        f"[{scale * interval.lower:+.6f}{suffix}, {scale * interval.upper:+.6f}{suffix}]  "
        f"{interval.verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--marker", type=Path, default=MARKER_PATH)
    parser.add_argument("--out", type=Path, default=RESULT_PATH)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    arguments = parser.parse_args()

    recorded = read_spend(arguments.marker)
    if recorded is None:
        raise FatalDefect(
            "final_holdout_not_frozen",
            f"{arguments.marker} does not exist, so there is no recorded holdout result to "
            "describe. This script describes a number that already exists; it does not produce one",
        )

    contract = load_contract()
    if contract.digest != recorded.configuration.contract_digest:
        raise FatalDefect(
            "contract_mismatch",
            f"the holdout was evaluated under contract "
            f"{recorded.configuration.contract_digest[:12]} and the contract now digests to "
            f"{contract.digest[:12]}; describing one result with another's protocol is not a "
            "description of it",
        )

    fold = holdout_fold(contract)
    bars = load_archive_backfill(arguments.snapshot).bars
    rows = build_decision_day_rows(
        bars, arm=recorded.configuration.arm.lower(), last_day=fold.test_last_day
    )
    walk, backtest = rederive_holdout(rows, contract)

    # The check that makes this a description rather than a second measurement.
    drift = {
        "log_loss": (walk.log_loss, recorded.result.log_loss),
        "net_return": (backtest.net_return, recorded.result.net_return),
        "buy_and_hold_return": (backtest.buy_and_hold_return, recorded.result.buy_and_hold_return),
        "scored_days": (len(walk.predictions), recorded.result.scored_days),
    }
    moved = {name: pair for name, pair in drift.items() if pair[0] != pair[1]}
    if moved:
        raise FatalDefect(
            "contract_mismatch",
            f"the re-derived holdout does not match the recorded one: {moved}. This would be a "
            "different result, not a description of the recorded one, so nothing is reported",
        )
    print(f"re-derivation matches the marker exactly on {list(drift)}\n")

    reported = [
        measure(
            "daily log loss minus the no-skill reference, Final Holdout",
            "natural",
            [daily_log_loss(p.label, p.probability) - NO_SKILL_LOG_LOSS for p in walk.predictions],
            statistics.fmean,
            resamples=arguments.resamples,
        ),
        measure(
            "net cumulative return after costs minus buy-and-hold, Final Holdout",
            "percent",
            daily_log_returns(backtest),
            excess_return,
            resamples=arguments.resamples,
        ),
    ]
    payload = {
        "describes_marker": str(arguments.marker),
        "spent_at": recorded.spent_at,
        "contract_digest": contract.digest,
        "days": recorded.result.scored_days,
        "resamples": arguments.resamples,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "reported_block_days": BLOCK_DAYS,
        "spends_a_trial": False,
        "is_second_evaluation": False,
        "quantities": reported,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {arguments.out}")
    print("No Trial and no evaluation spent: the recorded result was described, not remeasured.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FatalDefect as defect:
        print(f"{defect.kind}: {defect}", file=sys.stderr)
        raise SystemExit(1) from defect
