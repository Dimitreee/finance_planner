"""Spend the single permitted evaluation of the Final Holdout.

**This command can succeed exactly once.** It prints the frozen configuration and the contract
digest before it computes anything, so whoever runs it can see what is about to be spent and on
what. A second invocation refuses and exits non-zero without producing a number — ADR-0018, and
the refusal is the mechanism rather than the intention.

If a defect is discovered after this has run, the number is reported with the defect disclosed, or
the claim is withdrawn. It does not license a second run. That is the cost of the rule and it was
accepted in advance.

It spends no Trial. The configuration was frozen in the Experiment Contract beforehand and nothing
downstream may change because of the result, so there is no choice it could have influenced.

Run from the repository root:

    uv run python scripts/final_holdout.py --dry-run   # show what would be spent
    uv run python scripts/final_holdout.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptoguard_core.contract import contract_value, load_contract
from cryptoguard_core.dataset import build_decision_day_rows
from cryptoguard_core.holdout import (
    MARKER_PATH,
    evaluate_final_holdout,
    frozen_configuration,
    holdout_fold,
    read_spend,
    refuse_if_spent,
)
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.metrics import NO_SKILL_LOG_LOSS

SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--marker", type=Path, default=MARKER_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the frozen configuration and the state of the marker; spend nothing",
    )
    arguments = parser.parse_args()

    contract = load_contract()
    configuration = frozen_configuration(contract)
    fold = holdout_fold(contract)

    print("The single permitted evaluation of the Final Holdout.")
    print(f"  contract digest:   {configuration.contract_digest}")
    print(f"  arm:               {configuration.arm}")
    print(f"  lag:               {configuration.lag_hours}")
    print(f"  cost scenario:     {configuration.cost_scenario}")
    print(f"  policy thresholds: {configuration.to_usdt_at} / {configuration.to_btc_at}")
    print(
        f"  regularisation:    {list(configuration.regularisation_grid)}, "
        f"{configuration.regularisation_selected}"
    )
    print(f"  holdout window:    {fold.test_first_day} .. {fold.test_last_day}")
    print(f"  frozen on:         {contract_value(contract, 'final_holdout', 'frozen_on')}")
    print(f"  marker:            {arguments.marker}")

    already = read_spend(arguments.marker)
    if already is not None:
        print(f"\nAlready spent on {already.spent_at}. Refusing.", file=sys.stderr)
        refuse_if_spent(arguments.marker)

    if arguments.dry_run:
        print("\ndry run: the Final Holdout is untouched and nothing was spent")
        return 0

    bars = load_archive_backfill(arguments.snapshot).bars
    rows = build_decision_day_rows(bars, arm=configuration.arm.lower(), last_day=fold.test_last_day)
    spend = evaluate_final_holdout(rows, contract, marker_path=arguments.marker)
    result = spend.result

    print(f"\nscored days:          {result.scored_days}")
    print(
        f"log loss:             {result.log_loss:.6f}  "
        f"({result.log_loss - NO_SKILL_LOG_LOSS:+.6f} against no-skill ln 2)"
    )
    print(f"brier:                {result.brier:.6f}")
    print(f"accuracy:             {result.accuracy:.4f}")
    print(f"net return:           {100 * result.net_return:+.1f}%")
    print(f"buy-and-hold:         {100 * result.buy_and_hold_return:+.1f}%")
    print(f"trades:               {result.trades}")
    print(f"max drawdown:         {100 * result.max_drawdown:.1f}%")
    print(f"turnover:             {result.turnover:.2f}x")
    print(f"time invested:        {100 * result.time_invested:.1f}%")
    print(f"\nspends a Trial:       {spend.spends_a_trial}")
    print(f"  {spend.trial_reasoning}")
    print(f"\nThe Final Holdout is now spent. Marker: {arguments.marker}")
    print("A defect found later is disclosed or the claim is withdrawn; it is not re-run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FatalDefect as defect:
        print(f"{defect.kind}: {defect}", file=sys.stderr)
        raise SystemExit(1) from defect
