"""Entry point for the batch jobs that ingest data and publish runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import cryptoguard_core
from cryptoguard_core.contract import load_contract
from cryptoguard_core.dataset import build_decision_day_rows
from cryptoguard_core.decide import DEFAULT_ASSET, run_decision_job
from cryptoguard_core.evaluate import evaluate_arm_a
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.release import current_release, promote
from cryptoguard_core.store import RunStore
from cryptoguard_core.trials import TrialLog

DATABASE_URL_VAR = "CRYPTOGUARD_DATABASE_URL"
SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")
BUNDLE_DIR = Path("artifacts/run_bundles")
RELEASES_DIR = Path("artifacts/releases")
TRIALS_PATH = Path("artifacts/trials.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryptoguard-job")
    subcommands = parser.add_subparsers(dest="command")

    decide = subcommands.add_parser("decide", help="publish one Decision Day")
    decide.add_argument(
        "--day",
        required=True,
        type=date.fromisoformat,
        help="the Decision Day, YYYY-MM-DD",
    )
    decide.add_argument("--asset", default=DEFAULT_ASSET)
    decide.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    decide.add_argument("--bundles", type=Path, default=BUNDLE_DIR)
    decide.add_argument("--releases", type=Path, default=RELEASES_DIR)

    evaluate = subcommands.add_parser(
        "evaluate", help="fit Arm A across the folds and publish the Replay Result"
    )
    evaluate.add_argument("--asset", default=DEFAULT_ASSET)
    evaluate.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    evaluate.add_argument("--releases", type=Path, default=RELEASES_DIR)
    evaluate.add_argument("--trials", type=Path, default=TRIALS_PATH)
    evaluate.add_argument(
        "--corrects", type=int, default=None, help="the Trial number this rerun replaces"
    )
    evaluate.add_argument(
        "--promote",
        action="store_true",
        help="also make this release the one that produces Published Runs",
    )
    return parser


def _decide(namespace: argparse.Namespace) -> int:
    if namespace.asset != DEFAULT_ASSET:
        # The snapshot path is fixed to BTC/USDT, so another asset would publish BTC features under
        # its own name and corrupt that asset's position chain.
        print(
            f"cryptoguard-job: only {DEFAULT_ASSET} is supported; "
            f"{namespace.asset} has no data snapshot",
            file=sys.stderr,
        )
        return 2

    dsn = os.environ.get(DATABASE_URL_VAR)
    if not dsn:
        print(f"cryptoguard-job: {DATABASE_URL_VAR} is not set", file=sys.stderr)
        return 2

    day: date = namespace.day
    store = RunStore(dsn)
    store.migrate()

    bars = load_archive_backfill(namespace.snapshot).bars
    # A decision needs no Label; requiring one would make today's Decision Day impossible.
    rows = build_decision_day_rows(bars, day, day, require_label=False)
    # The promoted release, if there is one. Promotion that changed nothing would be theatre.
    promoted = current_release(namespace.releases)
    outcome = run_decision_job(
        store=store,
        contract=load_contract(),
        model=promoted if promoted is not None else PreviousDirectionBaseline(),
        rows=rows,
        day=day,
        bundle_dir=namespace.bundles,
        asset=namespace.asset,
    )
    # On a skip the computed run was never stored, so reporting it would describe advice that
    # exists nowhere. Report what the database actually holds for that day.
    shown = outcome.run if outcome.published else store.latest_published(namespace.asset)
    print(
        json.dumps(
            {
                "day": day.isoformat(),
                "model": outcome.run.model_version,
                "published": outcome.published,
                "action": shown.action if shown else None,
                "probability": shown.probability if shown else None,
                "position_after": (
                    {"btc": shown.position_after.btc, "usdt": shown.position_after.usdt}
                    if shown
                    else None
                ),
                "explanation": shown.explanation if shown else None,
            }
        )
    )
    return 0


def _evaluate(namespace: argparse.Namespace) -> int:
    if namespace.asset != DEFAULT_ASSET:
        print(f"cryptoguard-job: only {DEFAULT_ASSET} is supported", file=sys.stderr)
        return 2
    dsn = os.environ.get(DATABASE_URL_VAR)
    if not dsn:
        print(f"cryptoguard-job: {DATABASE_URL_VAR} is not set", file=sys.stderr)
        return 2

    contract = load_contract()
    store = RunStore(dsn)
    store.migrate()

    bars = load_archive_backfill(namespace.snapshot).bars
    development_last: date = contract.values["splits"]["development_last_day"]
    # Rows stop at the Development Period. One holdout price is still read, and legitimately so:
    # the Label of the last development day *is* the first holdout Decision Price, by definition of
    # the protocol. Nothing in the Final Holdout is scored, fitted on, or reported.
    rows = build_decision_day_rows(bars, last_day=development_last)
    trials = TrialLog(namespace.trials, budget=contract.values["trials"]["budget"])

    evaluation = evaluate_arm_a(
        rows,
        contract,
        trials,
        releases_dir=namespace.releases,
        asset=namespace.asset,
        corrects=namespace.corrects,
    )
    written = store.publish_replay(evaluation.replays)
    if namespace.promote:
        promote(namespace.releases, evaluation.release.version)

    print(
        json.dumps(
            {
                "trial": evaluation.trial_number,
                "trials_remaining": trials.remaining(),
                "release": evaluation.release.version,
                "promoted": bool(namespace.promote),
                "scored_days": evaluation.scored_days,
                "log_loss": evaluation.log_loss,
                "brier": evaluation.brier,
                "accuracy": evaluation.accuracy,
                "replay_rows_written": written,
                "scenarios": {
                    replay.cost_scenario: {
                        "net_return": replay.net_return,
                        "buy_and_hold_return": replay.buy_and_hold_return,
                        "trades": replay.trades,
                    }
                    for replay in evaluation.replays
                },
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Report what this build is, or run a subcommand.

    Unknown arguments are rejected rather than ignored: this runs from a scheduler, where a mistyped
    subcommand that exits zero reads as a successful run.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(json.dumps({"core_version": cryptoguard_core.__version__}))
        return 0

    parser = _build_parser()
    try:
        namespace = parser.parse_args(args)
    except SystemExit as exit_request:  # argparse writes its own message to stderr
        # `--help` exits 0 and must stay 0: a scheduler reads these codes.
        return 2 if exit_request.code is None else int(exit_request.code)

    runners = {"decide": _decide, "evaluate": _evaluate}
    runner = runners.get(namespace.command)
    if runner is not None:
        try:
            return runner(namespace)
        except FatalDefect as defect:
            print(f"cryptoguard-job: {defect.kind}: {defect}", file=sys.stderr)
            return 1
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
