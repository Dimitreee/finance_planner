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
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.store import RunStore

DATABASE_URL_VAR = "CRYPTOGUARD_DATABASE_URL"
SNAPSHOT = Path("data/raw/binance/klines/BTCUSDT/1h")
BUNDLE_DIR = Path("artifacts/run_bundles")


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
    outcome = run_decision_job(
        store=store,
        contract=load_contract(),
        model=PreviousDirectionBaseline(),
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

    if namespace.command == "decide":
        try:
            return _decide(namespace)
        except FatalDefect as defect:
            print(f"cryptoguard-job: {defect.kind}: {defect}", file=sys.stderr)
            return 1
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
