"""Entry point for the batch jobs that ingest data and publish runs."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

import cryptoguard_core


def main(argv: Sequence[str] | None = None) -> int:
    """Report what this build is.

    Real subcommands arrive with the ingest and decision tickets. Until then unknown arguments are
    rejected rather than ignored: this runs from a scheduler, where a mistyped subcommand that exits
    zero reads as a successful run.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print(f"cryptoguard-job: unrecognised arguments: {' '.join(args)}", file=sys.stderr)
        return 2
    print(json.dumps({"core_version": cryptoguard_core.__version__}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
