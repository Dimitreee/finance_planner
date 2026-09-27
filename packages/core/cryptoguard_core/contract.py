"""The Experiment Contract, and the manifest binding a dataset to the contract it was built under.

A choice absent from the contract at fit time is not pre-registered, whatever a document claims
afterwards. The digest is taken over the parsed *values*, so reformatting the file or adding a
comment leaves it alone while any changed value moves it — the digest tracks the pre-registration,
not the typography. The loaded values are deeply immutable, because a mutated value under an
unchanged digest would attest to a pre-registration that was never used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from cryptoguard_core.dataset import (
    ARM_A_FEATURES,
    RESEARCH_WINDOW_FIRST_DAY,
    RESEARCH_WINDOW_LAST_DAY,
    WARM_UP_DAYS,
)
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.protocol import (
    DECISION_DEADLINE_UTC,
    EXECUTION_UTC,
    FEATURE_CUTOFF_UTC,
    HORIZON_DAYS,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = _REPO_ROOT / "config/experiment.yaml"
MANIFEST_DIR = _REPO_ROOT / "data/manifests"

# The per-file digests live in these manifests, which are tracked; hashing the manifest binds a
# dataset to every source file at once without copying a hundred digests into every run.
SOURCE_MANIFESTS = {
    "archive_backfill": "binance_BTCUSDT_1h_monthly.sha256",
    "news_cryptopanic": "news_cryptopanic.sha256",
}


def _canonical(node: Any) -> Any:
    """A JSON-safe rendering that keeps types distinct.

    A `date` and its ISO string are different values and must not collapse to the same digest.
    """
    if isinstance(node, Mapping):
        return {str(key): _canonical(value) for key, value in node.items()}
    if isinstance(node, list | tuple):
        return [_canonical(value) for value in node]
    if isinstance(node, date):
        return {"__date__": node.isoformat()}
    if node is None or isinstance(node, str | bool | int | float):
        return node
    return {"__repr__": repr(node)}


def digest_of(values: Mapping[str, Any]) -> str:
    """SHA-256 over a canonical rendering of the values: key order and comments do not count."""
    canonical = json.dumps(_canonical(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _freeze(node: Any) -> Any:
    if isinstance(node, Mapping):
        return MappingProxyType({key: _freeze(value) for key, value in node.items()})
    if isinstance(node, list):
        return tuple(_freeze(value) for value in node)
    return node


def deep_unfreeze(node: Any) -> Any:
    """A plain, mutable copy — for tooling that needs to build a variant, never for a fit."""
    if isinstance(node, Mapping):
        return {key: deep_unfreeze(value) for key, value in node.items()}
    if isinstance(node, tuple):
        return [deep_unfreeze(value) for value in node]
    return node


def digest_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digests_from_manifests(manifest_dir: Path = MANIFEST_DIR) -> dict[str, str]:
    """One digest per source manifest, for the dataset manifest to record."""
    digests: dict[str, str] = {}
    for name, filename in SOURCE_MANIFESTS.items():
        path = manifest_dir / filename
        if not path.is_file():
            raise FatalDefect("contract_inconsistent", f"source manifest {path} is absent")
        digests[name] = digest_of_file(path)
    return digests


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    path: Path
    values: Mapping[str, Any]
    digest: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What a dataset is and what it was built from. A fit is handed this, not loose rows.

    The contract is identified by its digest alone: a filename would not distinguish two contracts
    and would read as provenance without being any.
    """

    contract_digest: str
    row_count: int
    first_day: date
    last_day: date
    source_digests: Mapping[str, str]


def _refuse(detail: str) -> None:
    raise FatalDefect("contract_inconsistent", detail)


def _check(values: Mapping[str, Any]) -> None:
    window = values["research_window"]
    if window["first_day"] != RESEARCH_WINDOW_FIRST_DAY:
        _refuse(f"research_window.first_day {window['first_day']} != {RESEARCH_WINDOW_FIRST_DAY}")
    if window["last_day"] != RESEARCH_WINDOW_LAST_DAY:
        _refuse(f"research_window.last_day {window['last_day']} != {RESEARCH_WINDOW_LAST_DAY}")
    if window["warm_up_days"] != WARM_UP_DAYS:
        _refuse(f"research_window.warm_up_days {window['warm_up_days']} != {WARM_UP_DAYS}")
    span = (window["last_day"] - window["first_day"]).days + 1
    if window["decision_days"] != span:
        _refuse(f"research_window.decision_days {window['decision_days']} != {span} actual days")

    protocol = values["decision_protocol"]
    expected_protocol = {
        "feature_cutoff": FEATURE_CUTOFF_UTC,
        "decision_deadline": DECISION_DEADLINE_UTC,
        "execution": EXECUTION_UTC,
        "horizon_days": HORIZON_DAYS,
    }
    for key, expected in expected_protocol.items():
        if protocol[key] != expected:
            _refuse(
                f"decision_protocol.{key} is {protocol[key]!r}, but the code states {expected!r}"
            )

    if tuple(values["features"]["arm_a"]) != ARM_A_FEATURES:
        _refuse("features.arm_a does not match the frozen Arm A feature list in the code")

    news = values["news"]
    if news["primary_lag_hours"] not in news["lag_grid_hours"]:
        _refuse(
            f"news.primary_lag_hours {news['primary_lag_hours']} is not in the lag grid "
            f"{list(news['lag_grid_hours'])}"
        )

    policy = values["policy"]
    if not policy["to_usdt_at"] < policy["to_btc_at"]:
        _refuse("policy thresholds must leave a hysteresis band: to_usdt_at < to_btc_at")

    trials = values["trials"]
    if trials["planned"] > trials["budget"]:
        _refuse(f"trials.planned {trials['planned']} exceeds the budget {trials['budget']}")

    splits = values["splits"]
    if splits["final_holdout_first_day"] <= splits["development_last_day"]:
        _refuse("the Final Holdout must start after the Development Period ends")

    costs = values["cost_scenarios"]
    headline = costs["headline"]
    if headline not in costs or not isinstance(costs[headline], Mapping):
        _refuse(f"cost_scenarios.headline {headline!r} does not name a scenario")


def _validate(values: Any) -> None:
    """Every malformed contract leaves through the same door: a named refusal, not a traceback."""
    if not isinstance(values, Mapping):
        _refuse("the contract must be a mapping of sections")
    try:
        _check(values)
    except (KeyError, TypeError, AttributeError) as error:
        raise FatalDefect(
            "contract_inconsistent", f"the contract is malformed: {error!r}"
        ) from error


def load_contract(path: Path = CONTRACT_PATH) -> ExperimentContract:
    """Read, validate and freeze the contract. Config and code must not drift apart silently."""
    if not path.is_file():
        raise FatalDefect("contract_inconsistent", f"no Experiment Contract at {path}")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate(values)
    return ExperimentContract(path=path, values=_freeze(values), digest=digest_of(values))


def build_dataset_manifest(
    *,
    contract: ExperimentContract,
    row_count: int,
    first_day: date,
    last_day: date,
    source_digests: Mapping[str, str],
) -> DatasetManifest:
    if set(source_digests) != set(SOURCE_MANIFESTS):
        _refuse(
            f"a dataset manifest must name exactly {sorted(SOURCE_MANIFESTS)}, "
            f"got {sorted(source_digests)}"
        )
    return DatasetManifest(
        contract_digest=contract.digest,
        row_count=row_count,
        first_day=first_day,
        last_day=last_day,
        source_digests=dict(source_digests),
    )


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.write_text(
        json.dumps(
            {
                "contract_digest": manifest.contract_digest,
                "row_count": manifest.row_count,
                "first_day": manifest.first_day.isoformat(),
                "last_day": manifest.last_day.isoformat(),
                "source_digests": dict(manifest.source_digests),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_dataset_manifest(path: Path) -> DatasetManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DatasetManifest(
        contract_digest=raw["contract_digest"],
        row_count=raw["row_count"],
        first_day=date.fromisoformat(raw["first_day"]),
        last_day=date.fromisoformat(raw["last_day"]),
        source_digests=dict(raw["source_digests"]),
    )


def verify_manifest_against(
    manifest: DatasetManifest,
    contract: ExperimentContract,
    *,
    manifest_dir: Path = MANIFEST_DIR,
) -> str:
    """Return the contract digest a fit must record, or refuse to let the fit happen.

    Both halves of the binding are checked. The contract must be the one the dataset was built
    under, and the source files must still be the ones it was built from: re-fetching an archive
    changes the sources while leaving the contract untouched.
    """
    if manifest.contract_digest != contract.digest:
        raise FatalDefect(
            "contract_mismatch",
            f"the dataset was built under contract {manifest.contract_digest[:12]} but the "
            f"contract in force is {contract.digest[:12]}",
        )
    current = source_digests_from_manifests(manifest_dir)
    changed = [
        name for name, value in current.items() if manifest.source_digests.get(name) != value
    ]
    if changed:
        raise FatalDefect(
            "source_digest_mismatch",
            f"the sources changed since this dataset was built: {', '.join(sorted(changed))}",
        )
    return manifest.contract_digest
