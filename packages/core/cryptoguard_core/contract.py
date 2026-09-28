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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from cryptoguard_core.annotation import (
    AGREEMENT_REFERENCE,
    ALLOCATION,
    LABELS,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    STRATA,
    VOLUME_BANDS,
    WORDING_WHEN_NOT,
    WORDING_WHEN_TRANSFERS,
)
from cryptoguard_core.dataset import (
    ARM_A_FEATURES,
    ARMS_NEEDING_NEWS,
    FEATURES_BY_ARM,
    RESEARCH_WINDOW_FIRST_DAY,
    RESEARCH_WINDOW_LAST_DAY,
    WARM_UP_DAYS,
    DecisionDayRow,
)
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news_features import (
    ARM_B_EXTRA_FEATURES,
    ARM_C_EXTRA_FEATURES,
    LAG_GRID_HOURS,
)
from cryptoguard_core.policy import COST_SCENARIOS, HEADLINE_SCENARIO
from cryptoguard_core.protocol import (
    DECISION_DEADLINE_UTC,
    EXECUTION_UTC,
    FEATURE_CUTOFF_UTC,
    HORIZON_DAYS,
)
from cryptoguard_core.sentiment import (
    EXTRACTOR_NAME,
    EXTRACTOR_REVISION,
    MAX_TOKENS,
    SCORE_DECIMALS,
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
    # Which Experiment Arm's features the rows carry, and under which Assumed Availability Lag. A
    # price-only arm records no lag, because it reads no news and a recorded lag would imply it did.
    arm: str
    lag_hours: int | None


_QUARTER_LABEL = re.compile(r"^(\d{4})Q([1-4])$")


def _quarter_start(label: object) -> date:
    """`2022Q1` to the first day of that quarter, refusing anything else by name."""
    match = _QUARTER_LABEL.match(label) if isinstance(label, str) else None
    if match is None:
        _refuse(f"{label!r} is not a quarter label such as '2022Q1'")
        raise AssertionError("unreachable")  # _refuse always raises
    year, quarter = int(match.group(1)), int(match.group(2))
    return date(year, 3 * (quarter - 1) + 1, 1)


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
    if tuple(values["features"]["arm_b_extra"]) != ARM_B_EXTRA_FEATURES:
        _refuse("features.arm_b_extra does not match the frozen Arm B feature list in the code")
    if tuple(values["features"]["arm_c_extra"]) != ARM_C_EXTRA_FEATURES:
        _refuse("features.arm_c_extra does not match the frozen Arm C feature list in the code")
    news = values["news"]
    if tuple(news["lag_grid_hours"]) != LAG_GRID_HOURS:
        _refuse(f"news.lag_grid_hours {news['lag_grid_hours']} != {list(LAG_GRID_HOURS)}")
    if news["primary_lag_hours"] not in LAG_GRID_HOURS:
        _refuse(f"news.primary_lag_hours {news['primary_lag_hours']} is not in the frozen lag grid")
    extractor = news["sentiment_extractor"]
    pinned = {
        "name": EXTRACTOR_NAME,
        "revision": EXTRACTOR_REVISION,
        "max_tokens": MAX_TOKENS,
        "score_decimals": SCORE_DECIMALS,
    }
    for key, expected in pinned.items():
        if extractor[key] != expected:
            _refuse(
                f"news.sentiment_extractor.{key} is {extractor[key]!r}, but the code states "
                f"{expected!r}"
            )
    if extractor["fine_tuned"] is not False:
        _refuse("the Sentiment Extractor is frozen by ADR-0015; fine_tuned must be false")
    if extractor["device"] != "cpu":
        _refuse("inference runs on CPU, because a dataset's numbers must be CPU-reproducible")

    annotation = values["annotation"]
    pinned_annotation = {
        "sample_size": SAMPLE_SIZE,
        "seed": SAMPLE_SEED,
        "agreement_reference": AGREEMENT_REFERENCE,
        "wording_when_agreement_beats_reference": WORDING_WHEN_TRANSFERS,
        "wording_otherwise": WORDING_WHEN_NOT,
    }
    for key, expected in pinned_annotation.items():
        if annotation[key] != expected:
            _refuse(f"annotation.{key} is {annotation[key]!r}, but the code states {expected!r}")
    if tuple(annotation["labels"]) != LABELS:
        _refuse(f"annotation.labels {list(annotation['labels'])} != {list(LABELS)}")
    if tuple(annotation["volume_bands"]) != VOLUME_BANDS:
        _refuse(
            f"annotation.volume_bands {list(annotation['volume_bands'])} != {list(VOLUME_BANDS)}"
        )
    # Checked rather than merely recorded: a key nothing validates can be edited to describe a
    # method the code does not implement, and the digest would move as if the change were real.
    if tuple(annotation["strata"]) != STRATA:
        _refuse(f"annotation.strata {list(annotation['strata'])} != {list(STRATA)}")
    if annotation["allocation"] != ALLOCATION:
        _refuse(
            f"annotation.allocation is {annotation['allocation']!r}, but the sampler implements "
            f"{ALLOCATION!r}"
        )
    lower, upper = annotation["sample_size_bounds"]
    if not lower <= annotation["sample_size"] <= upper:
        _refuse(
            f"annotation.sample_size {annotation['sample_size']} is outside the pre-registered "
            f"range [{lower}, {upper}]"
        )
    if annotation["unclear_excluded_from_agreement"] is not True:
        _refuse(
            "annotation.unclear_excluded_from_agreement must be true: mapping 'unclear' to neutral "
            "would report an uncertainty as a reading"
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
    # Counted, not asserted. The first version of this contract carried a number that was simply
    # wrong by one day and was repeated into four documents before anyone did the arithmetic.
    first_scored = _quarter_start(splits["first_test_quarter"])
    scored = (splits["development_last_day"] - first_scored).days + 1
    if splits["scored_development_days"] != scored:
        _refuse(
            f"splits.scored_development_days is {splits['scored_development_days']}, but "
            f"{first_scored} to {splits['development_last_day']} is {scored} days"
        )

    costs = values["cost_scenarios"]
    headline = costs["headline"]
    if headline not in costs or not isinstance(costs[headline], Mapping):
        _refuse(f"cost_scenarios.headline {headline!r} does not name a scenario")
    if headline != HEADLINE_SCENARIO:
        _refuse(
            f"cost_scenarios.headline is {headline!r}, but the code states {HEADLINE_SCENARIO!r}"
        )
    for name, scenario in COST_SCENARIOS.items():
        stated = costs.get(name)
        if stated is None:
            _refuse(f"cost_scenarios is missing {name!r}")
        elif (stated["fee_pct"], stated["slippage_bps"]) != (
            scenario.fee_pct,
            scenario.slippage_bps,
        ):
            _refuse(
                f"cost_scenarios.{name} is {stated['fee_pct']}% + {stated['slippage_bps']} bp, "
                f"but the code states {scenario.fee_pct}% + {scenario.slippage_bps} bp"
            )


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


def _arm_of(features: Mapping[str, float]) -> str:
    """Which arm a row belongs to, read off its feature names."""
    names = tuple(features)
    for arm, expected in FEATURES_BY_ARM.items():
        if names == expected:
            return arm
    _refuse(f"features {list(names)} match no known arm")
    raise AssertionError("unreachable")


def build_dataset_manifest(
    *,
    contract: ExperimentContract,
    rows: Sequence[DecisionDayRow],
    source_digests: Mapping[str, str],
) -> DatasetManifest:
    """Describe the dataset that was actually built, reading arm, lag and span off the rows.

    Taking these as arguments let a caller state a lag the rows were not built under, and nothing
    downstream could have caught it: the digest would assert the Primary Lag over a one-hour-lag
    dataset. Derived, the two cannot disagree.
    """
    if set(source_digests) != set(SOURCE_MANIFESTS):
        _refuse(
            f"a dataset manifest must name exactly {sorted(SOURCE_MANIFESTS)}, "
            f"got {sorted(source_digests)}"
        )
    if not rows:
        _refuse("a dataset manifest describes rows; there are none")

    arms = {_arm_of(row.features) for row in rows}
    if len(arms) > 1:
        _refuse(f"the rows carry more than one arm: {sorted(arms)}")
    lags = {row.news_lag_hours for row in rows}
    if len(lags) > 1:
        _refuse(f"the rows carry more than one Assumed Availability Lag: {sorted(map(str, lags))}")

    arm = arms.pop()
    lag_hours = lags.pop()
    _check_arm_and_lag(arm, lag_hours)
    return DatasetManifest(
        contract_digest=contract.digest,
        row_count=len(rows),
        first_day=rows[0].day,
        last_day=rows[-1].day,
        source_digests=dict(source_digests),
        arm=arm,
        lag_hours=lag_hours,
    )


def _check_arm_and_lag(arm: str, lag_hours: int | None) -> None:
    """The invariants binding an arm to a lag, applied on the way in and on the way back out."""
    if arm not in FEATURES_BY_ARM:
        _refuse(f"arm {arm!r} is not one of {sorted(FEATURES_BY_ARM)}")
    reads_news = arm in ARMS_NEEDING_NEWS
    if reads_news and lag_hours is None:
        _refuse(f"arm {arm!r} reads news, so its dataset must record the lag it was built under")
    if not reads_news and lag_hours is not None:
        _refuse(f"arm {arm!r} reads no news, so recording a lag would imply that it did")
    if lag_hours is not None and lag_hours not in LAG_GRID_HOURS:
        _refuse(f"lag {lag_hours}h is not in the frozen lag grid {list(LAG_GRID_HOURS)}")


def _manifest_mapping(manifest: DatasetManifest) -> dict[str, Any]:
    """The one serialisation of a manifest, so the file and its digest cannot disagree."""
    return {
        "contract_digest": manifest.contract_digest,
        "row_count": manifest.row_count,
        "first_day": manifest.first_day.isoformat(),
        "last_day": manifest.last_day.isoformat(),
        "source_digests": dict(manifest.source_digests),
        "arm": manifest.arm,
        "lag_hours": manifest.lag_hours,
    }


def digest_of_manifest(manifest: DatasetManifest) -> str:
    """What a dataset is, as one value. Two datasets differing in arm or lag differ here."""
    return digest_of(_manifest_mapping(manifest))


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.write_text(
        json.dumps(_manifest_mapping(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_dataset_manifest(path: Path) -> DatasetManifest:
    """Read a manifest and re-apply every invariant the build enforced.

    A fit is handed the manifest that was read, not the one that was built, so validating only on
    the way in leaves the invariants unenforced exactly where they matter. A hand-edited arm/lag
    pair is refused here, and a manifest written before these fields existed is named as such
    instead of raising a bare KeyError.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("arm", "lag_hours") if key not in raw]
    if missing:
        _refuse(f"the manifest at {path} predates the arm and lag fields: missing {missing}")
    _check_arm_and_lag(raw["arm"], raw["lag_hours"])
    return DatasetManifest(
        contract_digest=raw["contract_digest"],
        row_count=raw["row_count"],
        first_day=date.fromisoformat(raw["first_day"]),
        last_day=date.fromisoformat(raw["last_day"]),
        source_digests=dict(raw["source_digests"]),
        arm=raw["arm"],
        lag_hours=raw["lag_hours"],
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
