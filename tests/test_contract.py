"""The Experiment Contract: one file that freezes every choice, and a digest that notices edits.

The contract is the mechanism that makes pre-registration checkable rather than asserted. Its digest
is taken over the parsed values, not the file bytes, so reformatting or a new comment leaves it
alone while any changed value moves it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import SNAPSHOT
from cryptoguard_core.contract import (
    CONTRACT_PATH,
    DatasetManifest,
    ExperimentContract,
    build_dataset_manifest,
    deep_unfreeze,
    digest_of,
    load_contract,
    read_dataset_manifest,
    source_digests_from_manifests,
    verify_manifest_against,
    write_dataset_manifest,
)
from cryptoguard_core.dataset import (
    ARM_A_FEATURES,
    RESEARCH_WINDOW_FIRST_DAY,
    RESEARCH_WINDOW_LAST_DAY,
    WARM_UP_DAYS,
    build_decision_day_rows,
)
from cryptoguard_core.ingest import FatalDefect, load_archive_backfill


def leaf_paths(node: Any, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if isinstance(node, dict):
        return [p for key, value in node.items() for p in leaf_paths(value, (*prefix, key))]
    if isinstance(node, list):
        return [p for i, value in enumerate(node) for p in leaf_paths(value, (*prefix, i))]
    return [prefix]


def set_at(node: Any, path: tuple[object, ...], value: object) -> None:
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value


def test_the_shipped_contract_loads() -> None:
    contract = load_contract()
    assert contract.path == CONTRACT_PATH
    assert contract.digest


def test_contract_agrees_with_the_frozen_code_constants() -> None:
    """Config and code must not drift: the contract is the pre-registration, the code obeys it."""
    contract = load_contract()
    window = contract.values["research_window"]
    assert window["first_day"] == RESEARCH_WINDOW_FIRST_DAY
    assert window["last_day"] == RESEARCH_WINDOW_LAST_DAY
    assert window["warm_up_days"] == WARM_UP_DAYS
    assert tuple(contract.values["features"]["arm_a"]) == ARM_A_FEATURES


def test_the_primary_lag_measured_in_the_audit_is_frozen_here() -> None:
    news = load_contract().values["news"]
    assert news["primary_lag_hours"] == 24
    assert news["lag_grid_hours"] == (1, 6, 24)
    assert news["primary_lag_hours"] in news["lag_grid_hours"]


def test_every_single_value_change_moves_the_digest() -> None:
    """A silent edit must not pass as the same pre-registration."""
    contract = load_contract()
    baseline = contract.digest
    paths = leaf_paths(deep_unfreeze(contract.values))
    assert len(paths) > 40, "the contract should cover the whole protocol"

    for path in paths:
        mutated = deep_unfreeze(contract.values)
        set_at(mutated, path, "___changed___")
        assert digest_of(mutated) != baseline, f"digest ignored a change at {path}"


def test_the_digest_follows_values_not_formatting() -> None:
    contract = load_contract()
    reordered = dict(reversed(list(deep_unfreeze(contract.values).items())))
    assert digest_of(reordered) == contract.digest


def test_an_inconsistent_contract_is_refused(tmp_path: Path) -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        "primary_lag_hours: 24", "primary_lag_hours: 12"
    )
    broken = tmp_path / "experiment.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        load_contract(broken)
    assert caught.value.kind == "contract_inconsistent"


def _manifest(contract: ExperimentContract) -> DatasetManifest:
    return build_dataset_manifest(
        contract=contract,
        row_count=1_764,
        first_day=RESEARCH_WINDOW_FIRST_DAY,
        last_day=RESEARCH_WINDOW_LAST_DAY,
        source_digests=source_digests_from_manifests(),
    )


def test_manifest_records_the_contract_and_every_source() -> None:
    contract = load_contract()
    manifest = _manifest(contract)
    assert manifest.contract_digest == contract.digest
    assert manifest.row_count == 1_764
    assert set(manifest.source_digests) == {"archive_backfill", "news_cryptopanic"}
    assert manifest.first_day == date(2021, 2, 1)


def test_manifest_survives_a_round_trip(tmp_path: Path) -> None:
    contract = load_contract()
    path = tmp_path / "dataset.json"
    write_dataset_manifest(path, _manifest(contract))
    assert read_dataset_manifest(path) == _manifest(contract)


def test_a_dataset_built_under_another_contract_is_refused(tmp_path: Path) -> None:
    """This is what "no fitting without a recorded contract" means in practice."""
    contract = load_contract()
    manifest = _manifest(contract)
    verify_manifest_against(manifest, contract)  # the matching case passes

    text = CONTRACT_PATH.read_text(encoding="utf-8").replace("budget: 12", "budget: 11")
    other_path = tmp_path / "experiment.yaml"
    other_path.write_text(text, encoding="utf-8")
    other = load_contract(other_path)

    with pytest.raises(FatalDefect) as caught:
        verify_manifest_against(manifest, other)
    assert caught.value.kind == "contract_mismatch"


def test_source_digests_come_from_the_tracked_manifests() -> None:
    """The dataset is bound to every source file at once, through the manifests that list them."""
    digests = source_digests_from_manifests()
    assert set(digests) == {"archive_backfill", "news_cryptopanic"}
    assert all(len(value) == 64 for value in digests.values())
    assert digests["archive_backfill"] != digests["news_cryptopanic"]


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_contract_dataset_and_manifest_compose_on_real_data() -> None:
    """The whole chain: frozen contract, real bars, 1 764 rows, a manifest that binds them."""
    contract = load_contract()
    rows = build_decision_day_rows(load_archive_backfill(SNAPSHOT).bars)
    manifest = build_dataset_manifest(
        contract=contract,
        row_count=len(rows),
        first_day=rows[0].day,
        last_day=rows[-1].day,
        source_digests=source_digests_from_manifests(),
    )
    assert manifest.row_count == contract.values["research_window"]["decision_days"] == 1_764
    assert manifest.first_day == contract.values["research_window"]["first_day"]
    assert manifest.last_day == contract.values["research_window"]["last_day"]
    assert verify_manifest_against(manifest, contract) == contract.digest


def test_a_contract_missing_a_section_is_refused(tmp_path: Path) -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace("\npolicy:", "\npolicy_renamed:")
    broken = tmp_path / "experiment.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        load_contract(broken)
    assert caught.value.kind == "contract_inconsistent"


def test_an_empty_contract_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "experiment.yaml"
    empty.write_text("# nothing but a comment\n", encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        load_contract(empty)
    assert caught.value.kind == "contract_inconsistent"


def test_a_mistyped_value_is_refused(tmp_path: Path) -> None:
    """A quoted date is a string; comparing it to a date must refuse, not raise TypeError."""
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        "development_last_day: 2024-12-31", 'development_last_day: "2024-12-31"'
    )
    broken = tmp_path / "experiment.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        load_contract(broken)
    assert caught.value.kind == "contract_inconsistent"


def test_an_absent_contract_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FatalDefect) as caught:
        load_contract(tmp_path / "nowhere.yaml")
    assert caught.value.kind == "contract_inconsistent"


def test_a_manifest_must_name_every_source() -> None:
    """A dataset bound to no sources would pass every other check with no provenance at all."""
    contract = load_contract()
    with pytest.raises(FatalDefect) as caught:
        build_dataset_manifest(
            contract=contract,
            row_count=1,
            first_day=date(2021, 2, 1),
            last_day=date(2021, 2, 1),
            source_digests={},
        )
    assert caught.value.kind == "contract_inconsistent"


def test_sources_changing_under_a_dataset_is_refused() -> None:
    """The contract half of the binding is not the whole binding: the sources are checked too."""
    contract = load_contract()
    manifest = _manifest(contract)
    stale = DatasetManifest(
        contract_digest=manifest.contract_digest,
        row_count=manifest.row_count,
        first_day=manifest.first_day,
        last_day=manifest.last_day,
        source_digests={**manifest.source_digests, "news_cryptopanic": "cc" * 32},
    )
    with pytest.raises(FatalDefect) as caught:
        verify_manifest_against(stale, contract)
    assert caught.value.kind == "source_digest_mismatch"


def test_contract_values_cannot_be_mutated_after_loading() -> None:
    """A mutated value with an unchanged digest would attest to a pre-registration never used."""
    contract = load_contract()
    with pytest.raises(TypeError):
        contract.values["policy"]["to_btc_at"] = 0.99


def test_a_date_and_its_iso_string_are_different_values() -> None:
    assert digest_of({"day": date(2024, 12, 31)}) != digest_of({"day": "2024-12-31"})


def test_a_miscounted_scored_day_total_is_refused(tmp_path: Path) -> None:
    """The first version of this contract was wrong by one day and nobody noticed for four files."""
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        "scored_development_days: 1096", "scored_development_days: 1095"
    )
    broken = tmp_path / "experiment.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        load_contract(broken)
    assert caught.value.kind == "contract_inconsistent"
    assert "1096 days" in str(caught.value)
