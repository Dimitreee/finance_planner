"""The job seam: bars already turned into rows, in; one Published Run, out.

These run against a real PostgreSQL because the two properties that matter most here — the unique
key and the single-statement position transition — are properties of the schema. A fake store would
assert that the fake behaves, which is not the question. They skip when no database is reachable.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import psycopg
import pytest
from cryptoguard_core.contract import ExperimentContract, load_contract
from cryptoguard_core.dataset import DAY_MS, DecisionDayRow
from cryptoguard_core.decide import run_decision_job
from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.store import RunStore

ASSET = "BTCUSDT"
FIRST_DAY = date(2021, 2, 1)
FIRST_CUTOFF_MS = int(datetime(2021, 2, 1, tzinfo=UTC).timestamp() * 1000)


def row_for(offset: int, *, r_1: float, decision_price: float) -> DecisionDayRow:
    cutoff = FIRST_CUTOFF_MS + offset * DAY_MS
    return DecisionDayRow(
        day=datetime.fromtimestamp(cutoff / 1000, tz=UTC).date(),
        feature_cutoff_ms=cutoff,
        anchor_price=decision_price,
        decision_price=decision_price,
        label=1,
        label_end_ms=cutoff + DAY_MS + HOUR_MS,
        bars_in_day=24,
        features={
            "r_1": r_1,
            "r_3": 0.0,
            "r_7": 0.0,
            "r_14": 0.0,
            "rv_7": 0.01,
            "rv_7_over_30": 1.0,
            "volume_7_over_30": 1.0,
        },
    )


@pytest.fixture
def store() -> RunStore:
    dsn = os.environ.get("CRYPTOGUARD_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set CRYPTOGUARD_TEST_DATABASE_URL to run the job seam against PostgreSQL")
    try:
        with psycopg.connect(dsn, connect_timeout=3) as connection:
            connection.execute("DROP TABLE IF EXISTS published_runs, job_runs, published_replays")
            connection.commit()
    except psycopg.OperationalError as error:  # pragma: no cover - environment dependent
        pytest.skip(f"database not reachable: {error}")
    ready = RunStore(dsn)
    ready.migrate()
    return ready


@pytest.fixture
def contract() -> ExperimentContract:
    return load_contract()


def run(
    store: RunStore,
    contract: ExperimentContract,
    rows: tuple[DecisionDayRow, ...],
    day: date,
    bundle_dir: Path,
) -> object:
    return run_decision_job(
        store=store,
        contract=contract,
        model=PreviousDirectionBaseline(),
        rows=rows,
        day=day,
        bundle_dir=bundle_dir,
    )


def test_the_first_day_buys_from_genesis(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    outcome = run(store, contract, rows, FIRST_DAY, tmp_path)

    assert outcome.published is True  # type: ignore[attr-defined]
    published = store.latest_published(ASSET)
    assert published is not None
    assert published.action == "BUY"
    assert published.position_before.usdt == 1000.0
    assert published.position_before.btc == 0.0
    assert published.position_after.usdt == 0.0
    assert published.position_after.btc > 0.0
    assert published.probability == 0.95
    assert published.trade_fee is not None and published.trade_fee > 0.0
    assert published.decision_time == datetime(2021, 2, 1, 1, tzinfo=UTC)


def test_running_the_same_day_twice_publishes_once_and_trades_once(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    first = run(store, contract, rows, FIRST_DAY, tmp_path)
    second = run(store, contract, rows, FIRST_DAY, tmp_path)

    assert first.published is True  # type: ignore[attr-defined]
    assert second.published is False  # type: ignore[attr-defined]
    assert store.published_count(ASSET) == 1
    assert [status for status, _ in store.attempts(ASSET)] == ["skipped", "published"]


def test_the_position_carries_forward_and_a_fall_reduces(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (
        row_for(0, r_1=0.01, decision_price=100.0),
        row_for(1, r_1=-0.01, decision_price=120.0),
    )
    run(store, contract, rows, FIRST_DAY, tmp_path)
    bought = store.latest_published(ASSET)
    assert bought is not None

    run(store, contract, rows, date(2021, 2, 2), tmp_path)
    sold = store.latest_published(ASSET)
    assert sold is not None
    assert sold.action == "REDUCE"
    assert sold.position_before == bought.position_after
    assert sold.position_after.btc == 0.0
    assert sold.position_after.usdt > 1000.0, "a 20% rise should outrun two rounds of costs"


def test_an_unchanged_target_holds_and_books_no_trade(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (
        row_for(0, r_1=0.01, decision_price=100.0),
        row_for(1, r_1=0.02, decision_price=110.0),
    )
    run(store, contract, rows, FIRST_DAY, tmp_path)
    bought = store.latest_published(ASSET)
    run(store, contract, rows, date(2021, 2, 2), tmp_path)
    held = store.latest_published(ASSET)

    assert bought is not None and held is not None
    assert held.action == "HOLD"
    assert held.trade_notional is None
    assert held.trade_fee is None
    assert held.position_after == bought.position_after


def test_a_missing_row_fails_the_attempt_and_publishes_nothing(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    run(store, contract, rows, FIRST_DAY, tmp_path)
    before = store.latest_published(ASSET)

    with pytest.raises(FatalDefect) as caught:
        run(store, contract, rows, date(2021, 2, 9), tmp_path)
    assert caught.value.kind == "missing_decision_day_row"

    assert store.published_count(ASSET) == 1
    assert store.latest_published(ASSET) == before
    statuses = store.attempts(ASSET)
    assert statuses[0] == ("failed", "missing_decision_day_row")


def test_the_run_bundle_records_what_reproduction_needs(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    run(store, contract, rows, FIRST_DAY, tmp_path)
    published = store.latest_published(ASSET)
    assert published is not None

    bundle = json.loads(Path(published.run_bundle_path).read_text(encoding="utf-8"))
    assert bundle["feature_snapshot"]["r_1"] == 0.01
    assert bundle["contract_digest"] == contract.digest
    assert bundle["model_version"] == published.model_version
    assert bundle["rule_trace"]["rule"] == "enter_btc"
    assert bundle["cost_scenario"]["name"] == "base"


def test_the_explanation_names_only_what_was_used(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    run(store, contract, rows, FIRST_DAY, tmp_path)
    published = store.latest_published(ASSET)
    assert published is not None
    assert "0.95" in published.explanation
    for absent in ("news", "sentiment", "headline"):
        assert absent not in published.explanation.lower()


def test_a_promoted_model_may_not_republish_a_day_already_decided(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    """One shared position chain means one row per Decision Day, whatever the model version."""
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    run(store, contract, rows, FIRST_DAY, tmp_path)
    before = store.latest_published(ASSET)

    outcome = run_decision_job(
        store=store,
        contract=contract,
        model=PreviousDirectionBaseline(version="previous-direction-2.0"),
        rows=rows,
        day=FIRST_DAY,
        bundle_dir=tmp_path,
    )
    assert outcome.published is False
    assert store.published_count(ASSET) == 1
    assert store.latest_published(ASSET) == before


def test_a_skipped_run_does_not_write_a_bundle(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    rows = (row_for(0, r_1=0.01, decision_price=100.0),)
    run(store, contract, rows, FIRST_DAY, tmp_path)
    written_first = sorted(p.name for p in (tmp_path / ASSET).iterdir())

    run_decision_job(
        store=store,
        contract=contract,
        model=PreviousDirectionBaseline(version="previous-direction-2.0"),
        rows=rows,
        day=FIRST_DAY,
        bundle_dir=tmp_path,
    )
    assert sorted(p.name for p in (tmp_path / ASSET).iterdir()) == written_first


def test_the_asset_lock_excludes_a_second_holder(store: RunStore) -> None:
    """Two jobs for different days must not both read the same position and both publish."""
    with store.asset_lock(ASSET), psycopg.connect(store.dsn) as other:
        row = other.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (ASSET,)).fetchone()
        assert row is not None and row[0] is False

    with psycopg.connect(store.dsn) as after:
        row = after.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (ASSET,)).fetchone()
        assert row is not None and row[0] is True


def test_an_unexpected_error_still_closes_the_attempt(
    store: RunStore, contract: ExperimentContract, tmp_path: Path
) -> None:
    """A row left at 'running' is indistinguishable from a job still in flight."""
    incomplete = row_for(0, r_1=0.0, decision_price=100.0)
    without_r1 = DecisionDayRow(
        day=incomplete.day,
        feature_cutoff_ms=incomplete.feature_cutoff_ms,
        anchor_price=incomplete.anchor_price,
        decision_price=incomplete.decision_price,
        label=incomplete.label,
        label_end_ms=incomplete.label_end_ms,
        bars_in_day=incomplete.bars_in_day,
        features={"rv_7": 0.01},
    )
    with pytest.raises(KeyError):
        run(store, contract, (without_r1,), FIRST_DAY, tmp_path)
    assert store.attempts(ASSET)[0] == ("failed", "KeyError")
    assert store.published_count(ASSET) == 0
