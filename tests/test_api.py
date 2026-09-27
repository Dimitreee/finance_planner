"""The API seam: an HTTP request against seeded published state, and what comes back.

The three Freshness values and the independence of Model Mode are asserted here rather than in the
page, because they are properties of the response.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import psycopg
import pytest
from cryptoguard_api import create_app
from cryptoguard_core.policy import Position
from cryptoguard_core.store import PublishedReplay, PublishedRun, RunStore
from fastapi.testclient import TestClient

ASSET = "BTCUSDT"
API_PACKAGE = Path(__file__).resolve().parents[1] / "apps/api/cryptoguard_api"
FORBIDDEN_IMPORTS = {
    "cryptoguard_core.model",
    "cryptoguard_core.ingest",
    "cryptoguard_core.dataset",
    "cryptoguard_core.decide",
    "cryptoguard_core.news",
}


@pytest.fixture
def store() -> RunStore:
    dsn = os.environ.get("CRYPTOGUARD_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set CRYPTOGUARD_TEST_DATABASE_URL to run the API seam against PostgreSQL")
    try:
        with psycopg.connect(dsn, connect_timeout=3) as connection:
            connection.execute("DROP TABLE IF EXISTS published_runs, job_runs, published_replays")
            connection.commit()
    except psycopg.OperationalError as error:  # pragma: no cover - environment dependent
        pytest.skip(f"database not reachable: {error}")
    ready = RunStore(dsn)
    ready.migrate()
    return ready


def publish(store: RunStore, day: date, *, action: str = "BUY", mode: str = "price-only") -> None:
    cutoff = datetime(day.year, day.month, day.day, tzinfo=UTC)
    store.publish(
        PublishedRun(
            asset=ASSET,
            decision_day=day,
            feature_cutoff=cutoff,
            decision_time=cutoff.replace(hour=1),
            decision_price=100.0,
            probability=0.95,
            target_exposure=1.0,
            action=action,  # type: ignore[arg-type]
            position_before=Position(btc=0.0, usdt=1000.0),
            position_after=Position(btc=9.0, usdt=0.0),
            trade_notional=999.0 if action != "HOLD" else None,
            trade_fee=1.0 if action != "HOLD" else None,
            trade_price=100.05 if action != "HOLD" else None,
            explanation="the probability crossed the threshold",
            model_version="previous-direction-1.0",
            policy_version="abc123",
            contract_digest="deadbeef",
            model_mode=mode,
            run_bundle_path="/tmp/bundle.json",
        )
    )


def client(store: RunStore, today: date) -> TestClient:
    return TestClient(create_app(store, today=lambda: today))


def test_no_published_run_reports_unavailable_and_offers_no_advice(store: RunStore) -> None:
    body = client(store, date(2026, 9, 28)).get("/api/advice").json()
    assert body["freshness"] == "unavailable"
    assert body["action"] is None
    assert body["explanation"] is None
    assert body["decision_day"] is None


def test_a_run_for_today_is_served_as_current(store: RunStore) -> None:
    publish(store, date(2026, 9, 28))
    body = client(store, date(2026, 9, 28)).get("/api/advice").json()
    assert body["freshness"] == "current"
    assert body["decision_day"] == "2026-09-28"
    # UTC, not the server's zone: the same instant must render identically in every deployment.
    assert body["data_cutoff"] == "2026-09-28T00:00:00+00:00"
    assert body["action"] == "BUY"
    assert body["probability"] == 0.95
    assert body["position"] == {"btc": 9.0, "usdt": 0.0, "value_usdt": 900.0}
    assert body["versions"]["model"] == "previous-direction-1.0"


def test_a_missed_day_is_served_as_stale_with_the_date_being_shown(store: RunStore) -> None:
    publish(store, date(2026, 9, 27))
    body = client(store, date(2026, 9, 28)).get("/api/advice").json()
    assert body["freshness"] == "stale"
    assert body["decision_day"] == "2026-09-27"
    assert body["explanation"]


def test_no_response_describes_a_day_later_than_the_newest_run(store: RunStore) -> None:
    publish(store, date(2026, 9, 26))
    publish(store, date(2026, 9, 27))
    body = client(store, date(2026, 9, 28)).get("/api/advice").json()
    assert body["decision_day"] == "2026-09-27"


def test_model_mode_is_independent_of_freshness(store: RunStore) -> None:
    publish(store, date(2026, 9, 1), mode="fused")
    body = client(store, date(2026, 9, 28)).get("/api/advice").json()
    assert body["freshness"] == "stale"
    assert body["model_mode"] == "fused"


def test_the_paper_track_is_served_with_its_genesis_day(store: RunStore) -> None:
    publish(store, date(2026, 9, 26), action="BUY")
    publish(store, date(2026, 9, 27), action="HOLD")
    body = client(store, date(2026, 9, 28)).get("/api/paper-track").json()
    assert body["genesis_day"] == "2026-09-26"
    assert [point["day"] for point in body["points"]] == ["2026-09-26", "2026-09-27"]
    assert body["points"][0]["value_usdt"] == 900.0
    assert body["points"][0]["price"] == 100.0, "the page needs a price series to draw"


def test_health_answers_while_the_database_is_unreachable() -> None:
    """Liveness must not depend on the database being up, only on the process being alive."""
    unreachable = RunStore("postgresql://nowhere.invalid/none", connect_timeout=1)
    response = TestClient(create_app(unreachable)).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_an_unconfigured_process_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Better than passing a health check and returning 500 to every reader."""
    monkeypatch.delenv("CRYPTOGUARD_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="CRYPTOGUARD_DATABASE_URL"):
        create_app()


def test_the_api_package_imports_no_model_and_no_data_ingestion() -> None:
    """The direct half of the guarantee: nothing in this package names those modules."""
    imported: set[str] = set()
    for source in API_PACKAGE.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                # `from cryptoguard_core import model` names the module in the alias, not the
                # module clause, and would otherwise slip through.
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert imported.isdisjoint(FORBIDDEN_IMPORTS), sorted(imported & FORBIDDEN_IMPORTS)


def test_importing_the_api_pulls_in_no_model_or_ingestion_transitively() -> None:
    """The half the source scan cannot see: what a dependency imports on our behalf."""
    program = (
        "import sys, json; import cryptoguard_api; "
        f"print(json.dumps(sorted(set(sys.modules) & set({sorted(FORBIDDEN_IMPORTS)!r}))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []


def publish_replay(store: RunStore, *, version: str = "arm-a-logistic-1.0") -> None:
    store.publish_replay(
        [
            PublishedReplay(
                asset=ASSET,
                arm="A",
                model_version=version,
                cost_scenario=name,
                is_headline=name == "base",
                window_first_day=date(2022, 1, 1),
                window_last_day=date(2024, 12, 31),
                days=1096,
                trades=400,
                net_return=net,
                buy_and_hold_return=1.5,
                cash_return=0.0,
                max_drawdown=0.4,
                turnover=120.0,
                time_invested=0.5,
                total_fees=90.0,
                selection_metric="log_loss",
                selection_score=0.69,
                contract_digest="deadbeef",
            )
            for name, net in (("optimistic", 0.2), ("base", -0.1), ("pessimistic", -0.6))
        ]
    )


def test_no_replay_published_yet_is_reported_as_empty(store: RunStore) -> None:
    body = client(store, date(2026, 9, 28)).get("/api/replay").json()
    assert body["scenarios"] == []
    assert body["model_version"] is None


def test_the_replay_is_served_with_every_scenario_and_the_headline_first(store: RunStore) -> None:
    publish_replay(store)
    body = client(store, date(2026, 9, 28)).get("/api/replay").json()
    assert body["model_mode"] == "replay"
    assert body["arm"] == "A"
    assert body["window"] == {"first_day": "2022-01-01", "last_day": "2024-12-31", "days": 1096}
    assert body["selection"] == {"metric": "log_loss", "score": 0.69}
    assert [s["name"] for s in body["scenarios"]] == ["base", "optimistic", "pessimistic"]
    assert body["scenarios"][0]["is_headline"] is True
    assert body["scenarios"][0]["cash_return"] == 0.0


def test_publishing_the_same_replay_twice_writes_it_once(store: RunStore) -> None:
    publish_replay(store)
    publish_replay(store)
    body = client(store, date(2026, 9, 28)).get("/api/replay").json()
    assert len(body["scenarios"]) == 3


def test_a_newer_model_version_replaces_the_replay_shown(store: RunStore) -> None:
    publish_replay(store, version="arm-a-logistic-1.0")
    publish_replay(store, version="arm-a-logistic-2.0")
    body = client(store, date(2026, 9, 28)).get("/api/replay").json()
    assert body["model_version"] == "arm-a-logistic-2.0"
    assert len(body["scenarios"]) == 3
