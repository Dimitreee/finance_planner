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
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from cryptoguard_api import create_app
from cryptoguard_core.backtest import series_from
from cryptoguard_core.evaluate import published_replays_from
from cryptoguard_core.policy import COST_SCENARIOS, HEADLINE_SCENARIO, Position
from cryptoguard_core.replay import ReplaySeries, SeriesDay, SeriesScenarioDay
from cryptoguard_core.store import PublishedReplay, PublishedRun, RunStore
from fastapi.testclient import TestClient
from test_replay_series import sensitivity as simulated_sensitivity

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
            # One statement, so the foreign keys among the series tables go with them and no
            # CASCADE is needed. Left plain on purpose: a later table referencing one of these and
            # missing from this list fails here rather than surviving into the next test's reads.
            connection.execute(
                "DROP TABLE IF EXISTS published_runs, job_runs, published_replays,"
                " replay_scenario_days, replay_days, replay_series"
            )
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


# One made-up Headline Metric per pre-registered scenario, ordered as costs order them. Keyed by
# name rather than zipped against a list, because a pair aligned by position silently re-labels
# every figure the day a scenario is added; the assertion below is what makes a missing key loud.
NET_RETURNS = {"optimistic": 0.2, "base": -0.1, "pessimistic": -0.6}
assert set(NET_RETURNS) == set(COST_SCENARIOS)


def publish_replay(store: RunStore, *, version: str = "arm-a-logistic-1.0") -> None:
    store.publish_replay(
        [
            PublishedReplay(
                asset=ASSET,
                arm="A",
                model_version=version,
                cost_scenario=name,
                is_headline=name == HEADLINE_SCENARIO,
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
            for name, net in NET_RETURNS.items()
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


# --- The per-day Replay series --------------------------------------------------------------------


def a_series(days: int = 5, *, start: float = 1000.0) -> ReplaySeries:
    """A small series whose three scenarios agree on every decision and differ only in money.

    Shape only. Its money is arithmetic chosen to be readable in a failure message and does not
    reproduce the aggregates `publish_replay` writes — nothing here simulated anything. The tests
    below use it to check what the store and the endpoint do with a series' *structure*; the figures
    are checked against a real Cost Sensitivity in
    `test_the_published_series_reproduces_the_published_aggregate`.
    """
    first = date(2022, 1, 1)
    decisions = tuple(
        SeriesDay(
            day=first + timedelta(days=index),
            price=100.0 + index,
            probability=0.9 if index % 2 == 0 else 0.1,
            action="BUY" if index % 2 == 0 else "REDUCE",
        )
        for index in range(days)
    )
    return ReplaySeries(
        first_day=decisions[0].day,
        last_day=decisions[-1].day,
        start_value_usdt=start,
        headline_scenario="base",
        days=decisions,
        by_scenario={
            name: tuple(
                SeriesScenarioDay(
                    day=entry.day,
                    cost_scenario=name,
                    btc=0.01 * (index + 1),
                    usdt=0.0,
                    value_usdt=start * (1 + factor * (index + 1) / 100),
                    buy_and_hold_usdt=start * (1 + (index + 1) / 100),
                )
                for index, entry in enumerate(decisions)
            )
            for name, factor in (("optimistic", 2.0), ("base", 1.0), ("pessimistic", -1.0))
        },
    )


def test_a_published_series_reads_back_day_for_day(store: RunStore) -> None:
    written = store.publish_replay_series(ASSET, "arm-a-logistic-1.0", a_series())
    publish_replay(store)
    assert written == (5, 15)

    read = store.latest_replay_series(ASSET)
    assert read is not None
    assert [entry.day for entry in read.days] == [entry.day for entry in a_series().days]
    assert [entry.action for entry in read.days] == [entry.action for entry in a_series().days]
    assert set(read.by_scenario) == set(COST_SCENARIOS)
    assert read.start_value_usdt == 1000.0
    assert read.headline_scenario == "base"


def test_publishing_the_same_series_twice_writes_nothing_the_second_time(store: RunStore) -> None:
    """A frozen research result that drifts nightly is not a frozen research result."""
    store.publish_replay_series(ASSET, "arm-a-logistic-1.0", a_series())
    again = store.publish_replay_series(ASSET, "arm-a-logistic-1.0", a_series())
    assert again == (0, 0)


def test_a_series_without_its_aggregates_is_not_reachable(store: RunStore) -> None:
    """The version comes from the aggregates, so a series published alone is invisible."""
    store.publish_replay_series(ASSET, "arm-a-logistic-1.0", a_series())
    assert store.latest_replay_series(ASSET) is None


def test_the_series_of_the_newest_published_version_is_the_one_served(store: RunStore) -> None:
    store.publish_replay_series(ASSET, "arm-a-logistic-1.0", a_series(days=3, start=1000.0))
    publish_replay(store, version="arm-a-logistic-1.0")
    store.publish_replay_series(ASSET, "arm-a-logistic-2.0", a_series(days=4, start=2000.0))
    publish_replay(store, version="arm-a-logistic-2.0")

    read = store.latest_replay_series(ASSET)
    assert read is not None
    assert len(read.days) == 4
    assert read.start_value_usdt == 2000.0


def test_the_endpoint_serves_the_series_with_its_window_and_scenario_names(store: RunStore) -> None:
    store.publish_replay_series(ASSET, "arm-a-logistic-1.0", a_series())
    publish_replay(store)
    body = client(store, date(2026, 9, 28)).get("/api/replay-series").json()

    assert body["model_mode"] == "replay"
    assert body["window"] == {"first_day": "2022-01-01", "last_day": "2022-01-05", "days": 5}
    assert body["cost_scenarios"] == ["base", "optimistic", "pessimistic"]
    assert body["headline_scenario"] == "base"
    assert body["start_value_usdt"] == 1000.0
    assert len(body["days"]) == 5
    assert body["days"][0] == {
        "day": "2022-01-01",
        "price": 100.0,
        "probability": 0.9,
        "action": "BUY",
    }
    assert len(body["scenarios"]["base"]) == 5
    assert set(body["scenarios"]["base"][0]) == {
        "day",
        "btc",
        "usdt",
        "value_usdt",
        "buy_and_hold_usdt",
    }


def test_no_series_published_yet_is_reported_as_empty(store: RunStore) -> None:
    body = client(store, date(2026, 9, 28)).get("/api/replay-series").json()
    assert body["days"] == []
    assert body["scenarios"] == {}
    assert body["window"] is None


# --- The series and the aggregate, both after a round trip ----------------------------------------


def _deepest_trough(values: Sequence[float]) -> float:
    """The largest fall from a running peak, as a fraction of that peak.

    A second implementation of `max_drawdown` on purpose. Reading the figure the simulator produced
    and comparing it to itself would pass for any series at all; recomputing it from the rows the
    store handed back is what makes the comparison mean something.
    """
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, 1 - value / peak)
    return worst


def test_the_published_series_reproduces_the_published_aggregate(store: RunStore) -> None:
    """The invariant across the store, not inside one object: read both back, then recompute.

    `test_replay_series.py` checks the series against its aggregates while both are still one
    `BacktestResult` in memory. The page draws neither of those. It draws what
    `replay_scenario_days` and `published_replays` say, after a write and a read that can round a
    double, order rows by the wrong column, or join a scenario's money to another scenario's name —
    and a switcher showing a curve that ends somewhere other than the figure printed beside it is
    the one defect this panel exists to be incapable of.

    Both sides come from one real Cost Sensitivity, published through the same mapping the research
    job publishes through. If they disagree the series is wrong, not the aggregate: the aggregates
    are the Replay Result on the record.
    """
    simulated = simulated_sensitivity()
    version = "arm-a-logistic-1.0"
    store.publish_replay_series(ASSET, version, series_from(simulated))
    store.publish_replay(
        published_replays_from(
            simulated,
            asset=ASSET,
            arm="A",
            model_version=version,
            window_first_day=simulated.headline.series[0].day,
            window_last_day=simulated.headline.series[-1].day,
            selection_metric="log_loss",
            selection_score=0.69,
            contract_digest="deadbeef",
        )
    )

    series = store.latest_replay_series(ASSET)
    aggregates = {replay.cost_scenario: replay for replay in store.latest_replay(ASSET)}
    assert series is not None
    assert set(series.by_scenario) == set(aggregates) == set(COST_SCENARIOS)

    start = series.start_value_usdt
    assert start == simulated.headline.start_value_usdt
    for name, path in series.by_scenario.items():
        aggregate = aggregates[name]
        values = [point.value_usdt for point in path]
        assert len(path) == aggregate.days, name
        assert values[-1] / start - 1 == pytest.approx(aggregate.net_return), name
        assert path[-1].buy_and_hold_usdt / start - 1 == pytest.approx(
            aggregate.buy_and_hold_return
        ), name
        # The starting capital is the first peak, as it is in the simulator: a run whose worst point
        # is day one must not read as having had no drawdown.
        assert _deepest_trough([start, *values]) == pytest.approx(aggregate.max_drawdown), name
