"""Read-only HTTP surface over published state.

This process loads no model weights and holds no data provider credentials: it reads what a job has
already published, and nothing else. Two tests hold that line: one parses this package's imports,
and one imports it in a clean interpreter and inspects `sys.modules`, which also catches anything
pulled in transitively.

Each response is built from a single Published Run, which is why the Explanation lives on the row: a
request cannot pair a new forecast with an older explanation, because there is nothing to pair.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import cryptoguard_core
from cryptoguard_core.protocol import DECISION_DEADLINE_UTC, HORIZON_DAYS
from cryptoguard_core.serving import advice_view, build_paper_track
from cryptoguard_core.store import RunStore
from fastapi import FastAPI

DATABASE_URL_VAR = "CRYPTOGUARD_DATABASE_URL"
DEFAULT_ASSET = "BTCUSDT"


def _today_utc() -> date:
    return datetime.now(tz=UTC).date()


def _store_from_environment() -> RunStore:
    dsn = os.environ.get(DATABASE_URL_VAR)
    if not dsn:
        raise RuntimeError(f"{DATABASE_URL_VAR} is not set")
    return RunStore(dsn)


def create_app(
    store: RunStore | None = None,
    *,
    today: Callable[[], date] = _today_utc,
    asset: str = DEFAULT_ASSET,
) -> FastAPI:
    app = FastAPI(title="CryptoGuard API")
    # Resolved once, at construction: a process started without configuration must fail immediately
    # rather than pass its health check and return 500 to every reader. Connections are still opened
    # per request, so health needs no *reachable* database — only a configured one.
    published = store if store is not None else _store_from_environment()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        # Async on purpose: a sync handler runs in the bounded thread pool that the database-backed
        # endpoints also use, so a stalled database would stop liveness answering as well.
        return {"status": "ok", "core_version": cryptoguard_core.__version__}

    @app.get("/api/advice")
    def advice() -> dict[str, Any]:
        view = advice_view(published.latest_published(asset), today())
        return {
            "asset": asset,
            # Protocol constants the page states to the reader. Serving them keeps the client from
            # holding a second copy of values the contract already fixes.
            "protocol": {
                "horizon_days": HORIZON_DAYS,
                "decision_deadline_utc": DECISION_DEADLINE_UTC,
            },
            "freshness": view.freshness,
            "model_mode": view.model_mode,
            "decision_day": view.decision_day.isoformat() if view.decision_day else None,
            "data_cutoff": view.feature_cutoff.isoformat() if view.feature_cutoff else None,
            "decision_price": view.decision_price,
            "probability": view.probability,
            "action": view.action,
            "target_exposure": view.target_exposure,
            "position": {
                "btc": view.position_btc,
                "usdt": view.position_usdt,
                "value_usdt": view.portfolio_value_usdt,
            },
            "explanation": view.explanation,
            "versions": {
                "model": view.model_version,
                "policy": view.policy_version,
                "contract": view.contract_digest,
            },
        }

    @app.get("/api/paper-track")
    def paper_track() -> dict[str, Any]:
        track = build_paper_track(published.paper_track(asset))
        return {
            "asset": asset,
            "genesis_day": track.genesis_day.isoformat() if track.genesis_day else None,
            "points": [
                {
                    "day": point.day.isoformat(),
                    "action": point.action,
                    "btc": point.btc,
                    "usdt": point.usdt,
                    "price": point.price,
                    "value_usdt": point.value_usdt,
                }
                for point in track.points
            ],
        }

    @app.get("/api/replay-series")
    def replay_series() -> dict[str, Any]:
        """The day-by-day Replay behind the aggregates, for the curves and the underwater view.

        Its `model_mode` and `model_version` come from the aggregates rather than the series, so a
        reader can never be shown a curve belonging to one model beside figures from another. The
        Cost Scenario names are served rather than assumed by the page: a switcher hard-coding
        them would keep offering a scenario the contract had dropped.
        """
        series = published.latest_replay_series(asset)
        scenarios = published.latest_replay(asset)
        first = scenarios[0] if scenarios else None
        # The unpublished shape first, and the published one as an overlay on it. Written as two
        # complete dicts it was two hand-maintained key sets, and the failure they permit is silent:
        # a key added to one and forgotten in the other makes a field the page can read on a normal
        # day and cannot read on an empty one, which is the day nobody tests by hand.
        #
        # Nulls rather than zeros for the three scalars. A window of zero days and a starting
        # capital # of 0 are both readable as measurements, and neither was measured.
        empty: dict[str, Any] = {
            "asset": asset,
            "model_mode": first.model_mode if first else None,
            "model_version": first.model_version if first else None,
            "window": None,
            "start_value_usdt": None,
            "headline_scenario": None,
            "cost_scenarios": [],
            "days": [],
            "scenarios": {},
        }
        if series is None:
            return empty
        return empty | {
            "window": {
                "first_day": series.first_day.isoformat(),
                "last_day": series.last_day.isoformat(),
                "days": len(series.days),
            },
            "start_value_usdt": series.start_value_usdt,
            "headline_scenario": series.headline_scenario,
            "cost_scenarios": sorted(series.by_scenario),
            "days": [
                {
                    "day": entry.day.isoformat(),
                    "price": entry.price,
                    "probability": entry.probability,
                    "action": entry.action,
                }
                for entry in series.days
            ],
            "scenarios": {
                name: [
                    {
                        "day": value.day.isoformat(),
                        "btc": value.btc,
                        "usdt": value.usdt,
                        "value_usdt": value.value_usdt,
                        "buy_and_hold_usdt": value.buy_and_hold_usdt,
                    }
                    for value in values
                ]
                for name, values in series.by_scenario.items()
            },
        }

    @app.get("/api/replay")
    def replay() -> dict[str, Any]:
        scenarios = published.latest_replay(asset)
        first = scenarios[0] if scenarios else None
        return {
            "asset": asset,
            "arm": first.arm if first else None,
            "model_version": first.model_version if first else None,
            "model_mode": first.model_mode if first else None,
            "window": (
                {
                    "first_day": first.window_first_day.isoformat(),
                    "last_day": first.window_last_day.isoformat(),
                    "days": first.days,
                }
                if first
                else None
            ),
            "selection": (
                {"metric": first.selection_metric, "score": first.selection_score}
                if first
                else None
            ),
            "scenarios": [
                {
                    "name": scenario.cost_scenario,
                    "is_headline": scenario.is_headline,
                    "net_return": scenario.net_return,
                    "buy_and_hold_return": scenario.buy_and_hold_return,
                    "cash_return": scenario.cash_return,
                    "max_drawdown": scenario.max_drawdown,
                    "turnover": scenario.turnover,
                    "trades": scenario.trades,
                    "time_invested": scenario.time_invested,
                    "total_fees": scenario.total_fees,
                }
                for scenario in scenarios
            ],
        }

    return app


__all__ = ["create_app"]
