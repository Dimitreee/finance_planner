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

    return app


__all__ = ["create_app"]
