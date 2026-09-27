"""Read-only HTTP surface over published state.

This process loads no model weights and holds no data provider credentials: it reads what a
job has already published.
"""

from __future__ import annotations

import cryptoguard_core
from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="CryptoGuard API")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "core_version": cryptoguard_core.__version__}

    return app


__all__ = ["create_app"]
