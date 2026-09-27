"""The three Python packages exist and the two entry points genuinely reach the shared core."""

from __future__ import annotations

import json

import cryptoguard_core
import pytest
from cryptoguard_api import create_app
from cryptoguard_core.store import RunStore
from cryptoguard_jobs.cli import main
from fastapi.testclient import TestClient


def test_core_reports_a_version() -> None:
    assert cryptoguard_core.__version__


def test_api_serves_health_carrying_the_core_version() -> None:
    # A store is injected because the app refuses to start unconfigured; the point here is only
    # that the API app genuinely reaches the shared core.
    client = TestClient(
        create_app(RunStore("postgresql://nowhere.invalid/none", connect_timeout=1))
    )
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "core_version": cryptoguard_core.__version__}


def test_job_entry_point_reports_the_core_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["core_version"] == cryptoguard_core.__version__


def test_job_entry_point_rejects_unknown_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    """A mistyped subcommand must not read as a successful run once this is wired into cron."""
    exit_code = main(["--no-such-flag"])
    assert exit_code != 0
    assert "--no-such-flag" in capsys.readouterr().err
