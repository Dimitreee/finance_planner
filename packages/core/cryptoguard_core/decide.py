"""The decision job: from a Decision Day row to one durable, dated piece of advice.

Running it twice for the same day does nothing the second time, and that is a property of the
schema rather than of care taken here: the run and the position transition are one row under a
unique key. A Fatal Defect anywhere leaves the previous Published Run untouched, because nothing is
published until every step has succeeded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptoguard_core.contract import ExperimentContract, digest_of
from cryptoguard_core.dataset import DecisionDayRow
from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.model import ModelRelease, forecast_with
from cryptoguard_core.policy import (
    CostScenario,
    PolicyConfig,
    Position,
    RuleTrace,
    decide_target_exposure,
    explain,
    rebalance,
)
from cryptoguard_core.store import PublishedRun, RunStore

DEFAULT_ASSET = "BTCUSDT"
PRICE_ONLY = "price-only"


@dataclass(frozen=True, slots=True)
class JobOutcome:
    attempt_id: int
    published: bool
    run: PublishedRun


def policy_from(contract: ExperimentContract) -> PolicyConfig:
    policy = contract.values["policy"]
    return PolicyConfig(to_btc_at=policy["to_btc_at"], to_usdt_at=policy["to_usdt_at"])


def policy_version_of(contract: ExperimentContract) -> str:
    """Derived from the Policy's own values: changing a threshold starts a new strategy track."""
    return digest_of(contract.values["policy"])[:12]


def genesis_from(contract: ExperimentContract) -> Position:
    policy = contract.values["policy"]
    if policy["initial_position"] != "usdt":
        raise FatalDefect(
            "contract_inconsistent", "the Virtual Portfolio starts in USDT in this version"
        )
    return Position(btc=0.0, usdt=float(policy["initial_capital_usdt"]))


def headline_costs_from(contract: ExperimentContract) -> CostScenario:
    scenarios = contract.values["cost_scenarios"]
    name = scenarios["headline"]
    scenario = scenarios[name]
    return CostScenario(
        name=name, fee_pct=scenario["fee_pct"], slippage_bps=scenario["slippage_bps"]
    )


def run_bundle_path(
    bundle_dir: Path, *, asset: str, day: date, model_version: str, policy_version: str
) -> Path:
    """Deterministic and version-scoped, so two releases cannot overwrite each other's bundle."""
    return bundle_dir / asset / f"{day.isoformat()}__{model_version}__{policy_version}.json"


def write_run_bundle(
    path: Path,
    *,
    asset: str,
    day: date,
    row: DecisionDayRow,
    trace: RuleTrace,
    model_version: str,
    contract: ExperimentContract,
    costs: CostScenario,
) -> Path:
    """The immutable artifact a Published Run points at.

    What reproduction needs, not what the page needs. Written only after a successful publish, so a
    skipped rerun cannot overwrite the bundle of the run that did publish.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "asset": asset,
        "decision_day": day.isoformat(),
        "feature_snapshot": dict(row.features),
        "anchor_price": row.anchor_price,
        "decision_price": row.decision_price,
        "bars_in_day": row.bars_in_day,
        "model_version": model_version,
        "contract_digest": contract.digest,
        "cost_scenario": {
            "name": costs.name,
            "fee_pct": costs.fee_pct,
            "slippage_bps": costs.slippage_bps,
        },
        "rule_trace": {
            "probability": trace.probability,
            "to_btc_at": trace.to_btc_at,
            "to_usdt_at": trace.to_usdt_at,
            "current_exposure": trace.current_exposure,
            "target_exposure": trace.target_exposure,
            "rule": trace.rule,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_decision_job(
    *,
    store: RunStore,
    contract: ExperimentContract,
    model: ModelRelease,
    rows: tuple[DecisionDayRow, ...],
    day: date,
    bundle_dir: Path,
    asset: str = DEFAULT_ASSET,
    model_mode: str = PRICE_ONLY,
) -> JobOutcome:
    # Both instants come from the Decision Day itself, not from a row, so the attempt can be
    # recorded before anything that might fail. A crash that left no trace in job_runs would defeat
    # the table's purpose.
    feature_cutoff = datetime(day.year, day.month, day.day, tzinfo=UTC)
    decision_time = feature_cutoff + timedelta(milliseconds=HOUR_MS)
    attempt_id = store.start_attempt(asset, decision_time)

    try:
        with store.asset_lock(asset):
            row = {candidate.day: candidate for candidate in rows}.get(day)
            if row is None:
                raise FatalDefect(
                    "missing_decision_day_row", f"the dataset holds no Decision Day row for {day}"
                )
            position_before = store.current_position(asset, genesis_from(contract))
            costs = headline_costs_from(contract)
            forecast = forecast_with(model, row.features)
            trace = decide_target_exposure(
                forecast.probability,
                current_exposure=position_before.exposure(row.decision_price),
                policy=policy_from(contract),
            )
            result = rebalance(position_before, trace.target_exposure, row.decision_price, costs)
            policy_version = policy_version_of(contract)
            bundle_path = run_bundle_path(
                bundle_dir,
                asset=asset,
                day=day,
                model_version=forecast.model_version,
                policy_version=policy_version,
            )
            traded = result.action != "HOLD"
            run = PublishedRun(
                asset=asset,
                decision_day=day,
                feature_cutoff=feature_cutoff,
                decision_time=decision_time,
                decision_price=row.decision_price,
                probability=forecast.probability,
                target_exposure=trace.target_exposure,
                action=result.action,
                position_before=position_before,
                position_after=result.position_after,
                trade_notional=result.notional if traded else None,
                trade_fee=result.fee if traded else None,
                trade_price=result.execution_price if traded else None,
                explanation=explain(trace, result.action),
                model_version=forecast.model_version,
                policy_version=policy_version,
                contract_digest=contract.digest,
                model_mode=model_mode,
                run_bundle_path=str(bundle_path),
            )
            published = store.publish(run)
            if published:
                write_run_bundle(
                    bundle_path,
                    asset=asset,
                    day=day,
                    row=row,
                    trace=trace,
                    model_version=forecast.model_version,
                    contract=contract,
                    costs=costs,
                )
    except FatalDefect as defect:
        store.finish_attempt(
            attempt_id, "failed", failure_kind=defect.kind, failure_detail=str(defect)
        )
        raise
    except BaseException as error:
        # Anything else — a database error, a missing feature — must still close the attempt.
        # A row left at 'running' for ever is indistinguishable from a job still in flight.
        store.finish_attempt(
            attempt_id, "failed", failure_kind=type(error).__name__, failure_detail=repr(error)
        )
        raise

    store.finish_attempt(attempt_id, "published" if published else "skipped")
    return JobOutcome(attempt_id=attempt_id, published=published, run=run)
