"""What the read path serves: the advice being shown, how old it is, and the Paper Track.

Freshness and Model Mode are separate axes on purpose. Freshness answers whether today's decision
exists; Model Mode answers which model produced whichever decision is being shown. Collapsing them
into one word would make a fresh price-only run and a stale fused run indistinguishable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from cryptoguard_core.policy import Action
from cryptoguard_core.store import PublishedRun

Freshness = Literal["current", "stale", "unavailable"]


@dataclass(frozen=True, slots=True)
class AdviceView:
    """Everything the advice panel needs, from one Published Run and nothing else."""

    freshness: Freshness
    model_mode: str | None
    decision_day: date | None
    feature_cutoff: datetime | None
    decision_price: float | None
    probability: float | None
    action: Action | None
    target_exposure: float | None
    position_btc: float | None
    position_usdt: float | None
    portfolio_value_usdt: float | None
    explanation: str | None
    model_version: str | None
    policy_version: str | None
    contract_digest: str | None


@dataclass(frozen=True, slots=True)
class PaperTrackPoint:
    day: date
    action: Action
    btc: float
    usdt: float
    price: float
    value_usdt: float


@dataclass(frozen=True, slots=True)
class PaperTrack:
    """The live sequence since go-live. Evidence that the pipeline works, not of profitability."""

    genesis_day: date | None
    points: tuple[PaperTrackPoint, ...]


def freshness_of(latest: PublishedRun | None, today: date) -> Freshness:
    """One rule. A later day can only come from a deliberate backfill and is not stale."""
    if latest is None:
        return "unavailable"
    return "current" if latest.decision_day >= today else "stale"


def advice_view(latest: PublishedRun | None, today: date) -> AdviceView:
    freshness = freshness_of(latest, today)
    if latest is None:
        return AdviceView(
            freshness=freshness,
            model_mode=None,
            decision_day=None,
            feature_cutoff=None,
            decision_price=None,
            probability=None,
            action=None,
            target_exposure=None,
            position_btc=None,
            position_usdt=None,
            portfolio_value_usdt=None,
            explanation=None,
            model_version=None,
            policy_version=None,
            contract_digest=None,
        )
    return AdviceView(
        freshness=freshness,
        model_mode=latest.model_mode,
        decision_day=latest.decision_day,
        feature_cutoff=latest.feature_cutoff,
        decision_price=latest.decision_price,
        probability=latest.probability,
        action=latest.action,
        target_exposure=latest.target_exposure,
        position_btc=latest.position_after.btc,
        position_usdt=latest.position_after.usdt,
        portfolio_value_usdt=latest.position_after.value(latest.decision_price),
        explanation=latest.explanation,
        model_version=latest.model_version,
        policy_version=latest.policy_version,
        contract_digest=latest.contract_digest,
    )


TrackRow = tuple[date, Action, float, float, float]
"""(day, action, btc after, usdt after, Decision Price) — one published day, oldest first."""


def build_paper_track(rows: Iterable[TrackRow]) -> PaperTrack:
    points = tuple(
        PaperTrackPoint(
            day=day, action=action, btc=btc, usdt=usdt, price=price, value_usdt=btc * price + usdt
        )
        for day, action, btc, usdt, price in rows
    )
    return PaperTrack(genesis_day=points[0].day if points else None, points=points)
