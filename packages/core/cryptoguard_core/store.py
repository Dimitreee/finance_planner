"""Published state in PostgreSQL: one row per Decision Day, and every attempt to produce one.

The Virtual Portfolio's position is carried forward in these rows. What prevents a retry booking a
second virtual trade is the unique key in the schema, not care taken in the job: the run and the
position transition are the same row, written in one statement.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import psycopg

from cryptoguard_core.policy import Action, Position

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

AttemptStatus = str  # running | published | skipped | failed


@dataclass(frozen=True, slots=True)
class PublishedRun:
    """Everything a Decision Day is served from."""

    asset: str
    decision_day: date
    feature_cutoff: datetime
    decision_time: datetime
    probability: float
    target_exposure: float
    action: Action
    position_before: Position
    position_after: Position
    trade_notional: float | None
    trade_fee: float | None
    trade_price: float | None
    explanation: str
    model_version: str
    policy_version: str
    contract_digest: str
    model_mode: str
    run_bundle_path: str


class RunStore:
    """Every read and write of published state. One connection per operation; the job is a batch."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    @property
    def dsn(self) -> str:
        return self._dsn

    @contextmanager
    def asset_lock(self, asset: str) -> Iterator[None]:
        """Hold the right to advance this asset's position chain.

        The unique key stops a repeated day. It does not stop two jobs for *different* days reading
        the same position and both publishing, which would silently discard one transition. This
        serialises the read-modify-write that the schema cannot.
        """
        with psycopg.connect(self._dsn) as connection:
            connection.execute("SELECT pg_advisory_lock(hashtext(%s))", (asset,))
            connection.commit()
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (asset,))
                connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self._dsn) as connection:
            yield connection

    def migrate(self) -> None:
        with self._connect() as connection:
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                connection.execute(path.read_text(encoding="utf-8"))
            connection.commit()

    def start_attempt(self, asset: str, decision_time: datetime) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "INSERT INTO job_runs (asset, decision_time, status) "
                "VALUES (%s, %s, 'running') RETURNING id",
                (asset, decision_time),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("the database did not return an id for the new job_runs row")
        return int(row[0])

    def finish_attempt(
        self,
        attempt_id: int,
        status: AttemptStatus,
        *,
        failure_kind: str | None = None,
        failure_detail: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE job_runs SET status = %s, finished_at = now(), "
                "failure_kind = %s, failure_detail = %s WHERE id = %s",
                (status, failure_kind, failure_detail, attempt_id),
            )
            connection.commit()

    def attempts(self, asset: str) -> list[tuple[str, str | None]]:
        """Status and failure kind of every attempt, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, failure_kind FROM job_runs WHERE asset = %s ORDER BY id DESC",
                (asset,),
            ).fetchall()
        return [(str(status), kind) for status, kind in rows]

    def latest_published(self, asset: str) -> PublishedRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT asset, decision_day, feature_cutoff, decision_time, probability, "
                "target_exposure, action, position_before_btc, position_before_usdt, "
                "position_after_btc, position_after_usdt, trade_notional, trade_fee, trade_price, "
                "explanation, model_version, policy_version, contract_digest, model_mode, "
                "run_bundle_path FROM published_runs WHERE asset = %s "
                # decision_time is a function of the day, so it cannot break a tie on its own.
                "ORDER BY decision_time DESC, id DESC LIMIT 1",
                (asset,),
            ).fetchone()
        if row is None:
            return None
        return PublishedRun(
            asset=row[0],
            decision_day=row[1],
            feature_cutoff=row[2],
            decision_time=row[3],
            probability=row[4],
            target_exposure=row[5],
            action=row[6],
            position_before=Position(btc=row[7], usdt=row[8]),
            position_after=Position(btc=row[9], usdt=row[10]),
            trade_notional=row[11],
            trade_fee=row[12],
            trade_price=row[13],
            explanation=row[14],
            model_version=row[15],
            policy_version=row[16],
            contract_digest=row[17],
            model_mode=row[18],
            run_bundle_path=row[19],
        )

    def published_count(self, asset: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT count(*) FROM published_runs WHERE asset = %s", (asset,)
            ).fetchone()
        return int(row[0]) if row else 0

    def current_position(self, asset: str, genesis: Position) -> Position:
        """The position the last Published Run left behind, or genesis if there is none."""
        latest = self.latest_published(asset)
        return genesis if latest is None else latest.position_after

    def publish(self, run: PublishedRun) -> bool:
        """Write the run and its position transition. False means this key was already published."""
        with self._connect() as connection:
            row = connection.execute(
                "INSERT INTO published_runs ("
                " asset, decision_day, feature_cutoff, decision_time, probability, target_exposure,"
                " action, position_before_btc, position_before_usdt, position_after_btc,"
                " position_after_usdt, trade_notional, trade_fee, trade_price, explanation,"
                " model_version, policy_version, contract_digest, model_mode, run_bundle_path"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                # Any conflict means this Decision Day is already published: on the day-level
                # constraint or on the version-scoped one, the answer is the same.
                " %s, %s) ON CONFLICT DO NOTHING RETURNING id",
                (
                    run.asset,
                    run.decision_day,
                    run.feature_cutoff,
                    run.decision_time,
                    run.probability,
                    run.target_exposure,
                    run.action,
                    run.position_before.btc,
                    run.position_before.usdt,
                    run.position_after.btc,
                    run.position_after.usdt,
                    run.trade_notional,
                    run.trade_fee,
                    run.trade_price,
                    run.explanation,
                    run.model_version,
                    run.policy_version,
                    run.contract_digest,
                    run.model_mode,
                    run.run_bundle_path,
                ),
            ).fetchone()
            connection.commit()
        return row is not None
