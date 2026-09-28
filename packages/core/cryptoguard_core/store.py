"""Published state in PostgreSQL: one row per Decision Day, and every attempt to produce one.

The Virtual Portfolio's position is carried forward in these rows. What prevents a retry booking a
second virtual trade is the unique key in the schema, not care taken in the job: the run and the
position transition are the same row, written in one statement.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

import psycopg

from cryptoguard_core.policy import Action, Position
from cryptoguard_core.replay import ReplaySeries, SeriesDay, SeriesScenarioDay

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

AttemptStatus = str  # running | published | skipped | failed


@dataclass(frozen=True, slots=True)
class PublishedRun:
    """Everything a Decision Day is served from."""

    asset: str
    decision_day: date
    feature_cutoff: datetime
    decision_time: datetime
    decision_price: float
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


@dataclass(frozen=True, slots=True)
class PublishedReplay:
    """One Cost Scenario of a frozen backtest. Never joined to the Paper Track."""

    asset: str
    arm: str
    model_version: str
    cost_scenario: str
    is_headline: bool
    window_first_day: date
    window_last_day: date
    days: int
    trades: int
    net_return: float
    buy_and_hold_return: float
    cash_return: float
    max_drawdown: float
    turnover: float
    time_invested: float
    total_fees: float
    selection_metric: str
    selection_score: float
    contract_digest: str
    model_mode: str = "replay"


class RunStore:
    """Every read and write of published state. One connection per operation; the job is a batch."""

    # Two different stalls need two different bounds. `connect_timeout` covers a host that never
    # completes the handshake; `statement_timeout` covers one that accepts the connection and then
    # never answers. Without the second, a lock wait would hold a request thread indefinitely.
    CONNECT_TIMEOUT_SECONDS = 5
    STATEMENT_TIMEOUT_MS = 10_000

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout: int = CONNECT_TIMEOUT_SECONDS,
        statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
    ) -> None:
        self._dsn = dsn
        self._connect_timeout = connect_timeout
        self._options = f"-c timezone=UTC -c statement_timeout={statement_timeout_ms}"

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
        # The lock is held for the whole job, which is longer than a statement timeout should allow,
        # so this connection sets only the zone.
        with psycopg.connect(
            self._dsn, options="-c timezone=UTC", connect_timeout=self._connect_timeout
        ) as connection:
            connection.execute("SELECT pg_advisory_lock(hashtext(%s))", (asset,))
            connection.commit()
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (asset,))
                connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        # Every instant in this project is defined in UTC. Without pinning the session timezone,
        # `timestamptz` comes back rendered in the server's zone, so the same instant would be
        # served as 03:00+03:00 in one deployment and 00:00+00:00 in another.
        with psycopg.connect(
            self._dsn, options=self._options, connect_timeout=self._connect_timeout
        ) as connection:
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
                "SELECT asset, decision_day, feature_cutoff, decision_time, decision_price, "
                "probability, "
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
            decision_price=row[4],
            probability=row[5],
            target_exposure=row[6],
            action=row[7],
            position_before=Position(btc=row[8], usdt=row[9]),
            position_after=Position(btc=row[10], usdt=row[11]),
            trade_notional=row[12],
            trade_fee=row[13],
            trade_price=row[14],
            explanation=row[15],
            model_version=row[16],
            policy_version=row[17],
            contract_digest=row[18],
            model_mode=row[19],
            run_bundle_path=row[20],
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

    def paper_track(self, asset: str) -> list[tuple[date, Action, float, float, float]]:
        """Every published day, oldest first: what the live track is made of."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT decision_day, action, position_after_btc, position_after_usdt, "
                "decision_price FROM published_runs WHERE asset = %s "
                "ORDER BY decision_day ASC, id ASC",
                (asset,),
            ).fetchall()
        return [
            (day, action, float(btc), float(usdt), float(price))
            for day, action, btc, usdt, price in rows
        ]

    def publish_replay(self, replays: Sequence[PublishedReplay]) -> int:
        """Write a frozen backtest. Already-published scenarios are left alone."""
        written = 0
        with self._connect() as connection:
            for replay in replays:
                row = connection.execute(
                    "INSERT INTO published_replays ("
                    " asset, arm, model_version, cost_scenario, is_headline, window_first_day,"
                    " window_last_day, days, trades, net_return, buy_and_hold_return, cash_return,"
                    " max_drawdown, turnover, time_invested, total_fees, selection_metric,"
                    " selection_score, contract_digest, model_mode"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                    " %s, %s, %s) ON CONFLICT DO NOTHING RETURNING id",
                    (
                        replay.asset,
                        replay.arm,
                        replay.model_version,
                        replay.cost_scenario,
                        replay.is_headline,
                        replay.window_first_day,
                        replay.window_last_day,
                        replay.days,
                        replay.trades,
                        replay.net_return,
                        replay.buy_and_hold_return,
                        replay.cash_return,
                        replay.max_drawdown,
                        replay.turnover,
                        replay.time_invested,
                        replay.total_fees,
                        replay.selection_metric,
                        replay.selection_score,
                        replay.contract_digest,
                        replay.model_mode,
                    ),
                ).fetchone()
                written += row is not None
            connection.commit()
        return written

    def latest_replay(self, asset: str) -> tuple[PublishedReplay, ...]:
        """Every Cost Scenario of the most recently published backtest for this asset."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT asset, arm, model_version, cost_scenario, is_headline, window_first_day,"
                " window_last_day, days, trades, net_return, buy_and_hold_return, cash_return,"
                " max_drawdown, turnover, time_invested, total_fees, selection_metric,"
                " selection_score, contract_digest, model_mode FROM published_replays"
                " WHERE asset = %s AND model_version = ("
                "   SELECT model_version FROM published_replays WHERE asset = %s"
                "   ORDER BY published_at DESC, id DESC LIMIT 1)"
                " ORDER BY is_headline DESC, cost_scenario",
                (asset, asset),
            ).fetchall()
        return tuple(PublishedReplay(*row) for row in rows)

    def publish_replay_series(
        self, asset: str, model_version: str, series: ReplaySeries
    ) -> tuple[int, int]:
        """Write the day-by-day Replay. An already-published series is left exactly as it is.

        Returns the counts of decision days and scenario-days actually written, so a caller can say
        whether this run published anything or found the series already there. Published once for
        the same reason the aggregates are: the daily job must not be able to move a frozen result.
        """
        days_written = 0
        scenario_written = 0
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO replay_series"
                " (asset, model_version, first_day, last_day, start_value_usdt, headline_scenario)"
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    asset,
                    model_version,
                    series.first_day,
                    series.last_day,
                    series.start_value_usdt,
                    series.headline_scenario,
                ),
            )
            for entry in series.days:
                row = connection.execute(
                    "INSERT INTO replay_days"
                    " (asset, model_version, day, price, probability, action)"
                    " VALUES (%s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT DO NOTHING RETURNING day",
                    (asset, model_version, entry.day, entry.price, entry.probability, entry.action),
                ).fetchone()
                days_written += row is not None
            for scenario in series.by_scenario.values():
                for value in scenario:
                    row = connection.execute(
                        "INSERT INTO replay_scenario_days"
                        " (asset, model_version, cost_scenario, day, btc, usdt, value_usdt,"
                        "  buy_and_hold_usdt)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                        " ON CONFLICT DO NOTHING RETURNING day",
                        (
                            asset,
                            model_version,
                            value.cost_scenario,
                            value.day,
                            value.btc,
                            value.usdt,
                            value.value_usdt,
                            value.buy_and_hold_usdt,
                        ),
                    ).fetchone()
                    scenario_written += row is not None
            connection.commit()
        return days_written, scenario_written

    def latest_replay_series(self, asset: str) -> ReplaySeries | None:
        """The day-by-day Replay of the most recently published backtest, or None if there is none.

        The model version comes from `published_replays`, not from these tables, so the series a
        reader sees always belongs to the aggregates printed beside it. A series reachable without
        its aggregates is the failure worth making unreachable.
        """
        with self._connect() as connection:
            version_row = connection.execute(
                "SELECT model_version FROM published_replays WHERE asset = %s"
                " ORDER BY published_at DESC, id DESC LIMIT 1",
                (asset,),
            ).fetchone()
            if version_row is None:
                return None
            model_version = str(version_row[0])
            header = connection.execute(
                "SELECT first_day, last_day, start_value_usdt, headline_scenario"
                " FROM replay_series"
                " WHERE asset = %s AND model_version = %s",
                (asset, model_version),
            ).fetchone()
            if header is None:
                return None
            day_rows = connection.execute(
                "SELECT day, price, probability, action FROM replay_days"
                " WHERE asset = %s AND model_version = %s ORDER BY day",
                (asset, model_version),
            ).fetchall()
            scenario_rows = connection.execute(
                "SELECT cost_scenario, day, btc, usdt, value_usdt, buy_and_hold_usdt"
                " FROM replay_scenario_days WHERE asset = %s AND model_version = %s"
                " ORDER BY cost_scenario, day",
                (asset, model_version),
            ).fetchall()

        by_scenario: dict[str, list[SeriesScenarioDay]] = {}
        for name, day, btc, usdt, value_usdt, held in scenario_rows:
            by_scenario.setdefault(str(name), []).append(
                SeriesScenarioDay(
                    day=day,
                    cost_scenario=str(name),
                    btc=float(btc),
                    usdt=float(usdt),
                    value_usdt=float(value_usdt),
                    buy_and_hold_usdt=float(held),
                )
            )
        first_day, last_day, start_value_usdt, headline_scenario = header
        return ReplaySeries(
            first_day=first_day,
            last_day=last_day,
            start_value_usdt=float(start_value_usdt),
            headline_scenario=str(headline_scenario),
            days=tuple(
                SeriesDay(
                    day=day, price=float(price), probability=float(probability), action=action
                )
                for day, price, probability, action in day_rows
            ),
            by_scenario=MappingProxyType(
                {name: tuple(values) for name, values in sorted(by_scenario.items())}
            ),
        )

    def publish(self, run: PublishedRun) -> bool:
        """Write the run and its position transition. False means this key was already published."""
        with self._connect() as connection:
            row = connection.execute(
                "INSERT INTO published_runs ("
                " asset, decision_day, feature_cutoff, decision_time, decision_price,"
                " probability, target_exposure,"
                " action, position_before_btc, position_before_usdt, position_after_btc,"
                " position_after_usdt, trade_notional, trade_fee, trade_price, explanation,"
                " model_version, policy_version, contract_digest, model_mode, run_bundle_path"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                # Any conflict means this Decision Day is already published: on the day-level
                # constraint or on the version-scoped one, the answer is the same.
                " %s, %s, %s) ON CONFLICT DO NOTHING RETURNING id",
                (
                    run.asset,
                    run.decision_day,
                    run.feature_cutoff,
                    run.decision_time,
                    run.decision_price,
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
