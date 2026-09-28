"""Seed the throwaway dev database with DEMO state, so the page can be opened and looked at.

**Nothing this writes is a research result.** The prices are a seeded random walk, not BTC, and the
forecast is noise with a sliver of engineered skill. The Cost Sensitivity *is* a real simulation —
of that walk — so the charts get honest shapes to draw, but no number here measures anything. The
model version is `demo-not-a-result-0.0` and it is on every panel, so a screenshot cannot be
mistaken for a result later.

Why it exists: every UI ticket before this one was verified structurally and by test, and the
tickets said so, because there was no published state to point a browser at. Running the real API
against this makes the page checkable by eye without spending a Trial (ADR-0012) or touching the
Final Holdout (ADR-0018). It refuses any DSN that is not the throwaway cluster on 55432, so it
cannot overwrite the database `CRYPTOGUARD_DATABASE_URL` points at.

    CRYPTOGUARD_DATABASE_URL="$(./scripts/dev_postgres.sh url)" uv run python scripts/seed_demo.py
    CRYPTOGUARD_DATABASE_URL="$(./scripts/dev_postgres.sh url)" make api   # :8000
    make web                                                              # :5173
"""

from __future__ import annotations

import os
import random
from datetime import UTC, date, datetime, timedelta

import psycopg
from cryptoguard_core.backtest import run_cost_sensitivity, series_from
from cryptoguard_core.dataset import ARM_A_FEATURES, DecisionDayRow
from cryptoguard_core.evaluate import published_replays_from
from cryptoguard_core.model import PreviousDirectionBaseline
from cryptoguard_core.policy import PolicyConfig, Position
from cryptoguard_core.store import PublishedRun, RunStore

ASSET = "BTCUSDT"
VERSION = "demo-not-a-result-0.0"
FIRST = date(2022, 1, 1)
LAST = date(2024, 12, 31)
DAYS = (LAST - FIRST).days + 1

rng = random.Random(20260928)

prices: list[float] = [46_000.0]
for _ in range(DAYS - 1):
    prices.append(max(5_000.0, prices[-1] * (1 + rng.gauss(0.00035, 0.018))))

rows = tuple(
    DecisionDayRow(
        day=FIRST + timedelta(days=index),
        feature_cutoff_ms=0,
        anchor_price=price,
        decision_price=price,
        label=1 if index + 1 < DAYS and prices[index + 1] > price else 0,
        label_end_ms=0,
        bars_in_day=24,
        features=dict.fromkeys(ARM_A_FEATURES, 0.0),
        news_lag_hours=None,
    )
    for index, price in enumerate(prices)
)

# A forecast with a little skill and a lot of noise, so the Policy actually flips now and then.
probabilities = {
    row.day: min(0.99, max(0.01, (0.515 if row.label == 1 else 0.485) + rng.gauss(0, 0.075)))
    for row in rows
}

sensitivity = run_cost_sensitivity(
    rows,
    PreviousDirectionBaseline(),
    PolicyConfig(to_btc_at=0.55, to_usdt_at=0.45),
    initial=Position(usdt=1000.0, btc=0.0),
    arm="A",
    probabilities=probabilities,
)
series = series_from(sensitivity)

dsn = os.environ["CRYPTOGUARD_DATABASE_URL"]
assert "55432" in dsn, f"refusing to seed demo data into {dsn}"
store = RunStore(dsn)
with psycopg.connect(dsn) as connection:
    connection.execute(
        "DROP TABLE IF EXISTS published_runs, job_runs, published_replays,"
        " replay_scenario_days, replay_days, replay_series"
    )
    connection.commit()
store.migrate()

store.publish_replay(
    published_replays_from(
        sensitivity,
        asset=ASSET,
        arm="A",
        model_version=VERSION,
        window_first_day=FIRST,
        window_last_day=LAST,
        selection_metric="log_loss",
        selection_score=0.671,
        contract_digest="demo0000demo",
    )
)
print("series rows:", store.publish_replay_series(ASSET, VERSION, series))

# The Paper Track: five consecutive days ending today, so freshness reads `current`.
today = date.today()
position = Position(btc=0.0, usdt=1000.0)
for offset in range(4, -1, -1):
    day = today - timedelta(days=offset)
    price = 63_000.0 + 900.0 * (4 - offset) - 400.0 * ((4 - offset) % 3)
    probability = (0.93, 0.61, 0.58, 0.41, 0.88)[4 - offset]
    target = (
        1.0 if probability >= 0.55 else (0.0 if probability <= 0.45 else position.exposure(price))
    )
    before = position
    if target == 1.0 and before.btc == 0.0:
        action, btc, usdt = "BUY", before.usdt / price * 0.999, 0.0
    elif target == 0.0 and before.btc > 0.0:
        action, btc, usdt = "REDUCE", 0.0, before.btc * price * 0.999
    else:
        action, btc, usdt = "HOLD", before.btc, before.usdt
    position = Position(btc=btc, usdt=usdt)
    cutoff = datetime(day.year, day.month, day.day, tzinfo=UTC)
    store.publish(
        PublishedRun(
            asset=ASSET,
            decision_day=day,
            feature_cutoff=cutoff,
            decision_time=cutoff.replace(minute=8),
            decision_price=price,
            probability=probability,
            target_exposure=target,
            action=action,  # type: ignore[arg-type]
            position_before=before,
            position_after=position,
            trade_notional=None if action == "HOLD" else before.value(price),
            trade_fee=None if action == "HOLD" else before.value(price) * 0.001,
            trade_price=None if action == "HOLD" else price,
            explanation=(
                f"The model puts the probability of a higher price tomorrow at {probability:.2f}, "
                + (
                    "and that is at or above the 0.55 threshold for moving into BTC. "
                    if target == 1.0
                    else "and that is at or below the 0.45 threshold for moving into USDT. "
                    if target == 0.0
                    else "which is between the two thresholds, so the position is left alone. "
                )
                + f"The portfolio was holding {'USDT' if before.btc == 0 else 'BTC'}, "
                f"so the action is {action}."
            ),
            model_version=VERSION,
            policy_version="demo0policy",
            contract_digest="demo0000demo",
            model_mode="price-only",
            run_bundle_path=f"/tmp/demo-{day.isoformat()}.json",
        )
    )

print("headline net return:", round(sensitivity.headline.net_return, 4))
print("headline max drawdown:", round(sensitivity.headline.max_drawdown, 4))
print("trades:", sensitivity.headline.trades, "days:", sensitivity.headline.days)
print("paper track through:", today.isoformat())
