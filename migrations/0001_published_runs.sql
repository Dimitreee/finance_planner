-- Published state for the advisor: one row per Decision Day, plus every attempt to produce one.
--
-- The position is carried forward in these rows rather than replayed from the forecasts: replay
-- exists as a deliberate repair command, not as the daily mechanism (ADR-0010). What stops a retry
-- booking a second virtual trade is the unique key below, not discipline in the job.

CREATE TABLE IF NOT EXISTS job_runs (
    id              bigserial   PRIMARY KEY,
    asset           text        NOT NULL,
    decision_time   timestamptz NOT NULL,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    -- running | published | skipped | failed. A late job and a crashed job are distinguishable
    -- here even though both surface to the user as `stale`.
    status          text        NOT NULL,
    failure_kind    text,
    failure_detail  text
);

CREATE INDEX IF NOT EXISTS job_runs_asset_decision_time ON job_runs (asset, decision_time DESC);

CREATE TABLE IF NOT EXISTS published_runs (
    id                   bigserial        PRIMARY KEY,
    asset                text             NOT NULL,
    decision_day         date             NOT NULL,
    feature_cutoff       timestamptz      NOT NULL,
    decision_time        timestamptz      NOT NULL,
    probability          double precision NOT NULL,
    target_exposure      double precision NOT NULL,
    action               text             NOT NULL,
    position_before_btc  double precision NOT NULL,
    position_before_usdt double precision NOT NULL,
    position_after_btc   double precision NOT NULL,
    position_after_usdt  double precision NOT NULL,
    -- Null on HOLD: exposure is binary and there is one decision a day, so a day has at most one
    -- trade and it fits here rather than in a table of its own.
    trade_notional       double precision,
    trade_fee            double precision,
    trade_price          double precision,
    explanation          text             NOT NULL,
    model_version        text             NOT NULL,
    policy_version       text             NOT NULL,
    contract_digest      text             NOT NULL,
    model_mode           text             NOT NULL,
    run_bundle_path      text             NOT NULL,
    published_at         timestamptz      NOT NULL DEFAULT now(),

    -- The operative constraint while there is a single shared Virtual Portfolio: a Decision Day is
    -- published once, full stop. Promoting a model must not book a second trade against a day that
    -- already has one, because both would land in the same position chain.
    CONSTRAINT published_runs_one_per_day
        UNIQUE (asset, decision_day),
    -- The version-scoped key from ADR-0010. It is what a multi-track design will use, once a track
    -- identity exists to scope the position chain by; today it is implied by the constraint above.
    CONSTRAINT published_runs_idempotency
        UNIQUE (asset, decision_time, model_version, policy_version),
    CONSTRAINT published_runs_action
        CHECK (action IN ('BUY', 'HOLD', 'REDUCE')),
    CONSTRAINT published_runs_no_negative_legs
        CHECK (position_after_btc >= 0 AND position_after_usdt >= 0),
    CONSTRAINT published_runs_hold_is_free
        CHECK (
            action <> 'HOLD'
            OR (trade_notional IS NULL AND trade_fee IS NULL AND trade_price IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS published_runs_asset_day ON published_runs (asset, decision_day DESC);
