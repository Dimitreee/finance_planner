-- The Replay Result: a frozen backtest, published once and never recomputed by the daily job.
--
-- It is a different object from the Paper Track and must never be joined to it. The left half was
-- produced by a model evaluated on history; the right half is real operation. One series for both
-- would be a substitution, not a simplification, which is why `model_mode` is constrained here
-- rather than merely set by the writer.

CREATE TABLE IF NOT EXISTS published_replays (
    id                  bigserial        PRIMARY KEY,
    asset               text             NOT NULL,
    arm                 text             NOT NULL,
    model_version       text             NOT NULL,
    cost_scenario       text             NOT NULL,
    is_headline         boolean          NOT NULL,
    window_first_day    date             NOT NULL,
    window_last_day     date             NOT NULL,
    days                integer          NOT NULL,
    trades              integer          NOT NULL,
    net_return          double precision NOT NULL,
    buy_and_hold_return double precision NOT NULL,
    cash_return         double precision NOT NULL,
    max_drawdown        double precision NOT NULL,
    turnover            double precision NOT NULL,
    time_invested       double precision NOT NULL,
    total_fees          double precision NOT NULL,
    selection_metric    text             NOT NULL,
    selection_score     double precision NOT NULL,
    contract_digest     text             NOT NULL,
    model_mode          text             NOT NULL DEFAULT 'replay',
    published_at        timestamptz      NOT NULL DEFAULT now(),

    CONSTRAINT published_replays_once
        UNIQUE (asset, arm, model_version, cost_scenario),
    CONSTRAINT published_replays_is_replay
        CHECK (model_mode = 'replay')
);

CREATE INDEX IF NOT EXISTS published_replays_lookup
    ON published_replays (asset, published_at DESC);
