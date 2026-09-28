-- The per-day Replay series: the day-by-day path behind the aggregates in `published_replays`.
--
-- Three tables, and the split is the point.
--
-- The header holds what is true of the whole series, including the starting capital — the one figure
-- the three curves share and the reason they can be compared at all. Cash is not stored as a curve
-- because it is that constant: a flat line at the starting capital, and a column repeating one number
-- 1 096 times would invite it to stop being constant.
--
-- The day table holds what the Policy decided. Costs change what the portfolio is worth; they do not
-- change the decision, because the Action comes from the probability and from whether the portfolio was
-- in or out, and with a target exposure of exactly 0 or 1 a fee cannot move it. Storing the decision
-- once means a Cost Scenario switcher on the page *cannot* show an Action that flips when the reader
-- changes a fee — the schema makes that impossible rather than the UI making it unlikely.
--
-- The scenario table holds the money, which does differ: the strategy's value and buy-and-hold's, both
-- after that scenario's costs.
--
-- Published once, like the aggregates, and never recomputed by the daily job: a frozen research result
-- that drifts nightly is not a frozen research result.

CREATE TABLE IF NOT EXISTS replay_series (
    asset             text             NOT NULL,
    model_version     text             NOT NULL,
    first_day         date             NOT NULL,
    last_day          date             NOT NULL,
    start_value_usdt  double precision NOT NULL,
    headline_scenario text             NOT NULL,
    published_at      timestamptz      NOT NULL DEFAULT now(),

    CONSTRAINT replay_series_once PRIMARY KEY (asset, model_version),
    CONSTRAINT replay_series_window CHECK (first_day <= last_day),
    CONSTRAINT replay_series_capital CHECK (start_value_usdt > 0),
    -- ADR-0009 pre-registers the base scenario as the headline. Spelled out rather than left to the
    -- publisher because a headline that can be chosen after the numbers are in is not a
    -- pre-registration; moving it should cost a migration and a paragraph, not an argument.
    CONSTRAINT replay_series_headline CHECK (headline_scenario = 'base')
);

CREATE TABLE IF NOT EXISTS replay_days (
    asset         text             NOT NULL,
    model_version text             NOT NULL,
    day           date             NOT NULL,
    price         double precision NOT NULL,
    probability   double precision NOT NULL,
    action        text             NOT NULL,

    CONSTRAINT replay_days_once PRIMARY KEY (asset, model_version, day),
    CONSTRAINT replay_days_action CHECK (action IN ('BUY', 'HOLD', 'REDUCE')),
    -- A probability outside [0, 1] is not a probability, and the page prints this number.
    CONSTRAINT replay_days_probability CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT replay_days_has_a_series
        FOREIGN KEY (asset, model_version)
        REFERENCES replay_series (asset, model_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS replay_scenario_days (
    asset            text             NOT NULL,
    model_version    text             NOT NULL,
    cost_scenario    text             NOT NULL,
    day              date             NOT NULL,
    btc              double precision NOT NULL,
    usdt             double precision NOT NULL,
    value_usdt       double precision NOT NULL,
    buy_and_hold_usdt double precision NOT NULL,

    CONSTRAINT replay_scenario_days_once
        PRIMARY KEY (asset, model_version, cost_scenario, day),
    -- The three pre-registered points of the band and nothing else (ADR-0009). Without this a fourth
    -- rate could be published and the switcher would offer it, which is the free rate field the panel
    -- refuses to render — arriving through the back door.
    CONSTRAINT replay_scenario_days_scenario
        CHECK (cost_scenario IN ('optimistic', 'base', 'pessimistic')),
    -- A scenario's money for a day with no decision behind it would be a curve with no history.
    CONSTRAINT replay_scenario_days_has_a_day
        FOREIGN KEY (asset, model_version, day)
        REFERENCES replay_days (asset, model_version, day) ON DELETE CASCADE
);
