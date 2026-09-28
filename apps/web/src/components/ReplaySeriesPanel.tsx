import { useState } from "react";
import type { Replay, ReplaySeries } from "../api";
import { formatDay, formatReturn, formatUsdt } from "../format";
import { EquityChart } from "./EquityChart";
import { UnderwaterChart } from "./UnderwaterChart";

/**
 * How the advice's money moved across the Research Window, under each pre-registered Cost Scenario.
 *
 * Two things this panel refuses to do. It never joins these curves to the Paper Track (ADR-0013): the
 * left half was a model evaluated on history and the right half is real operation, and one series for
 * both would be a substitution rather than a simplification. And the switcher offers exactly the
 * scenarios the API served — there is no free rate field, because a rate a reader typed would produce
 * a number the pre-registration never covered.
 *
 * Switching a scenario changes the curves and the figures beside them. It cannot change the Action
 * history: costs alter what the portfolio is worth, never what the Policy decided, and the published
 * series stores each decision once for exactly that reason.
 */
export function ReplaySeriesPanel({
  series,
  replay,
}: {
  series: ReplaySeries;
  replay: Replay;
}) {
  const headline = series.headline_scenario;
  const [selected, setSelected] = useState(
    headline ?? series.cost_scenarios[0] ?? "",
  );

  if (
    !series.window ||
    series.days.length === 0 ||
    series.start_value_usdt === null
  ) {
    return (
      <section className="panel" aria-labelledby="series-heading">
        <h2 id="series-heading">How the money moved</h2>
        <p className="track__caveat">
          No day-by-day evaluation has been published yet.
        </p>
      </section>
    );
  }

  // Three lookups on one name, and each can miss for its own reason. `unknown` distinguishes "this
  // scenario was never served" from "nothing has been published": a generic empty state in front of
  // the first would send a reader looking for missing data that is not missing.
  const values = series.scenarios[selected];
  const aggregate = replay.scenarios.find(
    (scenario) => scenario.name === selected,
  );
  const unknown = values === undefined || aggregate === undefined;
  const acted = series.days.filter((day) => day.action !== "HOLD");
  const start = series.start_value_usdt;

  // One scale for every scenario. Fitting each scenario to its own extent makes three different ramps
  // normalise to the same shape, so the switcher changes the figures while the line stays put — which
  // reads as "costs do not matter", the opposite of what this panel is for. A test caught it.
  const everyValue = [
    start,
    ...Object.values(series.scenarios).flatMap((scenario) =>
      scenario.flatMap((point) => [point.value_usdt, point.buy_and_hold_usdt]),
    ),
  ];
  const domain = { low: Math.min(...everyValue), high: Math.max(...everyValue) };
  const deepestReported = Math.max(
    0,
    ...replay.scenarios.map((entry) => entry.max_drawdown),
  );

  if (unknown) {
    return (
      <section className="panel" aria-labelledby="series-heading">
        <div className="panel__head">
          <h2 id="series-heading">How the money moved</h2>
          <span className="badge badge--neutral">{series.model_mode}</span>
        </div>
        <p className="notice notice--bad" data-testid="series-scenario-missing">
          The {selected} scenario has a day-by-day series but no published
          aggregate, or the other way round. Drawing one without the other would
          show a curve with no figure to check it against, so neither is shown.
        </p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="series-heading">
      <div className="panel__head">
        <h2 id="series-heading">How the money moved</h2>
        <span className="badge badge--neutral">{series.model_mode}</span>
      </div>

      <p className="track__caveat">
        Day by day across {series.window.days} days,{" "}
        {formatDay(series.window.first_day)} to{" "}
        {formatDay(series.window.last_day)}, all three lines starting from{" "}
        {formatUsdt(start)}. Evaluated on history: results on past data, not
        money earned, and deliberately kept apart from the track since launch.
        The series stops at {formatDay(series.window.last_day)}, the last day of
        the development period — the final holdout is evaluated once and is not
        drawn here.
      </p>

      <fieldset className="switcher">
        <legend>Cost scenario</legend>
        {series.cost_scenarios.map((name) => (
          <label key={name} className="switcher__option">
            <input
              type="radio"
              name="cost-scenario"
              value={name}
              checked={name === selected}
              onChange={() => setSelected(name)}
            />
            <span>
              {name}
              {name === headline && (
                <span className="scenarios__tag"> headline</span>
              )}
            </span>
          </label>
        ))}
      </fieldset>

      <ul className="legend" aria-label="What each line is">
        <li>
          <span className="swatch swatch--strategy" aria-hidden="true" />
          The advice
        </li>
        <li>
          <span className="swatch swatch--market" aria-hidden="true" />
          Holding BTC
        </li>
        <li>
          <span className="swatch swatch--cash" aria-hidden="true" />
          Holding USDT
        </li>
      </ul>

      <EquityChart
        values={values}
        startValue={start}
        scenario={selected}
        domain={domain}
      />

      <dl className="facts" data-testid="series-figures">
        <div>
          <dt>The advice</dt>
          <dd>{formatReturn(aggregate.net_return)}</dd>
        </div>
        <div>
          <dt>Holding BTC</dt>
          <dd>{formatReturn(aggregate.buy_and_hold_return)}</dd>
        </div>
        <div>
          <dt>Holding USDT</dt>
          <dd>{formatReturn(aggregate.cash_return)}</dd>
        </div>
        <div>
          <dt>Deepest drawdown</dt>
          <dd>{formatReturn(-aggregate.max_drawdown)}</dd>
        </div>
      </dl>

      <UnderwaterChart
        values={values}
        startValue={start}
        reportedMaxDrawdown={aggregate.max_drawdown}
        deepestReported={deepestReported}
      />

      <p className="freshness-note" data-testid="action-history">
        The Policy acted on {acted.length} of {series.days.length} days. Costs
        change what the portfolio is worth, never what was decided, so this
        count is the same under every scenario.
      </p>
    </section>
  );
}
