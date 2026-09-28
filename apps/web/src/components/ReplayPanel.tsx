import type { Replay } from "../api";
import { formatDay, formatReturn } from "../format";

/** Log loss of a model that always answers 0.50. Anything at or above it has no skill. */
const NO_SKILL_LOG_LOSS = Math.log(2);

/** The frozen backtest. A separate panel on purpose: it is never joined to the Paper Track. */
export function ReplayPanel({ replay }: { replay: Replay }) {
  if (!replay.window || replay.scenarios.length === 0) {
    return (
      <section className="panel" aria-labelledby="replay-heading">
        <h2 id="replay-heading">Evaluated on history</h2>
        <p className="track__caveat">No evaluation has been published yet.</p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="replay-heading">
      <div className="panel__head">
        <h2 id="replay-heading">Evaluated on history</h2>
        <span className="badge badge--neutral">{replay.model_mode}</span>
      </div>

      <p className="track__caveat">
        A backtest over {replay.window.days} days,{" "}
        {formatDay(replay.window.first_day)} to{" "}
        {formatDay(replay.window.last_day)}. These are results on past data, not
        money earned, and they are deliberately kept apart from the track since
        launch.
      </p>

      {/* Six columns of figures do not fit a 360px screen, and dropping one would remove
          information. The table scrolls inside its own box instead, so the page never does. */}
      <div className="scenarios__scroll">
        <table className="scenarios">
          <thead>
            <tr>
              <th scope="col">Cost scenario</th>
              {/* The swatch carries which series this is; the heading keeps its text colour. A tinted
                heading would read as emphasis, and the hue is not emphasis. Empty and aria-hidden so
                it adds nothing to what the panel says. */}
              <th scope="col">
                <span className="swatch swatch--strategy" aria-hidden="true" />
                Strategy
              </th>
              <th scope="col">
                <span className="swatch swatch--market" aria-hidden="true" />
                Buy &amp; hold
              </th>
              <th scope="col">
                <span className="swatch swatch--cash" aria-hidden="true" />
                Cash
              </th>
              <th scope="col">Drawdown</th>
              <th scope="col">Trades</th>
            </tr>
          </thead>
          <tbody>
            {replay.scenarios.map((scenario) => (
              <tr
                key={scenario.name}
                className={scenario.is_headline ? "scenarios__headline" : ""}
              >
                <th scope="row">
                  {scenario.name}
                  {scenario.is_headline && (
                    <span className="scenarios__tag"> headline</span>
                  )}
                </th>
                <td>{formatReturn(scenario.net_return)}</td>
                <td>{formatReturn(scenario.buy_and_hold_return)}</td>
                <td>{formatReturn(scenario.cash_return)}</td>
                <td>{formatReturn(-scenario.max_drawdown)}</td>
                <td>{scenario.trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {replay.selection && (
        <>
          <p className="track__caveat">
            Selected on {replay.selection.metric.replace("_", " ")}:{" "}
            {replay.selection.score.toFixed(4)}. Model {replay.model_version},
            arm {replay.arm}.
          </p>
          {replay.selection.metric === "log_loss" &&
            replay.selection.score >= NO_SKILL_LOG_LOSS && (
              <p className="warning">
                A model that always answered 0.50 would score{" "}
                {NO_SKILL_LOG_LOSS.toFixed(4)}, so this forecast shows no skill.
                Whatever the table above says about returns came from a handful
                of decisions, not from predicting anything.
              </p>
            )}
        </>
      )}
    </section>
  );
}
