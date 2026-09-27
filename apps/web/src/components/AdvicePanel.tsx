import type { Advice } from "../api";
import {
  formatBtc,
  formatDay,
  formatInstant,
  formatProbability,
  formatUsdt,
  freshnessLabel,
  freshnessNote,
  headline,
  horizonSentence,
  meaningSentence,
} from "../format";

export function AdvicePanel({ advice }: { advice: Advice }) {
  const unavailable = advice.freshness === "unavailable" || advice.action === null;
  const meaning = meaningSentence(advice);
  const horizon = horizonSentence(advice);

  return (
    <section className="panel" aria-labelledby="advice-heading">
      <div className="panel__head">
        <h2 id="advice-heading">Today</h2>
        <span className={`badge badge--${advice.freshness}`}>{freshnessLabel(advice.freshness)}</span>
      </div>

      <p className="headline">{headline(advice)}</p>

      {!unavailable && advice.action && (
        <>
          {meaning && <p className="meaning">{meaning}</p>}
          {horizon && <p className="horizon">{horizon}</p>}

          <dl className="facts">
            <div>
              <dt>Probability of a higher price tomorrow</dt>
              <dd>{formatProbability(advice.probability ?? 0)}</dd>
            </div>
            <div>
              <dt>Decision day</dt>
              <dd>{advice.decision_day ? formatDay(advice.decision_day) : "—"}</dd>
            </div>
            <div>
              <dt>Data cutoff</dt>
              <dd>{advice.data_cutoff ? formatInstant(advice.data_cutoff) : "—"}</dd>
            </div>
            <div>
              <dt>Model mode</dt>
              <dd>{advice.model_mode ?? "—"}</dd>
            </div>
          </dl>

          <p className="explanation">{advice.explanation}</p>

          <div className="position">
            <h3>Simulated portfolio</h3>
            <p className="position__warning">
              A simulation. It holds no real funds, and no trade is ever placed on an exchange.
            </p>
            <p className="position__holdings">
              {formatBtc(advice.position.btc ?? 0)} and {formatUsdt(advice.position.usdt ?? 0)}
              {advice.position.value_usdt !== null && (
                <> — worth {formatUsdt(advice.position.value_usdt)} at the decision price</>
              )}
            </p>
          </div>
        </>
      )}

      <p className="freshness-note">{freshnessNote(advice)}</p>
    </section>
  );
}
