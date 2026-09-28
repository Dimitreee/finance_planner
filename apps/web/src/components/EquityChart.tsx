import type { SeriesScenarioDay } from "../api";
import { formatDay } from "../format";

const WIDTH = 600;
const HEIGHT = 200;
const PADDING = 8;
const MS_PER_DAY = 86_400_000;

const dayNumber = (iso: string) => Date.parse(`${iso}T00:00:00Z`) / MS_PER_DAY;

/**
 * Three curves from identical capital: the strategy, holding BTC, and holding USDT.
 *
 * Cash is drawn as the flat line at the starting capital rather than read from a stored column,
 * because that is what it is — a constant. Points are spaced by date, not by array index, so a gap in
 * published days draws as a gap.
 *
 * Interaction belongs to ticket 03, and so do labels at the end of each line. Identity here is
 * carried by the legend the panel renders above this chart, by the dashed stroke that marks cash as
 * the reference rather than a third contender, and by the `aria-label` — nothing a reader must
 * distinguish is available only as a colour.
 */
export function EquityChart({
  values,
  startValue,
  scenario,
  domain,
}: {
  values: SeriesScenarioDay[];
  startValue: number;
  scenario: string;
  /** Shared across every Cost Scenario, so switching moves the line instead of rescaling under it. */
  domain: { low: number; high: number };
}) {
  const first = values[0];
  const last = values[values.length - 1];
  if (!first || !last) {
    return (
      <p className="chart chart--empty">
        No days published yet, so there is nothing to draw.
      </p>
    );
  }

  // Taken from the caller, which computes it over *all* scenarios. Scaling to the selected scenario
  // alone was the first version, and it made the switcher useless: three curves that are each a
  // different ramp normalise to the same shape, so the line looked identical while the money changed.
  const { low, high } = domain;
  const span = high - low || 1;
  const daySpan = dayNumber(last.day) - dayNumber(first.day) || 1;

  const x = (day: string) =>
    PADDING +
    ((dayNumber(day) - dayNumber(first.day)) / daySpan) * (WIDTH - 2 * PADDING);
  const y = (value: number) =>
    HEIGHT - PADDING - ((value - low) / span) * (HEIGHT - 2 * PADDING);
  const line = (pick: (point: SeriesScenarioDay) => number) =>
    values
      .map((point) => `${x(point.day).toFixed(1)},${y(pick(point)).toFixed(1)}`)
      .join(" ");

  return (
    <figure className="chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={
          `Portfolio value under ${scenario} costs across ${values.length} days, ` +
          `against holding BTC and holding USDT, all from ${startValue} USDT`
        }
        data-testid="equity-chart"
      >
        <polyline
          className="series series--cash"
          points={`${x(first.day).toFixed(1)},${y(startValue).toFixed(1)} ${x(last.day).toFixed(1)},${y(startValue).toFixed(1)}`}
          fill="none"
        />
        <polyline
          className="series series--market"
          points={line((point) => point.buy_and_hold_usdt)}
          fill="none"
        />
        <polyline
          className="series series--strategy"
          points={line((point) => point.value_usdt)}
          fill="none"
        />
      </svg>
      <figcaption>
        Value under {scenario} costs, {formatDay(first.day)} to{" "}
        {formatDay(last.day)}, all three starting from the same capital.
      </figcaption>
    </figure>
  );
}
