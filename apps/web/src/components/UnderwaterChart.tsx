import type { SeriesScenarioDay } from "../api";
import { formatReturn } from "../format";

const WIDTH = 600;
const HEIGHT = 120;
const PADDING = 8;
const MS_PER_DAY = 86_400_000;

const dayNumber = (iso: string) => Date.parse(`${iso}T00:00:00Z`) / MS_PER_DAY;

/** Drawdown from the running peak, with the reported maximum marked on it.
 *
 * The maximum is the *published* figure, passed in rather than taken from the deepest point drawn:
 * if the two ever disagree the chart says so in its caption, instead of quietly redefining the
 * number the report quotes. Zero is the top of the plot, because a drawdown only ever goes down.
 *
 * The depth scale is the deepest *published* drawdown across the scenarios and nothing else. Folding
 * the drawn depths into it would let the axis rescale on a switch in exactly the case that matters —
 * a series deeper than its own aggregate — which is the defect this chart exists to expose.
 */
export function UnderwaterChart({
  values,
  startValue,
  reportedMaxDrawdown,
  deepestReported,
}: {
  values: SeriesScenarioDay[];
  startValue: number;
  reportedMaxDrawdown: number;
  /** The deepest drawdown any scenario reported, so the depth scale holds across a switch. */
  deepestReported: number;
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

  let peak = startValue;
  const drawdowns = values.map((point) => {
    peak = Math.max(peak, point.value_usdt);
    return { day: point.day, depth: 1 - point.value_usdt / peak };
  });
  // Shared across scenarios for the same reason the value scale is: a depth axis rescaling on every
  // switch would draw a shallower drawdown as the same picture. So it depends only on the published
  // figures, never on the selected scenario's drawn path.
  const deepest = deepestReported || 1;
  const drawnDeepest = drawdowns.reduce((worst, entry) => Math.max(worst, entry.depth), 0);
  // A hair of tolerance: these are floats that travelled through JSON.
  const disagrees = drawnDeepest > deepest + 1e-9;
  const daySpan = dayNumber(last.day) - dayNumber(first.day) || 1;

  const x = (day: string) =>
    PADDING +
    ((dayNumber(day) - dayNumber(first.day)) / daySpan) * (WIDTH - 2 * PADDING);
  const y = (depth: number) =>
    PADDING + (depth / deepest) * (HEIGHT - 2 * PADDING);
  const marked = y(reportedMaxDrawdown);

  return (
    <figure className="chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Drawdown from the running peak, deepest reported ${formatReturn(-reportedMaxDrawdown)}`}
        data-testid="underwater-chart"
      >
        <polyline
          className="series series--strategy"
          points={drawdowns
            .map(
              (entry) =>
                `${x(entry.day).toFixed(1)},${y(entry.depth).toFixed(1)}`,
            )
            .join(" ")}
          fill="none"
        />
        <line
          className="chart__marker"
          x1={PADDING}
          x2={WIDTH - PADDING}
          y1={marked.toFixed(1)}
          y2={marked.toFixed(1)}
          data-testid="underwater-reported-max"
        />
      </svg>
      <figcaption>
        Drawdown from the running peak. The marked line is the reported maximum,{" "}
        {formatReturn(-reportedMaxDrawdown)}.
        {disagrees && (
          <>
            {" "}
            The drawn path reaches {formatReturn(-drawnDeepest)}, deeper than any
            published maximum: the day-by-day series and the aggregate disagree,
            and the aggregate is the authority.
          </>
        )}
      </figcaption>
    </figure>
  );
}
