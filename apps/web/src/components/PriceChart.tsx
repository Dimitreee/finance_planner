import type { TrackPoint } from "../api";
import { formatDay } from "../format";

const WIDTH = 600;
const HEIGHT = 180;
const PADDING = 8;
const MS_PER_DAY = 86_400_000;

const dayNumber = (iso: string) => Date.parse(`${iso}T00:00:00Z`) / MS_PER_DAY;

/** BTC/USDT over the days the track covers. Scales with its container: no fixed pixel size. */
export function PriceChart({ points }: { points: TrackPoint[] }) {
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) {
    return <p className="chart chart--empty">No days published yet, so there is nothing to draw.</p>;
  }

  const prices = points.map((point) => point.price);
  const low = Math.min(...prices);
  const high = Math.max(...prices);
  const priceSpan = high - low || 1;
  // Spaced by date, not by array index. Only published days are returned, so an ordinal x-axis
  // would draw a track with a month-long outage exactly like a contiguous one.
  const daySpan = dayNumber(last.day) - dayNumber(first.day) || 1;

  const coordinates = points.map((point) => {
    const x =
      PADDING + ((dayNumber(point.day) - dayNumber(first.day)) / daySpan) * (WIDTH - 2 * PADDING);
    const y = HEIGHT - PADDING - ((point.price - low) / priceSpan) * (HEIGHT - 2 * PADDING);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <figure className="chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`BTC/USDT price across ${points.length} published days`}
        data-testid="price-chart"
      >
        <polyline className="series series--market" points={coordinates.join(" ")} fill="none" />
      </svg>
      <figcaption>
        BTC/USDT, {formatDay(first.day)} to {formatDay(last.day)}
      </figcaption>
    </figure>
  );
}
