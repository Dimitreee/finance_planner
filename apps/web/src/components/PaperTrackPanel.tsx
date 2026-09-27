import type { PaperTrack } from "../api";
import { formatDay, formatUsdt } from "../format";
import { PriceChart } from "./PriceChart";

export function PaperTrackPanel({ track }: { track: PaperTrack }) {
  const latest = track.points[track.points.length - 1];

  return (
    <section className="panel" aria-labelledby="track-heading">
      <div className="panel__head">
        <h2 id="track-heading">Since launch</h2>
        {track.genesis_day && <span className="badge badge--neutral">from {formatDay(track.genesis_day)}</span>}
      </div>

      <p className="track__caveat">
        This track began on{" "}
        {track.genesis_day ? formatDay(track.genesis_day) : "a date not yet set, as nothing has been published"}
        {track.genesis_day && ` and covers ${track.points.length} decision ${track.points.length === 1 ? "day" : "days"}`}
        . It is far too short to say anything about whether the advice makes money.
      </p>

      {latest && (
        <p className="track__value">
          The simulated portfolio is worth {formatUsdt(latest.value_usdt)}, valued at the most recent
          decision price.
        </p>
      )}

      <PriceChart points={track.points} />
    </section>
  );
}
