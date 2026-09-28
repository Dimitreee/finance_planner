import { useEffect, useState } from "react";
import type { Advice, PaperTrack, Replay, ReplaySeries } from "./api";
import {
  fetchAdvice,
  fetchPaperTrack,
  fetchReplay,
  fetchReplaySeries,
} from "./api";
import { AdvicePanel } from "./components/AdvicePanel";
import { PaperTrackPanel } from "./components/PaperTrackPanel";
import { ReplayPanel } from "./components/ReplayPanel";
import { ReplaySeriesPanel } from "./components/ReplaySeriesPanel";

const REFRESH_MS = 60_000;

// What a *failed* fetch falls back to, which is a different thing from what the endpoint serves when
// nothing has been published: this one never reaches the network. Spelled out rather than derived
// because `Replay` and `ReplaySeries` annotate them: a field added to either type stops the build
// here. The API's two payload shapes needed an overlay to get that guarantee; these get it free.
const EMPTY_REPLAY: Replay = {
  asset: "",
  arm: null,
  model_version: null,
  model_mode: null,
  window: null,
  selection: null,
  scenarios: [],
};

const EMPTY_SERIES: ReplaySeries = {
  asset: "",
  model_mode: null,
  model_version: null,
  window: null,
  start_value_usdt: null,
  headline_scenario: null,
  cost_scenarios: [],
  days: [],
  scenarios: {},
};

type State =
  | { status: "loading" }
  | { status: "failed"; detail: string }
  | {
      status: "ready";
      advice: Advice;
      track: PaperTrack;
      replay: Replay;
      series: ReplaySeries;
    };

export function App() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let live = true;
    const load = () =>
      Promise.all([
        fetchAdvice(),
        fetchPaperTrack(),
        // The evaluation is the newest surface and the first to exist on a fresh deployment. Losing
        // it must not cost the reader today's advice as well.
        fetchReplay().catch(() => EMPTY_REPLAY),
        // The series is the newest surface of all, so losing it must not cost the reader the
        // aggregates it illustrates, let alone today's advice.
        fetchReplaySeries().catch(() => EMPTY_SERIES),
      ])
        .then(
          ([advice, track, replay, series]) =>
            live &&
            setState({ status: "ready", advice, track, replay, series }),
        )
        .catch((error: unknown) =>
          live
            ? setState({ status: "failed", detail: String(error) })
            : undefined,
        );

    void load();
    // Freshness is computed against the request-time UTC date. A tab left open across 00:00 UTC
    // would otherwise keep showing yesterday's decision badged as today's, which defeats the one
    // mechanism this page exists to make honest.
    const timer = setInterval(() => void load(), REFRESH_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <main className="page">
      <header className="page__head">
        <h1>CryptoGuard</h1>
        <p className="page__subtitle">
          A BTC/USDT advisor. It explains what it decided and why, and keeps a
          simulated portfolio so the advice can be judged. It never places a
          trade.
        </p>
      </header>

      {state.status === "loading" && (
        <p className="notice">Loading the published decision…</p>
      )}
      {state.status === "failed" && (
        <p className="notice notice--bad">
          The published state could not be read, so there is no advice to show.
          This is a failure to reach the service, not a view about the market. (
          {state.detail})
        </p>
      )}
      {state.status === "ready" && (
        <>
          <AdvicePanel advice={state.advice} />
          <PaperTrackPanel track={state.track} />
          <ReplayPanel replay={state.replay} />
          <ReplaySeriesPanel series={state.series} replay={state.replay} />
        </>
      )}
    </main>
  );
}
