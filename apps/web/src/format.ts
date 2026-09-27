/** How the advice is worded. Kept apart from the markup because the wording is the requirement.
 *
 * The central rule: a decision from an earlier day is never phrased as an instruction. It is
 * reported, in the past tense, with its date in the sentence rather than in fine print.
 */

import type { Action, Advice, Freshness } from "./api";

const IMPERATIVE: Record<Action, string> = {
  BUY: "Buy BTC",
  HOLD: "Hold",
  REDUCE: "Reduce to USDT",
};

const PAST: Record<Action, string> = {
  BUY: "buy BTC",
  HOLD: "hold",
  REDUCE: "reduce to USDT",
};

const RELATIVE_TO_POSITION: Record<Action, string> = {
  BUY: "move the simulated portfolio out of USDT and into BTC",
  HOLD: "keep the simulated portfolio exactly as it is",
  REDUCE: "move the simulated portfolio out of BTC and into USDT",
};

export function formatDay(iso: string): string {
  // `Number("x")` is NaN, which is not nullish, so a `??` guard here would be dead and the reader
  // would be shown the literal text "Invalid Date" inside a sentence.
  const [year, month, day] = iso.split("-").map(Number);
  if (![year, month, day].every((part) => Number.isFinite(part))) return iso;
  return new Date(Date.UTC(year as number, (month as number) - 1, day as number)).toLocaleDateString(
    "en-GB",
    { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" },
  );
}

export function formatInstant(iso: string): string {
  return `${new Date(iso).toISOString().replace("T", " ").slice(0, 16)} UTC`;
}

export function freshnessLabel(freshness: Freshness): string {
  if (freshness === "current") return "Today's decision";
  if (freshness === "stale") return "From an earlier day";
  return "No decision available";
}

export function headline(advice: Advice): string {
  if (advice.freshness === "unavailable" || advice.action === null) {
    return "There is no advice to show";
  }
  if (advice.freshness === "stale" && advice.decision_day) {
    return `The decision of ${formatDay(advice.decision_day)} was to ${PAST[advice.action]}`;
  }
  return IMPERATIVE[advice.action];
}

export function meaningSentence(advice: Advice): string | null {
  if (advice.action === null || advice.freshness === "unavailable") return null;
  const move = RELATIVE_TO_POSITION[advice.action];
  // An earlier day's decision is reported, never handed over as something to do now. Degrading the
  // headline alone would leave the instruction standing in the sentence underneath it.
  return advice.freshness === "stale"
    ? `That decision was to ${move}. It is not advice for today.`
    : `This means: ${move}.`;
}

export function horizonSentence(advice: Advice): string | null {
  if (advice.action === null || advice.freshness === "unavailable") return null;
  const days = advice.protocol.horizon_days;
  if (advice.freshness === "stale") {
    return days === 1
      ? "Its one-day horizon has already passed."
      : `Its ${days}-day horizon has already passed.`;
  }
  return days === 1
    ? "The horizon is one day: this is about tomorrow, not about the months ahead."
    : `The horizon is ${days} days.`;
}

function beforeDeadline(now: Date, deadlineUtc: string): boolean {
  const current = `${String(now.getUTCHours()).padStart(2, "0")}:${String(
    now.getUTCMinutes(),
  ).padStart(2, "0")}`;
  return current < deadlineUtc;
}

export function freshnessNote(advice: Advice, now: Date = new Date()): string {
  if (advice.freshness === "current") {
    return "A decision was published for today.";
  }
  if (advice.freshness === "stale") {
    const note =
      "No decision has been published for today, so the one above is the most recent. " +
      "The simulated portfolio has not moved since.";
    // Before the deadline the absence is expected, not a miss. Saying so costs one clause and
    // avoids a daily few minutes of reading like a failure.
    return beforeDeadline(now, advice.protocol.decision_deadline_utc)
      ? `${note} Today's decision is due at ${advice.protocol.decision_deadline_utc} UTC.`
      : note;
  }
  return (
    "Nothing has been published yet, or the job that publishes it has not completed. " +
    "This is not a view that the market is neutral — there is simply no decision to show."
  );
}

export function formatUsdt(value: number): string {
  return `${value.toLocaleString("en-GB", { maximumFractionDigits: 2, minimumFractionDigits: 2 })} USDT`;
}

export function formatBtc(value: number): string {
  return `${value.toLocaleString("en-GB", { maximumFractionDigits: 8 })} BTC`;
}

export function formatProbability(value: number): string {
  return value.toLocaleString("en-GB", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}


export function formatReturn(value: number): string {
  // `-0 >= 0` is true and `(-0 * 100).toLocaleString()` is "-0.0", which would render "+-0.0%".
  const normalised = value === 0 ? 0 : value;
  return `${normalised >= 0 ? "+" : ""}${(normalised * 100).toLocaleString("en-GB", {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  })}%`;
}
