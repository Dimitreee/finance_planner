import { describe, expect, it } from "vitest";
import { formatDay, freshnessNote, headline, horizonSentence, meaningSentence } from "./format";
import { advice } from "./fixtures";

describe("wording", () => {
  it("states today's decision as an instruction", () => {
    expect(headline(advice({ freshness: "current", action: "BUY" }))).toBe("Buy BTC");
  });

  it("reports an earlier decision in the past tense, with its date in the sentence", () => {
    const text = headline(advice({ freshness: "stale", decision_day: "2026-09-26", action: "BUY" }));
    expect(text).toBe("The decision of 26 September 2026 was to buy BTC");
    expect(text).not.toBe("Buy BTC");
  });

  it("offers nothing to act on when there is no decision", () => {
    expect(headline(advice({ freshness: "unavailable", action: null }))).toBe(
      "There is no advice to show",
    );
  });

  it("says a missing decision is a failure to publish, not a neutral market", () => {
    const note = freshnessNote(advice({ freshness: "unavailable", action: null }));
    expect(note).toMatch(/not a view that the market is neutral/i);
  });

  it("formats a day in UTC regardless of the reader's timezone", () => {
    expect(formatDay("2026-01-01")).toBe("1 January 2026");
  });

  it("returns a malformed date unchanged rather than rendering 'Invalid Date'", () => {
    expect(formatDay("not-a-date")).toBe("not-a-date");
  });

  it("never phrases an earlier decision as something to do now", () => {
    const stale = advice({ freshness: "stale", decision_day: "2026-09-26", action: "BUY" });
    const meaning = meaningSentence(stale);
    expect(meaning).toMatch(/It is not advice for today/);
    expect(meaning).not.toMatch(/^This means:/);
    expect(horizonSentence(stale)).toMatch(/horizon has already passed/);
  });

  it("says the decision is still due when the deadline has not passed", () => {
    const stale = advice({ freshness: "stale", decision_day: "2026-09-27" });
    const before = new Date("2026-09-28T00:04:00Z");
    const after = new Date("2026-09-28T09:00:00Z");
    expect(freshnessNote(stale, before)).toMatch(/due at 00:10 UTC/);
    expect(freshnessNote(stale, after)).not.toMatch(/due at/);
  });
});
