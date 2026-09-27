import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AdvicePanel } from "./components/AdvicePanel";
import { PaperTrackPanel } from "./components/PaperTrackPanel";
import { ReplayPanel } from "./components/ReplayPanel";
import { advice, replay, track } from "./fixtures";

const pageText = () => document.body.textContent ?? "";

describe("the advice panel", () => {
  it("shows one action, what it means for the position, and the horizon", () => {
    render(<AdvicePanel advice={advice()} />);
    expect(screen.getByText("Buy BTC")).toBeInTheDocument();
    expect(pageText()).toMatch(/move the simulated portfolio out of USDT and into BTC/);
    expect(pageText()).toMatch(/horizon is one day/i);
  });

  it("shows the probability and the rule that produced the action", () => {
    render(<AdvicePanel advice={advice()} />);
    expect(screen.getByText("0.95")).toBeInTheDocument();
    expect(pageText()).toMatch(/at or above the 0.55 threshold/);
  });

  it("shows the data cutoff and the freshness state", () => {
    render(<AdvicePanel advice={advice()} />);
    expect(pageText()).toMatch(/2026-09-28 00:00 UTC/);
    expect(screen.getByText("Today's decision")).toBeInTheDocument();
  });

  it("puts the date beside a stale action and never phrases it as an instruction", () => {
    render(<AdvicePanel advice={advice({ freshness: "stale", decision_day: "2026-09-26" })} />);
    expect(screen.getByText(/The decision of 26 September 2026 was to buy BTC/)).toBeInTheDocument();
    expect(screen.queryByText("Buy BTC")).not.toBeInTheDocument();
    expect(screen.getByText("From an earlier day")).toBeInTheDocument();
    // The headline is not the only place an instruction can hide.
    expect(pageText()).not.toMatch(/This means:/);
    expect(pageText()).toMatch(/It is not advice for today/);
    expect(pageText()).toMatch(/horizon has already passed/);
    expect(pageText()).not.toMatch(/this is about tomorrow/);
  });

  it("says plainly that there is no advice, and does not dress failure as neutrality", () => {
    render(<AdvicePanel advice={advice({ freshness: "unavailable", action: null })} />);
    expect(screen.getByText("There is no advice to show")).toBeInTheDocument();
    expect(pageText()).toMatch(/not a view that the market is neutral/i);
    expect(pageText()).not.toMatch(/\bhold\b/i);
  });

  it("labels the portfolio as a simulation holding no real funds", () => {
    render(<AdvicePanel advice={advice()} />);
    expect(pageText()).toMatch(/holds no real funds/i);
    expect(pageText()).toMatch(/no trade is ever placed/i);
  });

  it("exposes nothing about training or model fitting", () => {
    render(<AdvicePanel advice={advice()} />);
    for (const word of [/\btrain/i, /\bfitting\b/i, /\bepoch/i, /hyperparameter/i, /\bdataset\b/i]) {
      expect(pageText()).not.toMatch(word);
    }
  });
});

describe("the paper track panel", () => {
  it("carries its start date and says it is too short to conclude from", () => {
    render(<PaperTrackPanel track={track()} />);
    expect(pageText()).toMatch(/began on 26 September 2026/);
    expect(pageText()).toMatch(/far too short to say anything about whether the advice makes money/i);
  });

  it("draws a price chart over the days it covers", () => {
    render(<PaperTrackPanel track={track()} />);
    const svg = screen.getByTestId("price-chart");
    expect(svg).toHaveAttribute("viewBox");
    expect(svg).not.toHaveAttribute("width");
    expect(svg.querySelector("polyline")?.getAttribute("points")?.split(" ")).toHaveLength(3);
  });

  it("spaces the chart by date, so a gap in publication is visible", () => {
    const gappy = track({
      points: [
        { day: "2026-09-01", action: "BUY", btc: 0.03, usdt: 0, price: 33000, value_usdt: 990 },
        { day: "2026-09-02", action: "HOLD", btc: 0.03, usdt: 0, price: 34000, value_usdt: 1020 },
        { day: "2026-09-30", action: "HOLD", btc: 0.03, usdt: 0, price: 33500, value_usdt: 1005 },
      ],
    });
    render(<PaperTrackPanel track={gappy} />);
    const xs = (screen.getByTestId("price-chart").querySelector("polyline")?.getAttribute("points") ?? "")
      .split(" ")
      .map((pair) => Number(pair.split(",")[0]));
    const [a, b, c] = xs as [number, number, number];
    expect(b - a).toBeLessThan((c - b) / 10);
  });

  it("puts no stray title on the chart that would read as its label", () => {
    render(<PaperTrackPanel track={track()} />);
    expect(screen.getByTestId("price-chart").querySelectorAll("title")).toHaveLength(0);
  });

  it("says so rather than drawing an empty chart when nothing is published", () => {
    render(<PaperTrackPanel track={track({ genesis_day: null, points: [] })} />);
    expect(screen.queryByTestId("price-chart")).not.toBeInTheDocument();
    expect(pageText()).toMatch(/nothing to draw/i);
  });
});


describe("the history evaluation panel", () => {
  it("names each cost scenario and compares against cash and buy-and-hold", () => {
    render(<ReplayPanel replay={replay()} />);
    for (const name of ["base", "optimistic", "pessimistic"]) {
      expect(screen.getByText(new RegExp(`^${name}$`))).toBeInTheDocument();
    }
    expect(screen.getByText("Buy & hold")).toBeInTheDocument();
    expect(screen.getByText("Cash")).toBeInTheDocument();
    expect(pageText()).toMatch(/\+150\.0%/);
  });

  it("marks the headline scenario", () => {
    render(<ReplayPanel replay={replay()} />);
    expect(pageText()).toMatch(/headline/);
  });

  it("says these are results on past data, not money earned", () => {
    render(<ReplayPanel replay={replay()} />);
    expect(pageText()).toMatch(/results on past data, not money earned/i);
    expect(pageText()).toMatch(/kept apart from the track since launch/i);
  });

  it("says the forecast has no skill when log loss is no better than answering 0.50", () => {
    render(<ReplayPanel replay={replay({ selection: { metric: "log_loss", score: 0.6944 } })} />);
    expect(pageText()).toMatch(/shows no skill/i);
    expect(pageText()).toMatch(/not from predicting anything/i);
  });

  it("does not cry no-skill when the forecast beats that reference", () => {
    render(<ReplayPanel replay={replay({ selection: { metric: "log_loss", score: 0.66 } })} />);
    expect(pageText()).not.toMatch(/shows no skill/i);
  });

  it("says so plainly when no evaluation has been published", () => {
    render(<ReplayPanel replay={replay({ window: null, scenarios: [] })} />);
    expect(pageText()).toMatch(/No evaluation has been published yet/i);
  });
});
