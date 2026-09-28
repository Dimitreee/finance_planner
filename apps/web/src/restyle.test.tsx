import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AdvicePanel } from "./components/AdvicePanel";
import { PaperTrackPanel } from "./components/PaperTrackPanel";
import { ReplayPanel } from "./components/ReplayPanel";
import { advice, replay, track } from "./fixtures";

/**
 * What each panel *says*, pinned so a restyle cannot change it.
 *
 * The visual system is free to move: tokens, spacing, dark mode, where a swatch sits. What may not
 * move is a word or a decimal place. A figure that quietly gains or loses a digit is a change to a
 * claim, and a sentence that disappears behind a nicer layout is a claim withdrawn without saying so.
 *
 * Inline snapshots on purpose. The expected text lives here rather than in a `.snap` file, so a diff
 * shows the sentence that changed instead of a hash that moved.
 */
const says = (node: HTMLElement) => (node.textContent ?? "").replace(/\s+/g, " ").trim();

describe("what the panels say", () => {
  it("today's advice reads exactly as it did before the restyle", () => {
    const { container } = render(<AdvicePanel advice={advice()} />);
    expect(says(container)).toMatchInlineSnapshot(`"TodayToday's decisionBuy BTCThis means: move the simulated portfolio out of USDT and into BTC.The horizon is one day: this is about tomorrow, not about the months ahead.Probability of a higher price tomorrow0.95Decision day28 September 2026Data cutoff2026-09-28 00:00 UTCModel modeprice-onlyThe model puts the probability of a higher price tomorrow at 0.95, and that is at or above the 0.55 threshold for moving into BTC. The portfolio was holding USDT, so the action is BUY.Simulated portfolioA simulation. It holds no real funds, and no trade is ever placed on an exchange.0.0295738 BTC and 0.00 USDT — worth 998.50 USDT at the decision priceA decision was published for today."`);
  });

  it("the stale advice reads exactly as it did before the restyle", () => {
    const { container } = render(
      <AdvicePanel advice={advice({ freshness: "stale", decision_day: "2026-09-26" })} />,
    );
    expect(says(container)).toMatchInlineSnapshot(`"TodayFrom an earlier dayThe decision of 26 September 2026 was to buy BTCThat decision was to move the simulated portfolio out of USDT and into BTC. It is not advice for today.Its one-day horizon has already passed.Probability of a higher price tomorrow0.95Decision day26 September 2026Data cutoff2026-09-28 00:00 UTCModel modeprice-onlyThe model puts the probability of a higher price tomorrow at 0.95, and that is at or above the 0.55 threshold for moving into BTC. The portfolio was holding USDT, so the action is BUY.Simulated portfolioA simulation. It holds no real funds, and no trade is ever placed on an exchange.0.0295738 BTC and 0.00 USDT — worth 998.50 USDT at the decision priceNo decision has been published for today, so the one above is the most recent. The simulated portfolio has not moved since."`);
  });

  it("the unavailable advice reads exactly as it did before the restyle", () => {
    const { container } = render(
      <AdvicePanel advice={advice({ freshness: "unavailable", action: null })} />,
    );
    expect(says(container)).toMatchInlineSnapshot(`"TodayNo decision availableThere is no advice to showNothing has been published yet, or the job that publishes it has not completed. This is not a view that the market is neutral — there is simply no decision to show."`);
  });

  it("the paper track reads exactly as it did before the restyle", () => {
    const { container } = render(<PaperTrackPanel track={track()} />);
    expect(says(container)).toMatchInlineSnapshot(`"Since launchfrom 26 September 2026This track began on 26 September 2026 and covers 3 decision days. It is far too short to say anything about whether the advice makes money.The simulated portfolio is worth 1,005.00 USDT, valued at the most recent decision price.BTC/USDT, 26 September 2026 to 28 September 2026"`);
  });

  it("the no-skill finding reads exactly as it did before the restyle", () => {
    // The sentence this whole page exists to make unmissable. It gained structure in the restyle and
    // must not have gained or lost a word.
    const { container } = render(
      <ReplayPanel replay={replay({ selection: { metric: "log_loss", score: 0.6944 } })} />,
    );
    expect(says(container)).toMatchInlineSnapshot(`"Evaluated on historyreplayA backtest over 1096 days, 1 January 2022 to 31 December 2024. These are results on past data, not money earned, and they are deliberately kept apart from the track since launch.Cost scenarioStrategyBuy & holdCashDrawdownTradesbase headline-12.0%+150.0%+0.0%-44.0%400optimistic+20.0%+150.0%+0.0%-40.0%400pessimistic-60.0%+150.0%+0.0%-70.0%400Selected on log loss: 0.6944. Model arm-a-logistic-1.0, arm A.A model that always answered 0.50 would score 0.6931, so this forecast shows no skill. Whatever the table above says about returns came from a handful of decisions, not from predicting anything."`);
  });

  it("the replay reads exactly as it did before the restyle", () => {
    const { container } = render(<ReplayPanel replay={replay()} />);
    expect(says(container)).toMatchInlineSnapshot(`"Evaluated on historyreplayA backtest over 1096 days, 1 January 2022 to 31 December 2024. These are results on past data, not money earned, and they are deliberately kept apart from the track since launch.Cost scenarioStrategyBuy & holdCashDrawdownTradesbase headline-12.0%+150.0%+0.0%-44.0%400optimistic+20.0%+150.0%+0.0%-40.0%400pessimistic-60.0%+150.0%+0.0%-70.0%400Selected on log loss: 0.6931. Model arm-a-logistic-1.0, arm A."`);
  });
});

describe("the visual system", () => {
  it("draws the same series in the same colour wherever it appears", () => {
    const { container } = render(<ReplayPanel replay={replay()} />);
    const swatches = [...container.querySelectorAll(".swatch")].map(
      (node) => node.className.replace("swatch ", ""),
    );
    // Strategy, the market, and cash — one identity each, in the order the columns read.
    expect(swatches).toEqual(["swatch--strategy", "swatch--market", "swatch--cash"]);
    // A swatch carries colour, never text: identity is never the only thing a reader gets.
    for (const node of container.querySelectorAll(".swatch")) {
      expect(node.textContent).toBe("");
      expect(node.getAttribute("aria-hidden")).toBe("true");
    }
  });

  it("scrolls the six-column table inside its own box rather than the page", () => {
    const { container } = render(<ReplayPanel replay={replay()} />);
    const scroller = container.querySelector(".scenarios__scroll");
    expect(scroller).not.toBeNull();
    expect(scroller?.querySelector("table.scenarios")).not.toBeNull();
  });

  it("carries no hard-coded colour in any panel's markup", () => {
    for (const node of [
      render(<AdvicePanel advice={advice()} />).container,
      render(<PaperTrackPanel track={track()} />).container,
      render(<ReplayPanel replay={replay()} />).container,
    ]) {
      // Colour lives in tokens. An inline style or a hex in the markup is a panel that will look
      // right in one scheme and washed out in the other.
      expect(node.innerHTML).not.toMatch(/#[0-9a-f]{3,8}\b/i);
      expect(node.innerHTML).not.toMatch(/style="[^"]*(color|background)/i);
    }
  });
});
