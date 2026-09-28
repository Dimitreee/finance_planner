import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AdvicePanel } from "./components/AdvicePanel";
import { PaperTrackPanel } from "./components/PaperTrackPanel";
import { ReplayPanel } from "./components/ReplayPanel";
import { ReplaySeriesPanel } from "./components/ReplaySeriesPanel";
import { advice, replay, replaySeries, track } from "./fixtures";
import styleSheet from "./styles.css?raw";

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
const says = (node: HTMLElement) =>
  (node.textContent ?? "").replace(/\s+/g, " ").trim();

describe("what the panels say", () => {
  it("today's advice reads exactly as it did before the restyle", () => {
    const { container } = render(<AdvicePanel advice={advice()} />);
    expect(says(container)).toMatchInlineSnapshot(
      `"TodayToday's decisionBuy BTCThis means: move the simulated portfolio out of USDT and into BTC.The horizon is one day: this is about tomorrow, not about the months ahead.Probability of a higher price tomorrow0.95Decision day28 September 2026Data cutoff2026-09-28 00:00 UTCModel modeprice-onlyThe model puts the probability of a higher price tomorrow at 0.95, and that is at or above the 0.55 threshold for moving into BTC. The portfolio was holding USDT, so the action is BUY.Simulated portfolioA simulation. It holds no real funds, and no trade is ever placed on an exchange.0.0295738 BTC and 0.00 USDT — worth 998.50 USDT at the decision priceA decision was published for today."`,
    );
  });

  it("the stale advice reads exactly as it did before the restyle", () => {
    const { container } = render(
      <AdvicePanel
        advice={advice({ freshness: "stale", decision_day: "2026-09-26" })}
      />,
    );
    expect(says(container)).toMatchInlineSnapshot(
      `"TodayFrom an earlier dayThe decision of 26 September 2026 was to buy BTCThat decision was to move the simulated portfolio out of USDT and into BTC. It is not advice for today.Its one-day horizon has already passed.Probability of a higher price tomorrow0.95Decision day26 September 2026Data cutoff2026-09-28 00:00 UTCModel modeprice-onlyThe model puts the probability of a higher price tomorrow at 0.95, and that is at or above the 0.55 threshold for moving into BTC. The portfolio was holding USDT, so the action is BUY.Simulated portfolioA simulation. It holds no real funds, and no trade is ever placed on an exchange.0.0295738 BTC and 0.00 USDT — worth 998.50 USDT at the decision priceNo decision has been published for today, so the one above is the most recent. The simulated portfolio has not moved since."`,
    );
  });

  it("the unavailable advice reads exactly as it did before the restyle", () => {
    const { container } = render(
      <AdvicePanel
        advice={advice({ freshness: "unavailable", action: null })}
      />,
    );
    expect(says(container)).toMatchInlineSnapshot(
      `"TodayNo decision availableThere is no advice to showNothing has been published yet, or the job that publishes it has not completed. This is not a view that the market is neutral — there is simply no decision to show."`,
    );
  });

  it("the paper track reads exactly as it did before the restyle", () => {
    const { container } = render(<PaperTrackPanel track={track()} />);
    expect(says(container)).toMatchInlineSnapshot(
      `"Since launchfrom 26 September 2026This track began on 26 September 2026 and covers 3 decision days. It is far too short to say anything about whether the advice makes money.The simulated portfolio is worth 1,005.00 USDT, valued at the most recent decision price.BTC/USDT, 26 September 2026 to 28 September 2026"`,
    );
  });

  it("the no-skill finding reads exactly as it did before the restyle", () => {
    // The sentence this whole page exists to make unmissable. It gained structure in the restyle and
    // must not have gained or lost a word.
    const { container } = render(
      <ReplayPanel
        replay={replay({ selection: { metric: "log_loss", score: 0.6944 } })}
      />,
    );
    expect(says(container)).toMatchInlineSnapshot(
      `"Evaluated on historyreplayA backtest over 1096 days, 1 January 2022 to 31 December 2024. These are results on past data, not money earned, and they are deliberately kept apart from the track since launch.Cost scenarioStrategyBuy & holdCashDrawdownTradesbase headline-12.0%+150.0%+0.0%-44.0%400optimistic+20.0%+150.0%+0.0%-40.0%400pessimistic-60.0%+150.0%+0.0%-70.0%400Selected on log loss: 0.6944. Model arm-a-logistic-1.0, arm A.A model that always answered 0.50 would score 0.6931, so this forecast shows no skill. Whatever the table above says about returns came from a handful of decisions, not from predicting anything."`,
    );
  });

  it("the replay reads exactly as it did before the restyle", () => {
    const { container } = render(<ReplayPanel replay={replay()} />);
    expect(says(container)).toMatchInlineSnapshot(
      `"Evaluated on historyreplayA backtest over 1096 days, 1 January 2022 to 31 December 2024. These are results on past data, not money earned, and they are deliberately kept apart from the track since launch.Cost scenarioStrategyBuy & holdCashDrawdownTradesbase headline-12.0%+150.0%+0.0%-44.0%400optimistic+20.0%+150.0%+0.0%-40.0%400pessimistic-60.0%+150.0%+0.0%-70.0%400Selected on log loss: 0.6931. Model arm-a-logistic-1.0, arm A."`,
    );
  });
});

describe("the visual system", () => {
  it("draws the same series in the same colour wherever it appears", () => {
    const { container } = render(<ReplayPanel replay={replay()} />);
    const swatches = [...container.querySelectorAll(".swatch")].map((node) =>
      node.className.replace("swatch ", ""),
    );
    // Strategy, the market, and cash — one identity each, in the order the columns read.
    expect(swatches).toEqual([
      "swatch--strategy",
      "swatch--market",
      "swatch--cash",
    ]);
    // A swatch carries colour, never text: identity is never the only thing a reader gets.
    for (const node of container.querySelectorAll(".swatch")) {
      expect(node.textContent).toBe("");
      expect(node.getAttribute("aria-hidden")).toBe("true");
    }
  });

  it("lets nothing outrank a series identity rule", () => {
    // The defect this exists for: `.chart polyline` is a type-plus-class selector and outranks the
    // single-class `.series--strategy`, so a stroke colour set there wins everywhere and quietly
    // draws all three curves in one colour. The switcher then looks broken precisely because the
    // data is right. A rendered test cannot see it — jsdom applies no stylesheet — so the guard
    // reads the stylesheet itself.
    const identities = new Set([
      ".series--strategy",
      ".series--market",
      ".series--cash",
    ]);
    const setters = new Set<string>();
    for (const rule of styleSheet.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const selector = rule[1] ?? "";
      const body = rule[2] ?? "";
      if (!/\bstroke:\s*var\(--series-/.test(body)) continue;
      // The last line of the selector, so a rule preceded by a comment reports the selector.
      setters.add(selector.trim().split("\n").slice(-1)[0]?.trim() ?? "");
    }
    expect(setters).toEqual(identities);
  });

  it("gives every drawn line an identity of its own", () => {
    // Nothing may rely on a stylesheet default for its colour: a line with no `.series--*` class is
    // a line whose identity depends on which rule happens to reach it.
    for (const node of [
      render(<PaperTrackPanel track={track()} />).container,
      render(<ReplaySeriesPanel series={replaySeries()} replay={replay()} />)
        .container,
    ]) {
      const lines = [...node.querySelectorAll("polyline")];
      expect(lines.length).toBeGreaterThan(0);
      for (const line of lines) {
        expect(line.getAttribute("class")).toMatch(
          /\bseries--(strategy|market|cash)\b/,
        );
      }
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

describe("the replay series panel", () => {
  it("draws three curves from identical capital, with cash flat at the start", () => {
    const { container } = render(
      <ReplaySeriesPanel series={replaySeries()} replay={replay()} />,
    );
    const chart = container.querySelector('[data-testid="equity-chart"]');
    expect(chart).not.toBeNull();
    // `className` on an SVG element is an SVGAnimatedString, so the attribute is what to read.
    const lines = [...(chart?.querySelectorAll("polyline") ?? [])].map((node) =>
      node.getAttribute("class"),
    );
    expect(lines).toEqual([
      "series series--cash",
      "series series--market",
      "series series--strategy",
    ]);
    // Cash never moves, so its two endpoints sit at the same height.
    const cash =
      chart?.querySelector(".series--cash")?.getAttribute("points") ?? "";
    const [left, right] = cash.split(" ").map((pair) => pair.split(",")[1]);
    expect(left).toBe(right);
  });

  it("marks the reported maximum drawdown rather than the deepest point it drew", () => {
    const { container } = render(
      <ReplaySeriesPanel series={replaySeries()} replay={replay()} />,
    );
    expect(
      container.querySelector('[data-testid="underwater-chart"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="underwater-reported-max"]'),
    ).not.toBeNull();
    // The base scenario's published drawdown is 44%, and the caption says so in the same units.
    expect(container.textContent).toMatch(/reported maximum, -44\.0%/);
  });

  it("offers exactly the pre-registered scenarios, names the headline, and takes no typed rate", () => {
    const { container } = render(
      <ReplaySeriesPanel series={replaySeries()} replay={replay()} />,
    );
    const options = [...container.querySelectorAll('input[type="radio"]')].map(
      (node) => (node as HTMLInputElement).value,
    );
    expect(options).toEqual(["base", "optimistic", "pessimistic"]);
    expect(container.textContent).toMatch(/base\s*headline/);
    // A free rate field would produce a number the pre-registration never covered.
    expect(
      container.querySelectorAll('input[type="number"], input[type="text"]'),
    ).toHaveLength(0);
  });

  it("changes the curves and the figures when the scenario changes, and never the actions", () => {
    const { container } = render(
      <ReplaySeriesPanel series={replaySeries()} replay={replay()} />,
    );
    const actionsBefore = container.querySelector(
      '[data-testid="action-history"]',
    )?.textContent;
    const curveBefore = container
      .querySelector(".series--strategy")
      ?.getAttribute("points");
    const figuresBefore = container.querySelector(
      '[data-testid="series-figures"]',
    )?.textContent;

    // fireEvent rather than a new dependency: one radio click does not earn @testing-library/user-event.
    fireEvent.click(screen.getByRole("radio", { name: /pessimistic/ }));

    expect(
      container.querySelector(".series--strategy")?.getAttribute("points"),
    ).not.toBe(curveBefore);
    expect(
      container.querySelector('[data-testid="series-figures"]')?.textContent,
    ).not.toBe(figuresBefore);
    // Costs change what the portfolio is worth, never what the Policy decided.
    expect(
      container.querySelector('[data-testid="action-history"]')?.textContent,
    ).toBe(actionsBefore);
  });

  // The published aggregate says the deepest drawdown was 70% at worst. This series says one
  // scenario fell further than that, which is the disagreement the underwater chart exists to make
  // visible instead of absorbing.
  function seriesThatOvershootsItsAggregate() {
    const built = replaySeries();
    return replaySeries({
      scenarios: {
        ...built.scenarios,
        pessimistic: built.scenarios.pessimistic!.map((point, index) => ({
          ...point,
          value_usdt: index === 2 ? 200 : point.value_usdt,
        })),
      },
    });
  }

  it("keeps one depth scale across a switch, so a shallower drawdown draws shallower", () => {
    const { container } = render(
      <ReplaySeriesPanel
        series={seriesThatOvershootsItsAggregate()}
        replay={replay()}
      />,
    );
    // The chart's own geometry. Duplicated here on purpose: a test reading the constants from the
    // component would follow it wherever it went, and the claim is about where a depth lands.
    const PADDING = 8;
    const PLOT = 120 - 2 * PADDING;
    const markerY = () =>
      Number(
        container
          .querySelector('[data-testid="underwater-reported-max"]')
          ?.getAttribute("y1"),
      );

    const published = replay().scenarios;
    const deepest = Math.max(...published.map((s) => s.max_drawdown));
    const atBase = published.find((s) => s.name === "base")!.max_drawdown;

    // The deepest *published* drawdown owns the bottom of the plot under every scenario, so a
    // shallower one sits proportionally higher — and stays exactly there when the reader switches.
    // The one scenario whose drawn path goes deeper than `deepest` must not stretch the axis to fit
    // itself; that is the switch on which the whole comparison would silently flatten.
    expect(markerY()).toBeCloseTo(PADDING + (atBase / deepest) * PLOT, 0);

    fireEvent.click(screen.getByRole("radio", { name: /pessimistic/ }));
    expect(markerY()).toBeCloseTo(PADDING + PLOT, 0);
  });

  it("says the series and the aggregate disagree when the drawn path goes deeper", () => {
    const { container } = render(
      <ReplaySeriesPanel
        series={seriesThatOvershootsItsAggregate()}
        replay={replay()}
      />,
    );
    // Base is untouched, so nothing is owed on arrival.
    expect(container.textContent).not.toMatch(/disagree/);

    fireEvent.click(screen.getByRole("radio", { name: /pessimistic/ }));
    expect(container.textContent).toMatch(
      /the day-by-day series and the aggregate disagree, and the aggregate is the authority/,
    );
  });

  it("refuses a scenario with a series but no published aggregate, rather than dropping it", () => {
    const built = replaySeries();
    const { container } = render(
      <ReplaySeriesPanel
        series={replaySeries({
          cost_scenarios: [...built.cost_scenarios, "hypothetical"],
          scenarios: { ...built.scenarios, hypothetical: built.scenarios.base! },
        })}
        replay={replay()}
      />,
    );
    // Reachable only because the switcher offers it: the headline is published on both sides.
    expect(
      container.querySelector('[data-testid="series-scenario-missing"]'),
    ).toBeNull();

    fireEvent.click(screen.getByRole("radio", { name: /hypothetical/ }));
    expect(
      container.querySelector('[data-testid="series-scenario-missing"]'),
    ).not.toBeNull();
    // Drawing one without the other would put a curve on the page with no figure to check it
    // against, so neither chart is drawn.
    expect(container.querySelector('[data-testid="equity-chart"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="underwater-chart"]'),
    ).toBeNull();
  });

  it("says so plainly when no day-by-day evaluation has been published", () => {
    const { container } = render(
      <ReplaySeriesPanel
        series={replaySeries({ window: null, days: [], scenarios: {} })}
        replay={replay()}
      />,
    );
    expect(container.textContent).toMatch(
      /No day-by-day evaluation has been published yet/,
    );
  });
});
