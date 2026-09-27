import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { advice, replay, track } from "./fixtures";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function stubApi(): () => number {
  let calls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      calls += 1;
      const body = url.includes("paper-track")
        ? track()
        : url.includes("replay")
          ? replay()
          : advice();
      return { ok: true, json: async () => body } as Response;
    }),
  );
  return () => calls;
}

describe("the page", () => {
  it("refetches, so a tab left open across UTC midnight stops claiming to be current", async () => {
    const calls = stubApi();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<App />);
    await waitFor(() => expect(screen.getByText("Buy BTC")).toBeInTheDocument());

    const before = calls();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(calls()).toBeGreaterThan(before);
  });

  it("keeps the page alive when only the evaluation is unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("replay")) return { ok: false, status: 500 } as Response;
        const body = url.includes("paper-track") ? track() : advice();
        return { ok: true, json: async () => body } as Response;
      }),
    );
    render(<App />);
    await waitFor(() => expect(screen.getByText("Buy BTC")).toBeInTheDocument());
    expect(screen.getByText(/No evaluation has been published yet/i)).toBeInTheDocument();
  });

  it("says a failure to reach the service is not a view about the market", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 }) as Response));
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/not a view about the market/i)).toBeInTheDocument(),
    );
  });
});


describe("the panels", () => {
  it("keeps the track since launch and the history evaluation apart", async () => {
    stubApi();
    render(<App />);
    await waitFor(() => expect(screen.getByText("Buy BTC")).toBeInTheDocument());

    // Two separate sections, and only one series is drawn: joining them would turn a backtest
    // into something a reader takes for earnings.
    expect(screen.getByRole("region", { name: /since launch/i })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /evaluated on history/i })).toBeInTheDocument();
    expect(screen.getAllByTestId("price-chart")).toHaveLength(1);
    const evaluation = screen.getByRole("region", { name: /evaluated on history/i });
    expect(evaluation.querySelector("svg")).toBeNull();
  });
});
