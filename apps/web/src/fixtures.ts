import type { Advice, PaperTrack } from "./api";

export function advice(overrides: Partial<Advice> = {}): Advice {
  return {
    asset: "BTCUSDT",
    protocol: { horizon_days: 1, decision_deadline_utc: "00:10" },
    freshness: "current",
    model_mode: "price-only",
    decision_day: "2026-09-28",
    data_cutoff: "2026-09-28T00:00:00+00:00",
    decision_price: 33779.94,
    probability: 0.95,
    action: "BUY",
    target_exposure: 1,
    position: { btc: 0.0295738, usdt: 0, value_usdt: 998.5 },
    explanation:
      "The model puts the probability of a higher price tomorrow at 0.95, and that is at or above " +
      "the 0.55 threshold for moving into BTC. The portfolio was holding USDT, so the action is BUY.",
    versions: { model: "previous-direction-1.0", policy: "5264b71ca960", contract: "fac3574a8922" },
    ...overrides,
  };
}

export function track(overrides: Partial<PaperTrack> = {}): PaperTrack {
  return {
    asset: "BTCUSDT",
    genesis_day: "2026-09-26",
    points: [
      { day: "2026-09-26", action: "BUY", btc: 0.03, usdt: 0, price: 33000, value_usdt: 990 },
      { day: "2026-09-27", action: "HOLD", btc: 0.03, usdt: 0, price: 34000, value_usdt: 1020 },
      { day: "2026-09-28", action: "HOLD", btc: 0.03, usdt: 0, price: 33500, value_usdt: 1005 },
    ],
    ...overrides,
  };
}
