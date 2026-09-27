/** The shapes the read API serves. Nothing here computes advice; it only describes what arrives. */

export type Freshness = "current" | "stale" | "unavailable";
export type Action = "BUY" | "HOLD" | "REDUCE";

export interface Advice {
  asset: string;
  protocol: { horizon_days: number; decision_deadline_utc: string };
  freshness: Freshness;
  model_mode: string | null;
  decision_day: string | null;
  data_cutoff: string | null;
  decision_price: number | null;
  probability: number | null;
  action: Action | null;
  target_exposure: number | null;
  position: { btc: number | null; usdt: number | null; value_usdt: number | null };
  explanation: string | null;
  versions: { model: string | null; policy: string | null; contract: string | null };
}

export interface TrackPoint {
  day: string;
  action: Action;
  btc: number;
  usdt: number;
  price: number;
  value_usdt: number;
}

export interface PaperTrack {
  asset: string;
  genesis_day: string | null;
  points: TrackPoint[];
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} responded ${response.status}`);
  return (await response.json()) as T;
}

export const fetchAdvice = () => getJson<Advice>("/api/advice");
export const fetchPaperTrack = () => getJson<PaperTrack>("/api/paper-track");
