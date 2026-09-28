"""The per-day Replay series as data, with nothing attached that could compute one.

This module exists because of a boundary the read API defends by test: importing the API may not
pull in `model`, `ingest`, `dataset`, `decide` or `news`. The store has to speak this vocabulary
to publish and read a series, and if the types lived beside the simulator that runs one, the
store's import would have dragged the whole modelling stack into the API process.

So the shapes live here, where the only dependency is what an Action is. `backtest` fills them in;
`store` writes and reads them; neither has to know about the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from cryptoguard_core.policy import Action


@dataclass(frozen=True, slots=True)
class SeriesDay:
    """What was decided on one Decision Day. The same under every Cost Scenario, by construction.

    Costs change what the portfolio is worth; they do not change what the Policy decided. The Action
    comes from the probability and from whether the portfolio was in or out, and with a target
    exposure of exactly 0 or 1 a fee cannot move it. So these facts are stored once rather than per
    scenario, which is what makes "switching scenario never changes the Action history" impossible
    to break instead of merely tested.
    """

    day: date
    price: float
    probability: float
    action: Action


@dataclass(frozen=True, slots=True)
class SeriesScenarioDay:
    """What one Cost Scenario's money did on one day: the strategy's, and holding's for comparison.

    Both are per-scenario because both pay costs: buy-and-hold pays a fee on the one purchase that
    starts it. Cash is not here: it is the flat line at the series' starting capital, and a column
    repeating one number for every day would invite it to stop being constant.
    """

    day: date
    cost_scenario: str
    btc: float
    usdt: float
    value_usdt: float
    buy_and_hold_usdt: float


@dataclass(frozen=True, slots=True)
class ReplaySeries:
    """The day-by-day Replay, assembled from the three scenarios that were simulated together."""

    first_day: date
    last_day: date
    start_value_usdt: float
    headline_scenario: str
    days: tuple[SeriesDay, ...]
    by_scenario: Mapping[str, tuple[SeriesScenarioDay, ...]]
