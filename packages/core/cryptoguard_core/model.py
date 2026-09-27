"""Model releases and the forecasts they produce.

A Model Release is promoted deliberately; a better backtest number promotes nothing by itself. The
first one through the whole path is a previous-direction baseline, chosen because a majority
baseline emits a constant near the training up-rate, never crosses the Policy thresholds, and leaves
`rebalance`, the costs and the trade columns unexercised while the path looked complete.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


class ModelRelease(Protocol):
    """What a promoted release must offer: a version to record and a probability to act on."""

    @property
    def version(self) -> str: ...

    def predict(self, features: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, slots=True)
class Forecast:
    """A probability with the version and the feature snapshot that produced it."""

    probability: float
    model_version: str
    feature_snapshot: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PreviousDirectionBaseline:
    """Predict a rise when yesterday rose, with probabilities clipped away from 0 and 1.

    The clipping belongs to this baseline alone: a hard 0 or 1 makes log loss infinite on a
    confident miss. It is never applied to a fitted model, where it would distort the calibration
    being measured.
    """

    clip_low: float = 0.05
    clip_high: float = 0.95
    version: str = field(default="previous-direction-1.0")

    def predict(self, features: Mapping[str, float]) -> float:
        return self.clip_high if features["r_1"] > 0 else self.clip_low


def forecast_with(model: ModelRelease, features: Mapping[str, float]) -> Forecast:
    return Forecast(
        probability=model.predict(features),
        model_version=model.version,
        feature_snapshot=dict(features),
    )
