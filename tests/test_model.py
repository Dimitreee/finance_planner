"""The first promoted Model Release: a baseline meant to exercise the path, not to be good."""

from __future__ import annotations

import pytest
from cryptoguard_core.model import PreviousDirectionBaseline, forecast_with


def test_the_baseline_follows_yesterdays_direction() -> None:
    model = PreviousDirectionBaseline()
    assert model.predict({"r_1": 0.01}) == 0.95
    assert model.predict({"r_1": -0.01}) == 0.05
    assert model.predict({"r_1": 0.0}) == 0.05, "an unchanged price is not a rise"


def test_clipping_keeps_log_loss_finite() -> None:
    """A hard 0 or 1 makes log loss infinite on a confident miss, so this baseline is clipped."""
    model = PreviousDirectionBaseline()
    assert model.clip_low == 0.05
    assert model.clip_high == 0.95
    assert 0.0 < model.predict({"r_1": 1.0}) < 1.0


def test_the_baseline_crosses_the_policy_thresholds_in_both_directions() -> None:
    """A majority baseline emits a constant near 0.53 and would never leave HOLD."""
    model = PreviousDirectionBaseline()
    assert model.predict({"r_1": 0.01}) >= 0.55
    assert model.predict({"r_1": -0.01}) <= 0.45


def test_a_forecast_records_the_model_and_the_features_it_saw() -> None:
    model = PreviousDirectionBaseline()
    features = {"r_1": 0.02, "rv_7": 0.001}
    forecast = forecast_with(model, features)
    assert forecast.probability == 0.95
    assert forecast.model_version == model.version
    assert forecast.feature_snapshot == features


def test_a_forecast_refuses_features_the_model_needs_and_did_not_get() -> None:
    model = PreviousDirectionBaseline()
    with pytest.raises(KeyError):
        forecast_with(model, {"rv_7": 0.001})
