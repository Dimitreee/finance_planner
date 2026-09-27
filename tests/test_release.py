"""The Trial budget and the act of promoting a release.

A budget that counts bug fixes against you encourages leaving bugs in, so a rerun that replaces an
earlier result is recorded as a correction and does not consume a Trial. Promotion is a separate,
deliberate act: a better number promotes nothing by itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.release import (
    LogisticRelease,
    current_release,
    load_release,
    promote,
    rollback_to,
)
from cryptoguard_core.training import FoldPreprocessing
from cryptoguard_core.trials import TrialLog

PREPROCESSING = FoldPreprocessing(
    features=("r_1", "rv_7"), medians=(0.0, 0.01), means=(0.0, 0.01), stds=(1.0, 1.0)
)


def release(version: str) -> LogisticRelease:
    return LogisticRelease(
        version=version,
        preprocessing=PREPROCESSING,
        coefficients=(0.5, -0.25),
        intercept=0.1,
        contract_digest="deadbeef",
        arm="A",
    )


class TestTrialBudget:
    def test_each_looked_at_result_spends_one_trial(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "trials.json", budget=3)
        log.record(arm="A", lag_hours=None, contract_digest="d", score=0.69)
        log.record(arm="B", lag_hours=24, contract_digest="d", score=0.68)
        assert log.spent() == 2
        assert log.remaining() == 1

    def test_a_correction_replaces_a_trial_rather_than_consuming_one(self, tmp_path: Path) -> None:
        """A budget that punishes fixing a bug would encourage leaving it in."""
        log = TrialLog(tmp_path / "trials.json", budget=2)
        first = log.record(arm="A", lag_hours=None, contract_digest="d", score=0.69)
        log.record(arm="A", lag_hours=None, contract_digest="d", score=0.66, corrects=first.number)
        assert log.spent() == 1
        assert log.remaining() == 1
        assert [entry.corrects for entry in log.entries()] == [None, first.number]

    def test_exhausting_the_budget_refuses_the_next_trial(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "trials.json", budget=1)
        log.record(arm="A", lag_hours=None, contract_digest="d", score=0.69)
        with pytest.raises(FatalDefect) as caught:
            log.record(arm="B", lag_hours=24, contract_digest="d", score=0.68)
        assert caught.value.kind == "trial_budget_exhausted"

    def test_correcting_a_trial_that_does_not_exist_is_refused(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "trials.json", budget=3)
        with pytest.raises(FatalDefect) as caught:
            log.record(arm="A", lag_hours=None, contract_digest="d", score=0.6, corrects=99)
        assert caught.value.kind == "unknown_trial"

    def test_the_next_number_advances_even_for_a_correction(self, tmp_path: Path) -> None:
        """A version derived from the spend would collide across two corrections."""
        log = TrialLog(tmp_path / "trials.json", budget=5)
        first = log.record(arm="A", lag_hours=None, contract_digest="d", score=0.69)
        assert log.next_number() == 2
        log.record(arm="A", lag_hours=None, contract_digest="d", score=0.68, corrects=first.number)
        assert log.spent() == 1
        assert log.next_number() == 3

    def test_the_log_survives_a_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "trials.json"
        TrialLog(path, budget=3).record(arm="A", lag_hours=None, contract_digest="d", score=0.69)
        assert TrialLog(path, budget=3).spent() == 1


class TestPromotion:
    def test_writing_a_release_does_not_promote_it(self, tmp_path: Path) -> None:
        """A better backtest number promotes nothing by itself."""
        stored = release("arm-a-logistic-1.0")
        stored.save(tmp_path)
        assert load_release(tmp_path, stored.version).version == stored.version
        assert current_release(tmp_path) is None

    def test_promotion_is_a_separate_deliberate_act(self, tmp_path: Path) -> None:
        stored = release("arm-a-logistic-1.0")
        stored.save(tmp_path)
        promote(tmp_path, stored.version)
        promoted = current_release(tmp_path)
        assert promoted is not None
        assert promoted.version == stored.version

    def test_promoting_a_release_that_was_never_written_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FatalDefect) as caught:
            promote(tmp_path, "arm-a-logistic-9.9")
        assert caught.value.kind == "unknown_release"

    def test_rollback_switches_the_pointer_and_keeps_both_releases(self, tmp_path: Path) -> None:
        first, second = release("arm-a-logistic-1.0"), release("arm-a-logistic-2.0")
        first.save(tmp_path)
        second.save(tmp_path)
        promote(tmp_path, second.version)
        rollback_to(tmp_path, first.version)
        promoted = current_release(tmp_path)
        assert promoted is not None and promoted.version == first.version
        assert load_release(tmp_path, second.version).version == second.version

    def test_a_release_predicts_without_clipping(self, tmp_path: Path) -> None:
        """Clipping belongs to the previous-direction baseline alone."""
        stored = release("arm-a-logistic-1.0")
        probability = stored.predict({"r_1": 5.0, "rv_7": 0.01})
        assert 0.0 < probability < 1.0
        assert probability > 0.9, "a large positive feature should not be clipped back to 0.95"
