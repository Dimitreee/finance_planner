"""The Final Holdout: one configuration, one evaluation, and a guard that makes a second an error.

The holdout is the only measurement in this project that cannot be repeated. Everything here exists
to make that structurally true rather than a matter of discipline: the refusal comes before any
number is computed, the frozen configuration is checked against the contract before the fit, and the
marker records what was spent so a later reader can tell whether the claim rests on it (ADR-0018).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from cryptoguard_core.contract import CONTRACT_PATH, load_contract
from cryptoguard_core.dataset import ARM_A_FEATURES, DAY_MS, DecisionDayRow
from cryptoguard_core.evaluate import development_last_day
from cryptoguard_core.holdout import (
    HOLDOUT_ARM,
    FrozenConfiguration,
    HoldoutResult,
    HoldoutSpend,
    evaluate_final_holdout,
    frozen_configuration,
    holdout_fold,
    read_spend,
    refuse_if_spent,
    write_spend,
)
from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.policy import HEADLINE_SCENARIO
from cryptoguard_core.release import LogisticRelease
from cryptoguard_core.store import PublishedReplay
from cryptoguard_core.training import FoldPreprocessing, admissible_rows, fit_fold


def configuration(digest: str = "f" * 64) -> FrozenConfiguration:
    return FrozenConfiguration(
        arm="A",
        lag_hours=None,
        cost_scenario="base",
        to_btc_at=0.55,
        to_usdt_at=0.45,
        regularisation_grid=(0.01, 0.1, 1.0),
        regularisation_selected="within_fold_inner_validation",
        contract_digest=digest,
    )


def result() -> HoldoutResult:
    return HoldoutResult(
        first_day=date(2025, 1, 1),
        last_day=date(2025, 11, 30),
        scored_days=333,
        log_loss=0.69,
        brier=0.25,
        accuracy=0.5,
        net_return=0.1,
        buy_and_hold_return=0.2,
        trades=7,
        max_drawdown=0.3,
        turnover=1.5,
        time_invested=0.4,
    )


def spend() -> HoldoutSpend:
    return HoldoutSpend(
        spent_at="2026-09-28T12:00:00+00:00",
        evaluations_allowed=1,
        configuration=configuration(),
        result=result(),
        spends_a_trial=False,
        trial_reasoning="confirmatory: the configuration was frozen before the run and nothing "
        "downstream may change because of the number",
    )


def test_an_unspent_holdout_reads_as_nothing_and_refuses_nothing(tmp_path: Path) -> None:
    marker = tmp_path / "final_holdout.json"
    assert read_spend(marker) is None
    refuse_if_spent(marker)  # does not raise


def test_a_spend_survives_a_round_trip_through_its_marker(tmp_path: Path) -> None:
    """The marker is the only evidence the holdout was spent, so it has to read back exactly."""
    marker = tmp_path / "final_holdout.json"
    write_spend(marker, spend())
    assert read_spend(marker) == spend()


def test_a_second_evaluation_is_refused_by_name(tmp_path: Path) -> None:
    """ADR-0018: one evaluation. The refusal is the mechanism, not the intention."""
    marker = tmp_path / "final_holdout.json"
    write_spend(marker, spend())
    with pytest.raises(FatalDefect) as caught:
        refuse_if_spent(marker)
    assert caught.value.kind == "final_holdout_spent"
    assert "2026-09-28" in str(caught.value)


def test_writing_a_second_marker_over_the_first_is_refused(tmp_path: Path) -> None:
    """Overwriting the marker would erase the only record that the holdout is gone."""
    marker = tmp_path / "final_holdout.json"
    write_spend(marker, spend())
    with pytest.raises(FatalDefect) as caught:
        write_spend(marker, spend())
    assert caught.value.kind == "final_holdout_spent"


# --- The single fold, and where its training stops ------------------------------------------------

FIRST_DAY = date(2021, 2, 1)
FIRST_CUTOFF_MS = 1_612_137_600_000  # 2021-02-01T00:00:00Z


def row(index: int) -> DecisionDayRow:
    cutoff = FIRST_CUTOFF_MS + index * DAY_MS
    return DecisionDayRow(
        day=FIRST_DAY + timedelta(days=index),
        feature_cutoff_ms=cutoff,
        anchor_price=100.0 + index * 0.01,
        decision_price=100.0 + index * 0.01,
        label=index % 2,
        label_end_ms=cutoff + DAY_MS + HOUR_MS,
        bars_in_day=24,
        features={
            name: index * 0.001 + position * 0.01 for position, name in enumerate(ARM_A_FEATURES)
        },
        news_lag_hours=None,
    )


def series_through_holdout() -> tuple[DecisionDayRow, ...]:
    """Every Decision Day of the Research Window, including the Final Holdout's."""
    contract = load_contract()
    last = contract.values["splits"]["final_holdout_last_day"]
    return tuple(row(index) for index in range((last - FIRST_DAY).days + 1))


def test_the_fold_is_exactly_the_holdout_range() -> None:
    contract = load_contract()
    fold = holdout_fold(contract)
    assert fold.test_first_day == contract.values["splits"]["final_holdout_first_day"]
    assert fold.test_last_day == contract.values["splits"]["final_holdout_last_day"]


def test_the_training_data_stops_before_the_holdout_opens() -> None:
    """The one measurement that must be clean. A row whose outcome resolves inside the holdout is
    excluded by the admissibility rule, not a date filter, so the boundary cannot be off by a day.
    """
    contract = load_contract()
    fold = holdout_fold(contract)
    rows = series_through_holdout()
    fitted = fit_fold(
        rows,
        fold,
        features=ARM_A_FEATURES,
        regularisation=tuple(contract.values["model"]["regularisation_grid"]),
        inner_validation_days=contract.values["splits"]["inner_validation_days"],
    )
    cutoff_ms = int(
        datetime(
            fold.test_first_day.year, fold.test_first_day.month, fold.test_first_day.day, tzinfo=UTC
        ).timestamp()
        * 1000
    )
    admitted = admissible_rows(rows, cutoff_ms)
    assert fitted.training_days == len(admitted)
    assert max(entry.day for entry in admitted) < fold.test_first_day


def test_a_second_evaluation_refuses_before_it_computes_anything(tmp_path: Path) -> None:
    """Ordering, not politeness: a run that computed the number and then declined to print it has
    already seen it. Handed no rows at all, the refusal still has to be the one that fires.
    """
    marker = tmp_path / "final_holdout.json"
    write_spend(marker, spend())
    with pytest.raises(FatalDefect) as caught:
        evaluate_final_holdout((), load_contract(), marker_path=marker)
    assert caught.value.kind == "final_holdout_spent"


# --- The freeze, and what may not carry a holdout number ------------------------------------------


def test_the_contract_and_the_code_agree_on_the_frozen_configuration() -> None:
    """The freeze is only a freeze if both sides say the same thing."""
    configuration = frozen_configuration(load_contract())
    assert configuration.arm == "A"
    assert configuration.lag_hours is None
    assert configuration.cost_scenario == HEADLINE_SCENARIO
    assert configuration.contract_digest == load_contract().digest


def test_a_configuration_that_drifted_from_the_code_is_refused(tmp_path: Path) -> None:
    """Checked where it is used rather than in the loader, because `contract.py` cannot import this
    module without a cycle — and this is the only place the frozen configuration is read.
    """
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        f"  arm: {HOLDOUT_ARM}\n  lag_hours: null", "  arm: c\n  lag_hours: 24", 1
    )
    drifted = tmp_path / "experiment.yaml"
    drifted.write_text(text, encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        frozen_configuration(load_contract(drifted))
    assert caught.value.kind == "final_holdout_not_frozen"


def test_a_release_record_carries_no_final_holdout_field(tmp_path: Path) -> None:
    """ADR-0004: a bundle retrained on later price history may not be attributed a holdout metric.

    The strongest form of that rule is that the record has nowhere to put one, so no later code can
    fill a field that a reader would take as a holdout number for a retrained model.
    """
    release = LogisticRelease(
        version="arm-a-logistic-99.0",
        preprocessing=FoldPreprocessing(
            features=ARM_A_FEATURES,
            medians=(0.0,) * len(ARM_A_FEATURES),
            means=(0.0,) * len(ARM_A_FEATURES),
            stds=(1.0,) * len(ARM_A_FEATURES),
        ),
        coefficients=(0.0,) * len(ARM_A_FEATURES),
        intercept=0.0,
        contract_digest="d" * 64,
        arm="A",
    )
    written = json.loads(release.save(tmp_path).read_text(encoding="utf-8"))
    assert not [key for key in written if "holdout" in key.lower()]
    assert not [field.name for field in fields(LogisticRelease) if "holdout" in field.name.lower()]


def test_a_published_replay_has_nowhere_to_put_a_holdout_number() -> None:
    """The holdout is its own row. A Replay Result that could carry it could also average it in."""
    assert not [field.name for field in fields(PublishedReplay) if "holdout" in field.name.lower()]


def test_the_holdout_window_begins_after_every_development_day() -> None:
    """The two windows cannot overlap, so no row can belong to both and be averaged across them."""
    contract = load_contract()
    assert holdout_fold(contract).test_first_day > development_last_day(contract)


# --- The command, invoked twice -------------------------------------------------------------------

COMMAND = Path(__file__).resolve().parents[1] / "scripts/final_holdout.py"
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data/raw/binance/klines/BTCUSDT/1h"


def run_command(marker: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(COMMAND), "--marker", str(marker)],
        capture_output=True,
        text=True,
        check=False,
        cwd=COMMAND.parents[1],
    )


def test_the_command_refuses_against_a_spent_marker_without_producing_a_number(
    tmp_path: Path,
) -> None:
    """The refusal, proved without the price snapshot — which `data/raw/` keeps out of a clone.

    The other test below covers success-then-refusal, but it can only run where the archive is
    present, so on a fresh checkout or in CI it skips. The refusal is the half that must never go
    unproven, so it is proved from a marker written by hand and nothing else.
    """
    marker = tmp_path / "final_holdout.json"
    write_spend(marker, spend())
    refused = run_command(marker)
    assert refused.returncode != 0
    assert "log loss:" not in refused.stdout
    assert "final_holdout_spent" in refused.stderr
    assert read_spend(marker) == spend()


@pytest.mark.skipif(
    not SNAPSHOT_DIR.exists(), reason="Archive Backfill snapshot is not in this checkout"
)
def test_the_command_succeeds_once_and_then_refuses_without_a_number(tmp_path: Path) -> None:
    """The ticket's own acceptance test: run it twice, and the second run must produce nothing.

    A subprocess rather than a function call, because what is checked is the command's contract
    with whoever types it — a non-zero exit and no number on stdout.
    """
    marker = tmp_path / "final_holdout.json"
    first = run_command(marker)
    assert first.returncode == 0, first.stderr
    assert "log loss:" in first.stdout
    assert marker.is_file()

    second = run_command(marker)
    assert second.returncode != 0
    assert "log loss:" not in second.stdout
    assert "final_holdout_spent" in second.stderr
    # The marker the first run wrote is untouched by the refusal.
    assert read_spend(marker) is not None
