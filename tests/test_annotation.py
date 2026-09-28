"""The Annotation Sample: the draw, the blind labelling surface, and the agreement figure.

Nothing here loads the Sentiment Extractor. Readings are supplied through a cache built by hand, so
the expected agreement is stated rather than recomputed by the code under test.

The sample measures the instrument (ADR-0016). It is never training data, never a tuning target and
gates nothing, so no test here asserts that a particular agreement figure is good enough.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest
from cryptoguard_core.annotation import (
    VOLUME_BANDS,
    WORDING_WHEN_NOT,
    WORDING_WHEN_TRANSFERS,
    AnnotationSample,
    SampledHeadline,
    Stratum,
    agreement,
    draw_annotation_sample,
    read_labels,
    read_sample,
    write_agreement,
    write_labelling_file,
    write_sample,
)
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news import NewsItem
from cryptoguard_core.sentiment import EXTRACTOR_REVISION, Reading, ReadingCache

FIRST_DAY = date(2021, 2, 1)
LAST_DAY = date(2023, 12, 31)


def item(item_id: str, *, stated: datetime, title: str | None = None) -> NewsItem:
    return NewsItem(
        item_id=item_id,
        title=title or f"Headline {item_id}",
        description=None,
        source_domain=None,
        source_url=None,
        stated_at=stated,
        stated_at_raw=stated.strftime("%Y-%m-%d %H:%M:%S"),
        stated_timezone_suffix=None,
        currencies=("BTC",),
    )


def corpus() -> list[NewsItem]:
    """Three years of BTC headlines whose daily volume varies, so the terciles are not degenerate.

    A day's volume cycles through 1, 4 and 12 headlines, which puts days in all three bands in every
    calendar year. Ordered by day so that a sampler reading file order rather than sorting is caught
    by the reproducibility test.
    """
    items: list[NewsItem] = []
    day = FIRST_DAY
    counter = 0
    next_id = 1000
    while day <= LAST_DAY:
        per_day = (1, 4, 12)[counter % 3]
        for index in range(per_day):
            # Opaque ids on purpose: an id carrying a date would leak one to the annotator, and the
            # blindness assertions below would pass for the wrong reason.
            items.append(
                item(str(next_id), stated=datetime(day.year, day.month, day.day, 6 + index))
            )
            next_id += 1
        counter += 1
        day = date.fromordinal(day.toordinal() + 1)
    return items


def test_the_draw_is_reproducible_from_the_seed_and_moves_when_it_changes() -> None:
    items = corpus()
    first = draw_annotation_sample(items, seed=7, size=30, first_day=FIRST_DAY, last_day=LAST_DAY)
    again = draw_annotation_sample(items, seed=7, size=30, first_day=FIRST_DAY, last_day=LAST_DAY)
    other = draw_annotation_sample(items, seed=8, size=30, first_day=FIRST_DAY, last_day=LAST_DAY)

    drawn = [sampled.item_id for sampled in first.items]
    assert len(drawn) == 30
    assert drawn == [sampled.item_id for sampled in again.items]
    assert drawn != [sampled.item_id for sampled in other.items]
    assert first.seed == 7


def test_every_stratum_is_represented_and_the_allocation_sums_to_the_request() -> None:
    sample = draw_annotation_sample(
        corpus(), seed=7, size=60, first_day=FIRST_DAY, last_day=LAST_DAY
    )
    assert sum(sample.allocation.values()) == 60
    assert len(sample.items) == 60
    assert set(sample.allocation) == set(sample.population)
    assert min(sample.allocation.values()) >= 1
    assert {stratum.calendar_year for stratum in sample.allocation} == {2021, 2022, 2023}
    assert {stratum.volume_band for stratum in sample.allocation} == set(VOLUME_BANDS)


def test_allocation_follows_the_corpus_rather_than_the_number_of_strata() -> None:
    """The corpus puts 1, 4 and 12 headlines on equally many days, so the bands are 1:4:12.

    Equal-per-cell allocation would give the three bands the same count; proportional allocation
    must not, or the figure would not transfer back to the corpus it is meant to describe.
    """
    sample = draw_annotation_sample(
        corpus(), seed=7, size=90, first_day=FIRST_DAY, last_day=LAST_DAY
    )
    by_band = {
        band: sum(
            count for stratum, count in sample.allocation.items() if stratum.volume_band == band
        )
        for band in VOLUME_BANDS
    }
    assert by_band["busy"] > by_band["typical"] > by_band["quiet"]


def test_a_request_smaller_than_the_strata_is_refused_rather_than_silently_reduced() -> None:
    with pytest.raises(FatalDefect) as caught:
        draw_annotation_sample(corpus(), seed=7, size=5, first_day=FIRST_DAY, last_day=LAST_DAY)
    assert caught.value.kind == "annotation_sample_too_small"


# --- The blind labelling surface ------------------------------------------------------------------


def sample_of(size: int = 30, seed: int = 7) -> AnnotationSample:
    return draw_annotation_sample(
        corpus(), seed=seed, size=size, first_day=FIRST_DAY, last_day=LAST_DAY
    )


def ids_in(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["item_id"] for row in csv.DictReader(handle)]


def test_the_labelling_file_carries_the_item_id_and_the_title_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Anything else on the row is a hint. The strata stay in the stored sample, never here."""
    sample = sample_of()
    path = tmp_path / "labelling.csv"
    write_labelling_file(sample, path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["item_id", "title"]
    assert len(rows) == len(sample.items) + 1
    assert {len(row) for row in rows[1:]} == {2}

    written = path.read_text(encoding="utf-8")
    for band in VOLUME_BANDS:
        assert band not in written
    for sampled in sample.items:
        assert sampled.stated_day.isoformat() not in written
        assert str(sampled.day_volume) not in written.split(",")


def test_the_labelling_order_is_shuffled_by_the_seed_so_time_does_not_leak(
    tmp_path: Path,
) -> None:
    """Ids rise with time in this archive, so id order would hand the annotator a chronology."""
    sample = sample_of()
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    write_labelling_file(sample, first)
    write_labelling_file(sample, second)

    order = ids_in(first)
    assert order == ids_in(second)
    assert order != [sampled.item_id for sampled in sample.items]
    assert sorted(order) == sorted(sampled.item_id for sampled in sample.items)


# --- Reading the labels back ----------------------------------------------------------------------


def labels_file(path: Path, rows: Sequence[Sequence[str]], header: tuple[str, ...]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_labels_are_read_back_by_item_id(tmp_path: Path) -> None:
    sample = sample_of()
    rows = [(sampled.item_id, "positive") for sampled in sample.items]
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "label"))
    assert read_labels(path, sample) == {sampled.item_id: "positive" for sampled in sample.items}


def test_a_title_column_is_accepted_and_checked_against_the_sample(tmp_path: Path) -> None:
    """The annotator will add a label column to a copy of the labelling file; that must work."""
    sample = sample_of()
    rows = [(sampled.item_id, sampled.title, "neutral") for sampled in sample.items]
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "title", "label"))
    assert set(read_labels(path, sample).values()) == {"neutral"}


def test_a_label_outside_the_permitted_four_is_refused(tmp_path: Path) -> None:
    sample = sample_of()
    rows = [(sampled.item_id, "bullish") for sampled in sample.items]
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "label"))
    with pytest.raises(FatalDefect) as caught:
        read_labels(path, sample)
    assert caught.value.kind == "annotation_labels_invalid"
    assert "bullish" in str(caught.value)


def test_a_headline_that_was_never_drawn_is_refused(tmp_path: Path) -> None:
    sample = sample_of()
    rows = [(sampled.item_id, "positive") for sampled in sample.items] + [("999999", "positive")]
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "label"))
    with pytest.raises(FatalDefect) as caught:
        read_labels(path, sample)
    assert caught.value.kind == "annotation_labels_invalid"


def test_the_same_headline_labelled_twice_is_refused(tmp_path: Path) -> None:
    sample = sample_of()
    rows = [(sampled.item_id, "positive") for sampled in sample.items]
    rows.append((sample.items[0].item_id, "negative"))
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "label"))
    with pytest.raises(FatalDefect) as caught:
        read_labels(path, sample)
    assert caught.value.kind == "annotation_labels_invalid"


def test_a_partially_labelled_sample_is_refused_rather_than_scored(tmp_path: Path) -> None:
    sample = sample_of()
    rows = [(sampled.item_id, "positive") for sampled in sample.items[:-1]]
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "label"))
    with pytest.raises(FatalDefect) as caught:
        read_labels(path, sample)
    assert caught.value.kind == "annotation_labels_invalid"


def test_a_shifted_row_is_caught_by_the_title_it_carries(tmp_path: Path) -> None:
    """The failure this guards against leaves the count intact and every label on the wrong item."""
    sample = sample_of()
    rows = [
        (sampled.item_id, shifted.title, "positive")
        for sampled, shifted in zip(sample.items, sample.items[1:], strict=False)
    ]
    rows.append((sample.items[-1].item_id, sample.items[-1].title, "positive"))
    path = labels_file(tmp_path / "labels.csv", rows, ("item_id", "title", "label"))
    with pytest.raises(FatalDefect) as caught:
        read_labels(path, sample)
    assert caught.value.kind == "annotation_labels_invalid"
    assert "shifted row" in str(caught.value)


# --- The agreement figure -------------------------------------------------------------------------

POSITIVE = Reading(p_positive=0.8, p_negative=0.1, p_neutral=0.1)
NEGATIVE = Reading(p_positive=0.1, p_negative=0.8, p_neutral=0.1)
NEUTRAL = Reading(p_positive=0.2, p_negative=0.2, p_neutral=0.6)


def hand_built_sample(titles: Sequence[str]) -> AnnotationSample:
    """A sample stated outright, so the expected agreement is arithmetic rather than a re-run."""
    items = tuple(
        SampledHeadline(
            item_id=str(index),
            title=title,
            stated_day=date(2022, 1, 1),
            volume_band=VOLUME_BANDS[1],
            day_volume=4,
        )
        for index, title in enumerate(titles)
    )
    only = Stratum(2022, VOLUME_BANDS[1])
    return AnnotationSample(
        seed=1,
        size=len(items),
        extractor_revision=EXTRACTOR_REVISION,
        volume_band_edges=(1, 4),
        population={only: len(items)},
        allocation={only: len(items)},
        items=items,
    )


def cache_with(tmp_path: Path, readings: Mapping[str, Reading]) -> ReadingCache:
    cache = ReadingCache(tmp_path / "readings.jsonl")
    for title, reading in readings.items():
        cache.put(title, reading)
    return cache


def test_unclear_is_counted_separately_and_never_scored_as_neutral(tmp_path: Path) -> None:
    """Five of six compared headlines agree; two `unclear` labels leave the figure alone."""
    titles = [f"t{index}" for index in range(8)]
    sample = hand_built_sample(titles)
    readings = {
        "t0": POSITIVE,
        "t1": POSITIVE,
        "t2": POSITIVE,
        "t3": NEUTRAL,
        "t4": NEGATIVE,
        "t5": NEUTRAL,
        "t6": NEUTRAL,
        "t7": NEUTRAL,
    }
    human = {
        "0": "positive",
        "1": "positive",
        "2": "positive",
        "3": "positive",
        "4": "negative",
        "5": "neutral",
        "6": "unclear",
        "7": "unclear",
    }
    result = agreement(sample, human, cache_with(tmp_path, readings))

    assert result.labelled == 8
    assert result.unclear == 2
    assert result.compared == 6
    assert result.agreed == 5
    assert result.agreement == pytest.approx(5 / 6)
    assert result.majority_class == "positive"
    assert result.majority_class_rate == pytest.approx(4 / 6)
    assert result.transfers is True
    assert result.permitted_wording == WORDING_WHEN_TRANSFERS


def test_agreement_that_does_not_beat_the_majority_rate_binds_the_weaker_wording(
    tmp_path: Path,
) -> None:
    """Against a 50% reference this would read as skill; against the right reference it does not."""
    titles = [f"t{index}" for index in range(4)]
    sample = hand_built_sample(titles)
    readings = {"t0": POSITIVE, "t1": POSITIVE, "t2": POSITIVE, "t3": POSITIVE}
    human = {"0": "positive", "1": "positive", "2": "positive", "3": "negative"}
    result = agreement(sample, human, cache_with(tmp_path, readings))

    assert result.agreement == pytest.approx(3 / 4)
    assert result.majority_class_rate == pytest.approx(3 / 4)
    assert result.transfers is False
    assert result.permitted_wording == WORDING_WHEN_NOT


def test_a_sample_drawn_under_another_revision_refuses_to_inherit_the_figure(
    tmp_path: Path,
) -> None:
    sample = replace(hand_built_sample(["t0"]), extractor_revision="0" * 40)
    cache = cache_with(tmp_path, {"t0": POSITIVE})
    with pytest.raises(FatalDefect) as caught:
        agreement(sample, {"0": "positive"}, cache)
    assert caught.value.kind == "annotation_revision_mismatch"


def test_a_sampled_headline_with_no_cached_reading_is_refused(tmp_path: Path) -> None:
    sample = hand_built_sample(["t0", "t1"])
    cache = cache_with(tmp_path, {"t0": POSITIVE})
    with pytest.raises(FatalDefect) as caught:
        agreement(sample, {"0": "positive", "1": "positive"}, cache)
    assert caught.value.kind == "annotation_scores_missing"


# --- What is kept on disk -------------------------------------------------------------------------


def test_a_sample_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    """The stored sample is the provenance of the figure, so it must read back unchanged."""
    sample = sample_of()
    path = tmp_path / "sample.json"
    write_sample(sample, path)
    assert read_sample(path) == sample


def test_the_stored_agreement_keeps_every_label_so_the_figure_can_be_recomputed(
    tmp_path: Path,
) -> None:
    """ADR-0016: a relabelling is a new sample, which is only checkable if the old one is kept."""
    sample = hand_built_sample(["t0", "t1", "t2", "t3"])
    readings = {"t0": POSITIVE, "t1": POSITIVE, "t2": POSITIVE, "t3": NEGATIVE}
    human = {"0": "positive", "1": "positive", "2": "unclear", "3": "negative"}
    cache = cache_with(tmp_path, readings)
    result = agreement(sample, human, cache)

    path = tmp_path / "agreement.json"
    write_agreement(path, sample=sample, labels=human, result=result)
    written = json.loads(path.read_text(encoding="utf-8"))

    assert written["extractor_revision"] == EXTRACTOR_REVISION
    assert written["seed"] == sample.seed
    assert written["agreement"] == pytest.approx(1.0)
    assert written["majority_class_rate"] == pytest.approx(2 / 3)
    assert written["permitted_wording"] == WORDING_WHEN_TRANSFERS
    assert {row["item_id"]: row["hand_label"] for row in written["labels"]} == human
    assert {row["item_id"]: row["extractor_label"] for row in written["labels"]} == {
        "0": "positive",
        "1": "positive",
        "2": "positive",
        "3": "negative",
    }


def test_a_missing_labels_file_is_refused_by_name(tmp_path: Path) -> None:
    """The handover point: the person running `score` first is the one who has not labelled yet."""
    with pytest.raises(FatalDefect) as caught:
        read_labels(tmp_path / "labels.csv", sample_of())
    assert caught.value.kind == "annotation_labels_invalid"
    assert "labels.csv" in str(caught.value)


def test_a_small_request_is_allocated_exactly_and_never_overshot() -> None:
    """The floor of one per stratum must not be able to push the total past the request.

    Measured before the fix: 9 strata and a request of 9 allocated 18, because the floor made the
    running total exceed the request and the largest-remainder pass then added to a negative slice.
    """
    items = corpus()
    for size in range(9, 40):
        sample = draw_annotation_sample(
            items, seed=7, size=size, first_day=FIRST_DAY, last_day=LAST_DAY
        )
        assert sum(sample.allocation.values()) == size, f"size {size}"
        assert len(sample.items) == size, f"size {size}"
        assert min(sample.allocation.values()) >= 1, f"size {size}"
        for key, drawn in sample.allocation.items():
            assert drawn <= sample.population[key], f"size {size}, stratum {key}"


def test_a_request_larger_than_the_corpus_is_refused() -> None:
    """Asking for more headlines than exist is a mistake, not a smaller sample."""
    items = corpus()
    with pytest.raises(FatalDefect) as caught:
        draw_annotation_sample(items, seed=7, size=10**6, first_day=FIRST_DAY, last_day=LAST_DAY)
    assert caught.value.kind == "annotation_sample_too_small"


def test_a_sample_labelled_entirely_unclear_is_refused_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """`unclear` is excluded by design, so "all of it" is reachable and must refuse by name."""
    sample = hand_built_sample(["t0", "t1"])
    cache = cache_with(tmp_path, {"t0": POSITIVE, "t1": NEGATIVE})
    with pytest.raises(FatalDefect) as caught:
        agreement(sample, {"0": "unclear", "1": "unclear"}, cache)
    assert caught.value.kind == "annotation_nothing_compared"


def test_no_stratum_is_drawn_more_than_one_headline_above_its_rounded_share() -> None:
    """Largest-remainder rounding moves a stratum by at most one seat; the sum alone misses that.

    A first attempt at the overshoot fix handed every spare seat to one stratum. That still summed
    to the request, so the exactness test above passed while a stratum whose rounded share was four
    drew ten.

    The bound is the rounded share plus one, not the share plus one, because the floor of one per
    stratum and the largest-remainder bonus do stack: the live draw gives `2021 busy` two headlines
    on a share of 0.67, which is 1.33 above it and correct. Weak monotonicity in population would be
    the tidier property, but largest-remainder allocation does not guarantee it once a floor and a
    population cap are in play, so asserting it would be asserting something false.
    """
    items = corpus()
    for size in (9, 12, 20, 33, 60, 90, 150):
        sample = draw_annotation_sample(
            items, seed=7, size=size, first_day=FIRST_DAY, last_day=LAST_DAY
        )
        total = sum(sample.population.values())
        for stratum, drawn in sample.allocation.items():
            share = size * sample.population[stratum] / total
            assert drawn <= max(1, math.floor(share)) + 1, (
                f"size {size}: stratum {stratum} holds {sample.population[stratum]} of {total}, "
                f"a share of {share:.2f}, and drew {drawn}"
            )
