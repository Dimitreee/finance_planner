"""The Annotation Sample: the draw, the blind labelling surface, and the agreement figure.

The sample measures the instrument and gates nothing (ADR-0016). Arm C runs whatever it shows; what
it decides is how the result may be *described*, and both permitted descriptions are frozen in the
Experiment Contract before the first label exists.

Two properties have to hold for the figure to be worth anything, and both are enforced here rather
than trusted. The draw is reproducible from a seed recorded in the contract, so re-rolling it until
the sample looks convenient is a visible contract edit. And the file the annotator reads carries the
title and the item id and nothing else: no date, no price, no volume band, no extractor output. The
strata live in the stored sample, never in the labelling surface, because the headline volume of a
day is itself a hint about the market.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news import NewsItem, mentions_asset
from cryptoguard_core.sentiment import EXTRACTOR_REVISION, READING_LABELS, ReadingCache

DEFAULT_ASSET = "BTC"

# Tracked, unlike the reading cache: the drawn sample and the labels are the provenance of a figure
# no rerun can reproduce, because the labels are human work.
ANNOTATION_DIR = Path(__file__).resolve().parents[3] / "data/annotation"

# Frozen and mirrored in `config/experiment.yaml`; the contract refuses to load if the two disagree.
SAMPLE_SIZE = 210
SAMPLE_SEED = 20260928
VOLUME_BANDS = ("quiet", "typical", "busy")
STRATA = ("calendar_year", "daily_headline_volume_tercile")
ALLOCATION = "proportional_largest_remainder"


@dataclass(frozen=True, slots=True, order=True)
class Stratum:
    """One cell of the draw. A named pair, because `key[1]` reads as nothing at the call site."""

    calendar_year: int
    volume_band: str


@dataclass(frozen=True, slots=True)
class SampledHeadline:
    """One drawn headline with the stratum it was drawn from. Never shown to the annotator."""

    item_id: str
    title: str
    stated_day: date
    volume_band: str
    day_volume: int

    @property
    def calendar_year(self) -> int:
        """Derived, not stored: a stored year could disagree with the day it was taken from."""
        return self.stated_day.year

    @property
    def stratum(self) -> Stratum:
        return Stratum(self.calendar_year, self.volume_band)


@dataclass(frozen=True, slots=True)
class AnnotationSample:
    """A drawn sample, with everything needed to redraw it and to recompute its figure."""

    seed: int
    size: int
    # Recorded, not inherited: a figure computed under one instrument says nothing about another, so
    # a revision change makes the agreement recomputable rather than quietly still true.
    extractor_revision: str
    volume_band_edges: tuple[int, int]
    population: Mapping[Stratum, int]
    allocation: Mapping[Stratum, int]
    items: tuple[SampledHeadline, ...]


def volume_band_edges(counts: Mapping[date, int]) -> tuple[int, int]:
    """The two tercile cuts of the daily headline volume, over the days that carry a headline."""
    values = sorted(counts.values())
    lower, upper = statistics.quantiles(values, n=3, method="inclusive")
    return int(lower), int(upper)


def band_of(volume: int, edges: tuple[int, int]) -> str:
    """The band a day's headline volume falls in, closed at the upper edge of each tercile.

    Closed at the upper edge so that a day sitting exactly on a cut lands in the lower band rather
    than in neither: the three bands have to partition the days, or the strata would not cover them.
    """
    lower, upper = edges
    if volume <= lower:
        return VOLUME_BANDS[0]
    if volume <= upper:
        return VOLUME_BANDS[1]
    return VOLUME_BANDS[2]


def draw_annotation_sample(
    items: Iterable[NewsItem],
    *,
    seed: int = SAMPLE_SEED,
    size: int = SAMPLE_SIZE,
    first_day: date,
    last_day: date,
    extractor_revision: str = EXTRACTOR_REVISION,
    asset: str = DEFAULT_ASSET,
) -> AnnotationSample:
    """Draw `size` headlines stratified by calendar year and by the volume band of their day."""
    eligible = [
        candidate
        for candidate in items
        if mentions_asset(candidate, asset) and first_day <= candidate.stated_at.date() <= last_day
    ]
    volumes = Counter(candidate.stated_at.date() for candidate in eligible)
    edges = volume_band_edges(volumes)

    cells: dict[Stratum, list[NewsItem]] = {}
    for candidate in eligible:
        day = candidate.stated_at.date()
        key = Stratum(day.year, band_of(volumes[day], edges))
        cells.setdefault(key, []).append(candidate)

    population = {key: len(members) for key, members in cells.items()}
    allocation = _allocate(population, size)

    rng = random.Random(seed)
    drawn: list[SampledHeadline] = []
    for key in sorted(cells):
        members = sorted(cells[key], key=lambda candidate: candidate.item_id)
        for candidate in rng.sample(members, allocation[key]):
            day = candidate.stated_at.date()
            drawn.append(
                SampledHeadline(
                    item_id=candidate.item_id,
                    title=candidate.title,
                    stated_day=day,
                    volume_band=key.volume_band,
                    day_volume=volumes[day],
                )
            )

    return AnnotationSample(
        seed=seed,
        size=size,
        extractor_revision=extractor_revision,
        volume_band_edges=edges,
        population=population,
        allocation=allocation,
        items=tuple(sorted(drawn, key=lambda sampled: sampled.item_id)),
    )


def _allocate(population: Mapping[Stratum, int], size: int) -> dict[Stratum, int]:
    """Proportional allocation by largest remainder, with at least one per non-empty stratum.

    Proportional rather than equal per cell: the figure is meant to transfer to the corpus, and
    equal allocation would weight a quiet day — about one per cent of the corpus — like a busy one.

    The floor of one pushes in both directions, so the correction has to go both ways. Rounding down
    can leave seats unplaced; the floor can take more seats than were asked for, which is not a
    smaller sample but an arithmetic error — measured at nine strata and a request of nine, where
    the uncorrected version allocated eighteen. Seats are therefore handed out and taken back until
    the total is the requested one, and a stratum is never allocated more headlines than it holds,
    which is the other way this used to end in a bare `ValueError` out of `random.sample`.
    """
    total = sum(population.values())
    if size < len(population):
        raise FatalDefect(
            "annotation_sample_too_small",
            f"{size} headlines cannot cover {len(population)} strata with at least one each; "
            "a sample that silently drops a stratum is not the stratified sample it claims to be",
        )
    if size > total:
        raise FatalDefect(
            "annotation_sample_too_small",
            f"{size} headlines were asked for and the strata hold {total}; drawing more than "
            "exists is a mistake in the request, not a reason to return a smaller sample",
        )

    exact = {key: size * count / total for key, count in population.items()}
    allocation = {key: min(max(1, int(value)), population[key]) for key, value in exact.items()}
    # Largest fractional part first when adding, smallest first when taking back, so the correction
    # lands on the strata whose rounding was least deserved in whichever direction it went.
    by_remainder = sorted(population, key=lambda key: (-(exact[key] - int(exact[key])), key))

    # One seat per stratum per pass, not all the spare seats onto the first one: handing them to a
    # single stratum is proportional to nothing. Passes repeat only when a stratum runs out of room.
    deficit = size - sum(allocation.values())
    while deficit > 0:
        placed = 0
        for key in by_remainder:
            if deficit == 0:
                break
            if allocation[key] < population[key]:
                allocation[key] += 1
                deficit -= 1
                placed += 1
        if placed == 0:  # pragma: no cover - unreachable while size <= total
            break
    while deficit < 0:
        taken = 0
        for key in reversed(by_remainder):
            if deficit == 0:
                break
            if allocation[key] > 1:
                allocation[key] -= 1
                deficit += 1
                taken += 1
        if taken == 0:  # pragma: no cover - unreachable while size >= len(population)
            break

    if sum(allocation.values()) != size:  # pragma: no cover - both guards above prevent it
        raise FatalDefect(
            "annotation_sample_too_small",
            f"the strata cannot be allocated {size} headlines: "
            f"{sum(allocation.values())} were placed",
        )
    return allocation


LABELLING_COLUMNS = ("item_id", "title")


def write_labelling_file(sample: AnnotationSample, path: Path) -> None:
    """Write the annotator's surface: the item id, the title, and nothing else.

    The rows are shuffled with the sample's seed. Item ids rise with time in this archive, so id
    order would hand the annotator a chronology, and stratum order would hand them the volume band.
    Shuffling is reproducible for the same reason the draw is.
    """
    order = list(sample.items)
    random.Random(sample.seed).shuffle(order)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABELLING_COLUMNS)
        for sampled in order:
            writer.writerow((sampled.item_id, sampled.title))


# Derived from the instrument, not restated beside it: agreement is a like-for-like comparison, so a
# hand-written copy of the three classes here could drift from the ones the extractor reports.
# `unclear` is the annotator's alone, is reported separately, and is never mapped to neutral —
# "I could not tell" and "the headline is neutral" are different readings.
UNCLEAR = "unclear"
COMPARED_LABELS = READING_LABELS
LABELS = (*READING_LABELS, UNCLEAR)


def _refuse_labels(detail: str) -> NoReturn:
    raise FatalDefect("annotation_labels_invalid", detail)


def read_labels(path: Path, sample: AnnotationSample) -> dict[str, str]:
    """Read the annotator's labels and check them against the sample they belong to.

    A `title` column is accepted and verified, because the natural way to produce this file is to
    add a label column to a copy of the labelling file — and a row inserted or deleted while doing
    that would shift every label onto the wrong headline, which is exactly the silent error that
    would destroy the measurement while leaving the count intact.
    """
    if not path.is_file():
        _refuse_labels(
            f"{path} does not exist yet. The labelling is human work: open the labelling file "
            f"beside it, add a label column, and save it here with the columns item_id,label"
        )
    titles = {sampled.item_id: sampled.title for sampled in sample.items}
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for number, row in enumerate(reader, start=2):
            item_id = (row.get("item_id") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if item_id not in titles:
                _refuse_labels(f"{path}:{number} labels {item_id!r}, which is not in the sample")
            if item_id in labels:
                _refuse_labels(f"{path}:{number} labels {item_id!r} a second time")
            if label not in LABELS:
                _refuse_labels(
                    f"{path}:{number} carries label {label!r}; the permitted labels are "
                    f"{list(LABELS)}"
                )
            stated_title = row.get("title")
            if stated_title is not None and stated_title != titles[item_id]:
                _refuse_labels(
                    f"{path}:{number} carries a title that is not the one drawn for {item_id!r}; "
                    "a shifted row would move every label onto the wrong headline"
                )
            labels[item_id] = label

    unlabelled = [sampled.item_id for sampled in sample.items if sampled.item_id not in labels]
    if unlabelled:
        _refuse_labels(
            f"{len(unlabelled)} of {len(sample.items)} sampled headlines carry no label "
            f"(first: {unlabelled[0]}); a partially labelled sample is a different sample"
        )
    return labels


# ADR-0016. The threshold is a rule, not a number: agreement is compared against the sample's own
# majority-class rate, because a three-class problem with skewed classes makes 50% meaningless.
AGREEMENT_REFERENCE = "sample_majority_class_rate"
WORDING_WHEN_TRANSFERS = "the tone of the news"
WORDING_WHEN_NOT = "the extractor's reading of the headlines"


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """What the Annotation Sample measured, and the wording that measurement permits."""

    extractor_revision: str
    labelled: int
    unclear: int
    compared: int
    agreed: int
    agreement: float
    majority_class: str
    majority_class_rate: float
    transfers: bool
    permitted_wording: str
    confusion: Mapping[tuple[str, str], int]
    # Every sampled headline's reading, including the ones the annotator marked unclear: the
    # extractor read those too, and dropping them here would make the stored file unrecomputable.
    extractor_labels: Mapping[str, str]


def agreement(
    sample: AnnotationSample, labels: Mapping[str, str], cache: ReadingCache
) -> AgreementResult:
    """Agreement between the extractor's three-way reading and the hand labels on this sample.

    `unclear` is excluded from the figure and reported beside it. The reference is the sample's
    majority-class rate over the compared headlines, so an instrument that answers "positive" to
    everything scores exactly the reference rather than looking skilful against 50%.
    """
    if cache.revision != sample.extractor_revision:
        raise FatalDefect(
            "annotation_revision_mismatch",
            f"the sample was drawn under {sample.extractor_revision[:12]} but the cache holds "
            f"{cache.revision[:12]}; the figure must be recomputed, not inherited",
        )

    human: list[str] = []
    machine: list[str] = []
    extractor_labels: dict[str, str] = {}
    unclear = 0
    for sampled in sample.items:
        label = labels[sampled.item_id]
        entry = cache.get(sampled.title)
        if entry is None:
            raise FatalDefect(
                "annotation_scores_missing",
                f"sampled headline {sampled.item_id} has no cached reading; score the corpus "
                "before measuring agreement on it",
            )
        read_as = entry.reading.reading_label
        extractor_labels[sampled.item_id] = read_as
        if label == UNCLEAR:
            unclear += 1
            continue
        human.append(label)
        machine.append(read_as)

    compared = len(human)
    if compared == 0:
        raise FatalDefect(
            "annotation_nothing_compared",
            f"all {len(sample.items)} labels are {UNCLEAR!r}, so there is nothing to compare; "
            "an agreement of zero over zero is not a measurement",
        )
    agreed = sum(1 for left, right in zip(human, machine, strict=True) if left == right)
    counts = Counter(human)
    majority_class, majority_count = max(
        counts.items(), key=lambda pair: (pair[1], -COMPARED_LABELS.index(pair[0]))
    )
    rate = majority_count / compared
    ratio = agreed / compared
    transfers = ratio > rate
    return AgreementResult(
        extractor_revision=sample.extractor_revision,
        labelled=len(sample.items),
        unclear=unclear,
        compared=compared,
        agreed=agreed,
        agreement=ratio,
        majority_class=majority_class,
        majority_class_rate=rate,
        transfers=transfers,
        permitted_wording=WORDING_WHEN_TRANSFERS if transfers else WORDING_WHEN_NOT,
        confusion=dict(Counter(zip(human, machine, strict=True))),
        extractor_labels=extractor_labels,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Both stored files are written the same way, so the way is stated once.

    Sorted keys and a trailing newline so that a diff of two draws shows what changed rather than
    how the dictionary happened to be built.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sample(sample: AnnotationSample, path: Path) -> None:
    """Store the sample as JSON: the provenance of every figure computed from it.

    JSON rather than CSV because this file is tracked and a CSV is not, and because the strata keys
    are pairs — a CSV would flatten them into a convention nobody can check.
    """
    payload = {
        "seed": sample.seed,
        "size": sample.size,
        "extractor_revision": sample.extractor_revision,
        "volume_band_edges": list(sample.volume_band_edges),
        "strata": [
            {
                "calendar_year": stratum.calendar_year,
                "volume_band": stratum.volume_band,
                "population": sample.population[stratum],
                "allocation": sample.allocation[stratum],
            }
            for stratum in sorted(sample.population)
        ],
        "items": [
            {
                "item_id": sampled.item_id,
                "title": sampled.title,
                "stated_day": sampled.stated_day.isoformat(),
                "calendar_year": sampled.calendar_year,
                "volume_band": sampled.volume_band,
                "day_volume": sampled.day_volume,
            }
            for sampled in sample.items
        ],
    }
    _write_json(path, payload)


def read_sample(path: Path) -> AnnotationSample:
    """Rebuild a stored sample, deriving what is derivable rather than trusting the file for it.

    The stratum keys come back as `Stratum` values and the calendar year is recomputed from the
    stated day, so a file edited by hand cannot present a headline as belonging to another year.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    lower, upper = raw["volume_band_edges"]
    return AnnotationSample(
        seed=raw["seed"],
        size=raw["size"],
        extractor_revision=raw["extractor_revision"],
        volume_band_edges=(lower, upper),
        population={
            Stratum(cell["calendar_year"], cell["volume_band"]): cell["population"]
            for cell in raw["strata"]
        },
        allocation={
            Stratum(cell["calendar_year"], cell["volume_band"]): cell["allocation"]
            for cell in raw["strata"]
        },
        items=tuple(
            SampledHeadline(
                item_id=entry["item_id"],
                title=entry["title"],
                stated_day=date.fromisoformat(entry["stated_day"]),
                volume_band=entry["volume_band"],
                day_volume=entry["day_volume"],
            )
            for entry in raw["items"]
        ),
    )


def write_agreement(
    path: Path,
    *,
    sample: AnnotationSample,
    labels: Mapping[str, str],
    result: AgreementResult,
) -> None:
    """Store the figure together with every label that produced it.

    The hand labels arrive in an untracked CSV, so this file is where they become durable. A later
    relabelling is a new sample recorded as such, which is only checkable because this one is kept.
    """
    payload = {
        "seed": sample.seed,
        "extractor_revision": result.extractor_revision,
        "agreement_reference": AGREEMENT_REFERENCE,
        "labelled": result.labelled,
        "unclear": result.unclear,
        "compared": result.compared,
        "agreed": result.agreed,
        "agreement": result.agreement,
        "majority_class": result.majority_class,
        "majority_class_rate": result.majority_class_rate,
        "transfers": result.transfers,
        "permitted_wording": result.permitted_wording,
        "confusion": [
            {"hand_label": hand, "extractor_label": machine, "count": count}
            for (hand, machine), count in sorted(result.confusion.items())
        ],
        "labels": [
            {
                "item_id": sampled.item_id,
                "hand_label": labels[sampled.item_id],
                "extractor_label": result.extractor_labels[sampled.item_id],
            }
            for sampled in sample.items
        ],
    }
    _write_json(path, payload)
