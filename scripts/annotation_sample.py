"""Draw the Annotation Sample, emit the blind labelling file, and score the labels when they exist.

The labelling in between is human work, and this script deliberately cannot do it. Run from the
repository root:

    uv run python scripts/annotation_sample.py draw     # the sample and the labelling file
    uv run python scripts/annotation_sample.py score    # once labels-<seed>.csv is filled in

`draw` refuses to overwrite an existing sample. The seed is pre-registered in
`config/experiment.yaml`, so re-rolling the draw until the sample looks convenient must be a visible
contract edit rather than a rerun (ADR-0016).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from cryptoguard_core.annotation import (
    ANNOTATION_DIR,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    AnnotationSample,
    agreement,
    draw_annotation_sample,
    read_labels,
    read_sample,
    write_agreement,
    write_labelling_file,
    write_sample,
)
from cryptoguard_core.contract import load_contract
from cryptoguard_core.dataset import RESEARCH_WINDOW_FIRST_DAY, RESEARCH_WINDOW_LAST_DAY
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news import NEWS_CSV, read_news_items
from cryptoguard_core.sentiment import READINGS_CACHE, ReadingCache


@dataclass(frozen=True, slots=True)
class AnnotationPaths:
    """The four files of one draw, named as fields.

    A dictionary with string keys would turn a typo into a runtime `KeyError` at the moment the
    script is trying to explain to a human where their labels go.
    """

    sample: Path
    labelling: Path
    labels: Path
    agreement: Path

    @classmethod
    def under(cls, directory: Path, seed: int) -> AnnotationPaths:
        return cls(
            sample=directory / f"sample-{seed}.json",
            labelling=directory / f"labelling-{seed}.csv",
            labels=directory / f"labels-{seed}.csv",
            agreement=directory / f"agreement-{seed}.json",
        )

    def superseded_agreement(self) -> Path:
        """The next free name to move an existing agreement aside to.

        ADR-0016: a relabelling is a new sample recorded as such, never a correction of the old one.
        Overwriting in place would destroy the only durable copy of the hand labels, which is the
        one thing in this whole measurement that no rerun can reproduce.
        """
        for number in range(1, 1000):
            candidate = self.agreement.with_name(f"{self.agreement.stem}.superseded-{number}.json")
            if not candidate.exists():
                return candidate
        raise FatalDefect(
            "annotation_agreement_exists",
            f"there are already 999 superseded agreements beside {self.agreement}",
        )


def describe(sample: AnnotationSample) -> list[str]:
    """The draw as a human-readable block: the seed, the bands, and the per-stratum allocation."""
    lines = [
        f"seed:                    {sample.seed}",
        f"size:                    {len(sample.items)}",
        f"extractor revision:      {sample.extractor_revision[:12]}",
        f"volume band cuts:        <= {sample.volume_band_edges[0]} / "
        f"<= {sample.volume_band_edges[1]} / above",
        "strata (year, band): population -> drawn",
    ]
    for stratum in sorted(sample.population):
        lines.append(
            f"  {stratum.calendar_year} {stratum.volume_band:<8} "
            f"{sample.population[stratum]:>6} -> {sample.allocation[stratum]:>3}"
        )
    return lines


def command_draw(arguments: argparse.Namespace) -> None:
    files = AnnotationPaths.under(arguments.directory, arguments.seed)
    if files.sample.exists() and not arguments.force:
        raise FatalDefect(
            "annotation_draw_exists",
            f"{files.sample} already exists. The seed is pre-registered, so redrawing is a "
            "contract edit, not a rerun; pass --force only when that edit has been made and "
            "recorded",
        )

    contract = load_contract().values["annotation"]
    size = arguments.size or int(contract["sample_size"])
    # The contract bounds `sample_size` and pins `seed`, but only for the values *in the file*. A
    # draw run with other numbers writes a file indistinguishable from a pre-registered one, so the
    # departure is refused here, where it happens, rather than left for a reader to notice.
    if not arguments.force and (size, arguments.seed) != (
        int(contract["sample_size"]),
        int(contract["seed"]),
    ):
        raise FatalDefect(
            "annotation_draw_exists",
            f"the contract pre-registers size {contract['sample_size']} and seed "
            f"{contract['seed']}; this run asks for size {size} and seed {arguments.seed}. Change "
            "the contract and record the change, or pass --force to draw something that is openly "
            "not the pre-registered sample",
        )
    items = list(read_news_items(NEWS_CSV))
    sample = draw_annotation_sample(
        items,
        seed=arguments.seed,
        size=size,
        first_day=RESEARCH_WINDOW_FIRST_DAY,
        last_day=RESEARCH_WINDOW_LAST_DAY,
    )
    write_sample(sample, files.sample)
    write_labelling_file(sample, files.labelling)

    for line in describe(sample):
        print(line)
    print(f"sample written:          {files.sample}")
    print(f"labelling file:          {files.labelling}")
    print(f"label into:              {files.labels}  (columns: item_id,label)")
    print(f"permitted labels:        {', '.join(contract['labels'])}")


def command_score(arguments: argparse.Namespace) -> None:
    files = AnnotationPaths.under(arguments.directory, arguments.seed)
    superseded: Path | None = None
    if files.agreement.exists():
        if not arguments.force:
            raise FatalDefect(
                "annotation_agreement_exists",
                f"{files.agreement} already exists. A relabelling after seeing model results is a "
                "new sample recorded as such, not a correction of this one (ADR-0016); pass "
                "--force only to repair a transcription error",
            )
        # Moved aside rather than overwritten: it holds the previous hand labels, and --force is
        # meant to add a corrected figure, never to make the earlier one unfindable.
        superseded = files.superseded_agreement()
        files.agreement.rename(superseded)

    sample = read_sample(files.sample)
    labels = read_labels(files.labels, sample)
    cache = ReadingCache(READINGS_CACHE, revision=sample.extractor_revision)
    result = agreement(sample, labels, cache)
    write_agreement(files.agreement, sample=sample, labels=labels, result=result)

    print(f"labelled:                {result.labelled}")
    print(f"unclear (excluded):      {result.unclear}")
    print(f"compared:                {result.compared}")
    print(f"agreed:                  {result.agreed}")
    print(f"agreement:               {result.agreement:.4f}")
    print(f"majority class:          {result.majority_class} at {result.majority_class_rate:.4f}")
    print(f"beats the reference:     {'yes' if result.transfers else 'no'}")
    print(f"permitted wording:       {result.permitted_wording!r}")
    print("confusion (hand -> extractor):")
    for (hand, machine), count in sorted(result.confusion.items()):
        print(f"  {hand:<9} -> {machine:<9} {count:>4}")
    print(f"written:                 {files.agreement}")
    if superseded is not None:
        print(f"previous figure kept as: {superseded}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--directory", type=Path, default=ANNOTATION_DIR)
    parser.add_argument("--force", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    drawer = commands.add_parser("draw", help="draw the sample and write the labelling file")
    drawer.add_argument("--size", type=int, default=None, help=f"default: {SAMPLE_SIZE}")
    drawer.set_defaults(handler=command_draw)
    scorer = commands.add_parser("score", help="measure agreement against the hand labels")
    scorer.set_defaults(handler=command_score)
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
