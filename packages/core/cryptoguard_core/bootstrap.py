"""The moving-block bootstrap: uncertainty around a result, without pretending days are independent.

Decision Days are not independent draws. Resampling them one at a time breaks the runs of trending
and quiet weather that make a daily series what it is, and the interval that comes back is narrower
than the experiment deserves. Blocks keep a day's neighbours travelling with it. The block lengths
were pre-registered: 20 days reported, 5 and 40 as sensitivity (ADR-0017).

**Two rules about the blocks, stated once so a test can pin them.** A block running past the end
of the series *wraps* to the front, and the last block drawn is *truncated* so a resample is
exactly as long as the original. Wrapping rather than stopping at the end, because truncating
would let the last `block_days - 1` days appear less often than the rest — and on a trending
asset those are precisely the days carrying the final return.

**What an interval may say depends on the quantity it covers, not on how it was built.** The
Primary Comparison's interval may inform a conclusion about forecasting skill; the Headline Metric's
may not inform anything, and exists so a return quoted alone cannot be read as repeatable. Both are
reported, and an interval spanning zero is Inconclusive — never "a small effect". This module
returns one type for both and ranks nothing; the asymmetry lives in the captions its callers supply.

Nothing here spends a Trial: it estimates uncertainty around a result already produced.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from cryptoguard_core.ingest import FatalDefect

# Frozen and mirrored in `config/experiment.yaml`; the contract refuses to load if the two disagree.
BLOCK_DAYS = 20
SENSITIVITY_BLOCK_DAYS = (5, 40)
RESAMPLES = 10_000
SEED = 20260928
CONFIDENCE = 0.95
BLOCK_RULE = "wrap_and_truncate"
# Seeds the conclusion is re-checked at. The reported seed is excluded by the contract, because a
# stability check run at the seed being checked would only report its agreement with itself.
STABILITY_SEEDS = (20260929, 20260930, 20261001, 20261002)

# A closed set, spelled as one, like `FatalKind`: these two strings reach a report and a page, and a
# third spelling of either would read as a third verdict.
Verdict = Literal["Inconclusive", "Separated"]
INCONCLUSIVE: Verdict = "Inconclusive"
SEPARATED: Verdict = "Separated"


def _require_series(series: Sequence[object]) -> None:
    if not series:
        raise FatalDefect(
            "bootstrap_series_empty", "an empty series has no uncertainty to estimate"
        )


def block_indices(length: int, block_days: int, rng: random.Random) -> list[int]:
    """Indices of one moving-block resample: whole blocks, wrapped at the end, cut to `length`."""
    if not 1 <= block_days <= length:
        raise FatalDefect(
            "bootstrap_block_invalid",
            f"a block of {block_days} days cannot be drawn from a series of {length}: a block "
            "longer than the series does not exist in it, and a block of no days draws nothing. A "
            "block equal to the length is allowed and makes every resample a rotation",
        )
    drawn: list[int] = []
    while len(drawn) < length:
        start = rng.randrange(length)
        drawn.extend((start + offset) % length for offset in range(block_days))
    return drawn[:length]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile interval, with everything needed to reproduce it recorded beside it."""

    estimate: float
    lower: float
    upper: float
    block_days: int | None
    resamples: int
    seed: int
    confidence: float

    @property
    def spans_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    @property
    def verdict(self) -> Verdict:
        """`Inconclusive` when the interval spans zero — never "a small effect" (ADR-0017)."""
        return INCONCLUSIVE if self.spans_zero else SEPARATED

    @property
    def width(self) -> float:
        return self.upper - self.lower


def _percentile_interval(
    resampled: Sequence[float],
    estimate: float,
    *,
    block_days: int | None,
    resamples: int,
    seed: int,
    confidence: float,
) -> BootstrapInterval:
    """The plain percentile interval: the resampled statistics' own tails."""
    if not 0.0 < confidence < 1.0:
        raise FatalDefect(
            "bootstrap_parameter_invalid",
            f"a confidence of {confidence} is not a proportion strictly between 0 and 1",
        )
    ordered = sorted(resampled)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=estimate,
        lower=_quantile(ordered, tail),
        upper=_quantile(ordered, 1.0 - tail),
        block_days=block_days,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Linear interpolation between order statistics.

    Written out rather than taken from `statistics.quantiles`, which cuts a distribution into equal
    parts and cannot be asked for one arbitrary fraction such as 0.025.
    """
    if not ordered:
        raise FatalDefect("bootstrap_series_empty", "no resampled statistics to take a quantile of")
    position = fraction * (len(ordered) - 1)
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def moving_block_bootstrap[Observation](
    series: Sequence[Observation],
    statistic: Callable[[Sequence[Observation]], float],
    *,
    block_days: int = BLOCK_DAYS,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> BootstrapInterval:
    """Resample contiguous blocks with replacement and take `statistic` on each resample."""
    _require_series(series)
    rng = random.Random(seed)
    drawn = [
        statistic([series[index] for index in block_indices(len(series), block_days, rng)])
        for _ in range(resamples)
    ]
    return _percentile_interval(
        drawn,
        statistic(series),
        confidence=confidence,
        block_days=block_days,
        resamples=resamples,
        seed=seed,
    )


def iid_bootstrap[Observation](
    series: Sequence[Observation],
    statistic: Callable[[Sequence[Observation]], float],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> BootstrapInterval:
    """The same thing one observation at a time. Here to be *compared against*, not to be reported.

    It exists so the choice of blocks is demonstrated rather than asserted: on an autocorrelated
    series this returns the narrower, flattering interval, and a test measures the gap.
    """
    _require_series(series)
    rng = random.Random(seed)
    length = len(series)
    drawn = [
        statistic([series[rng.randrange(length)] for _ in range(length)]) for _ in range(resamples)
    ]
    return _percentile_interval(
        drawn,
        statistic(series),
        confidence=confidence,
        block_days=None,
        resamples=resamples,
        seed=seed,
    )
