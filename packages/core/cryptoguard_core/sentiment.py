"""The Sentiment Extractor: a frozen instrument, its cache, and the score it is reduced to.

The extractor is never fitted, fine-tuned or adapted on this project's data (ADR-0015). It reads a
headline and returns three probabilities; the Sentiment Score is `p(positive) - p(negative)`, so a
headline the model reads as neutral and one it is torn between positive and negative both land near
zero and are not distinguished.

**Reproducibility is to 1e-6, not bitwise, and that is measured rather than assumed.** Padding and
the batch dimension change which kernels run: scoring a headline alone and inside a batch of sixteen
gives answers differing by about 5e-08, and two different batch sizes with fixed-length padding
agree exactly. The stored score is therefore rounded, and the cache is keyed by
`(text_hash, revision)` so any headline is scored once and reread thereafter: drift cannot enter a
second run through rebatching. See ADR-0021.

Inference runs on CPU. MPS would be faster and would answer differently, and the number that goes
into a dataset must be the one a Linux CPU reproduces.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from cryptoguard_core.ingest import HOUR_MS, FatalDefect
from cryptoguard_core.news import NewsItem, mentions_asset
from cryptoguard_core.news_features import HeadlineAvailability, headline_availability

EXTRACTOR_NAME = "ProsusAI/finbert"
# Pinned commit, not a moving tag. `main` would let the instrument change under a finished dataset.
EXTRACTOR_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"

MAX_TOKENS = 64
BATCH_SIZE = 32
# The last decimal a re-run is expected to reproduce; see the module docstring for the measurement.
SCORE_DECIMALS = 6
SCORE_TOLERANCE = 1e-6

# The extractor's own three classes, in the order the tie-break reads. Public because the Annotation
# Sample compares hand labels against exactly these, and a second copy of the list there would let
# the two drift apart silently.
READING_LABELS = ("positive", "negative", "neutral")

# Derived and gitignored, like every other cache: it is reproducible from the archive and the
# revision, and 30 000 readings do not belong in the history of a repository.
READINGS_CACHE = Path(__file__).resolve().parents[3] / "data/cache/sentiment/readings.jsonl"


@dataclass(frozen=True, slots=True)
class Reading:
    """One headline's three-way reading, as probabilities that sum to one."""

    p_positive: float
    p_negative: float
    p_neutral: float

    @property
    def score(self) -> float:
        """The Sentiment Score: positive mass minus negative mass, on [-1, +1]."""
        return round(self.p_positive - self.p_negative, SCORE_DECIMALS)

    @property
    def reading_label(self) -> str:
        """The class carrying the most mass — what the Annotation Sample compares against.

        The Sentiment Score cannot serve here: it collapses "neutral" and "torn between positive and
        negative" into the same neighbourhood of zero, so comparing it to a three-way hand label
        would not be like for like. Ties break on the frozen `READING_LABELS` order, so the answer
        does not depend on dictionary iteration.
        """
        mass = {
            "positive": self.p_positive,
            "negative": self.p_negative,
            "neutral": self.p_neutral,
        }
        return max(READING_LABELS, key=lambda name: (mass[name], -READING_LABELS.index(name)))


def text_key(title: str) -> str:
    """The cache key for a headline: a digest of its exact text, NFC-normalised.

    Normalised because the same headline can arrive with decomposed accents and would otherwise be
    scored twice under two keys, which is how a cache quietly stops being one.
    """
    normalised = unicodedata.normalize("NFC", title)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def is_latin_script(title: str) -> bool:
    """Whether every letter is Latin. Latin script is not proof of English and is not claimed to be.

    The archive states no language. This flags the headlines whose script alone rules English out,
    so they can be counted in the report rather than silently scored by an English model.
    """
    return all(
        "LATIN" in unicodedata.name(character, "") for character in title if character.isalpha()
    )


@dataclass(frozen=True, slots=True)
class CachedReading:
    reading: Reading
    latin_script: bool


class ReadingCache:
    """Readings by `(text_hash, revision)`, backed by a JSONL file it appends to.

    Append-only and one record per line so a partially written run is still a usable cache: scoring
    thirty thousand headlines is minutes of work that nobody should repeat because of one crash.
    """

    def __init__(self, path: Path, *, revision: str = EXTRACTOR_REVISION) -> None:
        self.path = path
        self.revision = revision
        self._entries: dict[str, CachedReading] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw: Mapping[str, Any] = json.loads(line)
                except json.JSONDecodeError as error:
                    raise FatalDefect(
                        "sentiment_cache_corrupt",
                        f"{self.path}:{number} is not JSON: {error}",
                    ) from error
                if raw.get("revision") != self.revision:
                    # A different instrument's answers, not this one's. Keeping them would let a
                    # revision change pass unnoticed inside a dataset.
                    continue
                self._entries[raw["text_hash"]] = CachedReading(
                    reading=Reading(
                        p_positive=raw["p_positive"],
                        p_negative=raw["p_negative"],
                        p_neutral=raw["p_neutral"],
                    ),
                    latin_script=raw["latin_script"],
                )

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, title: str) -> CachedReading | None:
        return self._entries.get(text_key(title))

    def put(self, title: str, reading: Reading) -> None:
        key = text_key(title)
        entry = CachedReading(reading=reading, latin_script=is_latin_script(title))
        self._entries[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "text_hash": key,
                        "revision": self.revision,
                        "p_positive": reading.p_positive,
                        "p_negative": reading.p_negative,
                        "p_neutral": reading.p_neutral,
                        "latin_script": entry.latin_script,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def missing(self, titles: Iterable[str]) -> list[str]:
        """The distinct titles this cache cannot answer, in first-seen order."""
        seen: dict[str, None] = {}
        for title in titles:
            if self.get(title) is None:
                seen.setdefault(title, None)
        return list(seen)


def _load_extractor(revision: str) -> tuple[Any, Any, Any]:
    """Load torch, tokeniser and model, or refuse by name if the research group is absent.

    `torch` is returned rather than imported by the caller so that this function is the single
    place the research stack is reached for. Importing it beside the call site put a bare
    `ModuleNotFoundError` in front of the named refusal, which is the one case the refusal exists
    for: a deployment that ships Arm A and installs no tensor libraries.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised by absence, not by the suite
        raise FatalDefect(
            "sentiment_extractor_unavailable",
            "the Sentiment Extractor needs the research dependency group: `uv sync`",
        ) from error

    tokeniser = AutoTokenizer.from_pretrained(EXTRACTOR_NAME, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(EXTRACTOR_NAME, revision=revision)
    model.eval()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    labels = {name.lower() for name in model.config.id2label.values()}
    if labels != set(READING_LABELS):
        raise FatalDefect(
            "sentiment_extractor_unavailable",
            f"{EXTRACTOR_NAME} at {revision} reports labels {sorted(labels)}, expected "
            f"{sorted(READING_LABELS)}; the label order is read by name, but an unknown set "
            "means a different model",
        )
    return torch, tokeniser, model


def score_titles(
    titles: Sequence[str],
    *,
    cache: ReadingCache,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Score every title the cache cannot answer, writing each result as it is produced.

    Returns the number of titles actually scored, so a caller can state how much of a run was
    computed and how much was reread.
    """
    outstanding = cache.missing(titles)
    if not outstanding:
        return 0

    torch, tokeniser, model = _load_extractor(cache.revision)
    order = {name.lower(): index for index, name in model.config.id2label.items()}
    scored = 0
    for start in range(0, len(outstanding), batch_size):
        batch = outstanding[start : start + batch_size]
        with torch.no_grad():
            encoded = tokeniser(
                batch,
                padding="max_length",
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).tolist()
        for title, row in zip(batch, probabilities, strict=True):
            cache.put(
                title,
                Reading(
                    p_positive=round(row[order["positive"]], SCORE_DECIMALS),
                    p_negative=round(row[order["negative"]], SCORE_DECIMALS),
                    p_neutral=round(row[order["neutral"]], SCORE_DECIMALS),
                ),
            )
            scored += 1
    return scored


def scores_for(items: Iterable[NewsItem], cache: ReadingCache) -> dict[str, float]:
    """Sentiment Score per item id, for the items the cache can answer."""
    out: dict[str, float] = {}
    for item in items:
        entry = cache.get(item.title)
        if entry is not None:
            out[item.item_id] = entry.reading.score
    return out


def news_signal(
    items: Iterable[NewsItem],
    *,
    lag_hours: int,
    covers_through_ms: int,
    cache: ReadingCache,
    asset: str = "BTC",
) -> HeadlineAvailability:
    """An availability set carrying a Sentiment Score beside every instant.

    Every attributed headline must already be in the cache. A missing score cannot be skipped: the
    headline would still be counted by Arm B and silently absent from Arm C's mean, so the two arms
    would no longer be reading the same corpus.
    """
    attributed = [item for item in items if mentions_asset(item, asset)]
    unscored = [item.item_id for item in attributed if cache.get(item.title) is None]
    if unscored:
        raise FatalDefect(
            "sentiment_scores_incomplete",
            f"{len(unscored)} attributed headlines have no cached reading "
            f"(first: {unscored[0]}); score the corpus before building Arm C",
        )

    shift_ms = lag_hours * HOUR_MS
    paired = sorted(
        (
            int(item.stated_at.replace(tzinfo=UTC).timestamp() * 1000) + shift_ms,
            _score_of(item, cache),
        )
        for item in attributed
    )
    bare = headline_availability(
        attributed, lag_hours=lag_hours, covers_through_ms=covers_through_ms, asset=asset
    )
    return HeadlineAvailability(
        asset=bare.asset,
        lag_hours=bare.lag_hours,
        instants_ms=bare.instants_ms,
        covers_through_ms=bare.covers_through_ms,
        scores=tuple(score for _, score in paired),
    )


def _score_of(item: NewsItem, cache: ReadingCache) -> float:
    entry = cache.get(item.title)
    assert entry is not None  # news_signal refuses beforehand; this keeps the type honest
    return entry.reading.score
