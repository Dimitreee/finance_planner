"""The Sentiment Extractor and its cache.

The cache, the key and the score are pure and always tested. The instrument itself is tested only
where it is present: the research dependency group and a local model download are not assumed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
from cryptoguard_core import sentiment
from cryptoguard_core.ingest import FatalDefect
from cryptoguard_core.news import NewsItem
from cryptoguard_core.news_features import sentiment_features
from cryptoguard_core.sentiment import (
    EXTRACTOR_NAME,
    EXTRACTOR_REVISION,
    SCORE_TOLERANCE,
    Reading,
    ReadingCache,
    is_latin_script,
    news_signal,
    score_titles,
    text_key,
)

MODEL_AVAILABLE = False
try:  # pragma: no cover - environment probe, not behaviour
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(
        EXTRACTOR_NAME, revision=EXTRACTOR_REVISION, local_files_only=True
    )
    MODEL_AVAILABLE = True
except Exception:  # noqa: BLE001 - any failure means "not available here"
    MODEL_AVAILABLE = False

needs_model = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="the Sentiment Extractor is not downloaded in this checkout"
)


def item(
    title: str,
    *,
    item_id: str = "1",
    currencies: tuple[str, ...] = ("BTC",),
    stated: datetime | None = None,
) -> NewsItem:
    stated = stated or datetime(2021, 2, 1, 6)
    return NewsItem(
        item_id=item_id,
        title=title,
        description=None,
        source_domain=None,
        source_url=None,
        stated_at=stated,
        stated_at_raw=stated.strftime("%Y-%m-%d %H:%M:%S"),
        stated_timezone_suffix=None,
        currencies=currencies,
    )


# --- The score, the key and the flag --------------------------------------------------------------


def test_the_score_is_positive_mass_minus_negative_mass() -> None:
    assert Reading(0.899, 0.029, 0.072).score == pytest.approx(0.87)
    assert Reading(0.016, 0.861, 0.123).score == pytest.approx(-0.845)


def test_neutral_and_torn_both_land_near_zero_and_are_not_distinguished() -> None:
    """The cost of reducing three numbers to one, stated as a test rather than only in prose."""
    neutral = Reading(0.05, 0.05, 0.90)
    torn = Reading(0.48, 0.48, 0.04)
    assert neutral.score == torn.score == 0.0


def test_the_key_normalises_so_one_headline_is_not_scored_twice() -> None:
    decomposed = "Bitcoin rallies in Zürich"
    composed = "Bitcoin rallies in Zürich"
    assert decomposed != composed
    assert text_key(decomposed) == text_key(composed)


def test_latin_script_is_flagged_and_is_not_called_english() -> None:
    assert is_latin_script("Bitcoin surges to a new high")
    assert is_latin_script("Bitcoin monte en flèche")  # Latin script, not English
    assert not is_latin_script("Биткоин вырос")
    assert not is_latin_script("比特币上涨")


# --- The cache ------------------------------------------------------------------------------------


def test_a_reading_survives_a_round_trip_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "readings.jsonl"
    cache = ReadingCache(path)
    cache.put("Bitcoin surges", Reading(0.9, 0.03, 0.07))
    reread = ReadingCache(path)
    entry = reread.get("Bitcoin surges")
    assert entry is not None
    assert entry.reading.score == pytest.approx(0.87)
    assert entry.latin_script is True


def test_readings_written_under_another_revision_are_ignored(tmp_path: Path) -> None:
    """A revision change must invalidate the cache, not be averaged into it."""
    path = tmp_path / "readings.jsonl"
    ReadingCache(path, revision="older-revision").put("Bitcoin surges", Reading(0.1, 0.8, 0.1))
    current = ReadingCache(path, revision=EXTRACTOR_REVISION)
    assert current.get("Bitcoin surges") is None
    assert len(current) == 0


def test_a_corrupt_cache_line_is_named_rather_than_skipped(tmp_path: Path) -> None:
    path = tmp_path / "readings.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(FatalDefect) as caught:
        ReadingCache(path)
    assert caught.value.kind == "sentiment_cache_corrupt"


def test_missing_returns_distinct_titles_in_first_seen_order(tmp_path: Path) -> None:
    cache = ReadingCache(tmp_path / "readings.jsonl")
    cache.put("known", Reading(0.5, 0.2, 0.3))
    assert cache.missing(["known", "b", "a", "b"]) == ["b", "a"]


# --- Building Arm C's signal ----------------------------------------------------------------------


def test_a_signal_carries_one_score_per_attributed_headline(tmp_path: Path) -> None:
    cache = ReadingCache(tmp_path / "readings.jsonl")
    cache.put("up", Reading(0.9, 0.05, 0.05))
    cache.put("down", Reading(0.05, 0.9, 0.05))
    items = [
        item("up", item_id="1"),
        item("down", item_id="2"),
        item("ignored", item_id="3", currencies=("ETH",)),
    ]
    signal = news_signal(items, lag_hours=1, covers_through_ms=10**13, cache=cache)
    assert signal.scores is not None
    assert len(signal.scores) == 2
    assert sorted(signal.scores) == pytest.approx([-0.85, 0.85])


def test_an_unscored_headline_refuses_the_build_rather_than_dropping_it(tmp_path: Path) -> None:
    """Dropping it would count the headline in Arm B and omit it from Arm C's mean."""
    cache = ReadingCache(tmp_path / "readings.jsonl")
    cache.put("up", Reading(0.9, 0.05, 0.05))
    with pytest.raises(FatalDefect) as caught:
        news_signal(
            [item("up", item_id="1"), item("unseen", item_id="2")],
            lag_hours=1,
            covers_through_ms=10**13,
            cache=cache,
        )
    assert caught.value.kind == "sentiment_scores_incomplete"


def test_the_signal_scores_line_up_with_their_instants(tmp_path: Path) -> None:
    """Parallel arrays: a mis-ordered pairing would attribute every score to the wrong day."""
    cache = ReadingCache(tmp_path / "readings.jsonl")
    cache.put("early", Reading(0.9, 0.0, 0.1))
    cache.put("late", Reading(0.0, 0.9, 0.1))
    early = item("early", item_id="1")
    later = item("late", item_id="2", stated=datetime(2021, 2, 1, 18))
    signal = news_signal([later, early], lag_hours=1, covers_through_ms=10**13, cache=cache)
    assert signal.scores == (0.9, -0.9)
    cutoff = int(datetime(2021, 2, 2).timestamp() * 1000)
    features = sentiment_features(signal, cutoff)
    assert features["sentiment_mean_24h"] == pytest.approx(0.0)


# --- What reaches the model -----------------------------------------------------------------------


class _ModelLoaded(Exception):
    """Raised by the stub loader: seeing it means the extractor was instantiated."""


def _refuse_to_load(revision: str) -> tuple[object, object, object]:
    raise _ModelLoaded(revision)


def test_a_fully_cached_run_never_instantiates_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache's whole purpose. Loading FinBERT to answer nothing is seconds and 400 MB."""
    cache = ReadingCache(tmp_path / "readings.jsonl")
    cache.put("known", Reading(0.5, 0.2, 0.3))
    monkeypatch.setattr(sentiment, "_load_extractor", _refuse_to_load)
    assert score_titles(["known"], cache=cache) == 0


def test_one_outstanding_title_does_reach_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The twin of the test above: a probe that can never fire proves nothing when it does not."""
    cache = ReadingCache(tmp_path / "readings.jsonl")
    cache.put("known", Reading(0.5, 0.2, 0.3))
    monkeypatch.setattr(sentiment, "_load_extractor", _refuse_to_load)
    with pytest.raises(_ModelLoaded):
        score_titles(["known", "unseen"], cache=cache)


def test_the_research_group_being_absent_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment that ships Arm A installs no tensor libraries; the refusal must say so."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(FatalDefect) as caught:
        score_titles(["unseen"], cache=ReadingCache(tmp_path / "readings.jsonl"))
    assert caught.value.kind == "sentiment_extractor_unavailable"


# --- The instrument itself ------------------------------------------------------------------------


@needs_model
def test_the_extractor_reads_direction_the_way_a_reader_would(tmp_path: Path) -> None:
    cache = ReadingCache(tmp_path / "readings.jsonl")
    titles = ["Bitcoin surges to new all-time high", "SEC sues major crypto exchange"]
    assert score_titles(titles, cache=cache) == 2
    up = cache.get(titles[0])
    down = cache.get(titles[1])
    assert up is not None and down is not None
    assert up.reading.score > 0.5
    assert down.reading.score < -0.5


@needs_model
def test_a_second_run_rereads_the_cache_instead_of_the_model(tmp_path: Path) -> None:
    cache = ReadingCache(tmp_path / "readings.jsonl")
    titles = ["Bitcoin surges to new all-time high"]
    assert score_titles(titles, cache=cache) == 1
    assert score_titles(titles, cache=cache) == 0


@needs_model
def test_batching_moves_a_score_by_less_than_the_stated_tolerance(tmp_path: Path) -> None:
    """Measured, not assumed: padding and the batch dimension change which kernels run."""
    title = "Bitcoin surges to new all-time high"
    filler = [f"Market commentary number {index} on trading volumes" for index in range(15)]

    alone = ReadingCache(tmp_path / "alone.jsonl")
    score_titles([title], cache=alone, batch_size=1)
    batched = ReadingCache(tmp_path / "batched.jsonl")
    score_titles([title, *filler], cache=batched, batch_size=16)

    one = alone.get(title)
    many = batched.get(title)
    assert one is not None and many is not None
    assert abs(one.reading.score - many.reading.score) <= SCORE_TOLERANCE


@needs_model
def test_two_batch_sizes_agree_exactly_under_fixed_length_padding(tmp_path: Path) -> None:
    title = "Bitcoin surges to new all-time high"
    filler = [f"Market commentary number {index} on trading volumes" for index in range(15)]

    eight = ReadingCache(tmp_path / "eight.jsonl")
    score_titles([title, *filler], cache=eight, batch_size=8)
    sixteen = ReadingCache(tmp_path / "sixteen.jsonl")
    score_titles([title, *filler], cache=sixteen, batch_size=16)

    left = eight.get(title)
    right = sixteen.get(title)
    assert left is not None and right is not None
    assert left.reading == right.reading


@needs_model
def test_the_cache_file_records_the_revision_that_produced_each_reading(tmp_path: Path) -> None:
    path = tmp_path / "readings.jsonl"
    score_titles(["Bitcoin surges to new all-time high"], cache=ReadingCache(path))
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert {row["revision"] for row in written} == {EXTRACTOR_REVISION}
