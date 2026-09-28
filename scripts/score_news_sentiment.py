"""Fill the Sentiment Extractor's cache over the news archive, and count what it read.

This script is the only place the instrument is run over real data. It is offline and idempotent:
every reading is keyed by `(text_hash, revision)`, so a second run loads no model and rescores
nothing. Nothing here fits anything, and no Trial is spent.

The counts it prints are the ones `reports/instrument_finbert.md` quotes. Run from the repository
root:

    uv run python scripts/score_news_sentiment.py --dry-run   # count the corpus, load no model
    uv run python scripts/score_news_sentiment.py             # score what the cache cannot answer
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from cryptoguard_core.news import NEWS_CSV, NewsItem, mentions_asset, read_news_items
from cryptoguard_core.sentiment import (
    BATCH_SIZE,
    EXTRACTOR_NAME,
    EXTRACTOR_REVISION,
    MAX_TOKENS,
    READINGS_CACHE,
    ReadingCache,
    is_latin_script,
    score_titles,
)

ASSET = "BTC"


def attributed(path: Path) -> list[NewsItem]:
    """Every archive record whose `currencies` carry an exact BTC token, in file order."""
    return [item for item in read_news_items(path) if mentions_asset(item, ASSET)]


def distinct_titles(items: Sequence[NewsItem]) -> list[str]:
    """Titles in first-seen order. The cache is keyed by text, so duplicates are one reading."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item.title, None)
    return list(seen)


def describe_corpus(items: Sequence[NewsItem], titles: Sequence[str]) -> list[str]:
    days = sorted({item.stated_at.date() for item in items})
    non_latin = [title for title in titles if not is_latin_script(title)]
    return [
        f"archive:                 {NEWS_CSV.name}",
        f"BTC-attributed records:  {len(items)}",
        f"distinct titles:         {len(titles)}",
        f"repeated titles:         {len(items) - len(titles)}",
        f"stated span:             {days[0]} .. {days[-1]}" if days else "stated span: empty",
        f"non-Latin-script titles: {len(non_latin)} "
        f"({100 * len(non_latin) / max(1, len(titles)):.2f}% of distinct)",
    ]


def describe_scores(cache: ReadingCache, titles: Sequence[str]) -> list[str]:
    scores = [entry.reading.score for title in titles if (entry := cache.get(title)) is not None]
    if not scores:
        return ["no readings cached yet"]
    buckets = Counter(
        "positive" if score > 0.1 else "negative" if score < -0.1 else "near zero"
        for score in scores
    )
    quantiles = statistics.quantiles(scores, n=4)
    return [
        f"readings:                {len(scores)}",
        f"mean Sentiment Score:    {statistics.fmean(scores):+.4f}",
        f"quartiles:               {quantiles[0]:+.4f} / {quantiles[1]:+.4f} / {quantiles[2]:+.4f}",
        f"range:                   {min(scores):+.4f} .. {max(scores):+.4f}",
        *(
            f"  {name:<22} {buckets[name]} ({100 * buckets[name] / len(scores):.2f}%)"
            for name in ("positive", "near zero", "negative")
        ),
    ]


def describe_truncation(titles: Sequence[str]) -> list[str]:
    """How many titles the 64-token limit actually cuts. Loads the tokeniser, not the model."""
    from transformers import AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(EXTRACTOR_NAME, revision=EXTRACTOR_REVISION)
    lengths = [len(tokeniser(title)["input_ids"]) for title in titles]
    cut = [length for length in lengths if length > MAX_TOKENS]
    return [
        f"token length median:     {statistics.median(lengths):.1f}",
        f"token length maximum:    {max(lengths)}",
        f"titles cut at {MAX_TOKENS}:       {len(cut)} "
        f"({100 * len(cut) / max(1, len(lengths)):.2f}% of distinct)",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count the corpus and the cache's coverage without loading the extractor",
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="also measure how many titles the token limit truncates (tokeniser only)",
    )
    parser.add_argument("--cache", type=Path, default=READINGS_CACHE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()

    items = attributed(NEWS_CSV)
    titles = distinct_titles(items)
    cache = ReadingCache(arguments.cache)
    outstanding = cache.missing(titles)

    print(f"extractor:               {EXTRACTOR_NAME} @ {EXTRACTOR_REVISION[:12]}")
    print(f"max tokens:              {MAX_TOKENS}")
    print(f"cache:                   {arguments.cache}")
    for line in describe_corpus(items, titles):
        print(line)
    print(f"already cached:          {len(titles) - len(outstanding)}")
    print(f"outstanding:             {len(outstanding)}")

    if arguments.dry_run:
        print("dry run: the extractor was not loaded")
    else:
        scored = score_titles(titles, cache=cache, batch_size=arguments.batch_size)
        print(f"scored this run:         {scored}")

    for line in describe_scores(cache, titles):
        print(line)

    if arguments.tokens:
        for line in describe_truncation(titles):
            print(line)


if __name__ == "__main__":
    main()
