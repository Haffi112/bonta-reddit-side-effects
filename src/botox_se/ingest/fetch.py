"""Streaming collection: (subreddit x term x time-chunk) -> JSONL.

Chunking limits API response size; JSONL append = no progress loss on crash.
"""

import json
from dataclasses import asdict
from pathlib import Path

import httpx

from botox_se.ingest.reddit_client import ArcticShiftClient, ArcticShiftError
from botox_se.logging import get_logger

log = get_logger(__name__)


def _chunks(start: int, end: int, step: int) -> list[tuple[int, int]]:
    """Split a time range [start, end) into step-sized chunks."""
    return [(t, min(t + step, end)) for t in range(start, end, step)]


async def collect_posts(
    *,
    http: httpx.AsyncClient,
    subreddits: list[str],
    terms: list[str],
    start: int,
    end: int,
    chunk_seconds: int = 30 * 24 * 3600,  # 30-day chunks
    output_path: Path,
) -> dict[str, int]:
    """Collect posts for all (subreddit, term, chunk) combos; append to JSONL.

    Returns a dict with ``total_posts`` and ``failed_chunks`` counts.
    Failed chunks are logged and skipped rather than aborting the run.
    """
    client = ArcticShiftClient(http=http)
    total = 0
    failed = 0
    with output_path.open("a") as f:
        for sub in subreddits:
            for term in terms:
                for chunk_start, chunk_end in _chunks(start, end, chunk_seconds):
                    try:
                        posts = await client.search_posts(
                            subreddit=sub,
                            text_query=term,
                            start=chunk_start,
                            end=chunk_end,
                        )
                    except (httpx.HTTPError, ArcticShiftError) as exc:
                        failed += 1
                        log.warning(
                            "chunk_failed",
                            subreddit=sub,
                            term=term,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            exc_type=type(exc).__name__,
                        )
                        continue
                    for p in posts:
                        f.write(json.dumps(asdict(p)) + "\n")
                    f.flush()
                    total += len(posts)
                    log.info(
                        "collected",
                        subreddit=sub,
                        term=term,
                        start=chunk_start,
                        end=chunk_end,
                        n=len(posts),
                        running_total=total,
                    )
    return {"total_posts": total, "failed_chunks": failed}


async def collect_comments(
    *,
    http: httpx.AsyncClient,
    subreddits: list[str],
    terms: list[str],
    start: int,
    end: int,
    chunk_seconds: int = 30 * 24 * 3600,
    output_path: Path,
) -> dict[str, int]:
    """Collect comments for all (subreddit, term, chunk) combos; append to JSONL.

    Returns a dict with ``total_comments`` and ``failed_chunks`` counts.
    Failed chunks are logged and skipped rather than aborting the run.
    """
    client = ArcticShiftClient(http=http)
    total = 0
    failed = 0
    with output_path.open("a") as f:
        for sub in subreddits:
            for term in terms:
                for chunk_start, chunk_end in _chunks(start, end, chunk_seconds):
                    try:
                        cmts = await client.search_comments(
                            subreddit=sub,
                            text_query=term,
                            start=chunk_start,
                            end=chunk_end,
                        )
                    except (httpx.HTTPError, ArcticShiftError) as exc:
                        failed += 1
                        log.warning(
                            "chunk_failed",
                            subreddit=sub,
                            term=term,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            exc_type=type(exc).__name__,
                        )
                        continue
                    for c in cmts:
                        f.write(json.dumps(asdict(c)) + "\n")
                    f.flush()
                    total += len(cmts)
                    log.info(
                        "collected",
                        subreddit=sub,
                        term=term,
                        start=chunk_start,
                        end=chunk_end,
                        n=len(cmts),
                        running_total=total,
                    )
    return {"total_comments": total, "failed_chunks": failed}
