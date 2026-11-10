"""PRAW-based enrichment: for each collected post in posts.jsonl, fetch
the full submission + comment tree and append to comments.jsonl.

Resumable: skips post IDs already present in comments.jsonl (by post_id).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from botox_se.ingest.praw_client import PrawClient
from botox_se.logging import get_logger

_log = get_logger(__name__)


def _already_enriched_post_ids(comments_path: Path) -> set[str]:
    """Return the set of post_ids that already have any comment row written."""
    if not comments_path.exists():
        return set()
    done: set[str] = set()
    with comments_path.open() as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pid := j.get("post_id"):
                done.add(pid)
    return done


def _load_post_ids(posts_path: Path) -> list[str]:
    ids: list[str] = []
    with posts_path.open() as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pid := j.get("id"):
                ids.append(pid)
    return ids


async def enrich_posts(
    *,
    client: PrawClient,
    posts_path: Path,
    comments_path: Path,
    enriched_posts_path: Path,
    resume: bool = True,
    concurrency: int = 4,
) -> dict[str, int]:
    """Fetch comment trees for every post in posts_path. Returns counters."""
    all_ids = _load_post_ids(posts_path)
    done_ids = _already_enriched_post_ids(comments_path) if resume else set()
    pending = [pid for pid in all_ids if pid not in done_ids]
    _log.info("enrich_start", total=len(all_ids), pending=len(pending), done=len(done_ids))

    sem = asyncio.Semaphore(concurrency)
    totals = {
        "posts_fetched": 0,
        "posts_missing": 0,
        "comments_written": 0,
    }
    comments_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_posts_path.parent.mkdir(parents=True, exist_ok=True)

    # Open both files in append mode; flush after each post for crash-safety.
    async def process(pid: str) -> None:
        async with sem:
            ep, comments = await client.fetch_post_with_comments(pid)
        # Single-threaded writer section (no other file I/O in this function body)
        if ep is None:
            totals["posts_missing"] += 1
            return
        with enriched_posts_path.open("a") as ef:
            ef.write(json.dumps(asdict(ep)) + "\n")
            ef.flush()
        if comments:
            with comments_path.open("a") as cf:
                for c in comments:
                    cf.write(json.dumps(asdict(c)) + "\n")
                cf.flush()
        totals["posts_fetched"] += 1
        totals["comments_written"] += len(comments)
        _log.info(
            "enriched",
            post_id=pid,
            n_comments=len(comments),
            running_posts=totals["posts_fetched"],
            running_comments=totals["comments_written"],
        )

    # Chunked gather to keep memory bounded
    CHUNK = 200
    for i in range(0, len(pending), CHUNK):
        await asyncio.gather(*(process(pid) for pid in pending[i : i + CHUNK]))

    return totals
