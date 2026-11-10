"""CLI: enrich Arctic-Shift-collected posts with PRAW fetching of the full
post + comment tree. Writes data/raw/enriched_posts.jsonl and data/raw/comments.jsonl.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from botox_se.config import load as load_settings
from botox_se.ingest.enrich import enrich_posts
from botox_se.ingest.praw_client import PrawClient
from botox_se.logging import configure_logging, get_logger
from botox_se.paths import Paths


async def _run() -> None:
    configure_logging(level="INFO")
    log = get_logger("enrich")
    settings = load_settings()
    paths = Paths(repo_root=Path(__file__).resolve().parents[1])
    paths.ensure_dirs()

    client_id, client_secret, user_agent = settings.require_reddit_credentials()
    client = PrawClient(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    posts_in = paths.raw / "posts.jsonl"
    if not posts_in.exists():
        raise FileNotFoundError(
            f"{posts_in} missing — run `make collect` (Arctic Shift posts) first."
        )

    totals = await enrich_posts(
        client=client,
        posts_path=posts_in,
        comments_path=paths.raw / "comments.jsonl",
        enriched_posts_path=paths.raw / "enriched_posts.jsonl",
        concurrency=settings.concurrent_requests,
    )
    log.info("enrich_done", **totals)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
