"""CLI: collect posts via Arctic Shift into data/raw/posts.jsonl.

Comment collection has been moved to scripts/run_enrich.py (PRAW-based).
"""

import asyncio
from pathlib import Path

import httpx

from botox_se.config import load as load_settings
from botox_se.ingest.config_loaders import load_search_terms
from botox_se.ingest.fetch import collect_posts
from botox_se.ingest.time_utils import iso_to_unix
from botox_se.logging import configure_logging, get_logger
from botox_se.paths import Paths


async def _run() -> None:
    configure_logging(level="INFO")
    log = get_logger("collect")
    settings = load_settings()
    paths = Paths(repo_root=Path(__file__).resolve().parents[1])
    paths.ensure_dirs()

    terms = load_search_terms(paths.config / "search_terms.yaml")
    finalists_path = paths.interim / "subreddits_final.txt"
    if not finalists_path.exists():
        raise FileNotFoundError(f"{finalists_path} not found — run `make inventory` first.")
    finalists = finalists_path.read_text().splitlines()
    finalists = [s for s in finalists if s.strip()]  # drop blank lines
    if not finalists:
        raise RuntimeError(
            f"{finalists_path} is empty. "
            "Run `make inventory` first, or check that at least one subreddit meets the threshold."
        )

    start = iso_to_unix(settings.collection_start)
    end = iso_to_unix(settings.collection_end)

    # One chunk spanning the whole window. Pagination inside the client handles >100-row results.
    # This minimizes API calls: fewer (sub, term, chunk) combos, fewer 429 wait cycles.
    window_seconds = end - start

    posts_out = paths.raw / "posts.jsonl"
    posts_out.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as http:
        n_posts = await collect_posts(
            http=http,
            subreddits=finalists,
            terms=terms,
            start=start,
            end=end,
            chunk_seconds=window_seconds,
            output_path=posts_out,
        )

    log.info(
        "collection_done",
        total_posts=n_posts["total_posts"],
        failed_post_chunks=n_posts["failed_chunks"],
        note="comments now collected via scripts/run_enrich.py (PRAW)",
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
