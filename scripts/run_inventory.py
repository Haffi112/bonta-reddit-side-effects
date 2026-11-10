"""CLI: run the subreddit inventory step.

Writes:
  data/interim/inventory.csv      (subreddit, match_count)
  data/interim/subreddits_final.txt   (newline list, filtered by threshold)
"""

import asyncio
from pathlib import Path

import httpx
import polars as pl

from botox_se.config import load as load_settings
from botox_se.ingest.config_loaders import load_search_terms, load_subreddits
from botox_se.ingest.inventory import filter_subreddits, inventory_subreddits
from botox_se.ingest.time_utils import iso_to_unix
from botox_se.logging import configure_logging, get_logger
from botox_se.paths import Paths


async def _run() -> None:
    configure_logging(level="INFO")
    log = get_logger("inventory")
    settings = load_settings()
    paths = Paths(repo_root=Path(__file__).resolve().parents[1])
    paths.ensure_dirs()

    terms = load_search_terms(paths.config / "search_terms.yaml")
    sub_cfg = load_subreddits(paths.config / "subreddits.yaml")

    start = iso_to_unix(settings.collection_start)
    end = iso_to_unix(settings.collection_end)

    async with httpx.AsyncClient() as http:
        counts = await inventory_subreddits(
            http=http,
            subreddits=sub_cfg.all_subreddits,
            terms=terms,
            start=start,
            end=end,
            concurrency=settings.concurrent_requests,
        )

    df = pl.DataFrame([{"subreddit": s, "match_count": n} for s, n in counts.items()])
    df = df.sort("match_count", descending=True)
    df.write_csv(paths.interim / "inventory.csv")

    finalists = filter_subreddits(counts, threshold=sub_cfg.min_posts_threshold)
    (paths.interim / "subreddits_final.txt").write_text("\n".join(finalists))

    log.info("inventory_done", n_candidates=len(counts), n_finalists=len(finalists))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
