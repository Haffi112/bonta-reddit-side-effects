"""Data inventory: count matching posts per (subreddit, term) combination."""

import asyncio
from collections import defaultdict

import httpx

from botox_se.ingest.reddit_client import ArcticShiftClient
from botox_se.logging import get_logger

log = get_logger(__name__)


async def inventory_subreddits(
    *,
    http: httpx.AsyncClient,
    subreddits: list[str],
    terms: list[str],
    start: int,
    end: int,
    concurrency: int = 8,
    full_count: bool = False,
) -> dict[str, int]:
    """Return {subreddit: total_matching_post_count_across_terms}.

    When full_count=False (default), each (subreddit, term) query is capped at 100 results per
    search field (200 combined across title + selftext).  This is an undercount sufficient for
    the >=50-post threshold check: a subreddit that reaches 50 in the sample is kept; one that
    returns fewer than 50 may be a false drop.  Set full_count=True to paginate exhaustively.
    """
    client = ArcticShiftClient(http=http)
    counts: dict[str, int] = defaultdict(int)
    sem = asyncio.Semaphore(concurrency)

    async def count_one(subreddit: str, term: str) -> None:
        async with sem:
            posts = await client.search_posts(
                subreddit=subreddit,
                text_query=term,
                start=start,
                end=end,
                paginate=full_count,
            )
            counts[subreddit] += len(posts)
            log.info("inventory_query", subreddit=subreddit, term=term, count=len(posts))

    tasks = [count_one(s, t) for s in subreddits for t in terms]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        log.warning("inventory_partial_failures", n_errors=len(errors), first=str(errors[0]))
    return dict(counts)


def filter_subreddits(counts: dict[str, int], threshold: int) -> list[str]:
    """Return sorted list of subreddits meeting the minimum post threshold."""
    return sorted([s for s, n in counts.items() if n >= threshold])
