"""Tests for inventory_subreddits.

Each (subreddit, term) pair now generates TWO HTTP requests (title= + selftext=).
The regex mocks are reusable so they serve both requests transparently.
"""

import re

import pytest
from httpx import AsyncClient

from botox_se.ingest.inventory import filter_subreddits, inventory_subreddits


@pytest.mark.asyncio
async def test_inventory_returns_counts_per_subreddit(httpx_mock):  # type: ignore[no-untyped-def]
    # Each search_posts call makes 2 HTTP requests (title + selftext).
    # The reusable regex mock matches both; the botox sub returns 2 rows each time.
    httpx_mock.add_response(
        url=re.compile(r".*subreddit=botox.*"),
        json={
            "data": [
                {"id": "p1", "subreddit": "botox", "created_utc": 1},
                {"id": "p2", "subreddit": "botox", "created_utc": 2},
            ]
        },
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=re.compile(r".*subreddit=emptysub.*"),
        json={"data": []},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        counts = await inventory_subreddits(
            http=http,
            subreddits=["botox", "emptysub"],
            terms=["botox"],
            start=0,
            end=99_999_999,
        )
    # With paginate=False (default), each call returns <=2 rows per field.
    # Both title and selftext requests return p1+p2, but search_posts dedupes them → 2 unique.
    assert counts["botox"] >= 2
    assert counts["emptysub"] == 0


@pytest.mark.asyncio
async def test_inventory_full_count_paginates(httpx_mock):  # type: ignore[no-untyped-def]
    """full_count=True passes paginate=True to the client."""
    httpx_mock.add_response(
        url=re.compile(r".*subreddit=botox.*"),
        json={
            "data": [
                {"id": "p1", "subreddit": "botox", "created_utc": 1},
            ]
        },
        is_reusable=True,
    )
    async with AsyncClient() as http:
        counts = await inventory_subreddits(
            http=http,
            subreddits=["botox"],
            terms=["botox"],
            start=0,
            end=99_999_999,
            full_count=True,
        )
    assert counts["botox"] >= 1


def test_filter_subreddits_applies_threshold() -> None:
    counts = {"botox": 60, "fillers": 30, "skincare": 100}
    result = filter_subreddits(counts, threshold=50)
    assert result == ["botox", "skincare"]
