"""Tests for collect_posts and collect_comments.

Each search_posts call now issues TWO requests (title= + selftext=).
The reusable regex mock handles both transparently.
"""

import json
import re

import pytest
from httpx import AsyncClient

from botox_se.ingest.fetch import collect_comments, collect_posts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WINDOW_START = 0
_WINDOW_END = 10
_CHUNK = 5  # produces 2 chunks: [0,5) and [5,10)


@pytest.mark.asyncio
async def test_collect_posts_writes_jsonl_and_returns_count(tmp_path, httpx_mock):  # type: ignore[no-untyped-def]
    # Returns same post for both title and selftext requests; collect_posts dedupes → 1 post.
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*"),
        json={
            "data": [
                {
                    "id": "a",
                    "subreddit": "botox",
                    "title": "t",
                    "selftext": "s",
                    "author": "u",
                    "created_utc": 1,
                    "permalink": "",
                    "score": 0,
                }
            ]
        },
        is_reusable=True,
    )
    out = tmp_path / "raw_posts.jsonl"
    async with AsyncClient() as http:
        result = await collect_posts(
            http=http,
            subreddits=["botox"],
            terms=["botox"],
            start=0,
            end=10,
            chunk_seconds=5,
            output_path=out,
        )
    assert result["total_posts"] >= 1
    assert result["failed_chunks"] == 0
    lines = out.read_text().splitlines()
    assert all(json.loads(line)["id"] for line in lines)


@pytest.mark.asyncio
async def test_collect_comments_writes_jsonl_and_returns_count(tmp_path, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        url=re.compile(r".*comments/search.*"),
        json={
            "data": [
                {
                    "id": "c1",
                    "link_id": "t3_abc",
                    "subreddit": "botox",
                    "body": "great results",
                    "author": "user1",
                    "created_utc": 5,
                }
            ]
        },
        is_reusable=True,
    )
    out = tmp_path / "raw_comments.jsonl"
    async with AsyncClient() as http:
        result = await collect_comments(
            http=http,
            subreddits=["botox"],
            terms=["botox"],
            start=0,
            end=10,
            chunk_seconds=5,
            output_path=out,
        )
    assert result["total_comments"] >= 1
    assert result["failed_chunks"] == 0
    lines = out.read_text().splitlines()
    assert all(json.loads(line)["id"] for line in lines)


# ---------------------------------------------------------------------------
# collect_posts: resilience to per-chunk failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_collect_posts_resilient_to_chunk_failure(tmp_path, httpx_mock, monkeypatch):  # type: ignore[no-untyped-def]
    """Permanent 422 responses exhaust retries; collect_posts logs and skips each chunk."""
    # 422 for every request — will exhaust _MAX_5XX_ATTEMPTS retries per chunk call
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*"),
        status_code=422,
        is_reusable=True,
    )

    async def fake_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("botox_se.ingest.reddit_client.asyncio.sleep", fake_sleep)

    out = tmp_path / "raw_posts.jsonl"
    async with AsyncClient() as http:
        result = await collect_posts(
            http=http,
            subreddits=["botox"],
            terms=["botox"],
            start=_WINDOW_START,
            end=_WINDOW_END,
            chunk_seconds=_CHUNK,
            output_path=out,
        )

    assert result["total_posts"] == 0
    assert result["failed_chunks"] >= 1
