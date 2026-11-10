"""Tests for ArcticShiftClient against the real API shape.

Key scenarios:
- search_posts issues TWO requests (title= and selftext=) and merges+dedupes by id.
- search_comments issues ONE request with body=.
- Cursor-based pagination: when first response fills the limit, a follow-up request uses
  after=<last.created_utc + 1>.
- API error payload raises ArcticShiftError.
- data=null in response treated as empty list (no crash).
- 429 with Retry-After header: sleep duration is honoured (with jitter) and request retried.
- 429 without Retry-After: falls back to exponential base waits.
- 429 exhaustion: after _MAX_429_ATTEMPTS raises HTTPStatusError.
"""

import random
import re

import pytest
from httpx import AsyncClient

from botox_se.ingest.reddit_client import (
    ArcticShiftClient,
    ArcticShiftError,
    RedditComment,
    RedditPost,
)

START = 1_577_836_800  # 2020-01-01 00:00:00 UTC
END = 1_578_009_600  # 2020-01-03 00:00:00 UTC


def _post(
    id: str,
    created_utc: int = 1_577_900_000,
    title: str = "Some title",
    selftext: str = "some selftext",
) -> dict:  # type: ignore[type-arg]
    return {
        "id": id,
        "subreddit": "botox",
        "title": title,
        "selftext": selftext,
        "author": "user1",
        "created_utc": created_utc,
        "permalink": f"/r/botox/comments/{id}",
        "score": 5,
    }


def _comment(id: str, post_id: str = "abc123", created_utc: int = 1_577_910_000) -> dict:  # type: ignore[type-arg]
    return {
        "id": id,
        "link_id": f"t3_{post_id}",
        "subreddit": "botox",
        "body": "I had ptosis for weeks",
        "author": "user2",
        "created_utc": created_utc,
    }


# ---------------------------------------------------------------------------
# search_posts: two requests, merge+dedupe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_posts_issues_title_and_selftext_requests(httpx_mock):  # type: ignore[no-untyped-def]
    """search_posts must send separate title= and selftext= requests."""
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*title=botox.*"),
        json={"data": [_post("p1")]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*selftext=botox.*"),
        json={"data": [_post("p2")]},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        posts = await client.search_posts(
            subreddit="botox",
            text_query="botox",
            start=START,
            end=END,
            paginate=False,
        )
    ids = {p.id for p in posts}
    assert "p1" in ids
    assert "p2" in ids
    assert all(isinstance(p, RedditPost) for p in posts)


@pytest.mark.asyncio
async def test_search_posts_dedupes_shared_id(httpx_mock):  # type: ignore[no-untyped-def]
    """A post that appears in both title and selftext responses is returned only once."""
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*title=botox.*"),
        json={"data": [_post("shared")]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*selftext=botox.*"),
        json={"data": [_post("shared")]},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        posts = await client.search_posts(
            subreddit="botox",
            text_query="botox",
            start=START,
            end=END,
            paginate=False,
        )
    assert len(posts) == 1
    assert posts[0].id == "shared"


# ---------------------------------------------------------------------------
# search_posts: pagination cursor advancement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_posts_paginates_when_full_page_returned(httpx_mock):  # type: ignore[no-untyped-def]
    """When a page fills the limit, a second request uses after=last.created_utc+1.

    search_posts runs title= first (two pages here), then selftext= (empty).
    We stage responses in the order the client will consume them:
      1. title page 1 → 2 items (fills limit → cursor advances)
      2. title page 2 → 1 item  (below limit → stops)
      3. selftext page 1 → empty
    """
    limit = 2
    page1 = [_post("p1", created_utc=1_000), _post("p2", created_utc=2_000)]
    page2 = [_post("p3", created_utc=3_000)]

    # Responses are consumed in registration order within matching URL pattern.
    # Register title responses first (two), then selftext (one empty).
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*title=.*"),
        json={"data": page1},
    )
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*title=.*"),
        json={"data": page2},
    )
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*selftext=.*"),
        json={"data": []},
    )

    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        posts = await client.search_posts(
            subreddit="botox",
            text_query="botox",
            start=0,
            end=99_999_999,
            limit=limit,
            paginate=True,
        )

    ids = [p.id for p in posts]
    assert "p1" in ids
    assert "p2" in ids
    assert "p3" in ids
    assert len(posts) == 3


# ---------------------------------------------------------------------------
# search_posts: data=null treated as empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_posts_handles_null_data(httpx_mock):  # type: ignore[no-untyped-def]
    """data=null in API response must be treated as empty list (no crash)."""
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*"),
        json={"data": None},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        posts = await client.search_posts(
            subreddit="botox",
            text_query="botox",
            start=START,
            end=END,
            paginate=False,
        )
    assert posts == []


# ---------------------------------------------------------------------------
# search_posts: API error raises ArcticShiftError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_posts_raises_arctic_shift_error_on_api_error(httpx_mock):  # type: ignore[no-untyped-def]
    """An API error payload must raise ArcticShiftError instead of silently returning []."""
    httpx_mock.add_response(
        url=re.compile(r".*posts/search.*"),
        json={"data": None, "error": "Unknown query parameter: 'q'"},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        with pytest.raises(ArcticShiftError, match="Unknown query parameter"):
            await client.search_posts(
                subreddit="botox",
                text_query="botox",
                start=START,
                end=END,
                paginate=False,
            )


# ---------------------------------------------------------------------------
# search_comments: uses body=, not q=
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_comments_uses_body_param(httpx_mock):  # type: ignore[no-untyped-def]
    """search_comments must send body=<query>, not q=<query>."""
    httpx_mock.add_response(
        url=re.compile(r".*comments/search.*body=botox.*"),
        json={"data": [_comment("cmt1")]},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        comments = await client.search_comments(
            subreddit="botox",
            text_query="botox",
            start=START,
            end=END,
            paginate=False,
        )
    assert len(comments) == 1
    assert comments[0].id == "cmt1"
    assert comments[0].post_id == "abc123"
    assert isinstance(comments[0], RedditComment)


@pytest.mark.asyncio
async def test_search_comments_dedupes(httpx_mock):  # type: ignore[no-untyped-def]
    """Duplicate comment ids are returned only once."""
    httpx_mock.add_response(
        url=re.compile(r".*comments/search.*"),
        json={"data": [_comment("cmt1"), _comment("cmt1")]},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        comments = await client.search_comments(
            subreddit="botox",
            text_query="botox",
            start=START,
            end=END,
            paginate=False,
        )
    assert len(comments) == 1


@pytest.mark.asyncio
async def test_search_comments_raises_on_api_error(httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        url=re.compile(r".*comments/search.*"),
        json={"data": None, "error": "limit out of range"},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        with pytest.raises(ArcticShiftError, match="limit out of range"):
            await client.search_comments(
                subreddit="botox",
                text_query="botox",
                start=START,
                end=END,
                paginate=False,
            )


# ---------------------------------------------------------------------------
# _get: 429 retry-after handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_honors_retry_after_on_429(httpx_mock, monkeypatch):  # type: ignore[no-untyped-def]
    """429 with Retry-After header: sleep ~header value (±20% jitter), then succeed."""
    random.seed(42)  # fix jitter so assertion range is deterministic
    monkeypatch.setattr("botox_se.ingest.reddit_client._MIN_REQUEST_INTERVAL_S", 0.0)

    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        status_code=429,
        headers={"retry-after": "2"},
    )
    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        json={"data": []},
    )

    slept: list[float] = []

    async def fake_sleep(x: float) -> None:
        slept.append(x)

    monkeypatch.setattr("botox_se.ingest.reddit_client.asyncio.sleep", fake_sleep)

    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        payload = await client._get("/posts/search", params={"subreddit": "x"})

    assert payload == {"data": []}
    assert len(slept) == 1
    # Base was 2 s; jitter ±20% → [1.6, 2.4]
    assert 1.6 <= slept[0] <= 2.4


@pytest.mark.asyncio
async def test_get_429_no_retry_after_uses_base_waits(httpx_mock, monkeypatch):  # type: ignore[no-untyped-def]
    """429 without Retry-After falls back to _BASE_429_WAITS[0] (5 s) with jitter."""
    random.seed(0)
    monkeypatch.setattr("botox_se.ingest.reddit_client._MIN_REQUEST_INTERVAL_S", 0.0)

    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        status_code=429,
    )
    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        json={"data": []},
    )

    slept: list[float] = []

    async def fake_sleep(x: float) -> None:
        slept.append(x)

    monkeypatch.setattr("botox_se.ingest.reddit_client.asyncio.sleep", fake_sleep)

    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        payload = await client._get("/posts/search", params={})

    assert payload == {"data": []}
    assert len(slept) == 1
    # Base is 5 s; jitter ±20% → [4.0, 6.0]
    assert 4.0 <= slept[0] <= 6.0


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_get_429_exhausts_attempts_and_raises(httpx_mock, monkeypatch):  # type: ignore[no-untyped-def]
    """After _MAX_429_ATTEMPTS (6) 429 responses the client raises HTTPStatusError.

    We register 10 responses but only 7 will be consumed (1 initial + 6 retries before
    the 7th attempt triggers the guard and raises). The marker suppresses teardown
    assertion about unused responses.
    """
    for _ in range(10):
        httpx_mock.add_response(
            url=re.compile(r".*/posts/search.*"),
            status_code=429,
            headers={"retry-after": "1"},
        )

    async def fake_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("botox_se.ingest.reddit_client.asyncio.sleep", fake_sleep)

    import httpx as _httpx

    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        with pytest.raises(_httpx.HTTPStatusError) as exc_info:
            await client._get("/posts/search", params={})

    assert exc_info.value.response.status_code == 429


# ---------------------------------------------------------------------------
# _get: 422 treated as retryable (transient server error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_422_is_retryable(httpx_mock, monkeypatch):  # type: ignore[no-untyped-def]
    """A 422 (non-soft-throttle body) is retried via the 5xx backoff path and succeeds."""
    monkeypatch.setattr("botox_se.ingest.reddit_client._MIN_REQUEST_INTERVAL_S", 0.0)
    # Body without the "timeout"/"slow down" tokens so it goes through the 5xx branch,
    # not the soft-throttle (429-equivalent) branch.
    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        status_code=422,
        json={"data": None, "error": "Some transient issue"},
    )
    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        json={"data": []},
    )

    slept: list[float] = []

    async def fake_sleep(x: float) -> None:
        slept.append(x)

    monkeypatch.setattr("botox_se.ingest.reddit_client.asyncio.sleep", fake_sleep)

    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        payload = await client._get("/posts/search", params={"subreddit": "x"})

    assert payload == {"data": []}
    assert len(slept) == 1


@pytest.mark.asyncio
async def test_get_422_with_timeout_body_treated_as_soft_throttle(httpx_mock, monkeypatch):  # type: ignore[no-untyped-def]
    """A 422 with body `{"error": "Timeout. Maybe slow down a bit"}` uses the 429
    long-backoff path (5 s base) rather than the 5xx short-backoff path (1 s base)."""
    random.seed(0)
    monkeypatch.setattr("botox_se.ingest.reddit_client._MIN_REQUEST_INTERVAL_S", 0.0)

    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        status_code=422,
        json={"data": None, "error": "Timeout. Maybe slow down a bit"},
    )
    httpx_mock.add_response(
        url=re.compile(r".*/posts/search.*"),
        json={"data": []},
    )

    slept: list[float] = []

    async def fake_sleep(x: float) -> None:
        slept.append(x)

    monkeypatch.setattr("botox_se.ingest.reddit_client.asyncio.sleep", fake_sleep)

    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        payload = await client._get("/posts/search", params={})

    assert payload == {"data": []}
    assert len(slept) == 1
    # Soft-throttle path uses _BASE_429_WAITS[0] = 5 s with ±20% jitter.
    assert 4.0 <= slept[0] <= 6.0
