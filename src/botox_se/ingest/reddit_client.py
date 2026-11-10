"""Arctic Shift client for historical Reddit search.

Live API at https://arctic-shift.photon-reddit.com/api. Docs: https://github.com/ArthurHeitmann/arctic_shift

Post search: matches title= OR selftext= (two separate requests, merged by id).
Comment search: matches body=.
Max limit per request is 100; pagination is cursor-based using `after` + `sort=asc`.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import httpx

from botox_se.logging import get_logger

_log = get_logger(__name__)

_MAX_429_ATTEMPTS = 6
_MAX_5XX_ATTEMPTS = 5
_BASE_429_WAITS = [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
# Minimum seconds between outgoing requests — keeps us well under Arctic Shift's
# per-client throttle ceiling. ~3 req/s is polite and empirically avoids the
# "Timeout. Maybe slow down a bit" soft-throttle response.
_MIN_REQUEST_INTERVAL_S = 0.35

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
MAX_LIMIT = 100


def _is_soft_throttle_422(resp: "httpx.Response") -> bool:
    """Arctic Shift sometimes returns 422 with a body like
    `{"error": "Timeout. Maybe slow down a bit"}` — treat as 429-equivalent."""
    try:
        payload = resp.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    err = (payload.get("error") or "").lower()
    return "timeout" in err or "slow down" in err


class ArcticShiftError(RuntimeError):
    """Raised when the Arctic Shift API returns an error payload."""


@dataclass(frozen=True)
class RedditPost:
    id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    created_utc: int
    permalink: str
    score: int

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "RedditPost":
        return cls(
            id=raw["id"],
            subreddit=raw.get("subreddit", "") or "",
            title=raw.get("title", "") or "",
            selftext=raw.get("selftext", "") or "",
            author=raw.get("author", "") or "",
            created_utc=int(raw.get("created_utc", 0)),
            permalink=raw.get("permalink", "") or "",
            score=int(raw.get("score") or 0),
        )


@dataclass(frozen=True)
class RedditComment:
    id: str
    post_id: str
    subreddit: str
    body: str
    author: str
    created_utc: int

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "RedditComment":
        link_id = raw.get("link_id", "") or ""
        post_id = link_id.removeprefix("t3_")
        return cls(
            id=raw["id"],
            post_id=post_id,
            subreddit=raw.get("subreddit", "") or "",
            body=raw.get("body", "") or "",
            author=raw.get("author", "") or "",
            created_utc=int(raw.get("created_utc", 0)),
        )


class ArcticShiftClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str = ARCTIC_BASE) -> None:
        self._http = http
        self._base = base_url
        self._rate_lock = asyncio.Lock()
        self._next_request_at: float = 0.0

    async def search_posts(
        self,
        *,
        subreddit: str,
        text_query: str,
        start: int,
        end: int,
        limit: int = MAX_LIMIT,
        paginate: bool = True,
    ) -> list[RedditPost]:
        """Search posts matching `text_query` in title OR selftext, in [start, end).

        If paginate=True (default), follows the cursor to return ALL matches in the window.
        If paginate=False, returns at most `limit` results per search field (<=200 combined).
        """
        title_rows = await self._collect_posts(
            subreddit=subreddit,
            field="title",
            text_query=text_query,
            start=start,
            end=end,
            limit=limit,
            paginate=paginate,
        )
        body_rows = await self._collect_posts(
            subreddit=subreddit,
            field="selftext",
            text_query=text_query,
            start=start,
            end=end,
            limit=limit,
            paginate=paginate,
        )
        seen: set[str] = set()
        out: list[RedditPost] = []
        for p in (*title_rows, *body_rows):
            if p.id in seen:
                continue
            seen.add(p.id)
            out.append(p)
        return out

    async def search_comments(
        self,
        *,
        subreddit: str,
        text_query: str,
        start: int,
        end: int,
        limit: int = MAX_LIMIT,
        paginate: bool = True,
    ) -> list[RedditComment]:
        rows = await self._collect_comments(
            subreddit=subreddit,
            text_query=text_query,
            start=start,
            end=end,
            limit=limit,
            paginate=paginate,
        )
        seen: set[str] = set()
        out: list[RedditComment] = []
        for c in rows:
            if c.id in seen:
                continue
            seen.add(c.id)
            out.append(c)
        return out

    # --- internals ---

    async def _collect_posts(
        self,
        *,
        subreddit: str,
        field: str,  # "title" or "selftext"
        text_query: str,
        start: int,
        end: int,
        limit: int,
        paginate: bool,
    ) -> list[RedditPost]:
        out: list[RedditPost] = []
        cursor = start
        while True:
            params: dict[str, Any] = {
                "subreddit": subreddit,
                field: text_query,
                "after": cursor,
                "before": end,
                "limit": limit,
                "sort": "asc",
            }
            payload = await self._get("/posts/search", params=params)
            rows = [RedditPost.from_api(r) for r in (payload.get("data") or [])]
            if not rows:
                break
            out.extend(rows)
            if not paginate or len(rows) < limit:
                break
            cursor = rows[-1].created_utc + 1
            if cursor >= end:
                break
        return out

    async def _collect_comments(
        self,
        *,
        subreddit: str,
        text_query: str,
        start: int,
        end: int,
        limit: int,
        paginate: bool,
    ) -> list[RedditComment]:
        out: list[RedditComment] = []
        cursor = start
        while True:
            params: dict[str, Any] = {
                "subreddit": subreddit,
                "body": text_query,
                "after": cursor,
                "before": end,
                "limit": limit,
                "sort": "asc",
            }
            payload = await self._get("/comments/search", params=params)
            rows = [RedditComment.from_api(r) for r in (payload.get("data") or [])]
            if not rows:
                break
            out.extend(rows)
            if not paginate or len(rows) < limit:
                break
            cursor = rows[-1].created_utc + 1
            if cursor >= end:
                break
        return out

    async def _throttle(self) -> None:
        """Enforce a per-client minimum interval between outgoing requests."""
        async with self._rate_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next_request_at:
                await asyncio.sleep(self._next_request_at - now)
            self._next_request_at = loop.time() + _MIN_REQUEST_INTERVAL_S

    async def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        attempt_429 = 0
        attempt_5xx = 0
        while True:
            await self._throttle()
            try:
                resp = await self._http.get(f"{self._base}{path}", params=params, timeout=30.0)
                resp.raise_for_status()
                payload: dict[str, Any] = resp.json()
                if payload.get("error"):
                    raise ArcticShiftError(f"{path} failed: {payload['error']} (params={params})")
                return payload
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # 429 OR 422-with-timeout-body → soft rate-limit, long backoff
                if status == 429 or (status == 422 and _is_soft_throttle_422(e.response)):
                    if attempt_429 >= _MAX_429_ATTEMPTS:
                        raise
                    retry_after_header = e.response.headers.get("retry-after")
                    base_wait = (
                        float(retry_after_header)
                        if retry_after_header and retry_after_header.isdigit()
                        else _BASE_429_WAITS[min(attempt_429, len(_BASE_429_WAITS) - 1)]
                    )
                    wait = base_wait * random.uniform(0.8, 1.2)
                    _log.warning(
                        "rate_limited",
                        path=path,
                        status=status,
                        attempt=attempt_429 + 1,
                        wait_seconds=round(wait, 2),
                    )
                    await asyncio.sleep(wait)
                    attempt_429 += 1
                    continue
                if 500 <= status < 600 or status == 422:
                    if attempt_5xx >= _MAX_5XX_ATTEMPTS:
                        raise
                    wait = min(30.0, 2.0**attempt_5xx) * random.uniform(0.8, 1.2)
                    _log.warning(
                        "server_error_retry",
                        path=path,
                        status=status,
                        attempt=attempt_5xx + 1,
                        wait_seconds=round(wait, 2),
                    )
                    await asyncio.sleep(wait)
                    attempt_5xx += 1
                    continue
                # Other 4xx: do not retry
                raise
            except httpx.HTTPError:
                # Network-level errors (connection, timeout)
                if attempt_5xx >= _MAX_5XX_ATTEMPTS:
                    raise
                wait = min(30.0, 2.0**attempt_5xx) * random.uniform(0.8, 1.2)
                _log.warning(
                    "network_error_retry",
                    path=path,
                    attempt=attempt_5xx + 1,
                    wait_seconds=round(wait, 2),
                )
                await asyncio.sleep(wait)
                attempt_5xx += 1
                continue
