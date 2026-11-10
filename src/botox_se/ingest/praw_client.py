"""PRAW wrapper for fetching full submission + comment tree by post ID.

PRAW is synchronous; wrap calls in asyncio.to_thread so this stays usable
from the rest of our async pipeline. Reddit's OAuth rate limit is 100 req/min
per token — PRAW handles Retry-After and sleeps internally, so we don't need
our own retry/backoff machinery here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import praw
from praw.models import Comment as PrawComment
from praw.models import Submission as PrawSubmission

from botox_se.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class EnrichedPost:
    id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    created_utc: int
    permalink: str
    score: int
    num_comments: int
    over_18: bool


@dataclass(frozen=True)
class EnrichedComment:
    id: str
    post_id: str
    parent_id: str  # "t3_<post>" or "t1_<comment>"
    subreddit: str
    body: str
    author: str
    created_utc: int
    score: int
    is_submitter: bool
    depth: int  # 0 = top-level, 1 = reply-to-top-level, etc.


class PrawClient:
    """Thin async-friendly facade over praw.Reddit."""

    def __init__(self, *, client_id: str, client_secret: str, user_agent: str) -> None:
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,  # we explicitly use to_thread, silence PRAW's warning
        )
        self._reddit.read_only = True

    async def fetch_post_with_comments(
        self, post_id: str
    ) -> tuple[EnrichedPost | None, list[EnrichedComment]]:
        """Fetch a submission by id (without the t3_ prefix) plus its full comment tree.

        Returns (None, []) if the post is deleted/removed/inaccessible.
        """
        return await asyncio.to_thread(self._fetch_sync, post_id)

    # --- sync internal ---

    def _fetch_sync(self, post_id: str) -> tuple[EnrichedPost | None, list[EnrichedComment]]:
        try:
            submission: PrawSubmission = self._reddit.submission(id=post_id)
            # Accessing an attribute forces PRAW to fetch.
            _ = submission.title
        except Exception as e:
            _log.warning("praw_fetch_failed", post_id=post_id, exc_type=type(e).__name__)
            return None, []

        ep = self._to_enriched_post(submission)

        # Expand "MoreComments" nodes recursively so we walk the full tree.
        try:
            submission.comments.replace_more(limit=None)
        except Exception as e:
            _log.warning("praw_replace_more_failed", post_id=post_id, exc_type=type(e).__name__)

        comments: list[EnrichedComment] = []
        for c in submission.comments.list():
            # Duck-typed check: any object with body + parent_id is a comment
            if hasattr(c, "body") and hasattr(c, "parent_id"):
                comments.append(self._to_enriched_comment(c, depth=_depth_of(c)))
        return ep, comments

    @staticmethod
    def _to_enriched_post(s: PrawSubmission) -> EnrichedPost:
        return EnrichedPost(
            id=s.id,
            subreddit=str(s.subreddit),
            title=s.title or "",
            selftext=s.selftext or "",
            author=str(s.author) if s.author else "",
            created_utc=int(s.created_utc),
            permalink=s.permalink or "",
            score=int(s.score or 0),
            num_comments=int(s.num_comments or 0),
            over_18=bool(s.over_18),
        )

    @staticmethod
    def _to_enriched_comment(c: PrawComment, *, depth: int) -> EnrichedComment:
        return EnrichedComment(
            id=c.id,
            post_id=str(c.link_id).removeprefix("t3_"),
            parent_id=str(c.parent_id),
            subreddit=str(c.subreddit),
            body=c.body or "",
            author=str(c.author) if c.author else "",
            created_utc=int(c.created_utc),
            score=int(c.score or 0),
            is_submitter=bool(c.is_submitter),
            depth=depth,
        )


def _depth_of(c: Any) -> int:
    """Walk up parent chain to compute thread depth. Top-level comment = 0."""
    depth = 0
    parent = c.parent() if hasattr(c, "parent") else None
    while parent is not None and hasattr(parent, "parent"):
        depth += 1
        try:
            p = parent.parent()
        except Exception:
            break
        if p is parent or not hasattr(p, "parent"):
            break
        parent = p
        if depth > 50:
            break  # pathological safeguard
    return depth
