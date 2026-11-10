from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from botox_se.ingest.praw_client import (
    EnrichedComment,
    EnrichedPost,
    PrawClient,
)


def _fake_submission(
    *,
    post_id: str = "abc",
    comments: list[MagicMock] | None = None,
) -> MagicMock:
    sub = MagicMock()
    sub.id = post_id
    sub.subreddit = MagicMock(__str__=lambda self: "botox")
    sub.title = "Test post"
    sub.selftext = "body text"
    sub.author = "user1"
    sub.created_utc = 1700000000
    sub.permalink = f"/r/botox/comments/{post_id}"
    sub.score = 10
    sub.num_comments = len(comments or [])
    sub.over_18 = False
    sub.comments.list.return_value = comments or []
    sub.comments.replace_more.return_value = None
    return sub


def _fake_comment(
    *,
    cid: str = "c1",
    post_id: str = "abc",
    parent: str | None = None,
    body: str = "nice",
) -> MagicMock:
    """Return a MagicMock that simulates a PRAW Comment.

    Duck-typed check (hasattr body + parent_id) means this works without
    patching isinstance.
    """
    c = MagicMock()
    c.id = cid
    c.link_id = f"t3_{post_id}"
    c.parent_id = parent or f"t3_{post_id}"
    c.subreddit = MagicMock(__str__=lambda self: "botox")
    c.body = body
    c.author = "commenter"
    c.created_utc = 1700000001
    c.score = 3
    c.is_submitter = False
    return c


@pytest.mark.asyncio
async def test_fetch_post_with_comments_returns_enriched_post_and_comments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reddit = MagicMock()
    c1 = _fake_comment(cid="c1")
    fake_sub = _fake_submission(post_id="abc", comments=[c1])
    fake_reddit.submission.return_value = fake_sub

    # Patch praw.Reddit to return our fake
    monkeypatch.setattr("botox_se.ingest.praw_client.praw.Reddit", lambda **kw: fake_reddit)

    client = PrawClient(client_id="x", client_secret="y", user_agent="ua")
    post, comments = await client.fetch_post_with_comments("abc")

    assert isinstance(post, EnrichedPost)
    assert post.id == "abc"
    assert post.subreddit == "botox"
    assert post.num_comments == 1
    assert len(comments) == 1
    assert isinstance(comments[0], EnrichedComment)
    assert comments[0].post_id == "abc"


@pytest.mark.asyncio
async def test_fetch_post_returns_none_on_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reddit = MagicMock()
    bad_sub = MagicMock()
    type(bad_sub).title = property(lambda self: (_ for _ in ()).throw(Exception("404 not found")))
    fake_reddit.submission.return_value = bad_sub

    monkeypatch.setattr("botox_se.ingest.praw_client.praw.Reddit", lambda **kw: fake_reddit)
    client = PrawClient(client_id="x", client_secret="y", user_agent="ua")
    post, comments = await client.fetch_post_with_comments("missing")
    assert post is None
    assert comments == []


@pytest.mark.asyncio
async def test_fetch_post_with_no_comments_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reddit = MagicMock()
    fake_sub = _fake_submission(post_id="xyz", comments=[])
    fake_reddit.submission.return_value = fake_sub

    monkeypatch.setattr("botox_se.ingest.praw_client.praw.Reddit", lambda **kw: fake_reddit)
    client = PrawClient(client_id="x", client_secret="y", user_agent="ua")
    post, comments = await client.fetch_post_with_comments("xyz")

    assert post is not None
    assert post.id == "xyz"
    assert comments == []


@pytest.mark.asyncio
async def test_fetch_post_comment_fields_are_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reddit = MagicMock()
    c1 = _fake_comment(cid="ccc", post_id="ppp", body="great post")
    fake_sub = _fake_submission(post_id="ppp", comments=[c1])
    fake_reddit.submission.return_value = fake_sub

    monkeypatch.setattr("botox_se.ingest.praw_client.praw.Reddit", lambda **kw: fake_reddit)
    client = PrawClient(client_id="x", client_secret="y", user_agent="ua")
    post, comments = await client.fetch_post_with_comments("ppp")

    assert post is not None
    ec = comments[0]
    assert ec.id == "ccc"
    assert ec.post_id == "ppp"
    assert ec.body == "great post"
    assert ec.subreddit == "botox"
    assert ec.score == 3
    assert ec.is_submitter is False
