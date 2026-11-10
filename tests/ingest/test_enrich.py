from __future__ import annotations

import json
from pathlib import Path

import pytest

from botox_se.ingest.enrich import enrich_posts
from botox_se.ingest.praw_client import EnrichedComment, EnrichedPost


class FakePrawClient:
    def __init__(
        self,
        responses: dict[str, tuple[EnrichedPost | None, list[EnrichedComment]]],
    ) -> None:
        self._r = responses

    async def fetch_post_with_comments(
        self, post_id: str
    ) -> tuple[EnrichedPost | None, list[EnrichedComment]]:
        return self._r.get(post_id, (None, []))


@pytest.mark.asyncio
async def test_enrich_writes_enriched_posts_and_comments(tmp_path: Path) -> None:
    posts_in = tmp_path / "posts.jsonl"
    posts_in.write_text(
        json.dumps({"id": "abc", "subreddit": "botox"})
        + "\n"
        + json.dumps({"id": "def", "subreddit": "migraine"})
        + "\n"
    )
    ep = EnrichedPost(
        id="abc",
        subreddit="botox",
        title="t",
        selftext="s",
        author="u",
        created_utc=1,
        permalink="",
        score=0,
        num_comments=1,
        over_18=False,
    )
    c1 = EnrichedComment(
        id="c1",
        post_id="abc",
        parent_id="t3_abc",
        subreddit="botox",
        body="nice",
        author="u2",
        created_utc=2,
        score=1,
        is_submitter=False,
        depth=0,
    )
    fake = FakePrawClient({"abc": (ep, [c1]), "def": (None, [])})

    totals = await enrich_posts(
        client=fake,  # type: ignore[arg-type]
        posts_path=posts_in,
        comments_path=tmp_path / "comments.jsonl",
        enriched_posts_path=tmp_path / "enriched_posts.jsonl",
    )
    assert totals["posts_fetched"] == 1
    assert totals["posts_missing"] == 1
    assert totals["comments_written"] == 1
    comments = (tmp_path / "comments.jsonl").read_text().splitlines()
    assert json.loads(comments[0])["id"] == "c1"


@pytest.mark.asyncio
async def test_enrich_is_resumable(tmp_path: Path) -> None:
    posts_in = tmp_path / "posts.jsonl"
    posts_in.write_text(json.dumps({"id": "abc"}) + "\n" + json.dumps({"id": "def"}) + "\n")
    # Pre-existing comments file: abc already enriched
    comments_in = tmp_path / "comments.jsonl"
    comments_in.write_text(json.dumps({"id": "c0", "post_id": "abc"}) + "\n")
    # Only define response for def; abc should be skipped
    ep_def = EnrichedPost(
        id="def",
        subreddit="migraine",
        title="",
        selftext="",
        author="u",
        created_utc=1,
        permalink="",
        score=0,
        num_comments=0,
        over_18=False,
    )
    fake = FakePrawClient({"def": (ep_def, [])})

    totals = await enrich_posts(
        client=fake,  # type: ignore[arg-type]
        posts_path=posts_in,
        comments_path=comments_in,
        enriched_posts_path=tmp_path / "enriched_posts.jsonl",
        resume=True,
    )
    assert totals["posts_fetched"] == 1  # only def
    assert totals["posts_missing"] == 0


@pytest.mark.asyncio
async def test_enrich_no_resume_reprocesses_all(tmp_path: Path) -> None:
    posts_in = tmp_path / "posts.jsonl"
    posts_in.write_text(json.dumps({"id": "abc"}) + "\n")
    ep = EnrichedPost(
        id="abc",
        subreddit="botox",
        title="t",
        selftext="",
        author="u",
        created_utc=1,
        permalink="",
        score=0,
        num_comments=0,
        over_18=False,
    )
    # Pre-existing file that would normally cause skip
    comments_in = tmp_path / "comments.jsonl"
    comments_in.write_text(json.dumps({"id": "c0", "post_id": "abc"}) + "\n")

    fake = FakePrawClient({"abc": (ep, [])})
    totals = await enrich_posts(
        client=fake,  # type: ignore[arg-type]
        posts_path=posts_in,
        comments_path=comments_in,
        enriched_posts_path=tmp_path / "enriched_posts.jsonl",
        resume=False,  # force reprocess
    )
    assert totals["posts_fetched"] == 1
    assert totals["posts_missing"] == 0


@pytest.mark.asyncio
async def test_enrich_enriched_posts_written(tmp_path: Path) -> None:
    posts_in = tmp_path / "posts.jsonl"
    posts_in.write_text(json.dumps({"id": "abc"}) + "\n")
    ep = EnrichedPost(
        id="abc",
        subreddit="botox",
        title="Test",
        selftext="body",
        author="user",
        created_utc=100,
        permalink="/r/botox/comments/abc",
        score=5,
        num_comments=0,
        over_18=False,
    )
    fake = FakePrawClient({"abc": (ep, [])})
    enriched_out = tmp_path / "enriched_posts.jsonl"

    await enrich_posts(
        client=fake,  # type: ignore[arg-type]
        posts_path=posts_in,
        comments_path=tmp_path / "comments.jsonl",
        enriched_posts_path=enriched_out,
    )

    rows = enriched_out.read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["id"] == "abc"
    assert row["title"] == "Test"
    assert row["score"] == 5
