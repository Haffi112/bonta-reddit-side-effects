import json
from pathlib import Path

import polars as pl

from botox_se.ingest.normalize import normalize_raw_to_interim


def test_normalize_combines_posts_and_comments_into_unified_parquet(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "interim").mkdir()

    posts_jsonl = tmp_path / "raw" / "posts.jsonl"
    comments_jsonl = tmp_path / "raw" / "comments.jsonl"
    posts_jsonl.write_text(
        json.dumps(
            {
                "id": "p1",
                "subreddit": "botox",
                "title": "hi",
                "selftext": "had botox for my forehead last week, great results!",
                "author": "u1",
                "created_utc": 1700000000,
                "permalink": "",
                "score": 5,
            }
        )
        + "\n"
    )
    comments_jsonl.write_text(
        json.dumps(
            {
                "id": "c1",
                "post_id": "p1",
                "subreddit": "botox",
                "body": "what dose did you use for your forehead treatment last week",
                "author": "u2",
                "created_utc": 1700000100,
            }
        )
        + "\n"
    )

    out = tmp_path / "interim" / "posts.parquet"
    n = normalize_raw_to_interim(
        posts_jsonl=posts_jsonl,
        comments_jsonl=comments_jsonl,
        out_path=out,
        apply_language_filter=False,
    )
    assert n == 2
    df = pl.read_parquet(out)
    assert set(df["record_type"].to_list()) == {"post", "comment"}
    assert set(df.columns) >= {"id", "subreddit", "author", "created_utc", "text", "record_type"}
