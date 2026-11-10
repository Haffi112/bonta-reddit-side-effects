"""Combine raw posts + comments JSONL into a single interim parquet.

Applies dedupe + basic filters. Produces `data/interim/posts.parquet`
with schema: id, subreddit, author, created_utc, text, record_type, post_id.
"""

from pathlib import Path

import polars as pl

from botox_se.ingest.dedupe import dedupe_posts
from botox_se.ingest.filters import apply_basic_filters, filter_english


def normalize_raw_to_interim(
    *,
    posts_jsonl: Path,
    comments_jsonl: Path,
    out_path: Path,
    apply_language_filter: bool = True,
) -> int:
    """Merge + dedupe + filter; return number of rows written."""
    if not posts_jsonl.exists():
        raise FileNotFoundError(f"Posts JSONL not found: {posts_jsonl}")
    if not comments_jsonl.exists():
        raise FileNotFoundError(f"Comments JSONL not found: {comments_jsonl}")
    posts = pl.read_ndjson(posts_jsonl)
    comments = pl.read_ndjson(comments_jsonl)

    # Unify schema
    post_rows = posts.select(
        id=pl.col("id"),
        subreddit=pl.col("subreddit"),
        author=pl.col("author"),
        created_utc=pl.col("created_utc"),
        text=(pl.col("title").fill_null("") + "\n\n" + pl.col("selftext").fill_null("")),
        record_type=pl.lit("post"),
        post_id=pl.col("id"),
    )
    comment_rows = comments.select(
        id=pl.col("id"),
        subreddit=pl.col("subreddit"),
        author=pl.col("author"),
        created_utc=pl.col("created_utc"),
        text=pl.col("body"),
        record_type=pl.lit("comment"),
        post_id=pl.col("post_id"),
    )
    merged = pl.concat([post_rows, comment_rows])
    merged = dedupe_posts(merged, id_col="id")
    merged = apply_basic_filters(merged, text_col="text", min_len=20)
    if apply_language_filter:
        merged = filter_english(merged, text_col="text")

    merged.write_parquet(out_path)
    return merged.height
