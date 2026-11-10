"""Deduplication for ingested posts/comments."""

import polars as pl


def dedupe_posts(df: pl.DataFrame, *, id_col: str = "id") -> pl.DataFrame:
    """Remove duplicate rows by id_col, keeping the first occurrence."""
    return df.unique(subset=id_col, keep="first", maintain_order=True)
