import polars as pl

from botox_se.ingest.dedupe import dedupe_posts


def test_dedupe_drops_duplicate_ids_keeps_first() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "a"],
            "text": ["first", "x", "second"],
        }
    )
    out = dedupe_posts(df, id_col="id")
    assert out["id"].to_list() == ["a", "b"]
    assert out.filter(pl.col("id") == "a")["text"].item() == "first"
