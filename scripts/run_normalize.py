"""CLI: normalize raw JSONL into filtered interim parquet.

Reads:
  data/raw/posts.jsonl
  data/raw/comments.jsonl

Writes:
  data/interim/posts.parquet
"""

from pathlib import Path

from botox_se.ingest.normalize import normalize_raw_to_interim
from botox_se.logging import configure_logging, get_logger
from botox_se.paths import Paths


def main() -> None:
    configure_logging(level="INFO")
    log = get_logger("normalize")
    paths = Paths(repo_root=Path(__file__).resolve().parents[1])
    paths.ensure_dirs()

    posts_jsonl = paths.raw / "posts.jsonl"
    comments_jsonl = paths.raw / "comments.jsonl"
    out_path = paths.interim / "posts.parquet"

    n = normalize_raw_to_interim(
        posts_jsonl=posts_jsonl,
        comments_jsonl=comments_jsonl,
        out_path=out_path,
    )
    log.info("normalize_done", rows_written=n, out_path=str(out_path))


if __name__ == "__main__":
    main()
