"""Text-level filters applied before LLM extraction.

Goals:
- Drop empty/short posts
- Drop obvious bot text (removed, deleted, mod-auto)
- Language filter (English) — applied separately for cost
"""

import polars as pl
from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore[import-untyped]

DetectorFactory.seed = 0  # deterministic


BOT_PATTERNS = [
    r"(?i)^\[removed\]$",
    r"(?i)^\[deleted\]$",
    r"(?i)^i am a bot",
    r"(?i)this is an automated (message|response)",
    r"(?i)performed automatically",
    r"(?i)contact the moderators of this subreddit",
    r"(?i)^\*I am a bot",
    r"(?i)automoderator",
]


def apply_basic_filters(
    df: pl.DataFrame,
    *,
    text_col: str,
    min_len: int = 20,
) -> pl.DataFrame:
    """Drop rows that are null, too short, or match bot patterns."""
    out = df.filter(pl.col(text_col).is_not_null() & (pl.col(text_col).str.len_chars() >= min_len))
    for pat in BOT_PATTERNS:
        out = out.filter(~pl.col(text_col).str.contains(pat))
    return out


def _is_english(text: str) -> bool:
    try:
        result: str = detect(text)
        return result == "en"
    except LangDetectException:
        return False


def filter_english(df: pl.DataFrame, *, text_col: str) -> pl.DataFrame:
    """Keep only rows where the text is detected as English."""
    mask = df[text_col].map_elements(_is_english, return_dtype=pl.Boolean)
    return df.filter(mask)
