import polars as pl

from botox_se.ingest.filters import apply_basic_filters, filter_english


def test_filters_drop_short_and_empty_posts() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "text": ["", "short", "this is a perfectly reasonable post about botox", "ok"],
        }
    )
    out = apply_basic_filters(df, text_col="text", min_len=10)
    assert out["id"].to_list() == ["c"]


def test_filter_bot_patterns_dropped() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "text": [
                "I am a bot and this is a compiled list of comments",
                "Just got botox for my forehead, loving the results.",
            ],
        }
    )
    out = apply_basic_filters(df, text_col="text", min_len=10)
    assert out["id"].to_list() == ["b"]


def test_filter_bot_patterns_automoderator_and_mod_contact_dropped() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": [
                "*I am a bot, and this action was performed automatically.",
                "Please contact the moderators of this subreddit if you have questions.",
                "Had great results with botox for my jaw tension after six weeks.",
            ],
        }
    )
    out = apply_basic_filters(df, text_col="text", min_len=10)
    assert out["id"].to_list() == ["c"]


def test_filter_english_keeps_english_drops_other() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": [
                "I had botox in my forehead last week and it worked well for my wrinkles",
                "Ich habe Botox fur meine Stirn letzte Woche genommen und bin sehr zufrieden",
                "Got great results from my first botox appointment for forehead lines",
            ],
        }
    )
    out = filter_english(df, text_col="text")
    assert set(out["id"].to_list()) == {"a", "c"}
