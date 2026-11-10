from pathlib import Path

from botox_se.ingest.config_loaders import load_search_terms, load_subreddits


def test_load_search_terms(tmp_path: Path) -> None:
    yaml = tmp_path / "search_terms.yaml"
    yaml.write_text("terms:\n  - botox\n  - dysport\n")
    assert load_search_terms(yaml) == ["botox", "dysport"]


def test_load_subreddits_flattens_all_buckets(tmp_path: Path) -> None:
    yaml = tmp_path / "subreddits.yaml"
    yaml.write_text(
        "medical:\n  - migraine\ncosmetic:\n  - Botox\nmixed:\n  - AskDocs\n"
        "min_posts_threshold: 50\n"
    )
    cfg = load_subreddits(yaml)
    assert cfg.all_subreddits == ["migraine", "Botox", "AskDocs"]
    assert cfg.min_posts_threshold == 50
