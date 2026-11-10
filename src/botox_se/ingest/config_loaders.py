"""Load YAML config files for ingestion."""

from dataclasses import dataclass
from pathlib import Path

import yaml


def load_search_terms(path: Path) -> list[str]:
    """Load and return the list of search terms from a YAML file."""
    data = yaml.safe_load(path.read_text())
    return list(data["terms"])


@dataclass(frozen=True)
class SubredditConfig:
    """Configuration for subreddit selection."""

    medical: list[str]
    cosmetic: list[str]
    mixed: list[str]
    min_posts_threshold: int

    @property
    def all_subreddits(self) -> list[str]:
        """Return all subreddits across all categories in order."""
        return [*self.medical, *self.cosmetic, *self.mixed]


def load_subreddits(path: Path) -> SubredditConfig:
    """Load and return the subreddit configuration from a YAML file."""
    data = yaml.safe_load(path.read_text())
    return SubredditConfig(
        medical=list(data.get("medical", [])),
        cosmetic=list(data.get("cosmetic", [])),
        mixed=list(data.get("mixed", [])),
        min_posts_threshold=int(data.get("min_posts_threshold", 50)),
    )
