"""Typed settings loader. Reads from environment (via .env) with defaults."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = (
    Path(__file__).resolve().parents[2]
)  # src/botox_se/config.py → src/botox_se → src → repo_root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = "gpt-5.3-chat-latest"
    llm_reranker_model: str = "gpt-5.3-chat-latest"
    llm_temperature: float = 0.0

    # External APIs
    openfda_api_key: SecretStr = Field(default=SecretStr(""))

    # Collection window
    collection_start: str = "2015-01-01"
    collection_end: str = "2026-03-31"

    # Reddit OAuth (PRAW)
    reddit_client_id: SecretStr = Field(default=SecretStr(""))
    reddit_client_secret: SecretStr = Field(default=SecretStr(""))
    reddit_user_agent: str = "botox-se-research/0.1 (by /u/anonymous)"

    # Runtime
    batch_api_enabled: bool = True
    concurrent_requests: int = 2

    def require_openai_key(self) -> str:
        key = self.openai_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it before "
                "running any LLM-dependent pipeline stage."
            )
        return key

    def require_openfda_key(self) -> str:
        key = self.openfda_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "OPENFDA_API_KEY is not set. Add it to .env before running FAERS ingestion."
            )
        return key

    def require_reddit_credentials(self) -> tuple[str, str, str]:
        client_id = self.reddit_client_id.get_secret_value()
        client_secret = self.reddit_client_secret.get_secret_value()
        if not client_id or not client_secret:
            raise RuntimeError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set. "
                "Create a Reddit 'script' app at https://www.reddit.com/prefs/apps "
                "and add to .env."
            )
        return client_id, client_secret, self.reddit_user_agent


def load() -> Settings:
    """Entrypoint used throughout code; tests can patch env vars."""
    return Settings()
