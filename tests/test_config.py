import pytest

from botox_se.config import Settings


def test_settings_reads_from_env(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENFDA_API_KEY", "openfda-test")
    s = Settings(_env_file=None)
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.openfda_api_key.get_secret_value() == "openfda-test"
    assert s.llm_model == "gpt-5.3-chat-latest"
    assert s.collection_start == "2015-01-01"
    assert s.collection_end == "2026-03-31"


def test_require_openai_key_raises_when_missing(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        s.require_openai_key()


def test_require_openai_key_returns_value_when_set(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    s = Settings(_env_file=None)
    assert s.require_openai_key() == "sk-abc"


def test_require_openfda_key_raises_when_missing(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
    s = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="OPENFDA_API_KEY is not set"):
        s.require_openfda_key()


def test_require_reddit_credentials_raises_when_missing(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    s = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set"):
        s.require_reddit_credentials()


def test_require_reddit_credentials_returns_values_when_set(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
    s = Settings(_env_file=None)
    cid, csecret, ua = s.require_reddit_credentials()
    assert cid == "test_id"
    assert csecret == "test_secret"
    assert "botox-se-research" in ua
