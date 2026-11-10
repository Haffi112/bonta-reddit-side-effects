import json

from botox_se.logging import configure_logging, get_logger


def test_logger_outputs_json_with_context(capsys):  # type: ignore[no-untyped-def]
    configure_logging(level="INFO", json=True)
    log = get_logger("test").bind(task="ingest", subreddit="r/botox")
    log.info("fetched", count=42)
    captured = capsys.readouterr().err.strip().splitlines()
    assert len(captured) == 1
    record = json.loads(captured[0])
    assert record["event"] == "fetched"
    assert record["count"] == 42
    assert record["task"] == "ingest"
    assert record["level"] == "info"
