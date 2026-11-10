"""Time conversion helpers for ingestion scripts."""

from datetime import UTC, datetime


def iso_to_unix(iso: str) -> int:
    """Parse an ISO 8601 timestamp and return Unix seconds (UTC).

    Naive input is assumed UTC; tz-aware input is converted to UTC.
    """
    dt = datetime.fromisoformat(iso)
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return int(dt.timestamp())
