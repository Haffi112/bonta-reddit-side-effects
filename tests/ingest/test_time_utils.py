from botox_se.ingest.time_utils import iso_to_unix


def test_iso_to_unix_naive_treated_as_utc():
    # 2024-01-01T00:00:00 UTC = 1704067200
    assert iso_to_unix("2024-01-01T00:00:00") == 1704067200
    assert iso_to_unix("2024-01-01") == 1704067200


def test_iso_to_unix_tz_aware_converted_to_utc():
    # 2024-01-01T02:00:00+02:00 = 2024-01-01T00:00:00 UTC = 1704067200
    assert iso_to_unix("2024-01-01T02:00:00+02:00") == 1704067200
    # 2024-01-01T00:00:00-05:00 = 2024-01-01T05:00:00 UTC = 1704085200
    assert iso_to_unix("2024-01-01T00:00:00-05:00") == 1704085200
