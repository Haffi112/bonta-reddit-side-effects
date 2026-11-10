"""Focused tests for the medical versus cosmetic comparison."""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "data_analysis"
    / "04_medical_cosmetic_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("medical_cosmetic_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def record(group: str, *side_effects: str):
    return analysis.ComparisonRecord(group, frozenset(side_effects))


def test_intervals_and_jensen_shannon_known_values():
    low, high = analysis.wilson_interval(50, 100)
    assert 0.40 < low < 0.41
    assert 0.59 < high < 0.60

    difference_low, difference_high = analysis.newcombe_difference_interval(
        50, 100, 50, 100
    )
    assert difference_low < 0 < difference_high
    assert math.isclose(abs(difference_low), difference_high)

    identical = analysis.jensen_shannon_divergence(
        np.asarray([2, 3]), np.asarray([2, 3])
    )
    disjoint = analysis.jensen_shannon_divergence(
        np.asarray([1, 0]), np.asarray([0, 1])
    )
    assert identical[0] == 0.0
    assert math.isclose(disjoint[0], 1.0)


def test_profile_permutation_detects_group_difference():
    records = [record("medical", "term a") for _ in range(20)] + [
        record("cosmetic", "term b") for _ in range(20)
    ]

    results = analysis.profile_permutation_tests(
        records, min_count=1, permutations=199, seed=7, batch_size=25
    )

    common = next(
        row for row in results if row["test"] == "record_label_permutation_jsd_common_terms"
    )
    assert common["statistic"] == 1.0
    assert common["p_value"] <= 0.01
    assert common["medical_records"] == 20
    assert common["cosmetic_records"] == 20


def test_term_comparisons_use_reporting_record_denominators_and_adjust_p_values():
    records = (
        [record("medical", "dysphagia") for _ in range(8)]
        + [record("medical", "headache") for _ in range(2)]
        + [record("cosmetic", "eyelid ptosis") for _ in range(8)]
        + [record("cosmetic", "headache") for _ in range(2)]
        + [record("medical"), record("cosmetic")]
    )

    rows = analysis.side_effect_comparison_rows(records, min_count=2)
    dysphagia = next(row for row in rows if row["candidate_pt"] == "dysphagia")
    headache = next(row for row in rows if row["candidate_pt"] == "headache")

    assert dysphagia["medical_denominator"] == 10
    assert dysphagia["cosmetic_denominator"] == 10
    assert dysphagia["higher_group"] == "medical"
    assert dysphagia["bh_q_value"] is not None
    assert headache["risk_difference_percentage_points"] == 0.0


def test_private_loader_discards_ids_spans_and_non_self_use_content():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE stage_results "
        "(stage TEXT, record_id TEXT, extraction_json TEXT, metadata_json TEXT, created_at TEXT)"
    )
    extraction = {
        "a_self_use": True,
        "c_indication_category": "medical",
        "d_side_effect_pts": [
            {"raw_span": "private verbatim material", "meddra_pt": "Headache"}
        ],
    }
    non_user = {
        "a_self_use": False,
        "c_indication_category": "unknown",
        "d_side_effect_pts": [],
    }
    connection.executemany(
        "INSERT INTO stage_results VALUES (?, ?, ?, '{}', '')",
        [
            ("stage1", "private-record-id", json.dumps(extraction)),
            ("stage1", "another-private-id", json.dumps(non_user)),
        ],
    )

    records = analysis.load_records(connection, "stage1")

    assert records == [record("medical", "headache")]
    assert "private verbatim material" not in repr(records)
    assert "private-record-id" not in repr(records)
