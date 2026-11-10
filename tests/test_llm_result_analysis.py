"""Focused tests for aggregate LLM result analysis."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "data_analysis" / "03_llm_result_analysis.py"
SPEC = importlib.util.spec_from_file_location("llm_result_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def extraction(**overrides):
    base = {
        "a_self_use": True,
        "b_products_mentioned": ["botox"],
        "c_indication_category": "medical",
        "c_indication_subtype": "chronic_migraine",
        "d_side_effect_pts": [{"raw_span": "private text", "meddra_pt": "Headache"}],
        "e1_post_overall_sentiment": "negative",
        "e2_sentiment_about_botox": "mixed",
        "f_sentiment_about_side_effect": "distressed",
        "g_treatment_area_freetext": "private treatment area",
        "h_dose_units": 20,
        "i_provider_type": "neurologist",
        "j_first_time_or_repeat": "repeat",
        "j_prior_treatment_count": 2,
        "k_time_to_onset": "1_3d",
        "l_duration_status": "ongoing",
        "m_sought_medical_attention": True,
        "n_discontinuation_intent": "unclear",
        "o_age_range": "30-39",
        "o_sex": "female",
    }
    base.update(overrides)
    return base


def test_analysis_view_removes_private_text_and_normalizes_sets():
    view = analysis.analysis_view(extraction())

    assert view["d_side_effect_pts"] == frozenset({"headache"})
    assert view["b_products_mentioned"] == frozenset({"botox"})
    assert view["g_treatment_area_freetext"] != "private treatment area"
    assert len(view["g_treatment_area_freetext"]) == 64
    assert "private text" not in repr(view)


def test_wilson_interval_and_kappa_known_values():
    low, high = analysis.wilson_interval(50, 100)
    assert 0.40 < low < 0.41
    assert 0.59 < high < 0.60
    assert analysis.cohen_kappa([True, False], [True, False]) == 1.0
    assert analysis.cohen_kappa([True, False], [False, True]) == -1.0


def test_applicability_prevents_structural_null_agreement():
    self_use = analysis.analysis_view(extraction())
    non_user = analysis.analysis_view(
        extraction(
            a_self_use=False,
            b_products_mentioned=[],
            c_indication_category="unknown",
            c_indication_subtype=None,
            d_side_effect_pts=[],
            f_sentiment_about_side_effect=None,
            g_treatment_area_freetext=None,
            h_dose_units=None,
        )
    )
    product_field = next(field for field in analysis.FIELDS if field.key == "b_products_mentioned")
    sentiment_field = next(
        field for field in analysis.FIELDS if field.key == "e1_post_overall_sentiment"
    )

    assert not analysis.is_applicable(product_field, (self_use, non_user))
    assert analysis.is_applicable(sentiment_field, (self_use, non_user))


def test_fleiss_kappa_and_jaccard():
    assert analysis.fleiss_kappa([("a", "a", "a"), ("b", "b", "b")]) == 1.0
    assert analysis.jaccard(frozenset(), frozenset()) == 1.0
    assert math.isclose(
        analysis.jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})),
        1 / 3,
    )


def test_stage1_descriptive_aggregation_is_aggregate_only():
    aggregate = analysis.StageAggregate()
    aggregate.observe(analysis.analysis_view(extraction()))

    rows = analysis.stage1_descriptive_rows(aggregate)
    numeric = analysis.numeric_summary_rows(aggregate)
    coverage = analysis.stage1_coverage_rows(aggregate)

    dose = next(row for row in numeric if row["field"] == "h_dose_units")
    treatment_area = next(
        row
        for row in rows
        if row["field"] == "g_treatment_area_freetext" and row["value"] == "reported"
    )
    assert dose["median"] == 20.0
    assert treatment_area["count"] == 1
    assert coverage
    assert "private text" not in repr(rows)
    assert "private treatment area" not in repr(rows)
