from __future__ import annotations

from eval_app.fields import (
    canonical_decision_key,
    required_decision_keys,
    review_rows,
    value_signature,
)

from .helpers import extraction


def test_side_effect_children_are_independent_review_fields():
    rows = review_rows(extraction())
    keys = [row.key for row in rows]
    assert len(keys) == 21
    assert "d_side_effect_pts" in keys
    assert "d_side_effect_pts[0].raw_span" in keys
    assert "d_side_effect_pts[0].meddra_pt" in keys
    raw_span = next(row for row in rows if row.key.endswith("raw_span"))
    assert raw_span.is_verbatim
    assert canonical_decision_key(raw_span.key) == "d_side_effect_pts[*].raw_span"
    assert required_decision_keys(extraction()) == tuple(keys)


def test_empty_side_effect_list_has_only_parent_decision():
    keys = required_decision_keys(extraction(effect=False))
    assert len(keys) == 19
    assert not any("raw_span" in key for key in keys)


def test_value_signatures_match_equal_values_and_normalize_product_order():
    assert value_signature("a_self_use", True) == value_signature("a_self_use", True)
    assert value_signature("a_self_use", True) != value_signature("a_self_use", False)
    assert value_signature("b_products_mentioned", ["botox", "dysport"]) == value_signature(
        "b_products_mentioned", ["dysport", "botox"]
    )
