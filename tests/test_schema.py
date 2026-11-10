"""The schema is a contract with three readers: the model, the analyst, and the
paper's Appendix A. These tests hold all three to the same values.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from botox_se.schema import (
    COSMETIC_SUBTYPES,
    MEDICAL_SUBTYPES,
    AgeRange,
    DiscontinuationIntent,
    DurationStatus,
    ExtractedFields,
    FirstTimeOrRepeat,
    IndicationCategory,
    IndicationSubtype,
    PostExtraction,
    Product,
    ProviderType,
    RecordKind,
    Sentiment,
    Sex,
    SideEffectPT,
    SideEffectSentiment,
    SourceField,
    TimeToOnset,
    load_enum_config,
    structured_output_schema,
)

CFG = load_enum_config()


def _values(enum_cls) -> set[str]:
    return {m.value for m in enum_cls}


@pytest.mark.parametrize(
    ("key", "enum_cls"),
    [
        ("products", Product),
        ("indication_category", IndicationCategory),
        ("sentiment", Sentiment),
        ("side_effect_sentiment", SideEffectSentiment),
        ("provider_type", ProviderType),
        ("first_time_or_repeat", FirstTimeOrRepeat),
        ("time_to_onset", TimeToOnset),
        ("duration_status", DurationStatus),
        ("discontinuation_intent", DiscontinuationIntent),
        ("age_range", AgeRange),
        ("sex", Sex),
        ("source_fields", SourceField),
    ],
)
def test_yaml_matches_python_enum(key, enum_cls):
    """config/extraction_enums.yaml is what we publish; it must not drift."""
    assert set(CFG[key]) == _values(enum_cls)


def test_yaml_subtypes_match_category_partition():
    assert set(CFG["indication_subtype"]["medical"]) == {s.value for s in MEDICAL_SUBTYPES}
    assert set(CFG["indication_subtype"]["cosmetic"]) == {s.value for s in COSMETIC_SUBTYPES}


def test_subtype_sets_partition_the_enum():
    """Every subtype belongs to exactly one category. No orphans, no overlap."""
    assert MEDICAL_SUBTYPES.isdisjoint(COSMETIC_SUBTYPES)
    assert set(IndicationSubtype) == MEDICAL_SUBTYPES | COSMETIC_SUBTYPES


def test_yaml_bounds_match_field_constraints():
    lo, hi = CFG["bounds"]["dose_units"]
    ExtractedFields(**_minimal(h_dose_units=hi))
    with pytest.raises(ValidationError):
        ExtractedFields(**_minimal(h_dose_units=hi + 1))
    with pytest.raises(ValidationError):
        ExtractedFields(**_minimal(h_dose_units=lo - 1))


def _minimal(**over):
    base = dict(
        a_self_use=True,
        c_indication_category=IndicationCategory.COSMETIC,
        e1_post_overall_sentiment=Sentiment.NEUTRAL,
        e2_sentiment_about_botox=Sentiment.NEUTRAL,
    )
    base.update(over)
    return base


# --- invariant 1: the self-use gate ---


def test_gate_off_resets_every_clinical_field():
    """A post about someone else's Botox must contribute no clinical content."""
    rec = ExtractedFields(
        **_minimal(
            a_self_use=False,
            b_products_mentioned=[Product.BOTOX],
            c_indication_category=IndicationCategory.MEDICAL,
            c_indication_subtype=IndicationSubtype.CHRONIC_MIGRAINE,
            d_side_effect_pts=[SideEffectPT(raw_span="droopy eyelid", meddra_pt="Ptosis")],
            f_sentiment_about_side_effect=SideEffectSentiment.DISTRESSED,
            g_treatment_area_freetext="left masseter",
            h_dose_units=50,
            i_provider_type=ProviderType.MEDSPA,
            o_age_range=AgeRange.A30_39,
            o_sex=Sex.FEMALE,
        )
    )
    assert rec.b_products_mentioned == []
    assert rec.d_side_effect_pts == []
    assert rec.c_indication_category is IndicationCategory.UNKNOWN
    assert rec.c_indication_subtype is None
    assert rec.h_dose_units is None
    assert rec.o_sex is None


def test_gate_off_keeps_the_record_and_its_sentiment():
    """Reset, not rejection: the record still counts in the denominator."""
    rec = ExtractedFields(
        **_minimal(a_self_use=False, e1_post_overall_sentiment=Sentiment.NEGATIVE)
    )
    assert rec.a_self_use is False
    assert rec.e1_post_overall_sentiment is Sentiment.NEGATIVE


# --- invariant 2: category governs subtype ---


def test_unknown_category_nulls_the_subtype():
    rec = ExtractedFields(
        **_minimal(
            c_indication_category=IndicationCategory.UNKNOWN,
            c_indication_subtype=IndicationSubtype.GLABELLAR,
        )
    )
    assert rec.c_indication_subtype is None


def test_mismatched_subtype_raises_rather_than_coercing():
    """A medical category with a cosmetic subtype is a model failure worth counting."""
    with pytest.raises(ValidationError, match="does not belong to category"):
        ExtractedFields(
            **_minimal(
                c_indication_category=IndicationCategory.MEDICAL,
                c_indication_subtype=IndicationSubtype.CROWS_FEET,
            )
        )


def test_matching_subtype_survives():
    rec = ExtractedFields(
        **_minimal(
            c_indication_category=IndicationCategory.MEDICAL,
            c_indication_subtype=IndicationSubtype.CHRONIC_MIGRAINE,
        )
    )
    assert rec.c_indication_subtype is IndicationSubtype.CHRONIC_MIGRAINE


# --- invariant 3: side-effect sentiment needs a side effect ---


def test_no_side_effect_nulls_its_sentiment():
    rec = ExtractedFields(
        **_minimal(
            d_side_effect_pts=[], f_sentiment_about_side_effect=SideEffectSentiment.ACCEPTING
        )
    )
    assert rec.f_sentiment_about_side_effect is None


def test_side_effect_keeps_its_sentiment():
    rec = ExtractedFields(
        **_minimal(
            d_side_effect_pts=[SideEffectPT(raw_span="headache", meddra_pt="Headache")],
            f_sentiment_about_side_effect=SideEffectSentiment.ACCEPTING,
        )
    )
    assert rec.f_sentiment_about_side_effect is SideEffectSentiment.ACCEPTING


# --- release form ---


def _full_record(**over) -> PostExtraction:
    base = _minimal(
        d_side_effect_pts=[
            SideEffectPT(raw_span="my eyelid drooped for weeks", meddra_pt="Ptosis")
        ],
        f_sentiment_about_side_effect=SideEffectSentiment.DISTRESSED,
        g_treatment_area_freetext="between the eyebrows",
        record_id="t1_def456",
        record_kind=RecordKind.COMMENT_THREAD,
        fields_seen=[
            SourceField.PARENT_TITLE,
            SourceField.PARENT_SELFTEXT,
            SourceField.COMMENT_BODY,
        ],
        prompt_version="1.0.0",
        model_fingerprint="gpt-x-2026-01-01",
    )
    base.update(over)
    return PostExtraction(**base)


def test_public_dict_carries_no_user_text():
    """The study publishes no user text, verbatim or paraphrased."""
    pub = _full_record().public_dict()
    blob = repr(pub)
    assert "my eyelid drooped for weeks" not in blob
    assert "between the eyebrows" not in blob
    assert "g_treatment_area_freetext" not in pub
    assert pub["d_side_effect_pts"] == [{"meddra_pt": "Ptosis"}]


def test_public_dict_keeps_the_analysable_fields():
    pub = _full_record().public_dict()
    assert pub["record_id"] == "t1_def456"
    assert pub["record_kind"] == "comment_thread"
    assert pub["model_fingerprint"] == "gpt-x-2026-01-01"


def test_thread_comments_record_that_they_saw_the_parent():
    """A thread-only reply is unanswerable for a_self_use without its parent."""
    rec = _full_record()
    assert rec.record_kind is RecordKind.COMMENT_THREAD
    assert SourceField.PARENT_TITLE in rec.fields_seen


# --- structured output ---


def test_structured_schema_excludes_provenance():
    """Provenance is attached, never requested: a model must not name its own prompt."""
    props = structured_output_schema()["properties"]
    for attached in ("prompt_version", "model_fingerprint", "record_id", "fields_seen"):
        assert attached not in props


def test_structured_schema_is_strict():
    schema = structured_output_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    for defn in schema["$defs"].values():
        if defn.get("type") == "object":
            assert defn["additionalProperties"] is False
