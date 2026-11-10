from __future__ import annotations

from typing import Any


def extraction(*, effect: bool = True) -> dict[str, Any]:
    return {
        "a_self_use": True,
        "b_products_mentioned": ["botox"],
        "c_indication_category": "medical",
        "c_indication_subtype": "chronic_migraine",
        "d_side_effect_pts": (
            [{"raw_span": "my eyelid drooped", "meddra_pt": "Ptosis"}] if effect else []
        ),
        "e1_post_overall_sentiment": "negative",
        "e2_sentiment_about_botox": "mixed",
        "f_sentiment_about_side_effect": "distressed" if effect else None,
        "g_treatment_area_freetext": "forehead",
        "h_dose_units": 30,
        "i_provider_type": "neurologist",
        "j_first_time_or_repeat": "repeat",
        "j_prior_treatment_count": 2,
        "k_time_to_onset": "1_3d",
        "l_duration_status": "ongoing",
        "m_sought_medical_attention": False,
        "n_discontinuation_intent": "unclear",
        "o_age_range": "30-39",
        "o_sex": "female",
        "record_id": "t3_test",
        "record_kind": "submission",
        "fields_seen": ["title", "selftext"],
        "prompt_version": "test",
        "model_fingerprint": "test/model",
    }

