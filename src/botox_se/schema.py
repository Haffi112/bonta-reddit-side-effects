"""The PostExtraction schema: one object per Reddit record.

Mirrors Appendix A of the manuscript. The value sets also live in
`config/extraction_enums.yaml`, which is the copy we publish; `tests/test_schema.py`
asserts the two never drift apart.

Two models rather than one. `ExtractedFields` is what the language model is asked
to return, and is the only thing rendered into the structured-output JSON Schema.
`PostExtraction` adds provenance that we attach ourselves, because a model that is
asked to report its own fingerprint can get it wrong, and a model asked to report
the prompt version has been handed a way to lie about which prompt produced it.

Three invariants are enforced after the call returns:

1.  `a_self_use` is a gate. Where it is false the record describes a friend's
    experience or quotes a news story, so every other field is reset to its null
    or empty value. A reset rather than a rejection: the record still counts in
    the denominator, it just contributes no clinical content, and no hallucinated
    side effect can survive a false gate.
2.  An `unknown` indication category forces a null subtype. A *mismatched* subtype
    (medical category, cosmetic subtype) is not coerced; it raises, because it is
    a model failure worth counting rather than a value worth guessing.
3.  No reported side effect forces a null side-effect sentiment. Sentiment about
    a side effect that was never reported has no referent.

`raw_span` holds verbatim user text. It exists so a human adjudicator can check the
model against the source during validation, and it must never reach a public
artifact: the study publishes no user text, verbatim or paraphrased. Use
`public_dict()` for anything released.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_REPO_ROOT = Path(__file__).resolve().parents[2]
ENUMS_PATH = _REPO_ROOT / "config" / "extraction_enums.yaml"


@lru_cache(maxsize=1)
def load_enum_config(path: Path | None = None) -> dict[str, Any]:
    """The published value sets, as written in config/extraction_enums.yaml."""
    loaded: dict[str, Any] = yaml.safe_load((path or ENUMS_PATH).read_text())
    return loaded


class Product(StrEnum):
    BOTOX = "botox"
    DYSPORT = "dysport"
    XEOMIN = "xeomin"
    JEUVEAU = "jeuveau"
    DAXXIFY = "daxxify"
    UNSPECIFIED_BONT_A = "unspecified_bont_a"


class IndicationCategory(StrEnum):
    MEDICAL = "medical"
    COSMETIC = "cosmetic"
    UNKNOWN = "unknown"


class IndicationSubtype(StrEnum):
    # medical
    CHRONIC_MIGRAINE = "chronic_migraine"
    CERVICAL_DYSTONIA = "cervical_dystonia"
    BLEPHAROSPASM = "blepharospasm"
    SPASTICITY = "spasticity"
    HYPERHIDROSIS = "hyperhidrosis"
    OVERACTIVE_BLADDER = "overactive_bladder"
    STRABISMUS = "strabismus"
    TMJ_MEDICAL = "tmj_medical"
    OTHER_MEDICAL = "other_medical"
    # cosmetic
    GLABELLAR = "glabellar"
    FOREHEAD = "forehead"
    CROWS_FEET = "crows_feet"
    MASSETER_JAWLINE = "masseter_jawline"
    NECK_PLATYSMA = "neck_platysma"
    LIP_FLIP = "lip_flip"
    BABY_BOTOX = "baby_botox"
    OTHER_COSMETIC = "other_cosmetic"


MEDICAL_SUBTYPES: frozenset[IndicationSubtype] = frozenset(
    {
        IndicationSubtype.CHRONIC_MIGRAINE,
        IndicationSubtype.CERVICAL_DYSTONIA,
        IndicationSubtype.BLEPHAROSPASM,
        IndicationSubtype.SPASTICITY,
        IndicationSubtype.HYPERHIDROSIS,
        IndicationSubtype.OVERACTIVE_BLADDER,
        IndicationSubtype.STRABISMUS,
        IndicationSubtype.TMJ_MEDICAL,
        IndicationSubtype.OTHER_MEDICAL,
    }
)
COSMETIC_SUBTYPES: frozenset[IndicationSubtype] = frozenset(
    {
        IndicationSubtype.GLABELLAR,
        IndicationSubtype.FOREHEAD,
        IndicationSubtype.CROWS_FEET,
        IndicationSubtype.MASSETER_JAWLINE,
        IndicationSubtype.NECK_PLATYSMA,
        IndicationSubtype.LIP_FLIP,
        IndicationSubtype.BABY_BOTOX,
        IndicationSubtype.OTHER_COSMETIC,
    }
)
SUBTYPES_BY_CATEGORY: dict[IndicationCategory, frozenset[IndicationSubtype]] = {
    IndicationCategory.MEDICAL: MEDICAL_SUBTYPES,
    IndicationCategory.COSMETIC: COSMETIC_SUBTYPES,
}


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class SideEffectSentiment(StrEnum):
    DISTRESSED = "distressed"
    ACCEPTING = "accepting"
    DISMISSIVE = "dismissive"


class ProviderType(StrEnum):
    MEDSPA = "medspa"
    DERMATOLOGIST = "dermatologist"
    PLASTIC_SURGEON = "plastic_surgeon"
    NEUROLOGIST = "neurologist"
    OTHER_MD = "other_md"
    NURSE_INJECTOR = "nurse_injector"
    DIY = "diy"
    UNSPECIFIED = "unspecified"


class FirstTimeOrRepeat(StrEnum):
    FIRST_TIME = "first_time"
    REPEAT = "repeat"


class TimeToOnset(StrEnum):
    UNDER_24H = "<24h"
    D1_3 = "1_3d"
    D4_7 = "4_7d"
    W1_2 = "1_2w"
    W2_4 = "2_4w"
    OVER_4W = ">4w"
    UNSPECIFIED = "unspecified"


class DurationStatus(StrEnum):
    ONGOING = "ongoing"
    RESOLVED_WEEKS = "resolved_weeks"
    RESOLVED_MONTHS = "resolved_months"
    PERMANENT = "permanent"
    UNSPECIFIED = "unspecified"


class DiscontinuationIntent(StrEnum):
    WILL_CONTINUE = "will_continue"
    CONSIDERING_STOPPING = "considering_stopping"
    STOPPED = "stopped"
    UNCLEAR = "unclear"


class AgeRange(StrEnum):
    UNDER_20 = "<20"
    A20_29 = "20-29"
    A30_39 = "30-39"
    A40_49 = "40-49"
    A50_59 = "50-59"
    A60_69 = "60-69"
    A70_PLUS = "70+"


class Sex(StrEnum):
    FEMALE = "female"
    MALE = "male"
    NONBINARY = "nonbinary"


class SourceField(StrEnum):
    """Which text the extractor was actually shown for a record."""

    TITLE = "title"
    SELFTEXT = "selftext"
    COMMENT_BODY = "comment_body"
    PARENT_TITLE = "parent_title"
    PARENT_SELFTEXT = "parent_selftext"


class RecordKind(StrEnum):
    """How the record entered the corpus.

    `comment_thread` is a reply kept because its parent submission matched, not
    because it names a product. It is roughly 56% of matched comments, and it is
    unanswerable without the parent, so those records carry parent text.
    """

    SUBMISSION = "submission"
    COMMENT_TERM = "comment_term"
    COMMENT_THREAD = "comment_thread"


class SideEffectPT(BaseModel):
    """One reported side effect.

    `meddra_pt` is the model's candidate Preferred Term, not a coded term. It is
    canonicalised against the MedDRA dictionary in the mapping step and kept here
    so the first guess can be scored against the adjudicated term.
    """

    model_config = ConfigDict(extra="forbid")

    raw_span: str = Field(description="Verbatim text naming the side effect. Never published.")
    meddra_pt: str = Field(description="Candidate MedDRA Preferred Term.")


class ExtractedFields(BaseModel):
    """Exactly what the model is asked to return. Nothing else."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    # Gate
    a_self_use: bool

    # Product
    b_products_mentioned: list[Product] = Field(default_factory=list)

    # Indication
    c_indication_category: IndicationCategory
    c_indication_subtype: IndicationSubtype | None = None

    # Primary outcome
    d_side_effect_pts: list[SideEffectPT] = Field(default_factory=list)

    # Sentiment
    e1_post_overall_sentiment: Sentiment
    e2_sentiment_about_botox: Sentiment
    f_sentiment_about_side_effect: SideEffectSentiment | None = None

    # Procedure and dose
    g_treatment_area_freetext: str | None = None
    h_dose_units: int | None = Field(default=None, ge=0, le=2000)

    # Clinical context
    i_provider_type: ProviderType | None = None
    j_first_time_or_repeat: FirstTimeOrRepeat | None = None
    j_prior_treatment_count: int | None = Field(default=None, ge=0, le=200)
    k_time_to_onset: TimeToOnset | None = None
    l_duration_status: DurationStatus | None = None
    m_sought_medical_attention: bool | None = None
    n_discontinuation_intent: DiscontinuationIntent | None = None

    # Demographics, recorded only where volunteered
    o_age_range: AgeRange | None = None
    o_sex: Sex | None = None

    @model_validator(mode="after")
    def _enforce_invariants(self) -> ExtractedFields:
        if not self.a_self_use:
            return self._gated_off()

        cat, sub = self.c_indication_category, self.c_indication_subtype
        if cat is IndicationCategory.UNKNOWN:
            self.c_indication_subtype = None
        elif sub is not None and sub not in SUBTYPES_BY_CATEGORY[cat]:
            raise ValueError(
                f"indication subtype {sub.value!r} does not belong to category {cat.value!r}"
            )

        if not self.d_side_effect_pts:
            self.f_sentiment_about_side_effect = None

        return self

    def _gated_off(self) -> ExtractedFields:
        """Reset every clinical field. The author did not receive the drug."""
        self.b_products_mentioned = []
        self.c_indication_category = IndicationCategory.UNKNOWN
        self.c_indication_subtype = None
        self.d_side_effect_pts = []
        self.f_sentiment_about_side_effect = None
        self.g_treatment_area_freetext = None
        self.h_dose_units = None
        self.i_provider_type = None
        self.j_first_time_or_repeat = None
        self.j_prior_treatment_count = None
        self.k_time_to_onset = None
        self.l_duration_status = None
        self.m_sought_medical_attention = None
        self.n_discontinuation_intent = None
        self.o_age_range = None
        self.o_sex = None
        return self


class PostExtraction(ExtractedFields):
    """An extracted record plus the provenance we attach ourselves."""

    record_id: str = Field(description="Reddit fullname, e.g. t3_abc123 or t1_def456.")
    record_kind: RecordKind
    fields_seen: list[SourceField] = Field(
        description="Which text the extractor was shown. Coverage is reported per field."
    )
    prompt_version: str = Field(description="Semantic version of the prompt file used.")
    model_fingerprint: str = Field(
        description="Snapshot id returned by the API, so a silent model update is detectable."
    )

    def public_dict(self) -> dict[str, Any]:
        """Release form. Drops every verbatim span of user text.

        The study publishes no user text, so `raw_span` is stripped and only the
        candidate Preferred Term survives. `g_treatment_area_freetext` is verbatim
        too and goes with it.
        """
        d = self.model_dump(mode="json")
        d["d_side_effect_pts"] = [{"meddra_pt": s["meddra_pt"]} for s in d["d_side_effect_pts"]]
        d.pop("g_treatment_area_freetext", None)
        return d


def structured_output_schema() -> dict[str, Any]:
    """JSON Schema for the model-facing fields, in OpenAI strict-mode form.

    Strict mode wants every property listed as required and `additionalProperties:
    false` on every object, so an omitted field is a protocol error rather than a
    silent null. Optionality is carried by the `null` branch of each type instead.
    """
    schema = ExtractedFields.model_json_schema()
    _strictify(schema)
    for defn in schema.get("$defs", {}).values():
        _strictify(defn)
    return schema


def _strictify(node: dict[str, Any]) -> None:
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        node["required"] = list(node.get("properties", {}))
    for prop in node.get("properties", {}).values():
        if isinstance(prop, dict):
            _strictify(prop)
