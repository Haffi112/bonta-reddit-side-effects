from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class ReviewRow:
    key: str
    canonical_key: str
    label: str
    description: str
    value: Any
    display_value: str
    value_signature: str
    is_verbatim: bool = False
    nested: bool = False


FIELD_SPECS = (
    FieldSpec("a_self_use", "Personal use", "The author explicitly reports personally receiving BoNT-A."),
    FieldSpec("b_products_mentioned", "Products mentioned", "Products tied to the author's own exposure."),
    FieldSpec("c_indication_category", "Indication category", "Medical, cosmetic, or unknown reason for treatment."),
    FieldSpec("c_indication_subtype", "Indication subtype", "The specific indication within the chosen category."),
    FieldSpec("d_side_effect_pts", "Side-effect list", "The complete set of unwanted outcomes the author reports after treatment."),
    FieldSpec("e1_post_overall_sentiment", "Overall post sentiment", "The overall emotional direction of the submission."),
    FieldSpec("e2_sentiment_about_botox", "Attitude toward BoNT-A", "The author's expressed attitude toward their treatment or BoNT-A."),
    FieldSpec("f_sentiment_about_side_effect", "Attitude toward side effect", "How the author responds emotionally to the reported side effect(s)."),
    FieldSpec("g_treatment_area_freetext", "Treatment area", "The explicitly stated anatomical treatment area, kept as source wording."),
    FieldSpec("h_dose_units", "Dose (units)", "Explicitly stated number of treatment units."),
    FieldSpec("i_provider_type", "Provider type", "The explicitly stated type of injector or provider."),
    FieldSpec("j_first_time_or_repeat", "First or repeat treatment", "Whether this was the first treatment or a repeat treatment."),
    FieldSpec("j_prior_treatment_count", "Prior treatment count", "Explicitly stated number of earlier treatments."),
    FieldSpec("k_time_to_onset", "Time to onset", "Time from treatment to the unwanted outcome."),
    FieldSpec("l_duration_status", "Duration status", "Whether the side effect is ongoing, resolved, or permanent."),
    FieldSpec("m_sought_medical_attention", "Sought medical attention", "Whether the author explicitly sought medical care for the outcome."),
    FieldSpec("n_discontinuation_intent", "Discontinuation intent", "Whether the author plans to continue or stop treatment."),
    FieldSpec("o_age_range", "Age range", "Age volunteered by the author, grouped into the study bands."),
    FieldSpec("o_sex", "Sex", "Sex explicitly volunteered by the author."),
)

FIELD_SPEC_BY_KEY = {field.key: field for field in FIELD_SPECS}
PROVENANCE_KEYS = {
    "record_id",
    "record_kind",
    "fields_seen",
    "prompt_version",
    "model_fingerprint",
}
DECISION_VALUES = {"correct", "incorrect"}
_SIDE_EFFECT_CHILD = re.compile(r"^d_side_effect_pts\[\d+\]\.(raw_span|meddra_pt)$")


def model_fields(extraction: dict[str, Any]) -> dict[str, Any]:
    """Keep only the exact model-facing schema, in its published field order."""
    return {spec.key: extraction.get(spec.key) for spec in FIELD_SPECS}


def _display(value: Any) -> str:
    if value is None:
        return "Not reported (null)"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, list):
        if not value:
            return "None (empty list)"
        return ", ".join(str(item) for item in value)
    return str(value)


def value_signature(key: str, value: Any) -> str:
    """Stable, non-revealing signature used to mirror equal field decisions."""
    normalised = value
    if key == "b_products_mentioned" and isinstance(value, list):
        normalised = sorted(value)
    elif key == "d_side_effect_pts" and isinstance(value, list):
        normalised = sorted(
            value,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    serialised = json.dumps(
        normalised, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialised.encode()).hexdigest()


def review_rows(extraction: dict[str, Any]) -> list[ReviewRow]:
    values = model_fields(extraction)
    rows: list[ReviewRow] = []
    for spec in FIELD_SPECS:
        value = values[spec.key]
        if spec.key == "d_side_effect_pts":
            effects = value if isinstance(value, list) else []
            count = len(effects)
            rows.append(
                ReviewRow(
                    key=spec.key,
                    canonical_key=spec.key,
                    label=spec.label,
                    description=spec.description,
                    value=value,
                    display_value=(
                        "None (empty list)"
                        if count == 0
                        else f"{count} side effect{'s' if count != 1 else ''} extracted"
                    ),
                    value_signature=value_signature(spec.key, value),
                )
            )
            for index, effect in enumerate(effects):
                if not isinstance(effect, dict):
                    continue
                for child_key, child_label, description, verbatim in (
                    (
                        "raw_span",
                        "Verbatim side-effect span",
                        "The shortest exact source wording that names this side effect.",
                        True,
                    ),
                    (
                        "meddra_pt",
                        "Candidate MedDRA term",
                        "The candidate Preferred Term accurately represents this side effect.",
                        False,
                    ),
                ):
                    key = f"d_side_effect_pts[{index}].{child_key}"
                    child_value = effect.get(child_key)
                    rows.append(
                        ReviewRow(
                            key=key,
                            canonical_key=f"d_side_effect_pts[*].{child_key}",
                            label=f"Effect {index + 1}: {child_label}",
                            description=description,
                            value=child_value,
                            display_value=_display(child_value),
                            value_signature=value_signature(key, child_value),
                            is_verbatim=verbatim,
                            nested=True,
                        )
                    )
            continue

        rows.append(
            ReviewRow(
                key=spec.key,
                canonical_key=spec.key,
                label=spec.label,
                description=spec.description,
                value=value,
                display_value=_display(value),
                value_signature=value_signature(spec.key, value),
                is_verbatim=spec.key == "g_treatment_area_freetext" and value is not None,
            )
        )
    return rows


def required_decision_keys(extraction: dict[str, Any]) -> tuple[str, ...]:
    return tuple(row.key for row in review_rows(extraction))


def canonical_decision_key(key: str) -> str:
    match = _SIDE_EFFECT_CHILD.fullmatch(key)
    return f"d_side_effect_pts[*].{match.group(1)}" if match else key


def field_label(key: str) -> str:
    if key == "d_side_effect_pts[*].raw_span":
        return "Verbatim side-effect span"
    if key == "d_side_effect_pts[*].meddra_pt":
        return "Candidate MedDRA term"
    return FIELD_SPEC_BY_KEY.get(key, FieldSpec(key, key, "")).label


def validate_decisions(
    extraction: dict[str, Any], submitted: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    required = required_decision_keys(extraction)
    cleaned = {key: submitted.get(key, "") for key in required}
    missing = [key for key, value in cleaned.items() if value not in DECISION_VALUES]
    return cleaned, missing
