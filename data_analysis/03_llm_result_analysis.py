"""Analyze agreement and outputs from the three-stage LLM extraction experiment.

The extraction checkpoint contains private model output, including verbatim Reddit
spans. This script reads that checkpoint but writes aggregate statistics only. It
never exports record IDs, raw spans, treatment-area text, or per-record results.

Outputs:

* ``03_llm_result_analysis.json``: complete aggregate analysis
* ``03_llm_result_analysis.md``: concise human-readable report
* ``03_pairwise_agreement.csv`` and ``03_three_way_agreement.csv``
* ``03_key_outcomes.csv``, ``03_operational_summary.csv``, and
  ``03_top_candidate_side_effects.csv``
* full-corpus DeepSeek no-reasoning descriptive tables prefixed with
  ``03_stage1_``
* publication figures in PNG and vector PDF under ``data_analysis/figures``

Example:

    uv run python data_analysis/03_llm_result_analysis.py
    uv run python data_analysis/03_llm_result_analysis.py --bootstrap 1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ANALYSIS_DIR / "02_llm_extraction_results"
DEFAULT_FIGURES_DIR = ANALYSIS_DIR / "figures"
DEFAULT_DATABASE = DEFAULT_RESULTS_DIR / "private_extraction.sqlite3"
DEFAULT_SEED = 20260715

Scope = Literal["all", "self_use", "side_effect"]
FieldKind = Literal["scalar", "set"]


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    short_label: str
    color: str


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind
    scope: Scope
    derived: bool = False


@dataclass(frozen=True)
class DescriptiveFieldSpec:
    key: str
    label: str
    scope: Scope
    order: tuple[str, ...]


STAGES = (
    StageSpec("stage1", "DeepSeek V4 Flash, no reasoning", "DeepSeek\nno reasoning", "#24557A"),
    StageSpec("stage2", "DeepSeek V4 Flash, high reasoning", "DeepSeek\nhigh reasoning", "#C45A2A"),
    StageSpec("stage3", "GPT-5.6 Luna, low reasoning", "GPT-5.6 Luna\nlow reasoning", "#278578"),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}

PAIRS = (
    ("stage1_vs_stage2", "stage1", "stage2", "DeepSeek: none vs high"),
    ("stage1_vs_stage3", "stage1", "stage3", "DeepSeek none vs Luna"),
    ("stage2_vs_stage3", "stage2", "stage3", "DeepSeek high vs Luna"),
)

# Stage 4 (the full-corpus OpenAI Batch Luna run) is activated in main() only
# when its checkpoint rows exist, so historical three-stage runs reproduce.
STAGE4_SPEC = StageSpec(
    "stage4", "GPT-5.6 Luna, low reasoning, batch", "GPT-5.6 Luna\nbatch", "#6B4F9E"
)
STAGE4_PAIRS = (
    ("stage1_vs_stage4", "stage1", "stage4", "DeepSeek none vs Luna batch"),
    ("stage2_vs_stage4", "stage2", "stage4", "DeepSeek high vs Luna batch"),
    ("stage3_vs_stage4", "stage3", "stage4", "Luna sync vs Luna batch"),
)

FIELDS = (
    FieldSpec("a_self_use", "Self-use", "scalar", "all"),
    FieldSpec("b_products_mentioned", "Products mentioned", "set", "self_use"),
    FieldSpec("c_indication_category", "Indication category", "scalar", "self_use"),
    FieldSpec("c_indication_subtype", "Indication subtype", "scalar", "self_use"),
    FieldSpec("side_effect_presence", "Any side effect", "scalar", "all", derived=True),
    FieldSpec("d_side_effect_pts", "Candidate side-effect PTs", "set", "self_use"),
    FieldSpec("e1_post_overall_sentiment", "Overall sentiment", "scalar", "all"),
    FieldSpec("e2_sentiment_about_botox", "Sentiment about BoNT-A", "scalar", "all"),
    FieldSpec("f_sentiment_about_side_effect", "Side-effect sentiment", "scalar", "side_effect"),
    FieldSpec("g_treatment_area_freetext", "Treatment area", "scalar", "self_use"),
    FieldSpec("h_dose_units", "Dose units", "scalar", "self_use"),
    FieldSpec("i_provider_type", "Provider type", "scalar", "self_use"),
    FieldSpec("j_first_time_or_repeat", "First or repeat treatment", "scalar", "self_use"),
    FieldSpec("j_prior_treatment_count", "Prior treatment count", "scalar", "self_use"),
    FieldSpec("k_time_to_onset", "Time to onset", "scalar", "self_use"),
    FieldSpec("l_duration_status", "Duration status", "scalar", "self_use"),
    FieldSpec("m_sought_medical_attention", "Sought medical attention", "scalar", "self_use"),
    FieldSpec("n_discontinuation_intent", "Discontinuation intent", "scalar", "self_use"),
    FieldSpec("o_age_range", "Age range", "scalar", "self_use"),
    FieldSpec("o_sex", "Sex", "scalar", "self_use"),
)
EXTRACTED_FIELDS = tuple(spec for spec in FIELDS if not spec.derived)

PRODUCT_ORDER = (
    "botox",
    "dysport",
    "xeomin",
    "jeuveau",
    "daxxify",
    "unspecified_bont_a",
)
INDICATION_ORDER = ("medical", "cosmetic", "unknown")
SENTIMENT_ORDER = ("positive", "mixed", "neutral", "negative")
SIDE_EFFECT_COUNT_ORDER = ("0", "1", "2", "3+")
SIDE_EFFECT_COUNT_FULL_ORDER = ("0", "1", "2", "3", "4", "5+")

DESCRIPTIVE_FIELDS = (
    DescriptiveFieldSpec("a_self_use", "Self-use", "all", ("true", "false")),
    DescriptiveFieldSpec(
        "c_indication_category",
        "Indication category",
        "self_use",
        ("medical", "cosmetic", "unknown"),
    ),
    DescriptiveFieldSpec(
        "c_indication_subtype",
        "Indication subtype",
        "self_use",
        (
            "chronic_migraine",
            "cervical_dystonia",
            "blepharospasm",
            "spasticity",
            "hyperhidrosis",
            "overactive_bladder",
            "strabismus",
            "tmj_medical",
            "other_medical",
            "glabellar",
            "forehead",
            "crows_feet",
            "masseter_jawline",
            "neck_platysma",
            "lip_flip",
            "baby_botox",
            "other_cosmetic",
            "not_reported",
        ),
    ),
    DescriptiveFieldSpec(
        "e1_post_overall_sentiment",
        "Overall post sentiment",
        "all",
        SENTIMENT_ORDER,
    ),
    DescriptiveFieldSpec(
        "e2_sentiment_about_botox",
        "Sentiment about BoNT-A",
        "all",
        SENTIMENT_ORDER,
    ),
    DescriptiveFieldSpec(
        "f_sentiment_about_side_effect",
        "Sentiment about side effect",
        "side_effect",
        ("distressed", "accepting", "dismissive", "not_reported"),
    ),
    DescriptiveFieldSpec(
        "i_provider_type",
        "Provider type",
        "self_use",
        (
            "medspa",
            "dermatologist",
            "plastic_surgeon",
            "neurologist",
            "other_md",
            "nurse_injector",
            "diy",
            "unspecified",
            "not_reported",
        ),
    ),
    DescriptiveFieldSpec(
        "j_first_time_or_repeat",
        "First or repeat treatment",
        "self_use",
        ("first_time", "repeat", "not_reported"),
    ),
    DescriptiveFieldSpec(
        "k_time_to_onset",
        "Time to onset",
        "self_use",
        ("<24h", "1_3d", "4_7d", "1_2w", "2_4w", ">4w", "unspecified", "not_reported"),
    ),
    DescriptiveFieldSpec(
        "l_duration_status",
        "Duration status",
        "self_use",
        (
            "ongoing",
            "resolved_weeks",
            "resolved_months",
            "permanent",
            "unspecified",
            "not_reported",
        ),
    ),
    DescriptiveFieldSpec(
        "m_sought_medical_attention",
        "Sought medical attention",
        "self_use",
        ("true", "false", "not_reported"),
    ),
    DescriptiveFieldSpec(
        "n_discontinuation_intent",
        "Discontinuation intent",
        "self_use",
        ("will_continue", "considering_stopping", "stopped", "unclear", "not_reported"),
    ),
    DescriptiveFieldSpec(
        "o_age_range",
        "Age range",
        "self_use",
        ("<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+", "not_reported"),
    ),
    DescriptiveFieldSpec(
        "o_sex",
        "Sex",
        "self_use",
        ("female", "male", "nonbinary", "not_reported"),
    ),
)

NUMERIC_FIELDS = (
    ("h_dose_units", "Dose units"),
    ("j_prior_treatment_count", "Prior treatment count"),
)

INFORMATIVE_COVERAGE_FIELDS = (
    ("c_indication_subtype", "Indication subtype", {"not_reported"}),
    ("g_treatment_area_freetext", "Treatment area", {"not_reported"}),
    ("h_dose_units", "Dose units", {"not_reported"}),
    ("i_provider_type", "Provider type", {"not_reported", "unspecified"}),
    ("j_first_time_or_repeat", "First or repeat treatment", {"not_reported"}),
    ("j_prior_treatment_count", "Prior treatment count", {"not_reported"}),
    ("k_time_to_onset", "Time to onset", {"not_reported", "unspecified"}),
    ("l_duration_status", "Duration status", {"not_reported", "unspecified"}),
    ("m_sought_medical_attention", "Sought medical attention", {"not_reported"}),
    ("n_discontinuation_intent", "Discontinuation intent", {"not_reported", "unclear"}),
    ("o_age_range", "Age range", {"not_reported"}),
    ("o_sex", "Sex", {"not_reported"}),
)


def category_key(value: Any) -> str:
    if value is None:
        return "not_reported"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def record_matches_scope(record: Mapping[str, Any], scope: Scope) -> bool:
    if scope == "all":
        return True
    if scope == "self_use":
        return bool(record["a_self_use"])
    return bool(record["side_effect_presence"])


@dataclass
class StageAggregate:
    n: int = 0
    self_use: int = 0
    side_effect_reports: int = 0
    side_effect_mentions: int = 0
    multiple_side_effect_reports: int = 0
    indication: Counter[str] = field(default_factory=Counter)
    sentiment: Counter[str] = field(default_factory=Counter)
    products: Counter[str] = field(default_factory=Counter)
    candidate_pts: Counter[str] = field(default_factory=Counter)
    product_records: int = 0
    descriptive_counts: dict[str, Counter[str]] = field(default_factory=dict)
    descriptive_denominators: Counter[str] = field(default_factory=Counter)
    numeric_values: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key, _label in NUMERIC_FIELDS}
    )
    numeric_denominators: Counter[str] = field(default_factory=Counter)
    field_informative_counts: Counter[str] = field(default_factory=Counter)
    side_effect_count_distribution: Counter[str] = field(default_factory=Counter)
    indication_self_use: Counter[str] = field(default_factory=Counter)
    indication_side_effect: Counter[str] = field(default_factory=Counter)

    def observe(self, record: Mapping[str, Any]) -> None:
        self.n += 1
        is_self_use = bool(record["a_self_use"])
        side_effects = set(record["d_side_effect_pts"])
        self.self_use += int(is_self_use)
        self.side_effect_reports += int(bool(side_effects))
        self.side_effect_mentions += len(side_effects)
        self.multiple_side_effect_reports += int(len(side_effects) >= 2)
        self.indication[str(record["c_indication_category"])] += 1
        self.sentiment[str(record["e2_sentiment_about_botox"])] += 1
        products = set(record["b_products_mentioned"])
        self.products.update(products)
        self.product_records += int(bool(products))
        self.candidate_pts.update(side_effects)

        for spec in DESCRIPTIVE_FIELDS:
            if not record_matches_scope(record, spec.scope):
                continue
            self.descriptive_denominators[spec.key] += 1
            counts = self.descriptive_counts.setdefault(spec.key, Counter())
            counts[category_key(record[spec.key])] += 1

        if is_self_use:
            side_effect_count = len(side_effects)
            side_effect_count_key = "5+" if side_effect_count >= 5 else str(side_effect_count)
            self.side_effect_count_distribution[side_effect_count_key] += 1
            indication = str(record["c_indication_category"])
            self.indication_self_use[indication] += 1
            if side_effects:
                self.indication_side_effect[indication] += 1
            for key, _label in NUMERIC_FIELDS:
                self.numeric_denominators[key] += 1
                value = record[key]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.numeric_values[key].append(float(value))
            for key, _label, excluded in INFORMATIVE_COVERAGE_FIELDS:
                value = record[key]
                value_key = category_key(value)
                if value_key not in excluded:
                    self.field_informative_counts[key] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "successful_records": self.n,
            "self_use_records": self.self_use,
            "self_use_prevalence": divide(self.self_use, self.n),
            "side_effect_report_records": self.side_effect_reports,
            "side_effect_report_prevalence": divide(self.side_effect_reports, self.n),
            "multiple_side_effect_report_records": self.multiple_side_effect_reports,
            "candidate_side_effect_mentions": self.side_effect_mentions,
            "mean_candidate_pts_per_side_effect_report": divide(
                self.side_effect_mentions, self.side_effect_reports
            ),
            "indication_counts": dict(self.indication),
            "sentiment_counts": dict(self.sentiment),
            "product_record_counts": dict(self.products),
            "candidate_pt_record_counts": dict(self.candidate_pts),
        }


@dataclass
class LoadedStage:
    records: dict[str, dict[str, Any]]
    aggregate: StageAggregate


def divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def normalize_text(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def treatment_area_digest(value: Any) -> str | None:
    """Retain equality information without retaining user-authored free text."""
    if value is None:
        return None
    normalized = normalize_text(value)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def analysis_view(extraction: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one private extraction to the normalized fields needed for analysis."""
    view: dict[str, Any] = {}
    for spec in EXTRACTED_FIELDS:
        value = extraction.get(spec.key)
        if spec.key == "b_products_mentioned":
            view[spec.key] = frozenset(normalize_text(item) for item in value or [])
        elif spec.key == "d_side_effect_pts":
            view[spec.key] = frozenset(
                normalize_text(item.get("meddra_pt"))
                for item in value or []
                if isinstance(item, Mapping) and item.get("meddra_pt")
            )
        elif spec.key == "g_treatment_area_freetext":
            view[spec.key] = treatment_area_digest(value)
        elif isinstance(value, str):
            view[spec.key] = normalize_text(value)
        else:
            view[spec.key] = value
    view["side_effect_presence"] = bool(view["d_side_effect_pts"])
    return view


def load_stage(
    connection: sqlite3.Connection,
    stage: str,
    retain_ids: set[str] | None = None,
) -> LoadedStage:
    total = int(
        connection.execute(
            "SELECT COUNT(*) FROM stage_results WHERE stage = ?", (stage,)
        ).fetchone()[0]
    )
    records: dict[str, dict[str, Any]] = {}
    aggregate = StageAggregate()
    rows = connection.execute(
        "SELECT record_id, extraction_json FROM stage_results WHERE stage = ?",
        (stage,),
    )
    for record_id, extraction_json in tqdm(
        rows, total=total, desc=f"load {stage}", unit="rec", unit_scale=True
    ):
        extraction = json.loads(extraction_json)
        view = analysis_view(extraction)
        aggregate.observe(view)
        if retain_ids is None or record_id in retain_ids:
            records[record_id] = view
    return LoadedStage(records, aggregate)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def cohen_kappa(left: Sequence[Hashable], right: Sequence[Hashable]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[value] / n) * (right_counts[value] / n)
        for value in left_counts.keys() | right_counts.keys()
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def bootstrap_kappa_interval(
    left: Sequence[Hashable],
    right: Sequence[Hashable],
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if resamples <= 0 or len(left) < 2:
        return None, None

    joint = Counter(zip(left, right, strict=True))
    category_index: dict[Hashable, int] = {}
    for value in (*left, *right):
        if value not in category_index:
            category_index[value] = len(category_index)
    pairs = list(joint)
    counts = np.asarray([joint[pair] for pair in pairs], dtype=float)
    probabilities = counts / counts.sum()
    left_cell = np.asarray([category_index[pair[0]] for pair in pairs])
    right_cell = np.asarray([category_index[pair[1]] for pair in pairs])
    diagonal = left_cell == right_cell
    n = len(left)
    category_count = len(category_index)
    estimates: list[float] = []

    for start in range(0, resamples, 50):
        batch_size = min(50, resamples - start)
        sampled = rng.multinomial(n, probabilities, size=batch_size)
        observed = sampled[:, diagonal].sum(axis=1) / n
        left_margins = np.zeros((batch_size, category_count), dtype=float)
        right_margins = np.zeros((batch_size, category_count), dtype=float)
        for cell in range(len(pairs)):
            left_margins[:, left_cell[cell]] += sampled[:, cell]
            right_margins[:, right_cell[cell]] += sampled[:, cell]
        expected = (left_margins * right_margins).sum(axis=1) / (n * n)
        denominator = 1.0 - expected
        batch_estimates = np.divide(
            observed - expected,
            denominator,
            out=np.full(batch_size, np.nan),
            where=~np.isclose(denominator, 0.0),
        )
        batch_estimates[np.isclose(denominator, 0.0) & np.isclose(observed, 1.0)] = 1.0
        estimates.extend(batch_estimates[np.isfinite(batch_estimates)].tolist())

    if not estimates:
        return None, None
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean_interval(
    values: Sequence[float],
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if resamples <= 0 or len(values) < 2:
        return None, None
    unique, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
    probabilities = counts / counts.sum()
    sampled = rng.multinomial(len(values), probabilities, size=resamples)
    estimates = sampled @ unique / len(values)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return np.random.default_rng((seed + offset) % (2**63 - 1))


def is_applicable(spec: FieldSpec, records: Sequence[Mapping[str, Any]]) -> bool:
    if spec.scope == "all":
        return True
    if spec.scope == "self_use":
        return all(bool(record["a_self_use"]) for record in records)
    return all(bool(record["side_effect_presence"]) for record in records)


def agreement_metrics(
    spec: FieldSpec,
    left_values: Sequence[Hashable],
    right_values: Sequence[Hashable],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(left_values)
    matches = sum(a == b for a, b in zip(left_values, right_values, strict=True))
    exact = divide(matches, n)
    exact_low, exact_high = wilson_interval(matches, n)
    result: dict[str, Any] = {
        "n": n,
        "exact_agreement": exact,
        "exact_ci_low": exact_low,
        "exact_ci_high": exact_high,
    }
    if spec.kind == "set":
        similarities = [
            jaccard(a, b)
            for a, b in zip(left_values, right_values, strict=True)
            if isinstance(a, frozenset) and isinstance(b, frozenset)
        ]
        low, high = bootstrap_mean_interval(similarities, bootstrap, rng)
        result.update(
            secondary_metric="mean_jaccard",
            secondary_value=float(np.mean(similarities)) if similarities else None,
            secondary_ci_low=low,
            secondary_ci_high=high,
        )
    else:
        low, high = bootstrap_kappa_interval(left_values, right_values, bootstrap, rng)
        result.update(
            secondary_metric="cohen_kappa",
            secondary_value=cohen_kappa(left_values, right_values),
            secondary_ci_low=low,
            secondary_ci_high=high,
        )
    return result


def pairwise_analysis(
    comparison: str,
    left_key: str,
    right_key: str,
    stages: Mapping[str, LoadedStage],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left_records = stages[left_key].records
    right_records = stages[right_key].records
    ids = sorted(left_records.keys() & right_records.keys())
    rows: list[dict[str, Any]] = []

    for spec in FIELDS:
        all_pairs = [(left_records[rid], right_records[rid]) for rid in ids]
        for scope_name, pairs in (
            ("all", all_pairs),
            (
                "applicable",
                [pair for pair in all_pairs if is_applicable(spec, pair)],
            ),
        ):
            left_values = [pair[0][spec.key] for pair in pairs]
            right_values = [pair[1][spec.key] for pair in pairs]
            metrics = agreement_metrics(
                spec,
                left_values,
                right_values,
                bootstrap,
                stable_rng(seed, comparison, spec.key, scope_name),
            )
            rows.append(
                {
                    "comparison": comparison,
                    "left_stage": left_key,
                    "right_stage": right_key,
                    "field": spec.key,
                    "field_label": spec.label,
                    "field_kind": spec.kind,
                    "scope": scope_name,
                    "applicability_rule": spec.scope,
                    **metrics,
                }
            )

    full_matches = sum(
        all(
            left_records[rid][spec.key] == right_records[rid][spec.key] for spec in EXTRACTED_FIELDS
        )
        for rid in ids
    )
    summary = {
        "comparison": comparison,
        "left_stage": left_key,
        "right_stage": right_key,
        "paired_records": len(ids),
        "full_record_exact_agreement": divide(full_matches, len(ids)),
    }
    return rows, summary


def fleiss_kappa(rows: Sequence[Sequence[Hashable]]) -> float | None:
    if not rows:
        return None
    ratings_per_record = len(rows[0])
    if ratings_per_record < 2 or any(len(row) != ratings_per_record for row in rows):
        return None
    category_totals: Counter[Hashable] = Counter()
    record_agreements: list[float] = []
    for row in rows:
        counts = Counter(row)
        category_totals.update(counts)
        numerator = sum(count * count for count in counts.values()) - ratings_per_record
        record_agreements.append(numerator / (ratings_per_record * (ratings_per_record - 1)))
    observed = float(np.mean(record_agreements))
    total_ratings = len(rows) * ratings_per_record
    expected = sum((count / total_ratings) ** 2 for count in category_totals.values())
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def three_way_analysis(
    stages: Mapping[str, LoadedStage],
) -> tuple[list[dict[str, Any]], set[str]]:
    common_ids = set.intersection(*(set(stages[stage.key].records) for stage in STAGES))
    rows: list[dict[str, Any]] = []
    for spec in FIELDS:
        all_records = [
            tuple(stages[stage.key].records[rid] for stage in STAGES) for rid in sorted(common_ids)
        ]
        for scope_name, records in (
            ("all", all_records),
            (
                "applicable",
                [record_set for record_set in all_records if is_applicable(spec, record_set)],
            ),
        ):
            values = [tuple(record[spec.key] for record in record_set) for record_set in records]
            unanimous = sum(len(set(row)) == 1 for row in values)
            exact_low, exact_high = wilson_interval(unanimous, len(values))
            result: dict[str, Any] = {
                "field": spec.key,
                "field_label": spec.label,
                "field_kind": spec.kind,
                "scope": scope_name,
                "applicability_rule": spec.scope,
                "n": len(values),
                "unanimous_exact_agreement": divide(unanimous, len(values)),
                "exact_ci_low": exact_low,
                "exact_ci_high": exact_high,
            }
            if spec.kind == "set":
                pairwise_similarities = [
                    np.mean(
                        (
                            jaccard(row[0], row[1]),
                            jaccard(row[0], row[2]),
                            jaccard(row[1], row[2]),
                        )
                    )
                    for row in values
                ]
                result.update(
                    secondary_metric="mean_pairwise_jaccard",
                    secondary_value=(
                        float(np.mean(pairwise_similarities)) if pairwise_similarities else None
                    ),
                )
            else:
                result.update(
                    secondary_metric="fleiss_kappa",
                    secondary_value=fleiss_kappa(values),
                )
            rows.append(result)
    return rows, common_ids


def outcome_value(record: Mapping[str, Any], outcome: str) -> bool:
    if outcome == "self_use":
        return bool(record["a_self_use"])
    if outcome == "side_effect_report":
        return bool(record["side_effect_presence"])
    if outcome == "medical_indication":
        return bool(record["a_self_use"]) and record["c_indication_category"] == "medical"
    if outcome == "cosmetic_indication":
        return bool(record["a_self_use"]) and record["c_indication_category"] == "cosmetic"
    if outcome == "multiple_side_effects":
        return len(record["d_side_effect_pts"]) >= 2
    raise ValueError(f"unknown outcome: {outcome}")


def key_outcome_rows(
    stages: Mapping[str, LoadedStage], common_ids: set[str]
) -> list[dict[str, Any]]:
    outcomes = (
        ("self_use", "Self-use"),
        ("side_effect_report", "Side-effect report"),
        ("medical_indication", "Medical indication"),
        ("cosmetic_indication", "Cosmetic indication"),
        ("multiple_side_effects", "Multiple side effects"),
    )
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        records = [stages[stage.key].records[rid] for rid in sorted(common_ids)]
        for outcome, label in outcomes:
            count = sum(outcome_value(record, outcome) for record in records)
            low, high = wilson_interval(count, len(records))
            rows.append(
                {
                    "stage": stage.key,
                    "approach": stage.label,
                    "outcome": outcome,
                    "outcome_label": label,
                    "n": len(records),
                    "count": count,
                    "prevalence": divide(count, len(records)),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return rows


def failure_counts(connection: sqlite3.Connection, stage: str) -> dict[str, int]:
    return {
        str(kind): int(count)
        for kind, count in connection.execute(
            """
            SELECT failure_kind, COUNT(*) FROM stage_failures
            WHERE stage = ? GROUP BY failure_kind
            """,
            (stage,),
        )
    }


def stage_summary_from_db(connection: sqlite3.Connection, stage: str) -> dict[str, Any]:
    """Operational summary computed from the checkpoint when run_summary predates a stage."""
    completed, prompt, completion, cached, cost, verbatim_num, verbatim_den = connection.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM(json_extract(metadata_json,'$.usage.prompt_tokens')), 0),
               COALESCE(SUM(json_extract(metadata_json,'$.usage.completion_tokens')), 0),
               COALESCE(SUM(json_extract(metadata_json,'$.usage.prompt_tokens_details.cached_tokens')), 0),
               COALESCE(SUM(json_extract(metadata_json,'$.usage.cost')), 0.0),
               COALESCE(SUM(json_extract(metadata_json,'$.raw_span_metrics.verbatim_raw_spans')), 0),
               COALESCE(SUM(json_extract(metadata_json,'$.raw_span_metrics.total_raw_spans')), 0)
        FROM stage_results WHERE stage = ?
        """,
        (stage,),
    ).fetchone()
    terminal = connection.execute(
        "SELECT COUNT(*) FROM stage_failures WHERE stage = ? AND terminal = 1", (stage,)
    ).fetchone()[0]
    retryable = connection.execute(
        "SELECT COUNT(*) FROM stage_failures WHERE stage = ? AND terminal = 0", (stage,)
    ).fetchone()[0]
    return {
        "target_records": completed + terminal + retryable,
        "completed_records": completed,
        "terminal_failed_records": terminal,
        "retryable_failed_records": retryable,
        "reported_cost_usd": cost,
        "raw_span_verbatim_rate": (verbatim_num / verbatim_den) if verbatim_den else None,
        "cache_hit_rate": (cached / prompt) if prompt else None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": 0,
    }


def full_corpus_gate_agreement(
    connection: sqlite3.Connection, left: str, right: str
) -> dict[str, Any]:
    """Exact agreement and kappa for the two gates over every paired record."""
    rows = connection.execute(
        """
        SELECT json_extract(a.extraction_json,'$.a_self_use'),
               json_array_length(a.extraction_json,'$.d_side_effect_pts') > 0,
               json_extract(b.extraction_json,'$.a_self_use'),
               json_array_length(b.extraction_json,'$.d_side_effect_pts') > 0,
               COUNT(*)
        FROM stage_results a
        JOIN stage_results b ON b.record_id = a.record_id AND b.stage = ?
        WHERE a.stage = ?
        GROUP BY 1, 2, 3, 4
        """,
        (right, left),
    ).fetchall()

    def gate_stats(index_left: int, index_right: int) -> dict[str, float]:
        counts: Counter[tuple[int, int]] = Counter()
        for row in rows:
            counts[(int(row[index_left]), int(row[index_right]))] += int(row[4])
        n = sum(counts.values())
        agree = counts[(0, 0)] + counts[(1, 1)]
        p_left = (counts[(1, 0)] + counts[(1, 1)]) / n
        p_right = (counts[(0, 1)] + counts[(1, 1)]) / n
        expected = p_left * p_right + (1 - p_left) * (1 - p_right)
        observed = agree / n
        kappa = (observed - expected) / (1 - expected) if expected < 1 else math.nan
        return {"n": n, "exact_agreement": observed, "kappa": kappa}

    return {
        "left": left,
        "right": right,
        "a_self_use": gate_stats(0, 2),
        "side_effect_presence": gate_stats(1, 3),
    }


def operational_rows(
    run_summary: Mapping[str, Any], connection: sqlite3.Connection
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries = run_summary["stages"]
    for stage in STAGES:
        summary = summaries.get(stage.key) or stage_summary_from_db(connection, stage.key)
        target = int(summary["target_records"])
        completed = int(summary["completed_records"])
        terminal = int(summary["terminal_failed_records"])
        cost = float(summary.get("reported_cost_usd") or 0.0)
        rows.append(
            {
                "stage": stage.key,
                "approach": stage.label,
                "target_records": target,
                "successful_records": completed,
                "terminal_failures": terminal,
                "retryable_failures": int(summary["retryable_failed_records"]),
                "failure_kinds": failure_counts(connection, stage.key),
                "success_rate": divide(completed, target),
                "handled_rate": divide(completed + terminal, target),
                "raw_span_verbatim_rate": summary.get("raw_span_verbatim_rate"),
                "cache_hit_rate": summary.get("cache_hit_rate"),
                "cost_usd": cost,
                "cost_per_1000_successes_usd": divide(cost * 1000.0, completed),
                "prompt_tokens": int(summary.get("prompt_tokens") or 0),
                "completion_tokens": int(summary.get("completion_tokens") or 0),
                "reasoning_tokens": int(summary.get("reasoning_tokens") or 0),
            }
        )
    return rows


def stage1_descriptive_rows(aggregate: StageAggregate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in DESCRIPTIVE_FIELDS:
        counts = aggregate.descriptive_counts.get(spec.key, Counter())
        denominator = aggregate.descriptive_denominators[spec.key]
        values = list(spec.order) + sorted(set(counts) - set(spec.order))
        for value in values:
            count = counts[value]
            rows.append(
                {
                    "field": spec.key,
                    "field_label": spec.label,
                    "value": value,
                    "value_label": pretty_label(value),
                    "scope": spec.scope,
                    "denominator": denominator,
                    "count": count,
                    "percent": (100.0 * count / denominator if denominator else None),
                    "multi_label": False,
                    "note": "",
                }
            )

    self_use_denominator = aggregate.self_use
    product_values = list(PRODUCT_ORDER) + sorted(set(aggregate.products) - set(PRODUCT_ORDER))
    for value in product_values:
        count = aggregate.products[value]
        rows.append(
            {
                "field": "b_products_mentioned",
                "field_label": "Products mentioned",
                "value": value,
                "value_label": pretty_label(value),
                "scope": "self_use",
                "denominator": self_use_denominator,
                "count": count,
                "percent": divide(100.0 * count, self_use_denominator),
                "multi_label": True,
                "note": "Percentages may sum above 100 because products are multi-label.",
            }
        )
    rows.append(
        {
            "field": "b_products_mentioned",
            "field_label": "Products mentioned",
            "value": "not_reported",
            "value_label": "Not reported",
            "scope": "self_use",
            "denominator": self_use_denominator,
            "count": self_use_denominator - aggregate.product_records,
            "percent": divide(
                100.0 * (self_use_denominator - aggregate.product_records),
                self_use_denominator,
            ),
            "multi_label": True,
            "note": "No product extracted for the record.",
        }
    )

    for value in SIDE_EFFECT_COUNT_FULL_ORDER:
        count = aggregate.side_effect_count_distribution[value]
        rows.append(
            {
                "field": "d_side_effect_count",
                "field_label": "Distinct candidate side-effect count",
                "value": value,
                "value_label": value,
                "scope": "self_use",
                "denominator": self_use_denominator,
                "count": count,
                "percent": divide(100.0 * count, self_use_denominator),
                "multi_label": False,
                "note": "Counts distinct normalized candidate Preferred Terms per record.",
            }
        )

    availability_fields = (
        ("g_treatment_area_freetext", "Treatment area"),
        *NUMERIC_FIELDS,
    )
    for key, label in availability_fields:
        reported = aggregate.field_informative_counts[key]
        for value, count in (
            ("reported", reported),
            ("not_reported", self_use_denominator - reported),
        ):
            rows.append(
                {
                    "field": key,
                    "field_label": label,
                    "value": value,
                    "value_label": pretty_label(value),
                    "scope": "self_use",
                    "denominator": self_use_denominator,
                    "count": count,
                    "percent": divide(100.0 * count, self_use_denominator),
                    "multi_label": False,
                    "note": "Availability only; no free text is exported.",
                }
            )
    return rows


def numeric_summary_rows(aggregate: StageAggregate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label in NUMERIC_FIELDS:
        values = np.asarray(aggregate.numeric_values[key], dtype=float)
        denominator = aggregate.numeric_denominators[key]
        if values.size:
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            summary = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else None,
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        else:
            summary = {
                "mean": None,
                "standard_deviation": None,
                "median": None,
                "q1": None,
                "q3": None,
                "minimum": None,
                "maximum": None,
            }
        rows.append(
            {
                "field": key,
                "field_label": label,
                "self_use_records": denominator,
                "reported_records": int(values.size),
                "reporting_rate": divide(int(values.size), denominator),
                **summary,
            }
        )
    return rows


def stage1_coverage_rows(aggregate: StageAggregate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label, excluded in INFORMATIVE_COVERAGE_FIELDS:
        count = aggregate.field_informative_counts[key]
        rows.append(
            {
                "field": key,
                "field_label": label,
                "denominator": aggregate.self_use,
                "informative_records": count,
                "informative_percent": divide(100.0 * count, aggregate.self_use),
                "excluded_values": sorted(excluded),
            }
        )
    return rows


def stage1_indication_side_effect_rows(aggregate: StageAggregate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for indication in INDICATION_ORDER:
        denominator = aggregate.indication_self_use[indication]
        count = aggregate.indication_side_effect[indication]
        low, high = wilson_interval(count, denominator)
        rows.append(
            {
                "indication_category": indication,
                "self_use_records": denominator,
                "side_effect_report_records": count,
                "side_effect_report_prevalence": divide(count, denominator),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def stage1_candidate_side_effect_rows(
    aggregate: StageAggregate, limit: int = 50
) -> list[dict[str, Any]]:
    ranked = sorted(aggregate.candidate_pts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {
            "rank": rank,
            "candidate_pt": term,
            "records_with_candidate_pt": count,
            "side_effect_reporting_records": aggregate.side_effect_reports,
            "records_per_100_side_effect_reports": divide(
                100.0 * count, aggregate.side_effect_reports
            ),
        }
        for rank, (term, count) in enumerate(ranked, start=1)
    ]


def common_distribution(
    records: Sequence[Mapping[str, Any]], distribution: str
) -> tuple[list[str], list[float], int]:
    if distribution == "indication":
        eligible = [record for record in records if record["a_self_use"]]
        counts = Counter(str(record["c_indication_category"]) for record in eligible)
        categories = list(INDICATION_ORDER)
    elif distribution == "sentiment":
        eligible = list(records)
        counts = Counter(str(record["e2_sentiment_about_botox"]) for record in eligible)
        categories = list(SENTIMENT_ORDER)
    elif distribution == "products":
        eligible = [record for record in records if record["a_self_use"]]
        counts = Counter(
            product for record in eligible for product in set(record["b_products_mentioned"])
        )
        categories = list(PRODUCT_ORDER)
    elif distribution == "side_effect_count":
        eligible = [record for record in records if record["a_self_use"]]
        counts = Counter(
            "3+" if len(record["d_side_effect_pts"]) >= 3 else str(len(record["d_side_effect_pts"]))
            for record in eligible
        )
        categories = list(SIDE_EFFECT_COUNT_ORDER)
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    denominator = len(eligible)
    values = [
        100.0 * counts[category] / denominator if denominator else 0.0 for category in categories
    ]
    return categories, values, denominator


def top_candidate_side_effect_rows(
    stages: Mapping[str, LoadedStage], common_ids: set[str], limit: int = 15
) -> list[dict[str, Any]]:
    stage_counts: dict[str, Counter[str]] = {}
    denominators: dict[str, int] = {}
    combined: Counter[str] = Counter()
    for stage in STAGES:
        records = [stages[stage.key].records[rid] for rid in sorted(common_ids)]
        counts: Counter[str] = Counter()
        denominator = 0
        for record in records:
            terms = set(record["d_side_effect_pts"])
            denominator += int(bool(terms))
            counts.update(terms)
        stage_counts[stage.key] = counts
        denominators[stage.key] = denominator
        combined.update(counts)
    top_terms = [
        term
        for term, _count in sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
    rows: list[dict[str, Any]] = []
    for rank, term in enumerate(top_terms, start=1):
        for stage in STAGES:
            count = stage_counts[stage.key][term]
            rows.append(
                {
                    "rank": rank,
                    "candidate_pt": term,
                    "stage": stage.key,
                    "approach": stage.label,
                    "side_effect_reporting_records": denominators[stage.key],
                    "records_with_candidate_pt": count,
                    "records_per_100_side_effect_reports": (
                        100.0 * count / denominators[stage.key] if denominators[stage.key] else None
                    ),
                }
            )
    return rows


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(figures_dir / f"{stem}.png", facecolor="white")
    fig.savefig(figures_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure_field_agreement(pairwise_rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> None:
    applicable = {
        (str(row["field"]), str(row["comparison"])): row
        for row in pairwise_rows
        if row["scope"] == "applicable"
    }
    comparison_keys = [pair[0] for pair in PAIRS]
    comparison_labels = [pair[3] for pair in PAIRS]
    field_labels = [spec.label + (" *" if spec.kind == "set" else "") for spec in FIELDS]
    exact = np.asarray(
        [
            [
                applicable[(spec.key, comparison)]["exact_agreement"]
                for comparison in comparison_keys
            ]
            for spec in FIELDS
        ],
        dtype=float,
    )
    secondary = np.asarray(
        [
            [
                applicable[(spec.key, comparison)]["secondary_value"]
                for comparison in comparison_keys
            ]
            for spec in FIELDS
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 9.2))
    sns.heatmap(
        exact,
        ax=axes[0],
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Agreement", "shrink": 0.65},
        xticklabels=comparison_labels,
        yticklabels=field_labels,
    )
    sns.heatmap(
        secondary,
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Agreement statistic", "shrink": 0.65},
        xticklabels=comparison_labels,
        yticklabels=False,
    )
    axes[0].set_title("Exact agreement", loc="left", fontweight="bold")
    axes[1].set_title("Chance-corrected / set agreement", loc="left", fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0].set_yticklabels(field_labels, rotation=0)
    fig.suptitle(
        "Pairwise field agreement on applicable records",
        x=0.08,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.08,
        -0.015,
        "Fields gated by self-use are compared only when both approaches mark self-use. "
        "Side-effect sentiment requires both to report a side effect. * Mean Jaccard; all "
        "other cells use Cohen's kappa.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_01_field_agreement")


def figure_key_outcomes(
    outcome_rows: Sequence[Mapping[str, Any]], common_n: int, figures_dir: Path
) -> None:
    outcomes = (
        "self_use",
        "side_effect_report",
        "medical_indication",
        "cosmetic_indication",
        "multiple_side_effects",
    )
    labels = {str(row["outcome"]): str(row["outcome_label"]) for row in outcome_rows}
    lookup = {(str(row["stage"]), str(row["outcome"])): row for row in outcome_rows}
    x = np.arange(len(outcomes), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(STAGES))
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    for stage, offset in zip(STAGES, offsets, strict=True):
        values = np.asarray(
            [100.0 * float(lookup[(stage.key, outcome)]["prevalence"]) for outcome in outcomes]
        )
        low = np.asarray(
            [100.0 * float(lookup[(stage.key, outcome)]["ci_low"]) for outcome in outcomes]
        )
        high = np.asarray(
            [100.0 * float(lookup[(stage.key, outcome)]["ci_high"]) for outcome in outcomes]
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack((values - low, high - values)),
            fmt="o",
            ms=6,
            capsize=3,
            elinewidth=1.2,
            color=stage.color,
            label=stage.label,
        )
    ax.set_xticks(x, [labels[outcome] for outcome in outcomes])
    ax.set_ylabel("Records (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Key extracted outcomes on the common three-model sample (n = {common_n:,})",
        loc="left",
        fontweight="bold",
    )
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_02_key_outcome_prevalence")


def grouped_distribution_panel(
    ax: Axes,
    stages: Mapping[str, LoadedStage],
    common_ids: set[str],
    distribution: str,
    title: str,
    ylabel: str,
) -> None:
    width = 0.24
    stage_distributions: list[tuple[StageSpec, list[str], list[float], int]] = []
    for stage in STAGES:
        records = [stages[stage.key].records[rid] for rid in sorted(common_ids)]
        categories, values, denominator = common_distribution(records, distribution)
        stage_distributions.append((stage, categories, values, denominator))
    denominators = [item[3] for item in stage_distributions]
    for index, (stage, categories, values, _denominator) in enumerate(stage_distributions):
        x = np.arange(len(categories)) + (index - 1) * width
        ax.bar(x, values, width=width, color=stage.color, label=stage.label)
    denominator_text = (
        f"denominator: n = {denominators[0]:,} for each approach"
        if len(set(denominators)) == 1
        else "denominators (legend order): " + " / ".join(f"{value:,}" for value in denominators)
    )
    ax.text(
        0.99,
        0.97,
        denominator_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    ax.set_xticks(np.arange(len(categories)), [pretty_label(category) for category in categories])
    ax.tick_params(axis="x", rotation=25, labelsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)


def figure_distributions(
    stages: Mapping[str, LoadedStage], common_ids: set[str], figures_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    grouped_distribution_panel(
        axes[0, 0], stages, common_ids, "indication", "Indication category", "% of self-use records"
    )
    grouped_distribution_panel(
        axes[0, 1], stages, common_ids, "sentiment", "Sentiment about BoNT-A", "% of records"
    )
    grouped_distribution_panel(
        axes[1, 0],
        stages,
        common_ids,
        "products",
        "Products mentioned (multi-label)",
        "% of self-use records",
    )
    grouped_distribution_panel(
        axes[1, 1],
        stages,
        common_ids,
        "side_effect_count",
        "Number of distinct candidate side effects",
        "% of self-use records",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.015), fontsize=8
    )
    fig.suptitle(
        "Distribution of extracted fields on matched records",
        x=0.07,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, figures_dir, "03_03_extracted_distributions")


def figure_top_side_effects(rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> None:
    terms = list(dict.fromkeys(str(row["candidate_pt"]) for row in rows))
    lookup = {(str(row["candidate_pt"]), str(row["stage"])): row for row in rows}
    y = np.arange(len(terms), dtype=float)
    height = 0.23
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    for index, stage in enumerate(STAGES):
        values = [
            float(lookup[(term, stage.key)]["records_per_100_side_effect_reports"] or 0.0)
            for term in terms
        ]
        ax.barh(
            y + (index - 1) * height,
            values,
            height=height,
            color=stage.color,
            label=stage.label,
        )
    ax.set_yticks(y, [pretty_label(term) for term in terms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Records with candidate PT per 100 side-effect-reporting records")
    ax.set_title(
        "Most frequent model-generated candidate side-effect terms",
        loc="left",
        fontweight="bold",
    )
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        -0.015,
        "Candidate Preferred Terms are model outputs and have not yet been validated against "
        "the licensed MedDRA dictionary.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_04_top_candidate_side_effects")


def figure_operational(rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> None:
    lookup = {str(row["stage"]): row for row in rows}
    labels = [stage.short_label for stage in STAGES]
    colors = [stage.color for stage in STAGES]
    metrics = (
        ("success_rate", "Successful extractions", "% of target", 100.0),
        ("raw_span_verbatim_rate", "Verbatim raw-span compliance", "% of spans", 100.0),
        ("cache_hit_rate", "Prompt cache utilization", "% of prompt tokens", 100.0),
        ("cost_per_1000_successes_usd", "Inference cost", "USD per 1,000 successes", 1.0),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (metric, title, ylabel, multiplier) in zip(axes, metrics, strict=True):
        values = [float(lookup[stage.key].get(metric) or 0.0) * multiplier for stage in STAGES]
        bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7.5)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="x", visible=False)
        padding = max(values) * 0.03 if max(values) else 0.05
        for bar, value in zip(bars, values, strict=True):
            label = f"{value:.1f}%" if multiplier == 100.0 else f"${value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + padding,
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_ylim(0, max(values) * 1.16 if max(values) else 1)
    fig.suptitle(
        "Operational comparison of extraction approaches",
        x=0.03,
        y=1.03,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_05_operational_comparison")


def plot_percentage_bars(
    ax: Axes,
    labels: Sequence[str],
    counts: Sequence[int],
    denominator: int,
    title: str,
    ylabel: str = "Records (%)",
    horizontal: bool = False,
) -> None:
    values = [100.0 * count / denominator if denominator else 0.0 for count in counts]
    colors = [
        "#9AA5AD"
        if label in {"not_reported", "unspecified", "unknown", "unclear"}
        else STAGES[0].color
        for label in labels
    ]
    display_labels = [pretty_label(label) for label in labels]
    if horizontal:
        positions = np.arange(len(labels))
        ax.barh(positions, values, color=colors)
        ax.set_yticks(positions, display_labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.grid(axis="y", visible=False)
        for y, value, count in zip(positions, values, counts, strict=True):
            ax.annotate(
                f"{count:,}",
                (value, y),
                textcoords="offset points",
                xytext=(3, 0),
                va="center",
                fontsize=6,
                color="#333333",
            )
        ax.margins(x=0.12)
    else:
        positions = np.arange(len(labels))
        ax.bar(positions, values, color=colors)
        ax.set_xticks(positions, display_labels, rotation=28, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        for x, value, count in zip(positions, values, counts, strict=True):
            ax.annotate(
                f"{count:,}",
                (x, value),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=6,
                color="#333333",
            )
        ax.margins(y=0.14)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.99,
        0.97,
        f"n = {denominator:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )


def stage1_counter_values(
    aggregate: StageAggregate, field_key: str, order: Sequence[str]
) -> tuple[list[str], list[int], int]:
    counts = aggregate.descriptive_counts[field_key]
    labels = [value for value in order if counts[value] > 0]
    labels.extend(sorted(value for value in counts if value not in order and counts[value] > 0))
    return (
        labels,
        [counts[value] for value in labels],
        aggregate.descriptive_denominators[field_key],
    )


def figure_stage1_overview(
    aggregate: StageAggregate,
    figures_dir: Path,
    stem: str = "03_06_stage1_full_corpus_overview",
    title_label: str = "DeepSeek V4 Flash without reasoning",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    outcome_labels = ["self_use", "side_effect_report", "multiple_side_effects"]
    outcome_counts = [
        aggregate.self_use,
        aggregate.side_effect_reports,
        aggregate.multiple_side_effect_reports,
    ]
    plot_percentage_bars(
        axes[0, 0],
        outcome_labels,
        outcome_counts,
        aggregate.n,
        "Nested record classifications",
        "% of all submissions",
    )

    indication_counts = [aggregate.indication_self_use[value] for value in INDICATION_ORDER]
    plot_percentage_bars(
        axes[0, 1],
        INDICATION_ORDER,
        indication_counts,
        aggregate.self_use,
        "Indication category",
        "% of self-use records",
    )

    sentiment_labels, sentiment_counts, sentiment_denominator = stage1_counter_values(
        aggregate, "f_sentiment_about_side_effect", ("distressed", "accepting", "dismissive")
    )
    plot_percentage_bars(
        axes[1, 0],
        sentiment_labels,
        sentiment_counts,
        sentiment_denominator,
        "Attitude toward the side effect",
        "% of side-effect reports",
    )

    side_effect_counts = [
        aggregate.side_effect_count_distribution[value] for value in SIDE_EFFECT_COUNT_FULL_ORDER
    ]
    plot_percentage_bars(
        axes[1, 1],
        SIDE_EFFECT_COUNT_FULL_ORDER,
        side_effect_counts,
        aggregate.self_use,
        "Distinct candidate side effects per record",
        "% of self-use records",
    )
    fig.suptitle(
        f"{title_label}: full-corpus overview",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def figure_stage1_clinical_context(
    aggregate: StageAggregate,
    figures_dir: Path,
    stem: str = "03_07_stage1_clinical_context",
    title_label: str = "DeepSeek V4 Flash without reasoning",
) -> None:
    panels = (
        ("c_indication_subtype", "Indication subtype"),
        ("i_provider_type", "Provider type"),
        ("j_first_time_or_repeat", "First or repeat treatment"),
        ("k_time_to_onset", "Time to onset"),
        ("l_duration_status", "Duration status"),
        ("n_discontinuation_intent", "Discontinuation intent"),
    )
    specs = {spec.key: spec for spec in DESCRIPTIVE_FIELDS}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        spec = specs[key]
        labels, counts, denominator = stage1_counter_values(aggregate, key, spec.order)
        if key == "c_indication_subtype":
            ranked = sorted(zip(labels, counts, strict=True), key=lambda item: (-item[1], item[0]))
            labels = [item[0] for item in ranked]
            counts = [item[1] for item in ranked]
        plot_percentage_bars(
            ax,
            labels,
            counts,
            denominator,
            title,
            "% of self-use records",
            horizontal=True,
        )
    fig.suptitle(
        f"{title_label}: clinical context",
        x=0.04,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def figure_stage1_sentiment_demographics(
    aggregate: StageAggregate,
    figures_dir: Path,
    stem: str = "03_08_stage1_sentiment_demographics",
    title_label: str = "DeepSeek V4 Flash without reasoning",
) -> None:
    panels = (
        ("e1_post_overall_sentiment", "Overall post sentiment"),
        ("e2_sentiment_about_botox", "Sentiment about BoNT-A"),
        ("f_sentiment_about_side_effect", "Side-effect sentiment"),
        ("m_sought_medical_attention", "Sought medical attention"),
        ("o_age_range", "Age range"),
        ("o_sex", "Sex"),
    )
    specs = {spec.key: spec for spec in DESCRIPTIVE_FIELDS}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        spec = specs[key]
        labels, counts, denominator = stage1_counter_values(aggregate, key, spec.order)
        scope_label = (
            "% of all submissions"
            if spec.scope == "all"
            else "% of side-effect reports"
            if spec.scope == "side_effect"
            else "% of self-use records"
        )
        plot_percentage_bars(ax, labels, counts, denominator, title, scope_label)
    fig.suptitle(
        f"{title_label}: sentiment and demographics",
        x=0.04,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def figure_stage1_side_effects_coverage(
    candidate_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str = "03_09_stage1_side_effects_and_coverage",
    title_label: str = "DeepSeek V4 Flash without reasoning",
) -> None:
    top = list(candidate_rows[:20])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.8))

    terms = [str(row["candidate_pt"]) for row in top]
    term_values = [float(row["records_per_100_side_effect_reports"] or 0.0) for row in top]
    y_terms = np.arange(len(terms))
    axes[0].barh(y_terms, term_values, color=STAGES[0].color)
    axes[0].set_yticks(y_terms, [pretty_label(term) for term in terms], fontsize=7.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Records per 100 side-effect reports")
    axes[0].set_title("Most frequent candidate side-effect terms", loc="left", fontweight="bold")
    axes[0].grid(axis="y", visible=False)

    coverage = sorted(
        coverage_rows,
        key=lambda row: float(row["informative_percent"] or 0.0),
        reverse=True,
    )
    coverage_labels = [str(row["field_label"]) for row in coverage]
    coverage_values = [float(row["informative_percent"] or 0.0) for row in coverage]
    y_coverage = np.arange(len(coverage))
    axes[1].barh(y_coverage, coverage_values, color="#C45A2A")
    axes[1].set_yticks(y_coverage, coverage_labels, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Self-use records with informative value (%)")
    axes[1].set_title("Informative field coverage", loc="left", fontweight="bold")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(
        f"{title_label}: side effects and field coverage",
        x=0.04,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.04,
        -0.01,
        "Candidate Preferred Terms are unvalidated model outputs. Coverage excludes null, "
        "unknown, unspecified, and unclear values as applicable.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def pretty_label(value: str) -> str:
    replacements = {
        "unspecified_bont_a": "Unspecified BoNT-A",
        "not_reported": "Not reported",
        "self_use": "Self-use",
        "side_effect_report": "Any side effect",
        "multiple_side_effects": "Multiple side effects",
        "1_3d": "1-3 d",
        "4_7d": "4-7 d",
        "1_2w": "1-2 wk",
        "2_4w": "2-4 wk",
    }
    if value in replacements:
        return replacements[value]
    return value.replace("_", " ").capitalize()


def json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def find_row(
    rows: Sequence[Mapping[str, Any]],
    **criteria: str,
) -> Mapping[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def write_markdown_report(
    path: Path,
    stages: Mapping[str, LoadedStage],
    pairwise_rows: Sequence[Mapping[str, Any]],
    pair_summaries: Sequence[Mapping[str, Any]],
    common_n: int,
    operational: Sequence[Mapping[str, Any]],
    stage1_numeric: Sequence[Mapping[str, Any]],
    stage1_indication_side_effects: Sequence[Mapping[str, Any]],
    stage1_candidate_side_effects: Sequence[Mapping[str, Any]],
    stage1_coverage: Sequence[Mapping[str, Any]],
    primary_key: str = "stage1",
    primary_label: str = "DeepSeek V4 Flash, no reasoning",
    gate_full: Mapping[str, Any] | None = None,
) -> None:
    lines = [
        "# LLM Extraction Result Analysis",
        "",
        "This report contains aggregate model outputs only. Agreement measures consistency, "
        "not validity against a human gold standard.",
        "",
        "## Extraction yield",
        "",
        "| Approach | Successful | Self-use | Side-effect reports |",
        "|---|---:|---:|---:|",
    ]
    for stage in STAGES:
        aggregate = stages[stage.key].aggregate
        lines.append(
            f"| {stage.label} | {aggregate.n:,} | {aggregate.self_use:,} "
            f"({100 * aggregate.self_use / aggregate.n:.1f}%) | "
            f"{aggregate.side_effect_reports:,} "
            f"({100 * aggregate.side_effect_reports / aggregate.n:.1f}%) |"
        )

    self_use_rows = [
        find_row(pairwise_rows, comparison=pair[0], field="a_self_use", scope="all")
        for pair in PAIRS
    ]
    side_effect_rows = [
        find_row(
            pairwise_rows,
            comparison=pair[0],
            field="side_effect_presence",
            scope="all",
        )
        for pair in PAIRS
    ]
    candidate_pt_rows = [
        find_row(
            pairwise_rows,
            comparison=pair[0],
            field="d_side_effect_pts",
            scope="applicable",
        )
        for pair in PAIRS
    ]
    full_record_values = [
        float(summary["full_record_exact_agreement"]) for summary in pair_summaries
    ]
    self_use_kappas = [float(row["secondary_value"]) for row in self_use_rows]
    side_effect_kappas = [float(row["secondary_value"]) for row in side_effect_rows]
    candidate_pt_jaccards = [float(row["secondary_value"]) for row in candidate_pt_rows]
    candidate_pt_exact = [float(row["exact_agreement"]) for row in candidate_pt_rows]

    lines.extend(
        [
            "",
            "## Principal findings",
            "",
            f"- Self-use classification was consistent across approaches: pairwise Cohen's "
            f"kappa ranged from {min(self_use_kappas):.3f} to {max(self_use_kappas):.3f}.",
            f"- Presence of any side effect agreed more often than the specific candidate "
            f"terms: presence kappa ranged from {min(side_effect_kappas):.3f} to "
            f"{max(side_effect_kappas):.3f}.",
            f"- Among records both approaches classified as self-use, candidate side-effect "
            f"set exact agreement ranged from {min(candidate_pt_exact):.3f} to "
            f"{max(candidate_pt_exact):.3f}, with mean Jaccard from "
            f"{min(candidate_pt_jaccards):.3f} to {max(candidate_pt_jaccards):.3f}.",
            f"- Exact agreement across every extracted field was substantially lower, ranging "
            f"from {min(full_record_values):.3f} to {max(full_record_values):.3f}.",
        ]
    )

    stage1 = stages[primary_key].aggregate
    indication_lookup = {
        str(row["indication_category"]): row for row in stage1_indication_side_effects
    }
    dose = next(row for row in stage1_numeric if row["field"] == "h_dose_units")
    prior = next(row for row in stage1_numeric if row["field"] == "j_prior_treatment_count")
    top_terms = ", ".join(
        f"{pretty_label(str(row['candidate_pt']))} ({int(row['records_with_candidate_pt']):,})"
        for row in stage1_candidate_side_effects[:10]
    )
    best_coverage = sorted(
        stage1_coverage,
        key=lambda row: float(row["informative_percent"] or 0.0),
        reverse=True,
    )[:3]
    coverage_text = ", ".join(
        f"{row['field_label']} ({float(row['informative_percent']):.1f}%)" for row in best_coverage
    )

    lines.extend(
        [
            "",
            f"## {primary_label}: full-corpus description",
            "",
            f"Of {stage1.n:,} submissions, {stage1.self_use:,} ({100 * stage1.self_use / stage1.n:.1f}%) "
            f"were classified as self-use and {stage1.side_effect_reports:,} "
            f"({100 * stage1.side_effect_reports / stage1.n:.1f}%) contained at least one "
            "candidate side effect.",
            "",
            "| Indication | Self-use records | Side-effect reports | Prevalence (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for indication in INDICATION_ORDER:
        row = indication_lookup[indication]
        prevalence = float(row["side_effect_report_prevalence"] or 0.0)
        low = float(row["ci_low"] or 0.0)
        high = float(row["ci_high"] or 0.0)
        lines.append(
            f"| {pretty_label(indication)} | {int(row['self_use_records']):,} | "
            f"{int(row['side_effect_report_records']):,} | "
            f"{100 * prevalence:.1f}% ({100 * low:.1f}-{100 * high:.1f}%) |"
        )
    lines.extend(
        [
            "",
            f"Dose was explicitly extracted for {int(dose['reported_records']):,} self-use "
            f"records; median {float(dose['median'] or 0):.0f} units "
            f"(IQR {float(dose['q1'] or 0):.0f}-{float(dose['q3'] or 0):.0f}). Prior treatment "
            f"count was available for {int(prior['reported_records']):,} records; median "
            f"{float(prior['median'] or 0):.0f} (IQR {float(prior['q1'] or 0):.0f}-"
            f"{float(prior['q3'] or 0):.0f}).",
            "",
            f"The ten most frequent unvalidated candidate terms were: {top_terms}.",
            "",
            f"The most frequently informative optional fields were {coverage_text}. Complete "
            "field distributions, numeric summaries, and coverage estimates are available in "
            f"the {primary_key} CSV tables.",
        ]
    )

    if gate_full is not None:
        self_use_gate = gate_full["a_self_use"]
        presence_gate = gate_full["side_effect_presence"]
        lines.extend(
            [
                "",
                "## Full-corpus gate agreement, DeepSeek vs Luna batch",
                "",
                f"Across all {int(self_use_gate['n']):,} submissions extracted by both "
                "full-corpus runs, self-use agreement was "
                f"{self_use_gate['exact_agreement']:.3f} (kappa {self_use_gate['kappa']:.3f}) "
                "and side-effect presence agreement was "
                f"{presence_gate['exact_agreement']:.3f} (kappa {presence_gate['kappa']:.3f}).",
            ]
        )

    lines.extend(
        [
            "",
            "## Agreement",
            "",
            f"All compared approaches have {common_n:,} successful records in common. Downstream "
            "field agreement is reported on applicable records to avoid inflating agreement "
            "with structural nulls from the self-use gate.",
            "",
            "| Comparison | Paired n | Self-use exact / kappa | Side-effect presence exact / kappa | Full-record exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in pair_summaries:
        comparison = str(summary["comparison"])
        self_use = find_row(pairwise_rows, comparison=comparison, field="a_self_use", scope="all")
        side_effect = find_row(
            pairwise_rows,
            comparison=comparison,
            field="side_effect_presence",
            scope="all",
        )
        label = next(pair[3] for pair in PAIRS if pair[0] == comparison)
        lines.append(
            f"| {label} | {int(summary['paired_records']):,} | "
            f"{float(self_use['exact_agreement']):.3f} / "
            f"{float(self_use['secondary_value']):.3f} | "
            f"{float(side_effect['exact_agreement']):.3f} / "
            f"{float(side_effect['secondary_value']):.3f} | "
            f"{float(summary['full_record_exact_agreement']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Operational summary",
            "",
            "| Approach | Success | Failures | Raw spans verbatim | Cost | Cost / 1,000 successes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in operational:
        failures = int(row["terminal_failures"]) + int(row["retryable_failures"])
        lines.append(
            f"| {row['approach']} | {100 * float(row['success_rate']):.2f}% | "
            f"{failures:,} | {100 * float(row['raw_span_verbatim_rate'] or 0):.2f}% | "
            f"${float(row['cost_usd']):.2f} | "
            f"${float(row['cost_per_1000_successes_usd']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- Pairwise sample sizes differ because the design is nested and some calls failed.",
            "- Agreement on downstream fields is partly determined by the self-use gate; both "
            "unconditional and applicable-subset estimates are retained in the CSV/JSON outputs.",
            "- Candidate MedDRA Preferred Terms are unvalidated model suggestions at this stage.",
            "- High exact agreement with modest kappa can occur for sparse fields dominated by null values.",
            "- Human-labelled validation remains necessary; no model is treated as ground truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Bootstrap resamples for kappa and Jaccard confidence intervals",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--primary",
        choices=("stage1", "stage4"),
        default="stage4",
        help="Stage whose full-corpus aggregate feeds the descriptive tables and figures",
    )
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    database = results_dir / "private_extraction.sqlite3"
    run_summary_path = results_dir / "run_summary.json"
    if not database.exists():
        raise SystemExit(f"extraction checkpoint not found: {database}")
    if not run_summary_path.exists():
        raise SystemExit(f"run summary not found: {run_summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        global STAGES, STAGE_BY_KEY, PAIRS
        stage4_present = bool(
            connection.execute(
                "SELECT 1 FROM stage_results WHERE stage = 'stage4' LIMIT 1"
            ).fetchone()
        )
        if stage4_present:
            STAGES = (*STAGES, STAGE4_SPEC)
            STAGE_BY_KEY = {stage.key: stage for stage in STAGES}
            PAIRS = (*PAIRS, *STAGE4_PAIRS)
        if args.primary != "stage1" and not stage4_present:
            raise SystemExit(f"--primary {args.primary} requires stage4 checkpoint rows")

        stage3 = load_stage(connection, "stage3")
        stage2 = load_stage(connection, "stage2")
        retained_stage1_ids = set(stage2.records) | set(stage3.records)
        stage1 = load_stage(connection, "stage1", retain_ids=retained_stage1_ids)
        stages = {"stage1": stage1, "stage2": stage2, "stage3": stage3}
        if stage4_present:
            stages["stage4"] = load_stage(connection, "stage4", retain_ids=retained_stage1_ids)
        gate_full = (
            full_corpus_gate_agreement(connection, "stage1", "stage4") if stage4_present else None
        )
        primary_aggregate = stages[args.primary].aggregate
        primary_label = STAGE_BY_KEY[args.primary].label

        pairwise_rows: list[dict[str, Any]] = []
        pair_summaries: list[dict[str, Any]] = []
        for comparison, left, right, _label in PAIRS:
            rows, summary = pairwise_analysis(
                comparison,
                left,
                right,
                stages,
                args.bootstrap,
                args.seed,
            )
            pairwise_rows.extend(rows)
            pair_summaries.append(summary)
        three_way_rows, common_ids = three_way_analysis(stages)
        outcomes = key_outcome_rows(stages, common_ids)
        operational = operational_rows(run_summary, connection)
        top_side_effects = top_candidate_side_effect_rows(stages, common_ids)
        stage1_descriptive = stage1_descriptive_rows(primary_aggregate)
        stage1_numeric = numeric_summary_rows(primary_aggregate)
        stage1_coverage = stage1_coverage_rows(primary_aggregate)
        stage1_indication_side_effects = stage1_indication_side_effect_rows(primary_aggregate)
        stage1_candidate_side_effects = stage1_candidate_side_effect_rows(primary_aggregate)
    finally:
        connection.close()

    prefix = args.primary
    configure_plotting()
    figure_field_agreement(pairwise_rows, figures_dir)
    figure_key_outcomes(outcomes, len(common_ids), figures_dir)
    figure_distributions(stages, common_ids, figures_dir)
    figure_top_side_effects(top_side_effects, figures_dir)
    figure_operational(operational, figures_dir)
    figure_stage1_overview(
        primary_aggregate, figures_dir, f"03_06_{prefix}_full_corpus_overview", primary_label
    )
    figure_stage1_clinical_context(
        primary_aggregate, figures_dir, f"03_07_{prefix}_clinical_context", primary_label
    )
    figure_stage1_sentiment_demographics(
        primary_aggregate, figures_dir, f"03_08_{prefix}_sentiment_demographics", primary_label
    )
    figure_stage1_side_effects_coverage(
        stage1_candidate_side_effects,
        stage1_coverage,
        figures_dir,
        f"03_09_{prefix}_side_effects_and_coverage",
        primary_label,
    )

    full_sample = {
        stage.key: {
            "approach": stage.label,
            **stages[stage.key].aggregate.as_dict(),
        }
        for stage in STAGES
    }
    analysis = json_ready(
        {
            "analysis_version": "1.0.0",
            "bootstrap_resamples": args.bootstrap,
            "seed": args.seed,
            "privacy_note": (
                "Aggregate output only; no record IDs, raw spans, treatment-area text, or "
                "per-record extractions are included."
            ),
            "full_sample_extraction_summary": full_sample,
            "common_three_way_records": len(common_ids),
            "pairwise_comparison_summary": pair_summaries,
            "pairwise_field_agreement": pairwise_rows,
            "three_way_field_agreement": three_way_rows,
            "common_sample_key_outcomes": outcomes,
            "operational_summary": operational,
            "top_candidate_side_effects_common_sample": top_side_effects,
            "full_corpus_gate_agreement_stage1_vs_stage4": gate_full,
            "primary_descriptive_analysis": {
                "primary_stage": args.primary,
                "primary_label": primary_label,
                "field_distributions": stage1_descriptive,
                "numeric_summaries": stage1_numeric,
                "informative_field_coverage": stage1_coverage,
                "side_effect_prevalence_by_indication": stage1_indication_side_effects,
                "top_candidate_side_effects": stage1_candidate_side_effects,
            },
        }
    )

    write_csv(output_dir / "03_pairwise_agreement.csv", pairwise_rows)
    write_csv(output_dir / "03_three_way_agreement.csv", three_way_rows)
    write_csv(output_dir / "03_key_outcomes.csv", outcomes)
    write_csv(output_dir / "03_operational_summary.csv", operational)
    write_csv(output_dir / "03_top_candidate_side_effects.csv", top_side_effects)
    write_csv(output_dir / f"03_{prefix}_descriptive_counts.csv", stage1_descriptive)
    write_csv(output_dir / f"03_{prefix}_numeric_summary.csv", stage1_numeric)
    write_csv(output_dir / f"03_{prefix}_informative_coverage.csv", stage1_coverage)
    write_csv(
        output_dir / f"03_{prefix}_side_effects_by_indication.csv",
        stage1_indication_side_effects,
    )
    write_csv(
        output_dir / f"03_{prefix}_candidate_side_effects.csv",
        stage1_candidate_side_effects,
    )
    (output_dir / "03_llm_result_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "03_llm_result_analysis.md",
        stages,
        pairwise_rows,
        pair_summaries,
        len(common_ids),
        operational,
        stage1_numeric,
        stage1_indication_side_effects,
        stage1_candidate_side_effects,
        stage1_coverage,
        primary_key=args.primary,
        primary_label=primary_label,
        gate_full=gate_full,
    )

    print("\n=== LLM extraction analysis ===")
    for stage in STAGES:
        aggregate = stages[stage.key].aggregate
        print(
            f"{stage.label}: {aggregate.n:,} successes; "
            f"self-use={aggregate.self_use:,}; side-effect reports={aggregate.side_effect_reports:,}"
        )
    print(f"common successful records across all stages: {len(common_ids):,}")
    print(f"aggregate tables and report: {output_dir}")
    print(f"9 figures in PNG and PDF: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
