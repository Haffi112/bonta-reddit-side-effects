"""Analyses requested in the 2026-10-23 friendly review of the manuscript.

Everything here is recomputed from data the paper already uses: the human verification of the
1,168 shared submissions, the stage-4 (GPT-5.6 Luna batch) extraction, the stage-1 and stage-2
DeepSeek runs and the filtered corpus. No model is called. Only aggregates are written; community
names are public, but Reddit user-profile pages are named after their owner, so they are reported
only as a group.

* ``self_use_gate``      -- sensitivity, specificity, PPV and NPV of each verified model's self-use
  decision. Self-use is binary, so the true value follows from the majority verdict: the model's
  value when it was judged correct, the opposite value when it was judged wrong.
* ``deployed_runs``      -- the deployed batch run (stage4) and the high-reasoning DeepSeek run
  (stage2) scored against that human ground truth on self-use, side-effect presence and indication,
  without any new annotation.
* ``sparse_field_misses`` -- for each optional field, how often a null output was judged wrong,
  that is, how often the model missed a value the post stated.
* ``authors`` and ``coverage`` -- submissions whose author cannot be linked, and corpus coverage
  over the eligible and extracted sets.
* ``unknown_stratum``    -- the excluded unknown-indication submissions by community, the verified
  split of the misclassified ones, and two reallocations of the stratum to the compared groups.
* ``vocabulary_yield``   -- yield, attitude and the distinct-term test after merging synonyms and
  removing terms that restate an indication or the intended effect (the vocabulary of ``08``).
* ``jsd_bootstrap``      -- bootstrap mean and basic interval of the divergence (``08``'s bootstrap).
* ``term_table``         -- the largest per-term differences with their author-clustered intervals,
  flagged where the term restates an indication or the intended effect.
* ``batch_list_identity`` and ``full_corpus_agreement`` -- how often the deployed batch run's
  candidate-term set equals the verified interactive one, and the full-corpus row of
  Supplementary Table 2.
* ``adjudication``       -- when the private file ``author_adjudication.json`` is present: the
  post-level presence labels for the verified outputs counted as side-effect false positives, which
  the three annotators gave independently (majority vote; all nine judged to describe no side
  effect), and an author's reading of the unknown indications neither configuration got right. The
  error correction is re-run with ``09``'s seed on the post-level counts. The file holds record
  identifiers, so it stays with the private extraction records and is never published.

Outputs: ``13_review_analyses.{json,md}`` and ``13_term_table.csv``.

Example:

    uv run python data_analysis/13_review_analyses.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
EXPORT_DIR = ANALYSIS_DIR / "02_llm_extraction_results"
DATABASE = EXPORT_DIR / "private_extraction.sqlite3"
CORPUS = ROOT / "filtered_dataset" / "submissions.jsonl"
ADJUDICATION = EXPORT_DIR / "author_adjudication.json"
SEED = 20261023
REPLICATES = 2_000
UNLINKABLE = ("[deleted]", "[removed]", "AutoModerator")


def _load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


a04 = _load_module("a04", "04_medical_cosmetic_comparison.py")
a05 = _load_module("a05", "05_annotation_analysis.py")
a08 = _load_module("a08", "08_robustness_analyses.py")
a09 = _load_module("a09", "09_verification_by_indication.py")


def record_id(row: dict[str, Any]) -> str:
    rid = str(row.get("name") or "t3_" + str(row.get("id")))
    return rid if rid.startswith("t3_") else "t3_" + rid


def terms_of(extraction: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        a04.normalize_text(effect.get("meddra_pt"))
        for effect in extraction.get("d_side_effect_pts") or []
        if isinstance(effect, dict) and effect.get("meddra_pt")
    )


def arm_of(extraction: dict[str, Any]) -> str | None:
    """The comparison arm of a self-use record; None when the record is not self-use."""
    if not extraction.get("a_self_use"):
        return None
    arm = a04.normalize_text(extraction.get("c_indication_category"))
    return arm if arm in a04.GROUPS else "unknown"


def rates(k: int, n: int) -> dict[str, float | int | None]:
    low, high = a04.wilson_interval(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci_low": low, "ci_high": high}


def difference(med_k: int, med_n: int, cos_k: int, cos_n: int) -> dict[str, float | int]:
    low, high = a04.newcombe_difference_interval(med_k, med_n, cos_k, cos_n)
    return {
        "medical_k": med_k,
        "medical_n": med_n,
        "cosmetic_k": cos_k,
        "cosmetic_n": cos_n,
        "prevalence_medical": med_k / med_n,
        "prevalence_cosmetic": cos_k / cos_n,
        "difference": med_k / med_n - cos_k / cos_n,
        "ci_low": low,
        "ci_high": high,
    }


# ------------------------------------------------------------------ verification ---


def verified_outputs() -> dict[str, dict[str, dict[str, Any]]]:
    """Per record and verified model: the output values and the majority verdict per field."""
    decisions, annotators, _ = a05.load_annotations(EXPORT_DIR)
    n_raters = len(annotators)
    votes: dict[tuple[int, str], list[bool]] = defaultdict(list)
    owner: dict[int, tuple[str, str]] = {}
    for d in decisions:
        votes[(d["model_output_id"], d["key"])].append(d["correct"])
        owner[d["model_output_id"]] = (d["record_id"], d["model_key"])
    values = a05.load_model_values(EXPORT_DIR)
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for output_id, (rid, model) in owner.items():
        verdict = {
            key: sum(v) * 2 > n_raters for (oid, key), v in votes.items() if oid == output_id
        }
        spans = [ok for key, ok in verdict.items() if key.endswith(".raw_span")]
        out[rid][model] = {
            "value": values[output_id],
            "verdict": verdict,
            "any_span_correct": any(spans),
        }
    return out


def truth_tables(verified: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Human ground truth per record, as far as the binary verdicts determine it."""
    truth: dict[str, dict[str, Any]] = {}
    for rid, models in verified.items():
        luna = models["stage3"]
        claimed = bool(luna["value"].get("a_self_use"))
        self_use = claimed if luna["verdict"].get("a_self_use") else not claimed
        presence = None
        if claimed and self_use:  # the list was judged as a self-use list
            listed = bool(luna["value"].get("d_side_effect_pts"))
            presence = (
                luna["any_span_correct"]
                if listed
                else luna["verdict"].get("d_side_effect_pts") is False
            )
        indication = None
        for model in (
            ("stage3", "stage1") if self_use else ()
        ):  # the first configuration judged right fixes the value
            o = models.get(model)
            if o and o["value"].get("a_self_use") and o["verdict"].get("c_indication_category"):
                indication = arm_of(o["value"])
                break
        truth[rid] = {"self_use": self_use, "presence": presence, "indication": indication}
    return truth


def gate_metrics(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Predicted against true values of a binary decision."""
    tp = sum(1 for p, t in pairs if p and t)
    fp = sum(1 for p, t in pairs if p and not t)
    fn = sum(1 for p, t in pairs if not p and t)
    tn = sum(1 for p, t in pairs if not p and not t)
    return {
        "n": len(pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": rates(tp + tn, len(pairs)),
        "sensitivity": rates(tp, tp + fn),
        "specificity": rates(tn, tn + fp),
        "ppv": rates(tp, tp + fp),
        "npv": rates(tn, tn + fn),
    }


def self_use_gate(verified: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for model in ("stage1", "stage3"):
        pairs = []
        for models in verified.values():
            o = models[model]
            claimed = bool(o["value"].get("a_self_use"))
            pairs.append((claimed, claimed if o["verdict"].get("a_self_use") else not claimed))
        result[model] = gate_metrics(pairs)
    return result


def deployed_runs(truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score runs that were never annotated against the verified ground truth."""
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    result: dict[str, Any] = {}
    for stage in ("stage4", "stage2", "stage3", "stage1"):
        values = {
            rid: json.loads(js)
            for rid, js in connection.execute(
                "SELECT record_id, extraction_json FROM stage_results WHERE stage = ?", (stage,)
            )
            if rid in truth
        }
        self_use = [
            (bool(e.get("a_self_use")), truth[rid]["self_use"]) for rid, e in values.items()
        ]
        presence = [
            (bool(e.get("a_self_use")) and bool(e.get("d_side_effect_pts")), truth[rid]["presence"])
            for rid, e in values.items()
            if truth[rid]["presence"] is not None
        ]
        indication = [
            arm_of(e) == truth[rid]["indication"]
            for rid, e in values.items()
            if truth[rid]["indication"] is not None
        ]
        # What matters for the comparison is whether an arm label is right when one is given:
        # assigning a medical post to the cosmetic arm, or the reverse, would bias the contrast.
        arm_labels = {
            arm: [
                truth[rid]["indication"] == arm
                for rid, e in values.items()
                if truth[rid]["indication"] is not None and arm_of(e) == arm
            ]
            for arm in a04.GROUPS
        }
        result[stage] = {
            "records": len(values),
            "self_use": gate_metrics(self_use),
            "presence": gate_metrics(presence),
            "indication_accuracy": rates(sum(indication), len(indication)),
            **{f"{arm}_label_precision": rates(sum(v), len(v)) for arm, v in arm_labels.items()},
        }
    connection.close()
    return result


def sparse_field_misses(verified: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Null outputs judged wrong, per optional field, among correct self-use outputs."""
    result: dict[str, Any] = {}
    for model in ("stage1", "stage3"):
        rows = {}
        for field in a05.CONDITIONAL_FIELDS:
            null, missed = 0, 0
            for models in verified.values():
                o = models[model]
                if not (o["value"].get("a_self_use") and o["verdict"].get("a_self_use")):
                    continue
                if a05.is_informative(o["value"].get(field)):
                    continue
                null += 1
                missed += o["verdict"].get(field) is False
            rows[field] = {"label": a05.FIELD_LABELS[field], **rates(missed, null)}
        result[model] = rows
    return result


# --------------------------------------------------------------------- corpus ---


def corpus_tables() -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    eligible = {
        rid
        for (rid,) in connection.execute(
            "SELECT record_id FROM stage_results WHERE stage = 'stage1'"
        )
    }
    extracted = {
        rid: json.loads(js)
        for rid, js in connection.execute(
            "SELECT record_id, extraction_json FROM stage_results WHERE stage = 'stage4'"
        )
    }
    connection.close()
    coverage = {
        key: {"communities": Counter(), "authors": Counter(), "lengths": []}
        for key in ("eligible", "extracted")
    }
    authors: Counter[tuple[str, str]] = Counter()
    records: list[dict[str, Any]] = []
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rid = record_id(row)
            author = row.get("author") or "[deleted]"
            community = str(row.get("subreddit") or "").casefold()
            for key, members in (("eligible", eligible), ("extracted", extracted)):
                if rid in members:
                    coverage[key]["communities"][community] += 1
                    if author not in UNLINKABLE:
                        coverage[key]["authors"][author] += 1
                    coverage[key]["lengths"].append(len(row.get("selftext") or ""))
            if rid not in extracted:
                continue
            e = extracted[rid]
            arm = arm_of(e) or "not_self_use"
            authors[(arm, "all")] += 1
            if author in UNLINKABLE:
                authors[(arm, author)] += 1
            records.append(
                {
                    "arm": arm,
                    "terms": terms_of(e),
                    "attitude": a04.normalize_text(e.get("f_sentiment_about_side_effect")) or None,
                    "community": community,
                    "unlinked": author in UNLINKABLE,
                    "automoderator": author == "AutoModerator",
                }
            )
    return {
        "coverage": {
            key: {
                "submissions": len(c["lengths"]),
                "communities": len(c["communities"]),
                "named_authors": len(c["authors"]),
                "median_body_characters": statistics.median(c["lengths"]),
            }
            for key, c in coverage.items()
        },
        "authors": {f"{arm}|{kind}": n for (arm, kind), n in sorted(authors.items())},
        "records": records,
    }


def unknown_stratum(
    records: list[dict[str, Any]],
    verified: dict[str, dict[str, dict[str, Any]]],
    rng: np.random.Generator,
    override: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Where the excluded unknown-indication submissions come from, and two reallocations."""
    unknown = [r for r in records if r["arm"] == "unknown"]
    by_community = Counter(
        "user profile pages" if r["community"].startswith("u_") else r["community"] for r in unknown
    )
    with_effect = Counter(
        "user profile pages" if r["community"].startswith("u_") else r["community"]
        for r in unknown
        if r["terms"]
    )
    top = [
        {"community": c, "submissions": n, "with_side_effect": with_effect[c]}
        for c, n in by_community.most_common(8)
    ]
    outside = [r for r in unknown if r["community"] != "noburp"]

    # Verified unknown-indication outputs of GPT-5.6 Luna: was "unknown" right, and if not, which arm
    # does the other verified configuration establish? Split by r/noburp, where it concentrates.
    community_of = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if (rid := record_id(row)) in verified:
                community_of[rid] = str(row.get("subreddit") or "").casefold()
    split: dict[str, Counter[str]] = {"noburp": Counter(), "other": Counter()}
    for rid, models in verified.items():
        luna = models["stage3"]
        if arm_of(luna["value"]) != "unknown":
            continue
        stratum = "noburp" if community_of.get(rid) == "noburp" else "other"
        if luna["verdict"].get("c_indication_category"):
            split[stratum]["stays_unknown"] += 1
            continue
        if override and rid in override:  # an author's reading settles what neither model got right
            split[stratum][override[rid]] += 1
            continue
        other = models.get("stage1")
        arm = (
            arm_of(other["value"])
            if other and other["verdict"].get("c_indication_category")
            else None
        )
        split[stratum][str(arm) if arm in a04.GROUPS else "undetermined"] += 1

    med = [r for r in records if r["arm"] == "medical"]
    cos = [r for r in records if r["arm"] == "cosmetic"]
    med_k, cos_k = sum(bool(r["terms"]) for r in med), sum(bool(r["terms"]) for r in cos)

    # (a) Community context: in r/noburp the only BoNT-A use discussed is cricopharyngeal injection.
    noburp_unknown = [r for r in unknown if r["community"] == "noburp"]
    context = difference(
        med_k + sum(bool(r["terms"]) for r in noburp_unknown),
        len(med) + len(noburp_unknown),
        cos_k,
        len(cos),
    )

    # (b) Verified proportions: each unknown record moves to medical or cosmetic, or stays excluded, with
    # the probabilities its stratum showed in verification (undetermined records stay excluded).
    # Each replicate first resamples the verified outputs, so both sources of uncertainty enter.
    has = {
        s: np.array(
            [bool(r["terms"]) for r in unknown if (r["community"] == "noburp") == (s == "noburp")]
        )
        for s in ("noburp", "other")
    }
    outcomes = {s: np.array([k for k, n in split[s].items() for _ in range(n)]) for s in split}
    draws = []
    for _ in range(REPLICATES):
        add_med_n = add_med_k = add_cos_n = add_cos_k = 0
        for s in ("noburp", "other"):
            sample = rng.choice(outcomes[s], size=len(outcomes[s]), replace=True)
            p_med = np.mean(sample == "medical")
            p_cos = np.mean(sample == "cosmetic")
            u = rng.random(len(has[s]))
            to_med, to_cos = u < p_med, (u >= p_med) & (u < p_med + p_cos)
            add_med_n += int(to_med.sum())
            add_med_k += int(has[s][to_med].sum())
            add_cos_n += int(to_cos.sum())
            add_cos_k += int(has[s][to_cos].sum())
        draws.append(
            (med_k + add_med_k) / (len(med) + add_med_n)
            - (cos_k + add_cos_k) / (len(cos) + add_cos_n)
        )
    draws_a = np.array(draws)

    return {
        "submissions": len(unknown),
        "with_side_effect": sum(bool(r["terms"]) for r in unknown),
        "noburp": {
            "submissions": len(noburp_unknown),
            "with_side_effect": sum(bool(r["terms"]) for r in noburp_unknown),
        },
        "outside_noburp": rates(sum(bool(r["terms"]) for r in outside), len(outside)),
        "top_communities": top,
        "verified_split": {s: dict(c) for s, c in split.items()},
        "noburp_to_medical": context,
        "verified_reallocation": {
            "replicates": REPLICATES,
            "median": float(np.median(draws_a)),
            "p2_5": float(np.percentile(draws_a, 2.5)),
            "p97_5": float(np.percentile(draws_a, 97.5)),
        },
    }


def noburp_without_eructation(records: list[dict[str, Any]]) -> dict[str, Any]:
    med = [r for r in records if r["arm"] == "medical" and r["community"] == "noburp"]
    return {
        "medical_self_use": len(med),
        "any": rates(sum(bool(r["terms"]) for r in med), len(med)),
        "without_eructation": rates(sum(bool(r["terms"] - {"eructation"}) for r in med), len(med)),
        "without_excluded_terms": rates(
            sum(bool(r["terms"] - a08.SYMMETRIC_EXCLUSIONS) for r in med), len(med)
        ),
    }


def vocabulary_yield(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fig. 2 quantities after merging synonyms and removing indication and intended-effect terms."""
    self_use = [r for r in records if r["arm"] != "not_self_use"]
    cleaned = [a08.merge_and_exclude(r["terms"]) for r in self_use]

    def attitudes(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(r["attitude"] for r in rows)
        n = len(rows)
        return {
            "n": n,
            **{a: rates(counts[a], n) for a in ("distressed", "accepting", "dismissive")},
        }

    raw_reporters = [r for r in self_use if r["terms"]]
    clean_reporters = [r for r, c in zip(self_use, cleaned, strict=True) if c]
    only_excluded = [r for r, c in zip(self_use, cleaned, strict=True) if r["terms"] and not c]
    return {
        "self_use": len(self_use),
        "any_raw": len(raw_reporters),
        "multiple_raw": sum(len(r["terms"]) > 1 for r in self_use),
        "any_cleaned": len(clean_reporters),
        "multiple_cleaned": sum(len(c) > 1 for c in cleaned),
        "only_excluded_terms_by_arm": dict(Counter(r["arm"] for r in only_excluded)),
        "attitude_raw": attitudes(raw_reporters),
        "attitude_cleaned": attitudes(clean_reporters),
        "attitude_only_excluded": attitudes(only_excluded),
    }


def distinct_terms_cleaned(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The Mann-Whitney comparison of distinct terms per report under the cleaned vocabulary."""
    comparison = [
        a04.ComparisonRecord(indication=r["arm"], side_effects=a08.merge_and_exclude(r["terms"]))
        for r in records
        if r["arm"] in a04.GROUPS
    ]
    test = a04.side_effect_burden_test(comparison)
    counts = {
        arm: np.array(
            [len(c.side_effects) for c in comparison if c.indication == arm and c.side_effects]
        )
        for arm in a04.GROUPS
    }
    return {
        **{f"{arm}_mean": float(v.mean()) for arm, v in counts.items()},
        **{f"{arm}_single_term_share": float(np.mean(v == 1)) for arm, v in counts.items()},
        "u": test["statistic"],
        "p": test["p_value"],
        "rank_biserial": test["effect_estimate"],
        "medical_n": test["medical_n"],
        "cosmetic_n": test["cosmetic_n"],
    }


def jsd_bootstrap() -> dict[str, Any]:
    """Re-run 08's author-clustered bootstrap with its seed; the first draw from that generator."""
    summary, _ = a08.cluster_bootstrap(
        a08.load(), 2_000, np.random.default_rng(a08.RESAMPLING_SEED)
    )
    keys = (
        "jsd",
        "jsd_ci_low",
        "jsd_ci_high",
        "jsd_bootstrap_mean",
        "jsd_basic_ci_low",
        "jsd_basic_ci_high",
    )
    return {k: summary[k] for k in keys}


def term_table(rows_wanted: int = 12) -> list[dict[str, Any]]:
    """Largest absolute per-term differences, with clustered intervals and an exclusion flag."""
    with (ANALYSIS_DIR / "04_side_effect_comparison.csv").open(encoding="utf-8") as fh:
        comparison = list(csv.DictReader(fh))
    with (ANALYSIS_DIR / "08_cluster_terms.csv").open(encoding="utf-8") as fh:
        clustered = {r["candidate_pt"]: r for r in csv.DictReader(fh)}
    tested = [r for r in comparison if r["candidate_pt"] in clustered]
    tested.sort(key=lambda r: -abs(float(r["risk_difference_percentage_points"])))
    return [
        {
            "rank": i,
            "term": r["candidate_pt"],
            "medical_per_100": float(r["medical_records_per_100_reports"]),
            "medical_records": int(r["medical_records"]),
            "cosmetic_per_100": float(r["cosmetic_records_per_100_reports"]),
            "cosmetic_records": int(r["cosmetic_records"]),
            "difference": float(r["risk_difference_percentage_points"]),
            "clustered_ci_low": float(clustered[r["candidate_pt"]]["ci_low"]),
            "clustered_ci_high": float(clustered[r["candidate_pt"]]["ci_high"]),
            "bh_q": float(r["bh_q_value"]),
            "restates_indication_or_intended_effect": r["candidate_pt"] in a08.SYMMETRIC_EXCLUSIONS,
        }
        for i, r in enumerate(tested[:rows_wanted], 1)
    ]


def correction_ranges(
    luna: dict[str, list[dict[str, Any]]], corpus: dict[str, float]
) -> dict[str, list[float]]:
    """Lowest and highest corrected prevalence per group across 09's bootstrap replicates (same seed)."""
    rng = np.random.default_rng(a09.SEED)
    seen: dict[str, list[float]] = {g: [] for g in a04.GROUPS}

    def factor(outputs: list[dict[str, Any]]) -> float:
        c = a09.cell(outputs)
        return (c["tp"] / (c["tp"] + c["fp"])) / (c["tp"] / (c["tp"] + c["fn"]))

    for _ in range(a09.REPLICATES):
        resampled = {
            g: [luna[g][i] for i in rng.integers(0, len(luna[g]), len(luna[g]))] for g in a04.GROUPS
        }
        try:
            values = {g: corpus[g] * factor(resampled[g]) for g in a04.GROUPS}
        except ZeroDivisionError:
            continue
        for g, v in values.items():
            seen[g].append(v)
    return {g: [min(v), max(v)] for g, v in seen.items()}


def adjudication(
    records: list[dict[str, Any]], verified: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any] | None:
    """Re-run the error correction and the reallocation with an author's re-reading, if present."""
    if not ADJUDICATION.exists():
        return None
    adjudicated = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    genuine = {
        rid
        for rid, v in adjudicated["presence_false_positives"].items()
        if v == "genuine side effect"
    }
    csv.field_size_limit(sys.maxsize)
    with (EXPORT_DIR / "model_outputs_export.csv").open(encoding="utf-8") as fh:
        record_of = {int(r["model_output_id"]): r["record_id"] for r in csv.DictReader(fh)}

    # Same construction and order as 09, so the bootstrap draws match and only the flags differ.
    luna: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reclassified: Counter[str] = Counter()
    for output_id, o in a09.load_outputs().items():
        if o["model"] != "stage3" or o["group"] not in a04.GROUPS:
            continue
        if record_of[output_id] in genuine and o["listed"] and not o["any_span_correct"]:
            o = {**o, "any_span_correct": True}
            reclassified[o["group"]] += 1
        luna[o["group"]].append(o)
    with (ANALYSIS_DIR / "04_group_summary.csv").open(encoding="utf-8") as fh:
        corpus = {
            r["indication"]: float(r["side_effect_report_prevalence"]) for r in csv.DictReader(fh)
        }
    cells = {g: a09.cell(luna[g]) for g in a04.GROUPS}
    override = {rid: v for rid, v in adjudicated["unknown_indication"].items()}
    reallocation = unknown_stratum(records, verified, np.random.default_rng(SEED), override)
    return {
        "false_positives_reread": len(adjudicated["presence_false_positives"]),
        "annotator_votes": dict(
            Counter(
                v
                for votes in adjudicated.get("presence_annotator_votes", {}).values()
                for v in votes.values()
            )
        ),
        "reclassified_as_true_positive": dict(reclassified),
        "cells": cells,
        "sensitivity": {g: c["tp"] / (c["tp"] + c["fn"]) for g, c in cells.items()},
        "ppv": {g: c["tp"] / (c["tp"] + c["fp"]) for g, c in cells.items()},
        "sensitivity_ci": {
            g: a04.wilson_interval(c["tp"], c["tp"] + c["fn"]) for g, c in cells.items()
        },
        "ppv_ci": {g: a04.wilson_interval(c["tp"], c["tp"] + c["fp"]) for g, c in cells.items()},
        "correction": a09.corrected_difference(dict(luna), corpus, np.random.default_rng(a09.SEED)),
        "corrected_prevalence_range": correction_ranges(dict(luna), corpus),
        "unknown_verified_split": reallocation["verified_split"],
        "unknown_verified_reallocation": reallocation["verified_reallocation"],
    }


def batch_list_identity(verified: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """How often the deployed batch run returned the same candidate-term set as the verified run.

    Both runs are the same model at the same settings, with provider-default sampling and no seed,
    so where the sets differ the difference is sampling variation. Where they agree, the human
    verdicts on the interactive list apply to the batch list unchanged.
    """
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    batch = {
        rid: json.loads(js)
        for rid, js in connection.execute(
            "SELECT record_id, extraction_json FROM stage_results WHERE stage = 'stage4'"
        )
        if rid in verified
    }
    connection.close()
    counts: Counter[str] = Counter()
    for rid, models in verified.items():
        interactive, deployed = models["stage3"]["value"], batch.get(rid)
        if deployed is None or not (interactive.get("a_self_use") and deployed.get("a_self_use")):
            continue
        if arm_of(deployed) not in a04.GROUPS:
            continue
        counts["both_self_use"] += 1
        counts["same_term_set"] += terms_of(interactive) == terms_of(deployed)
    return {**counts, "share_same": counts["same_term_set"] / counts["both_self_use"]}


def full_corpus_agreement() -> dict[str, Any]:
    """Supplementary Table 2's comparison of the two full-corpus runs over every shared record."""
    a03 = _load_module("a03", "03_llm_result_analysis.py")
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    stages = {stage: a03.load_stage(connection, stage) for stage in ("stage1", "stage4")}
    connection.close()
    rows, summary = a03.pairwise_analysis("stage1_vs_stage4_full", "stage1", "stage4", stages, 0, 0)

    def metric(field: str, scope: str, key: str) -> float:
        return next(r[key] for r in rows if r["field"] == field and r["scope"] == scope)

    return {
        "paired_records": summary["paired_records"],
        "self_use_agreement": metric("a_self_use", "all", "exact_agreement"),
        "self_use_kappa": metric("a_self_use", "all", "secondary_value"),
        "presence_agreement": metric("side_effect_presence", "all", "exact_agreement"),
        "presence_kappa": metric("side_effect_presence", "all", "secondary_value"),
        "term_set_exact": metric("d_side_effect_pts", "applicable", "exact_agreement"),
        "term_set_jaccard": metric("d_side_effect_pts", "applicable", "secondary_value"),
        "full_record": summary["full_record_exact_agreement"],
    }


# --------------------------------------------------------------------- output ---


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def markdown(result: dict[str, Any]) -> list[str]:
    lines = ["# Analyses for the 2026-10-23 review", ""]
    lines += ["## Self-use decision against the human ground truth", ""]
    for model, m in result["self_use_gate"].items():
        lines.append(
            f"- {model}: sensitivity {pct(m['sensitivity']['rate'])} ({m['tp']}/{m['tp'] + m['fn']}), "
            f"PPV {pct(m['ppv']['rate'])} ({m['tp']}/{m['tp'] + m['fp']}), "
            f"specificity {pct(m['specificity']['rate'])}, NPV {pct(m['npv']['rate'])}"
        )
    lines += ["", "## Runs scored against the verified ground truth", ""]
    for stage, m in result["deployed_runs"].items():
        lines.append(
            f"- {stage} ({m['records']} records): self-use accuracy {pct(m['self_use']['accuracy']['rate'])}, "
            f"presence accuracy {pct(m['presence']['accuracy']['rate'])} (n={m['presence']['n']}), "
            f"indication accuracy {pct(m['indication_accuracy']['rate'])} (n={m['indication_accuracy']['n']}); "
            f"medical label right {m['medical_label_precision']['k']}/{m['medical_label_precision']['n']}, "
            f"cosmetic label right {m['cosmetic_label_precision']['k']}/{m['cosmetic_label_precision']['n']}"
        )
    lines += ["", "## Missed values in optional fields (null judged wrong)", ""]
    for m in result["sparse_field_misses"]["stage3"].values():
        lines.append(f"- {m['label']}: {m['k']}/{m['n']} ({pct(m['rate'])})")
    u = result["unknown_stratum"]
    lines += [
        "",
        "## Unknown-indication stratum",
        "",
        f"- {u['submissions']:,} submissions, {u['with_side_effect']:,} with a side effect; "
        f"r/noburp {u['noburp']['submissions']:,} ({u['noburp']['with_side_effect']:,} with a side effect); "
        f"outside r/noburp {pct(u['outside_noburp']['rate'])}",
        f"- verified split: {u['verified_split']}",
        f"- r/noburp unknowns to medical: {100 * u['noburp_to_medical']['difference']:+.2f} pp "
        f"({100 * u['noburp_to_medical']['ci_low']:+.2f} to {100 * u['noburp_to_medical']['ci_high']:+.2f})",
        f"- verified reallocation: median {100 * u['verified_reallocation']['median']:+.2f} pp "
        f"({100 * u['verified_reallocation']['p2_5']:+.2f} to {100 * u['verified_reallocation']['p97_5']:+.2f})",
    ]
    v = result["vocabulary_yield"]
    lines += [
        "",
        "## Yield under the cleaned vocabulary",
        "",
        f"- any side effect {v['any_raw']:,} -> {v['any_cleaned']:,}; multiple {v['multiple_raw']:,} -> {v['multiple_cleaned']:,}",
        f"- reports whose only terms were excluded, by arm: {v['only_excluded_terms_by_arm']}",
    ]
    j = result["jsd_bootstrap"]
    lines += [
        "",
        "## Divergence bootstrap",
        "",
        f"- JSD {j['jsd']:.3f}; bootstrap mean {j['jsd_bootstrap_mean']:.3f}; percentile "
        f"{j['jsd_ci_low']:.3f}-{j['jsd_ci_high']:.3f}; basic {j['jsd_basic_ci_low']:.3f}-{j['jsd_basic_ci_high']:.3f}",
    ]
    a = result.get("adjudication")
    if a:
        c = a["correction"]
        lines += [
            "",
            "## Author re-reading (private adjudication file)",
            "",
            f"- false positives re-read: {a['false_positives_reread']}; reclassified as true positives: {a['reclassified_as_true_positive']}",
            f"- PPV medical {pct(a['ppv']['medical'])}, cosmetic {pct(a['ppv']['cosmetic'])}; "
            f"corrected difference {100 * c['corrected_difference']:+.2f} pp "
            f"({100 * c['corrected_difference_ci_low']:+.2f} to {100 * c['corrected_difference_ci_high']:+.2f})",
            f"- unknown split {a['unknown_verified_split']}; reallocation median "
            f"{100 * a['unknown_verified_reallocation']['median']:+.2f} pp "
            f"({100 * a['unknown_verified_reallocation']['p2_5']:+.2f} to {100 * a['unknown_verified_reallocation']['p97_5']:+.2f})",
        ]
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    verified = verified_outputs()
    truth = truth_tables(verified)
    corpus = corpus_tables()
    records = corpus.pop("records")
    result: dict[str, Any] = {
        "self_use_gate": self_use_gate(verified),
        "deployed_runs": deployed_runs(truth),
        "sparse_field_misses": sparse_field_misses(verified),
        **corpus,
        "automoderator_self_use": dict(
            Counter(r["arm"] for r in records if r["automoderator"] and r["arm"] != "not_self_use")
        ),
        "unknown_stratum": unknown_stratum(records, verified, rng),
        "noburp_medical": noburp_without_eructation(records),
        "vocabulary_yield": vocabulary_yield(records),
        "distinct_terms_cleaned": distinct_terms_cleaned(records),
        "jsd_bootstrap": jsd_bootstrap(),
        "term_table": term_table(),
        "adjudication": adjudication(records, verified),
        "batch_list_identity": batch_list_identity(verified),
        "full_corpus_agreement": full_corpus_agreement(),
    }
    (ANALYSIS_DIR / "13_review_analyses.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    with (ANALYSIS_DIR / "13_term_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["term_table"][0]))
        writer.writeheader()
        writer.writerows(result["term_table"])
    lines = markdown(result)
    (ANALYSIS_DIR / "13_review_analyses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
