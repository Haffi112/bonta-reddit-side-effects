"""Robustness analyses requested in the 2026-09-26 manuscript review.

Six additions to the primary medical/cosmetic comparison, recomputed from the
stage-4 (GPT-5.6 Luna batch) extraction with the machinery of
``04_medical_cosmetic_comparison.py``:

* ``loco``            -- leave out each of the 20 largest communities in turn,
  and four community families: the harm-focused support community, the two
  DIY-injection communities, user-profile pages, and r/noburp together with
  the support community (one outcome-focused community from each side);
* ``cluster``         -- author-clustered bootstrap intervals for the
  prevalence difference, the Jensen--Shannon divergence and every per-term
  difference, with the per-term Benjamini--Hochberg count recomputed from the
  clustered standard errors;
* ``author_draws``    -- 200 random one-submission-per-author draws, replacing
  the single lexicographic draw of ``07``;
* ``periods``         -- calendar strata, including the split at July 2023
  when the archive's collection source changed;
* ``within_site``     -- medical and cosmetic subtypes injected in the same
  region (for example chronic migraine against forehead and glabellar lines);
* ``vocabulary``      -- synonym merges alone, and merges plus a symmetric
  exclusion of terms that name the treated condition, the intended effect or
  a lack of effect in either group.

Only aggregates are written. Community names are public, but Reddit
user-profile pages are named after their owner, so they are only ever
reported as a group.

Outputs: ``08_robustness_analyses.{json,md}``, ``08_loco.csv``,
``08_periods.csv``, ``08_within_site.csv``, ``08_vocabulary.csv``,
``08_cluster_terms.csv``.

Example:

    uv run python data_analysis/08_robustness_analyses.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse, stats  # type: ignore[import-untyped]
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = ANALYSIS_DIR / "02_llm_extraction_results"


def _load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a04 = _load_module("a04", "04_medical_cosmetic_comparison.py")

STAGE = "stage4"
PERMUTATION_SEED = 20260715  # same seed as 04 and 07
RESAMPLING_SEED = 20260927
MIN_COUNT = 20
MIN_COUNT_SMALL = 10  # within-site strata are an order of magnitude smaller
TOP_COMMUNITIES = 20
COLLECTION_CHANGE = datetime(2023, 7, 1, tzinfo=UTC).timestamp()

FAMILIES: dict[str, str] = {
    "support_community": "harm-focused support community (r/BotoxSupportCommunity)",
    "diy_communities": "DIY-injection communities (r/DIYAesthetics, r/DIYCosmeticProcedures)",
    "user_profile_pages": "user-profile pages",
    "noburp_and_support": "r/noburp and r/BotoxSupportCommunity together",
}

WITHIN_SITE_PAIRS: dict[str, tuple[set[str], set[str], str]] = {
    "upper_face": (
        {"chronic_migraine"},
        {"forehead", "glabellar"},
        "chronic migraine vs forehead or glabellar lines",
    ),
    "periocular": ({"blepharospasm"}, {"crows_feet"}, "blepharospasm vs lateral canthal lines"),
    "jaw": (
        {"tmj_medical"},
        {"masseter_jawline"},
        "temporomandibular disorder vs masseter or jawline",
    ),
    "neck": ({"cervical_dystonia"}, {"neck_platysma"}, "cervical dystonia vs neck or platysma"),
}

# Spelling variants and synonyms of the same complaint, merged in both groups.
SYNONYMS: dict[str, str] = {
    "muscular weakness": "muscle weakness",
    "gastrooesophageal reflux disease": "gastroesophageal reflux disease",
    "brow ptosis": "eyebrow ptosis",
    "muscle spasm": "muscle spasms",
    "hypoesthesia": "hypoaesthesia",
    "paresthesia": "paraesthesia",
    "blurred vision": "vision blurred",
    "smile abnormal": "abnormal smile",
    "smile impairment": "abnormal smile",
    "facial expression abnormal": "abnormal facial expression",
    "facial expression disorder": "abnormal facial expression",
    "facial appearance abnormal": "abnormal facial appearance",
    "facial appearance change": "abnormal facial appearance",
    "facial appearance alteration": "abnormal facial appearance",
    "appearance altered": "abnormal facial appearance",
    "periorbital oedema": "periorbital swelling",
    "feeling of foreign body in throat": "globus sensation",
    "foreign body sensation in throat": "globus sensation",
    "throat pain": "oropharyngeal pain",
    "sore throat": "oropharyngeal pain",
    "pain exacerbated": "pain aggravated",
    "choking sensation": "choking",
    "abdominal bloating": "abdominal distension",
    "borborygmi": "abdominal sounds",
    "bruising": "contusion",
    "bleeding": "haemorrhage",
    "muscle tension": "muscle tightness",
    "heaviness": "feeling of heaviness",
    "eye heaviness": "eyelid heaviness",
    "facial paralysis": "facial paresis",
    "skin laxity": "skin sagging",
    "facial sagging": "skin sagging",
    "lightheadedness": "dizziness",
    "depression": "depressed mood",
    "wrinkle": "wrinkles",
    "skin wrinkling": "wrinkles",
    "facial wrinkles": "wrinkles",
    "facial lines": "wrinkles",
    "skin lines": "wrinkles",
    "eructation abnormal": "eructation decreased",
    "eructation disorder": "eructation decreased",
}

# After merging: terms that name the treated condition, the intended effect of
# treatment or a lack of effect. Excluded from both groups alike.
SYMMETRIC_EXCLUSIONS: set[str] = {
    # lack of effect
    "treatment failure",
    "drug ineffective",
    "therapeutic response decreased",
    "drug effect decreased",
    "drug effect incomplete",
    "therapeutic product effect decreased",
    "therapeutic product effect incomplete",
    # medical indications restated
    "migraine",
    "migraine aggravated",
    "condition aggravated",
    "pain aggravated",
    "hyperhidrosis",
    "bruxism",
    "anal fissure",
    "eructation decreased",
    "neuralgia",
    "urinary urgency",
    "urinary incontinence",
    "dystonia",
    "muscle spasticity",
    # intended effect of cricopharyngeal injection
    "eructation",
    # cosmetic indications restated
    "wrinkles",
    "aged appearance",
    "skin crease",
    "skin fold",
    "jowls",
    "double chin",
}


@dataclass(frozen=True)
class Rec:
    arm: str
    side_effects: frozenset[str]
    subtype: str
    author: str | None
    community: str
    created: float | None

    def comparison(self, terms: frozenset[str] | None = None) -> Any:
        return a04.ComparisonRecord(
            indication=self.arm, side_effects=self.side_effects if terms is None else terms
        )


def load() -> list[Rec]:
    connection = sqlite3.connect(
        f"file:{RESULTS_DIR / 'private_extraction.sqlite3'}?mode=ro", uri=True
    )
    keyed: dict[str, tuple[str, frozenset[str], str]] = {}
    for record_id, extraction_json in connection.execute(
        "SELECT record_id, extraction_json FROM stage_results WHERE stage = ?", (STAGE,)
    ):
        extraction = json.loads(extraction_json)
        if not extraction.get("a_self_use"):
            continue
        arm = a04.normalize_text(extraction.get("c_indication_category"))
        if arm not in a04.GROUPS:
            continue
        side_effects = frozenset(
            a04.normalize_text(effect.get("meddra_pt"))
            for effect in extraction.get("d_side_effect_pts") or []
            if isinstance(effect, dict) and effect.get("meddra_pt")
        )
        subtype = a04.normalize_text(extraction.get("c_indication_subtype") or "")
        keyed[record_id] = (arm, side_effects, subtype)
    connection.close()

    records: list[Rec] = []
    with (ROOT / "filtered_dataset" / "submissions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rid = str(row.get("name") or "t3_" + str(row.get("id")))
            if not rid.startswith("t3_"):
                rid = "t3_" + rid
            if rid not in keyed:
                continue
            arm, side_effects, subtype = keyed.pop(rid)
            author = row.get("author")
            if not author or author in ("[deleted]", "[removed]", "AutoModerator"):
                author = None
            try:
                created = float(row.get("created_utc"))
            except (TypeError, ValueError):
                created = None
            records.append(
                Rec(
                    arm,
                    side_effects,
                    subtype,
                    author,
                    str(row.get("subreddit") or "").casefold(),
                    created,
                )
            )
    if keyed:
        print(f"warning: {len(keyed):,} extraction records had no submission metadata")
    return records


def display_community(name: str) -> str:
    return "user-profile page" if name.startswith("u_") else f"r/{name}"


def summarise(
    name: str,
    records: Sequence[Rec],
    permutations: int,
    min_count: int = MIN_COUNT,
    terms_of: Any = None,
) -> dict[str, Any]:
    comparison = [
        r.comparison(None if terms_of is None else terms_of(r.side_effects)) for r in records
    ]
    prevalence = a04.side_effect_prevalence_test(comparison)
    profile = next(
        row
        for row in a04.profile_permutation_tests(
            comparison, min_count, permutations, PERMUTATION_SEED
        )
        if row["test"].endswith("common_terms")
    )
    n_medical = sum(1 for r in comparison if r.indication == "medical")
    se_medical = sum(1 for r in comparison if r.indication == "medical" and r.side_effects)
    se_cosmetic = sum(1 for r in comparison if r.indication == "cosmetic" and r.side_effects)
    return {
        "variant": name,
        "n_medical": n_medical,
        "n_cosmetic": len(comparison) - n_medical,
        "prevalence_medical": se_medical / n_medical if n_medical else None,
        "prevalence_cosmetic": se_cosmetic / (len(comparison) - n_medical)
        if len(comparison) > n_medical
        else None,
        "prevalence_difference": prevalence["effect_estimate"],
        "prevalence_ci_low": prevalence["effect_ci_low"],
        "prevalence_ci_high": prevalence["effect_ci_high"],
        "prevalence_p": prevalence["p_value"],
        "jsd": profile["statistic"],
        "jsd_terms": profile["candidate_terms"],
        "jsd_null_mean": profile["null_mean"],
        "jsd_p": profile["p_value"],
        "permutations": permutations,
        "min_count": min_count,
    }


# ---------------------------------------------------------------- leave-out ---


def leave_out(
    records: Sequence[Rec], permutations: int, family_permutations: int
) -> list[dict[str, Any]]:
    sizes = Counter(r.community for r in records)
    rows: list[dict[str, Any]] = []
    for community, _n in sizes.most_common(TOP_COMMUNITIES):
        removed = [r for r in records if r.community == community]
        kept = [r for r in records if r.community != community]
        row = summarise(f"without {display_community(community)}", kept, permutations)
        row.update(kind="community", **removed_counts(removed))
        rows.append(row)
    families = {
        "support_community": lambda c: c == "botoxsupportcommunity",
        "diy_communities": lambda c: c in {"diyaesthetics", "diycosmeticprocedures"},
        "user_profile_pages": lambda c: c.startswith("u_"),
        "noburp_and_support": lambda c: c in {"noburp", "botoxsupportcommunity"},
    }
    for key, test in families.items():
        removed = [r for r in records if test(r.community)]
        kept = [r for r in records if not test(r.community)]
        row = summarise(f"without {FAMILIES[key]}", kept, family_permutations)
        row.update(kind="family", family=key, **removed_counts(removed))
        rows.append(row)
    return rows


def removed_counts(removed: Sequence[Rec]) -> dict[str, Any]:
    med = [r for r in removed if r.arm == "medical"]
    cos = [r for r in removed if r.arm == "cosmetic"]
    return {
        "removed_medical": len(med),
        "removed_cosmetic": len(cos),
        "removed_prevalence_medical": sum(bool(r.side_effects) for r in med) / len(med)
        if med
        else None,
        "removed_prevalence_cosmetic": sum(bool(r.side_effects) for r in cos) / len(cos)
        if cos
        else None,
    }


# --------------------------------------------------------- author clustering ---


def common_terms(records: Sequence[Rec], min_count: int = MIN_COUNT) -> list[str]:
    pooled = Counter()
    for r in records:
        pooled.update(r.side_effects)
    return sorted(t for t, n in pooled.items() if n >= min_count)


def cluster_bootstrap(
    records: Sequence[Rec], replicates: int, rng: np.random.Generator
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    terms = common_terms(records)
    term_index = {t: i for i, t in enumerate(terms)}
    cluster_of: dict[str, int] = {}
    cluster_idx = np.empty(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        key = r.author if r.author is not None else f"\0single{i}"
        cluster_idx[i] = cluster_of.setdefault(key, len(cluster_of))
    n_clusters = len(cluster_of)
    is_med = np.array([r.arm == "medical" for r in records], dtype=float)
    has_se = np.array([bool(r.side_effects) for r in records], dtype=float)
    rows, cols = [], []
    for i, r in enumerate(records):
        for t in r.side_effects:
            if t in term_index:
                rows.append(i)
                cols.append(term_index[t])
    x = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(records), len(terms)))
    assign = sparse.csr_matrix(
        (np.ones(len(records)), (cluster_idx, np.arange(len(records)))),
        shape=(n_clusters, len(records)),
    )
    c_med_n = assign @ is_med
    c_cos_n = assign @ (1 - is_med)
    c_med_se = assign @ (is_med * has_se)
    c_cos_se = assign @ ((1 - is_med) * has_se)
    c_med_terms = (assign @ sparse.diags(is_med) @ x).tocsc()
    c_cos_terms = (assign @ sparse.diags(1 - is_med) @ x).tocsc()

    def statistics(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med_n, cos_n = w @ c_med_n, w @ c_cos_n
        med_se, cos_se = w @ c_med_se, w @ c_cos_se
        med_t = np.asarray((c_med_terms.T @ w.T).T)
        cos_t = np.asarray((c_cos_terms.T @ w.T).T)
        prev = med_se / med_n - cos_se / cos_n
        jsd = a04.jensen_shannon_divergence(med_t, cos_t)
        term_diff = 100 * (med_t / med_se[:, None] - cos_t / cos_se[:, None])
        return prev, jsd, term_diff

    obs_prev, obs_jsd, obs_terms = statistics(np.ones((1, n_clusters)))
    prev_b, jsd_b, terms_b = [], [], []
    batch = 50
    for start in range(0, replicates, batch):
        m = min(batch, replicates - start)
        draws = rng.integers(0, n_clusters, size=(m, n_clusters))
        w = np.vstack([np.bincount(d, minlength=n_clusters) for d in draws]).astype(float)
        p, j, t = statistics(w)
        prev_b.append(p)
        jsd_b.append(j)
        terms_b.append(t)
    prev_b = np.concatenate(prev_b)
    jsd_b = np.concatenate(jsd_b)
    terms_b = np.vstack(terms_b)

    n_med, n_cos = c_med_n.sum(), c_cos_n.sum()
    p_med, p_cos = c_med_se.sum() / n_med, c_cos_se.sum() / n_cos
    naive_se = float(np.sqrt(p_med * (1 - p_med) / n_med + p_cos * (1 - p_cos) / n_cos))
    cluster_se = float(prev_b.std(ddof=1))
    summary = {
        "clusters": n_clusters,
        "named_author_clusters": sum(1 for k in cluster_of if not k.startswith("\0")),
        "replicates": replicates,
        "prevalence_difference": float(obs_prev[0]),
        "prevalence_ci_low": float(np.percentile(prev_b, 2.5)),
        "prevalence_ci_high": float(np.percentile(prev_b, 97.5)),
        "prevalence_se_clustered": cluster_se,
        "prevalence_se_independent": naive_se,
        "design_effect": (cluster_se / naive_se) ** 2,
        "prevalence_p_clustered": float(2 * stats.norm.sf(abs(obs_prev[0]) / cluster_se)),
        "jsd": float(obs_jsd[0]),
        "jsd_terms": len(terms),
        "jsd_ci_low": float(np.percentile(jsd_b, 2.5)),
        "jsd_ci_high": float(np.percentile(jsd_b, 97.5)),
        # The plug-in divergence is biased upward under resampling, so the percentile interval sits
        # above the estimate. The basic interval (2 * estimate - percentile) reflects it back.
        "jsd_bootstrap_mean": float(jsd_b.mean()),
        "jsd_basic_ci_low": float(2 * obs_jsd[0] - np.percentile(jsd_b, 97.5)),
        "jsd_basic_ci_high": float(2 * obs_jsd[0] - np.percentile(jsd_b, 2.5)),
    }
    se_terms = terms_b.std(axis=0, ddof=1)
    z = np.divide(obs_terms[0], se_terms, out=np.zeros_like(se_terms), where=se_terms > 0)
    p_terms = 2 * stats.norm.sf(np.abs(z))
    reject, q_terms, _, _ = multipletests(p_terms, alpha=0.05, method="fdr_bh")
    lo = np.percentile(terms_b, 2.5, axis=0)
    hi = np.percentile(terms_b, 97.5, axis=0)
    term_rows = [
        {
            "candidate_pt": terms[k],
            "difference_per_100": float(obs_terms[0, k]),
            "clustered_se": float(se_terms[k]),
            "ci_low": float(lo[k]),
            "ci_high": float(hi[k]),
            "p_clustered": float(p_terms[k]),
            "q_clustered": float(q_terms[k]),
            "significant_fdr_0_05": bool(reject[k]),
        }
        for k in range(len(terms))
    ]
    summary["terms_significant_clustered_bh"] = int(reject.sum())
    summary["terms_ci_excludes_zero"] = int(((lo > 0) | (hi < 0)).sum())
    return summary, term_rows


def author_draws(records: Sequence[Rec], draws: int, rng: np.random.Generator) -> dict[str, Any]:
    terms = set(common_terms(records))
    by_author: dict[str, list[int]] = {}
    singles: list[int] = []
    for i, r in enumerate(records):
        (by_author.setdefault(r.author, []) if r.author is not None else singles).append(i)
    groups = list(by_author.values())
    diffs, lows, highs, jsds = [], [], [], []
    for _ in range(draws):
        chosen = singles + [g[int(rng.integers(len(g)))] for g in groups]
        subset = [records[i].comparison(records[i].side_effects & terms) for i in chosen]
        prevalence = a04.side_effect_prevalence_test([records[i].comparison() for i in chosen])
        diffs.append(prevalence["effect_estimate"])
        lows.append(prevalence["effect_ci_low"])
        highs.append(prevalence["effect_ci_high"])
        med = Counter()
        cos = Counter()
        for c in subset:
            (med if c.indication == "medical" else cos).update(c.side_effects)
        ordered = sorted(terms)
        jsds.append(
            float(
                a04.jensen_shannon_divergence(
                    np.array([med[t] for t in ordered]), np.array([cos[t] for t in ordered])
                )[0]
            )
        )
    diffs_a, jsds_a = np.array(diffs), np.array(jsds)
    return {
        "draws": draws,
        "records_per_draw": len(singles) + len(groups),
        "prevalence_difference_median": float(np.median(diffs_a)),
        "prevalence_difference_p2_5": float(np.percentile(diffs_a, 2.5)),
        "prevalence_difference_p97_5": float(np.percentile(diffs_a, 97.5)),
        "share_negative": float(np.mean(diffs_a < 0)),
        "share_ci_excludes_zero": float(np.mean(np.array(highs) < 0) + np.mean(np.array(lows) > 0)),
        "jsd_median": float(np.median(jsds_a)),
        "jsd_p2_5": float(np.percentile(jsds_a, 2.5)),
        "jsd_p97_5": float(np.percentile(jsds_a, 97.5)),
        "jsd_terms": len(terms),
    }


# ---------------------------------------------------------- strata and terms ---


def periods(records: Sequence[Rec], permutations: int) -> list[dict[str, Any]]:
    def year(r: Rec) -> int:
        """Calendar year, or 0 when the timestamp is missing (excluded from every stratum)."""
        return datetime.fromtimestamp(r.created, tz=UTC).year if r.created else 0

    strata = {
        "2015-2019": lambda r: 0 < year(r) <= 2019,
        "2020-2022": lambda r: 2020 <= year(r) <= 2022,
        "2023-2025": lambda r: year(r) >= 2023,
        "before July 2023": lambda r: r.created is not None and r.created < COLLECTION_CHANGE,
        "from July 2023": lambda r: r.created is not None and r.created >= COLLECTION_CHANGE,
    }
    return [
        summarise(name, [r for r in records if test(r)], permutations)
        for name, test in strata.items()
    ]


def within_site(records: Sequence[Rec], permutations: int) -> list[dict[str, Any]]:
    rows = []
    for key, (medical, cosmetic, label) in WITHIN_SITE_PAIRS.items():
        subset = [
            r
            for r in records
            if (r.arm == "medical" and r.subtype in medical)
            or (r.arm == "cosmetic" and r.subtype in cosmetic)
        ]
        n_med = sum(r.arm == "medical" for r in subset)
        reporting = [r for r in subset if r.side_effects]
        if n_med == 0 or n_med == len(subset) or len({r.arm for r in reporting}) < 2:
            rows.append(
                {
                    "variant": label,
                    "pair": key,
                    "n_medical": n_med,
                    "n_cosmetic": len(subset) - n_med,
                    "note": "too few records",
                }
            )
            continue
        try:
            row = summarise(label, subset, permutations, min_count=MIN_COUNT_SMALL)
        except ValueError:  # no term reaches the threshold in one of the two groups
            rows.append(
                {
                    "variant": label,
                    "pair": key,
                    "n_medical": n_med,
                    "n_cosmetic": len(subset) - n_med,
                    "note": f"no term in at least {MIN_COUNT_SMALL} reports in both groups",
                }
            )
            continue
        row["pair"] = key
        row["reporting_medical"] = sum(r.arm == "medical" for r in reporting)
        row["reporting_cosmetic"] = sum(r.arm == "cosmetic" for r in reporting)
        rows.append(row)
    return rows


def merge_synonyms(terms: frozenset[str]) -> frozenset[str]:
    return frozenset(SYNONYMS.get(t, t) for t in terms)


def merge_and_exclude(terms: frozenset[str]) -> frozenset[str]:
    return frozenset(t for t in merge_synonyms(terms) if t not in SYMMETRIC_EXCLUSIONS)


def vocabulary(records: Sequence[Rec], permutations: int) -> list[dict[str, Any]]:
    return [
        summarise("synonym merges", records, permutations, terms_of=merge_synonyms),
        summarise(
            "synonym merges and symmetric exclusions",
            records,
            permutations,
            terms_of=merge_and_exclude,
        ),
    ]


# ------------------------------------------------------------------- output ---


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pp(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f}"


def markdown(result: dict[str, Any]) -> list[str]:
    lines = [
        "# Robustness analyses (08)",
        "",
        "## Leave-out analyses",
        "",
        "| Variant | Removed med/cos | Prevalence diff pp (95% CI) | JSD (terms) | JSD p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in result["loco"]:
        lines.append(
            f"| {r['variant']} | {r['removed_medical']:,}/{r['removed_cosmetic']:,} | {pp(r['prevalence_difference'])} "
            f"({pp(r['prevalence_ci_low'])} to {pp(r['prevalence_ci_high'])}) | {r['jsd']:.3f} ({r['jsd_terms']}) | {r['jsd_p']:.4g} |"
        )
    c = result["cluster"]
    lines += [
        "",
        "## Author-clustered bootstrap",
        "",
        f"{c['clusters']:,} clusters ({c['named_author_clusters']:,} named authors), {c['replicates']:,} replicates.",
        f"Prevalence difference {pp(c['prevalence_difference'])} pp, clustered 95% CI {pp(c['prevalence_ci_low'])} to {pp(c['prevalence_ci_high'])}; "
        f"design effect {c['design_effect']:.2f}; clustered P {c['prevalence_p_clustered']:.3g}.",
        f"JSD {c['jsd']:.3f} ({c['jsd_terms']} terms), clustered 95% CI {c['jsd_ci_low']:.3f} to {c['jsd_ci_high']:.3f}.",
        f"Per-term: {c['terms_significant_clustered_bh']} of {c['jsd_terms']} significant at BH q<0.05 with clustered SEs; "
        f"{c['terms_ci_excludes_zero']} with a clustered percentile CI excluding zero.",
    ]
    d = result["author_draws"]
    lines += [
        "",
        "## Random one-submission-per-author draws",
        "",
        f"{d['draws']} draws of {d['records_per_draw']:,} records: prevalence difference median {pp(d['prevalence_difference_median'])} pp "
        f"(2.5-97.5%: {pp(d['prevalence_difference_p2_5'])} to {pp(d['prevalence_difference_p97_5'])}); negative in {100 * d['share_negative']:.1f}% of draws; "
        f"CI excludes zero in {100 * d['share_ci_excludes_zero']:.1f}%. JSD median {d['jsd_median']:.3f} ({d['jsd_p2_5']:.3f} to {d['jsd_p97_5']:.3f}).",
    ]
    for title, key in (
        ("Calendar periods", "periods"),
        ("Within-site contrasts", "within_site"),
        ("Vocabulary variants", "vocabulary"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Variant | Med/cos n | Prevalence diff pp (95% CI) | JSD (terms) | JSD p |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in result[key]:
            if "note" in r:
                lines.append(
                    f"| {r['variant']} | {r['n_medical']:,}/{r['n_cosmetic']:,} | {r['note']} | | |"
                )
                continue
            lines.append(
                f"| {r['variant']} | {r['n_medical']:,}/{r['n_cosmetic']:,} | {pp(r['prevalence_difference'])} "
                f"({pp(r['prevalence_ci_low'])} to {pp(r['prevalence_ci_high'])}) | {r['jsd']:.3f} ({r['jsd_terms']}) | {r['jsd_p']:.4g} |"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--permutations", type=int, default=10_000, help="for family, vocabulary and baseline tests"
    )
    parser.add_argument(
        "--stratum-permutations",
        type=int,
        default=2_000,
        help="for single-community, period and site strata",
    )
    parser.add_argument(
        "--replicates", type=int, default=2_000, help="author-clustered bootstrap replicates"
    )
    parser.add_argument("--draws", type=int, default=200, help="random one-per-author draws")
    args = parser.parse_args()

    records = load()
    print(f"loaded {len(records):,} comparison records")
    rng = np.random.default_rng(RESAMPLING_SEED)

    result: dict[str, Any] = {
        "method": {
            "stage": STAGE,
            "permutation_seed": PERMUTATION_SEED,
            "resampling_seed": RESAMPLING_SEED,
            "permutations": args.permutations,
            "stratum_permutations": args.stratum_permutations,
            "bootstrap_replicates": args.replicates,
            "author_draws": args.draws,
            "synonyms": SYNONYMS,
            "symmetric_exclusions": sorted(SYMMETRIC_EXCLUSIONS),
            "within_site_pairs": {
                k: [sorted(m), sorted(c), label] for k, (m, c, label) in WITHIN_SITE_PAIRS.items()
            },
        },
        "baseline": summarise("baseline", records, args.permutations),
    }
    print(
        "baseline",
        {k: result["baseline"][k] for k in ("prevalence_difference", "jsd", "jsd_terms")},
    )
    result["loco"] = leave_out(records, args.stratum_permutations, args.permutations)
    print("leave-out done")
    result["cluster"], cluster_terms = cluster_bootstrap(records, args.replicates, rng)
    print("cluster bootstrap done")
    result["author_draws"] = author_draws(records, args.draws, rng)
    print("author draws done")
    result["periods"] = periods(records, args.stratum_permutations)
    result["within_site"] = within_site(records, args.stratum_permutations)
    result["vocabulary"] = vocabulary(records, args.permutations)

    (ANALYSIS_DIR / "08_robustness_analyses.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(ANALYSIS_DIR / "08_loco.csv", result["loco"])
    write_csv(ANALYSIS_DIR / "08_periods.csv", result["periods"])
    write_csv(ANALYSIS_DIR / "08_within_site.csv", result["within_site"])
    write_csv(ANALYSIS_DIR / "08_vocabulary.csv", result["vocabulary"])
    write_csv(ANALYSIS_DIR / "08_cluster_terms.csv", cluster_terms)
    lines = markdown(result)
    (ANALYSIS_DIR / "08_robustness_analyses.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
