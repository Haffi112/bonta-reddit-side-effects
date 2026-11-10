"""Sensitivity analyses for the medical/cosmetic comparison.

Three concerns could distort the primary comparison: repeat posters supply
47.0% of submissions; r/noburp contributes a large, unusual block of
medical-use records; and the candidate terms are unvalidated model proposals
in which indication-shaped labels and near-synonyms occur. Each concern gets
one variant of the primary analysis, recomputed with the same machinery as
``04_medical_cosmetic_comparison.py``:

* ``baseline``      -- all stage-4 self-use records with a known indication.
* ``author_dedup``  -- one submission per named author, chosen as the
  lexicographically smallest record id; records whose author is deleted or a
  bot are kept as single observations because they cannot be linked.
* ``no_noburp``     -- baseline minus submissions posted in r/noburp.
* ``term_validity`` -- baseline after excluding indication- and
  failure-shaped terms (migraine, migraine aggravated, condition aggravated,
  pain aggravated, treatment failure, drug ineffective, therapeutic response
  decreased) and merging near-synonyms (all ptosis variants; regurgitation
  with reflux disease).

For every variant the script reports the prevalence difference with its
Newcombe--Wilson interval and the 20-record-threshold Jensen--Shannon
permutation test. Outputs: ``07_sensitivity_analyses.{csv,json,md}``.

Example:

    uv run python data_analysis/07_sensitivity_analyses.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

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
SEED = 20260715
MIN_COUNT = 20

EXCLUDED_TERMS = {
    "migraine",
    "migraine aggravated",
    "condition aggravated",
    "pain aggravated",
    "treatment failure",
    "drug ineffective",
    "therapeutic response decreased",
}
MERGED_TERMS = {
    "ptosis": "ptosis (any)",
    "eyelid ptosis": "ptosis (any)",
    "eyebrow ptosis": "ptosis (any)",
    "brow ptosis": "ptosis (any)",
    "regurgitation": "reflux or regurgitation",
    "gastroesophageal reflux disease": "reflux or regurgitation",
    "gastrooesophageal reflux disease": "reflux or regurgitation",
}


def load_records_with_meta(
    connection: sqlite3.Connection,
) -> list[tuple[str, a04.ComparisonRecord]]:
    """Stage-4 comparison records keyed by record id."""
    records: list[tuple[str, a04.ComparisonRecord]] = []
    rows = connection.execute(
        "SELECT record_id, extraction_json FROM stage_results WHERE stage = ?",
        (STAGE,),
    )
    for record_id, extraction_json in rows:
        extraction = json.loads(extraction_json)
        if not extraction.get("a_self_use"):
            continue
        indication = a04.normalize_text(extraction.get("c_indication_category"))
        if indication not in a04.GROUPS:
            continue
        side_effects = frozenset(
            a04.normalize_text(effect.get("meddra_pt"))
            for effect in extraction.get("d_side_effect_pts") or []
            if isinstance(effect, dict) and effect.get("meddra_pt")
        )
        records.append(
            (record_id, a04.ComparisonRecord(indication=indication, side_effects=side_effects))
        )
    return records


def load_author_subreddit() -> dict[str, tuple[str | None, str]]:
    meta: dict[str, tuple[str | None, str]] = {}
    with (ROOT / "filtered_dataset" / "submissions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            rid = str(record.get("name") or "t3_" + str(record.get("id")))
            if not rid.startswith("t3_"):
                rid = "t3_" + rid
            author = record.get("author")
            if not author or author in ("[deleted]", "[removed]", "AutoModerator"):
                author = None
            meta[rid] = (author, str(record.get("subreddit") or "").casefold())
    return meta


def clean_terms(record: a04.ComparisonRecord) -> a04.ComparisonRecord:
    remapped = frozenset(
        MERGED_TERMS.get(term, term) for term in record.side_effects if term not in EXCLUDED_TERMS
    )
    return a04.ComparisonRecord(indication=record.indication, side_effects=remapped)


def analyse(name: str, records: list[a04.ComparisonRecord], permutations: int) -> dict[str, Any]:
    prevalence = a04.side_effect_prevalence_test(records)
    jsd_rows = a04.profile_permutation_tests(records, MIN_COUNT, permutations, SEED)
    common = next(r for r in jsd_rows if r["test"].endswith("common_terms"))
    n_med = sum(1 for r in records if r.indication == "medical")
    n_cos = len(records) - n_med
    return {
        "variant": name,
        "n_records": len(records),
        "n_medical": n_med,
        "n_cosmetic": n_cos,
        "prevalence_difference": prevalence["effect_estimate"],
        "prevalence_ci_low": prevalence["effect_ci_low"],
        "prevalence_ci_high": prevalence["effect_ci_high"],
        "prevalence_p": prevalence["p_value"],
        "jsd_common_terms": common["statistic"],
        "jsd_terms": common["candidate_terms"],
        "jsd_p": common["p_value"],
        "jsd_permutations": permutations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    connection = sqlite3.connect(
        f"file:{RESULTS_DIR / 'private_extraction.sqlite3'}?mode=ro", uri=True
    )
    keyed = load_records_with_meta(connection)
    connection.close()
    meta = load_author_subreddit()

    baseline = [record for _rid, record in keyed]

    unattributable: list[a04.ComparisonRecord] = []
    by_author: dict[str, tuple[str, a04.ComparisonRecord]] = {}
    for rid, record in keyed:
        author = meta.get(rid, (None, ""))[0]
        if author is None:
            unattributable.append(record)
        elif author not in by_author or rid < by_author[author][0]:
            by_author[author] = (rid, record)
    author_dedup = [record for _rid, record in by_author.values()] + unattributable

    no_noburp = [record for rid, record in keyed if meta.get(rid, (None, ""))[1] != "noburp"]
    term_validity = [clean_terms(record) for record in baseline]

    results = [
        analyse("baseline", baseline, args.permutations),
        analyse("author_dedup", author_dedup, args.permutations),
        analyse("no_noburp", no_noburp, args.permutations),
        analyse("term_validity", term_validity, args.permutations),
    ]

    with (ANALYSIS_DIR / "07_sensitivity_analyses.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (ANALYSIS_DIR / "07_sensitivity_analyses.json").write_text(
        json.dumps(
            {
                "excluded_terms": sorted(EXCLUDED_TERMS),
                "merged_terms": MERGED_TERMS,
                "variants": results,
            },
            indent=2,
        )
    )

    lines = [
        "# Sensitivity Analyses",
        "",
        "| Variant | Records (med/cos) | Prevalence diff (95% CI) | JSD (terms) | JSD p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['variant']} | {row['n_medical']:,} / {row['n_cosmetic']:,} "
            f"| {100 * row['prevalence_difference']:.2f} pp "
            f"({100 * row['prevalence_ci_low']:.2f} to {100 * row['prevalence_ci_high']:.2f}) "
            f"| {row['jsd_common_terms']:.3f} ({row['jsd_terms']}) "
            f"| {row['jsd_p']:.4f} |"
        )
    (ANALYSIS_DIR / "07_sensitivity_analyses.md").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
