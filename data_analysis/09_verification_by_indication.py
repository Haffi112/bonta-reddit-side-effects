"""Human verification broken down by the indication group the model assigned.

The pooled accuracies of ``05`` cannot show whether extraction error differs
between the medical and cosmetic groups, which is what a between-group
comparison needs. This script re-reads the same blinded annotations and, for
each model, restricts to outputs the model classified as self-use and splits
them by the assigned indication (medical, cosmetic, unknown). Per group it
reports:

* majority-vote accuracy of the self-use, indication-category and
  side-effect-list fields;
* side-effect presence as a two-by-two table against the annotators' verdicts:
  a non-empty list with at least one span judged correct is a true positive,
  a non-empty list whose every span was judged incorrect a false positive, an
  empty list judged incorrect (a missed side effect) a false negative, and an
  empty list judged correct a true negative;
* sensitivity and positive predictive value for side-effect presence, with
  Wilson intervals, and the human-judged prevalence in the sample.

It then carries the per-group sensitivity and positive predictive value of the
primary model over to the full-corpus counts in ``04_group_summary.csv`` as a
simple misclassification correction (corrected prevalence = observed
prevalence x PPV / sensitivity), with a percentile bootstrap over verified
outputs within each group for the interval.

The verification arm is the interactive GPT-5.6 Luna run (``stage3``); its
batch counterpart (``stage4``) agreed with it on self-use and side-effect
presence in 98.5% and 99.1% of the same records. Only aggregates are written;
no source text, notes, record identifiers or model values leave the guarded
directory.

Outputs: ``09_verification_by_indication.{csv,json,md}``.

Example:

    uv run python data_analysis/09_verification_by_indication.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
EXPORT_DIR = ANALYSIS_DIR / "02_llm_extraction_results"
SEED = 20260927
REPLICATES = 2_000
GROUPS = ("medical", "cosmetic", "unknown")


def _load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


a05 = _load_module("a05", "05_annotation_analysis.py")
_SPAN = re.compile(r"^d_side_effect_pts\[(\d+)\]\.raw_span$")


def load_outputs() -> dict[int, dict[str, Any]]:
    """Per verified model output: model, assigned group, and majority verdicts."""
    decisions, annotators, _notes = a05.load_annotations(EXPORT_DIR)
    n_raters = len(annotators)
    votes: dict[tuple[int, str], list[bool]] = defaultdict(list)
    model_of: dict[int, str] = {}
    for d in decisions:
        votes[(d["model_output_id"], d["key"])].append(d["correct"])
        model_of[d["model_output_id"]] = d["model_key"]
    values = a05.load_model_values(EXPORT_DIR)

    outputs: dict[int, dict[str, Any]] = {}
    for output_id, model in model_of.items():
        value = values[output_id]
        if not value.get("a_self_use"):
            continue
        group = str(value.get("c_indication_category") or "unknown").casefold().strip()
        group = group if group in GROUPS else "unknown"

        def majority(key: str, output_id: int = output_id) -> bool | None:
            ratings = votes.get((output_id, key))
            if ratings is None:
                return None
            if len(ratings) != n_raters:
                raise SystemExit(f"unit without a full set of ratings: {key}")
            return sum(ratings) * 2 > n_raters

        spans = [majority(key) for (oid, key) in votes if oid == output_id and _SPAN.fullmatch(key)]
        outputs[output_id] = {
            "model": model,
            "group": group,
            "self_use_correct": majority("a_self_use"),
            "indication_correct": majority("c_indication_category"),
            "list_correct": majority("d_side_effect_pts"),
            "listed": bool(value.get("d_side_effect_pts")),
            "any_span_correct": any(s for s in spans if s is not None),
        }
    return outputs


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    return a05.wilson_interval(k, n)


def cell(outputs: list[dict[str, Any]]) -> dict[str, int]:
    tp = sum(1 for o in outputs if o["listed"] and o["any_span_correct"])
    fp = sum(1 for o in outputs if o["listed"] and not o["any_span_correct"])
    fn = sum(1 for o in outputs if not o["listed"] and o["list_correct"] is False)
    tn = sum(1 for o in outputs if not o["listed"] and o["list_correct"] is True)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def group_row(model: str, group: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(outputs)
    c = cell(outputs)
    row: dict[str, Any] = {"model": model, "group": group, "n_self_use_outputs": n, **c}
    for field in ("self_use_correct", "indication_correct", "list_correct"):
        k = sum(1 for o in outputs if o[field])
        lo, hi = wilson(k, n)
        row[f"{field}_rate"] = k / n if n else None
        row[f"{field}_ci_low"], row[f"{field}_ci_high"] = lo, hi
    sens_n, ppv_n = c["tp"] + c["fn"], c["tp"] + c["fp"]
    row["sensitivity"] = c["tp"] / sens_n if sens_n else None
    row["sensitivity_ci_low"], row["sensitivity_ci_high"] = wilson(c["tp"], sens_n)
    row["ppv"] = c["tp"] / ppv_n if ppv_n else None
    row["ppv_ci_low"], row["ppv_ci_high"] = wilson(c["tp"], ppv_n)
    row["observed_prevalence_sample"] = ppv_n / n if n else None
    row["human_judged_prevalence_sample"] = sens_n / n if n else None
    return row


def corrected_difference(
    luna: dict[str, list[dict[str, Any]]], corpus: dict[str, float], rng: np.random.Generator
) -> dict[str, Any]:
    def factor(outputs: list[dict[str, Any]]) -> float:
        c = cell(outputs)
        return (c["tp"] / (c["tp"] + c["fp"])) / (c["tp"] / (c["tp"] + c["fn"]))

    point = {g: corpus[g] * factor(luna[g]) for g in ("medical", "cosmetic")}
    draws = []
    for _ in range(REPLICATES):
        resampled = {
            g: [luna[g][i] for i in rng.integers(0, len(luna[g]), len(luna[g]))]
            for g in ("medical", "cosmetic")
        }
        try:
            draws.append(
                corpus["medical"] * factor(resampled["medical"])
                - corpus["cosmetic"] * factor(resampled["cosmetic"])
            )
        except ZeroDivisionError:
            continue
    draws_a = np.array(draws)
    return {
        "observed_prevalence_medical": corpus["medical"],
        "observed_prevalence_cosmetic": corpus["cosmetic"],
        "observed_difference": corpus["medical"] - corpus["cosmetic"],
        "corrected_prevalence_medical": point["medical"],
        "corrected_prevalence_cosmetic": point["cosmetic"],
        "corrected_difference": point["medical"] - point["cosmetic"],
        "corrected_difference_ci_low": float(np.percentile(draws_a, 2.5)),
        "corrected_difference_ci_high": float(np.percentile(draws_a, 97.5)),
        "share_of_replicates_negative": float(np.mean(draws_a < 0)),
        "replicates_used": len(draws_a),
    }


def main() -> None:
    outputs = load_outputs()
    by_model_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in outputs.values():
        by_model_group[(o["model"], o["group"])].append(o)
    rows = [
        group_row(model, group, by_model_group[(model, group)])
        for model in a05.MODEL_ORDER
        for group in GROUPS
    ]
    corpus = {}
    with (ANALYSIS_DIR / "04_group_summary.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            corpus[r["indication"]] = float(r["side_effect_report_prevalence"])
    luna = {g: by_model_group[("stage3", g)] for g in GROUPS}
    correction = corrected_difference(luna, corpus, np.random.default_rng(SEED))

    with (ANALYSIS_DIR / "09_verification_by_indication.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ANALYSIS_DIR / "09_verification_by_indication.json").write_text(
        json.dumps(
            {
                "groups": rows,
                "luna_misclassification_correction": correction,
                "seed": SEED,
                "replicates": REPLICATES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}"

    lines = [
        "# Verification by assigned indication (09)",
        "",
        "| Model | Group | n | Self-use ok | Indication ok | List ok | TP/FP/FN/TN | Sensitivity (95% CI) | PPV (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['group']} | {r['n_self_use_outputs']} | {pct(r['self_use_correct_rate'])} "
            f"| {pct(r['indication_correct_rate'])} ({pct(r['indication_correct_ci_low'])}-{pct(r['indication_correct_ci_high'])}) "
            f"| {pct(r['list_correct_rate'])} | {r['tp']}/{r['fp']}/{r['fn']}/{r['tn']} "
            f"| {pct(r['sensitivity'])} ({pct(r['sensitivity_ci_low'])}-{pct(r['sensitivity_ci_high'])}) "
            f"| {pct(r['ppv'])} ({pct(r['ppv_ci_low'])}-{pct(r['ppv_ci_high'])}) |"
        )
    c = correction
    lines += [
        "",
        "## Misclassification-corrected full-corpus prevalence (GPT-5.6 Luna)",
        "",
        f"Observed: medical {pct(c['observed_prevalence_medical'])}%, cosmetic {pct(c['observed_prevalence_cosmetic'])}%, "
        f"difference {100 * c['observed_difference']:+.2f} pp.",
        f"Corrected: medical {pct(c['corrected_prevalence_medical'])}%, cosmetic {pct(c['corrected_prevalence_cosmetic'])}%, "
        f"difference {100 * c['corrected_difference']:+.2f} pp (bootstrap 95% CI {100 * c['corrected_difference_ci_low']:+.2f} "
        f"to {100 * c['corrected_difference_ci_high']:+.2f}; negative in {100 * c['share_of_replicates_negative']:.1f}% of "
        f"{c['replicates_used']:,} replicates).",
    ]
    (ANALYSIS_DIR / "09_verification_by_indication.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
