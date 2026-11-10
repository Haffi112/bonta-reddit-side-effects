"""Analyze the human annotations from the blinded evaluation interface.

Three annotators independently marked every field of both blinded model
outputs (DeepSeek V4 Flash no reasoning, ``stage1``; GPT-5.6 Luna low
reasoning, ``stage3``) for 1,168 submissions as correct or incorrect.
This script computes, per model and per canonical field:

* pooled and per-annotator marked-correct rates with Wilson intervals;
* majority-vote (2 of 3) human-judged accuracy with Wilson intervals;
* inter-annotator reliability (unanimous rate, pairwise agreement,
  Fleiss' kappa) on units rated by every complete annotator;
* a paired McNemar comparison of majority-vote accuracy between the two
  models on the top-level fields shared by every form.

Inputs are the private database exports in
``data_analysis/02_llm_extraction_results/`` (annotations_export.csv,
model_outputs_export.csv). Only aggregate statistics are written; no
source text, notes, or model values leave the guarded directory.

Outputs:

* ``05_annotation_analysis.json`` and ``.md``
* ``05_field_correctness.csv``, ``05_annotator_agreement.csv``,
  ``05_model_comparison.csv``
* figures ``05_01_annotation_correctness`` and ``05_02_annotator_agreement``

Example:

    uv run python data_analysis/05_annotation_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.figure import Figure
from scipy import stats  # type: ignore[import-untyped]

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = ANALYSIS_DIR / "02_llm_extraction_results"
DEFAULT_FIGURES_DIR = ANALYSIS_DIR / "figures"

MODEL_ORDER = ("stage1", "stage3")
MODEL_NAMES = {
    "stage1": "DeepSeek V4 Flash, no reasoning",
    "stage3": "GPT-5.6 Luna, low reasoning",
}

# Canonical field order and labels, mirroring evaluation_interface fields.py.
FIELD_LABELS: dict[str, str] = {
    "a_self_use": "Personal use",
    "b_products_mentioned": "Products mentioned",
    "c_indication_category": "Indication category",
    "c_indication_subtype": "Indication subtype",
    "d_side_effect_pts": "Side-effect list",
    "d_side_effect_pts[*].raw_span": "Verbatim side-effect span",
    "d_side_effect_pts[*].meddra_pt": "Candidate MedDRA term",
    "e1_post_overall_sentiment": "Overall post sentiment",
    "e2_sentiment_about_botox": "Attitude toward BoNT-A",
    "f_sentiment_about_side_effect": "Attitude toward side effect",
    "g_treatment_area_freetext": "Treatment area",
    "h_dose_units": "Dose (units)",
    "i_provider_type": "Provider type",
    "j_first_time_or_repeat": "First or repeat treatment",
    "j_prior_treatment_count": "Prior treatment count",
    "k_time_to_onset": "Time to onset",
    "l_duration_status": "Duration status",
    "m_sought_medical_attention": "Sought medical attention",
    "n_discontinuation_intent": "Discontinuation intent",
    "o_age_range": "Age range",
    "o_sex": "Sex",
}
FIELD_ORDER = tuple(FIELD_LABELS)
_SIDE_EFFECT_CHILD = re.compile(r"^d_side_effect_pts\[\d+\]\.(raw_span|meddra_pt)$")


def canonical_key(key: str) -> str:
    match = _SIDE_EFFECT_CHILD.fullmatch(key)
    return f"d_side_effect_pts[*].{match.group(1)}" if match else key


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fleiss_kappa(unit_correct_counts: list[int], n_raters: int) -> float:
    """Binary Fleiss' kappa from per-unit counts of 'correct' ratings."""
    n_units = len(unit_correct_counts)
    if n_units == 0 or n_raters < 2:
        return float("nan")
    total = n_units * n_raters
    p_correct = sum(unit_correct_counts) / total
    p_incorrect = 1.0 - p_correct
    pe = p_correct * p_correct + p_incorrect * p_incorrect
    po = sum((c * c + (n_raters - c) * (n_raters - c) - n_raters) for c in unit_correct_counts) / (
        n_units * n_raters * (n_raters - 1)
    )
    if math.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1.0 - pe)


def pairwise_agreement(unit_correct_counts: list[int], n_raters: int) -> float:
    """Fraction of agreeing rater pairs across units."""
    if not unit_correct_counts:
        return float("nan")
    pairs_per_unit = n_raters * (n_raters - 1) / 2
    agree = 0.0
    for c in unit_correct_counts:
        d = n_raters - c
        agree += (c * (c - 1) + d * (d - 1)) / 2
    return agree / (pairs_per_unit * len(unit_correct_counts))


def load_annotations(export_dir: Path) -> tuple[list[dict], list[str], dict[str, int]]:
    """Return decision rows from complete annotators, their labels, and note counts."""
    path = export_dir / "annotations_export.csv"
    per_annotator_outputs: dict[str, set[int]] = defaultdict(set)
    raw_rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["model_output_id"] = int(row["model_output_id"])
            raw_rows.append(row)
            per_annotator_outputs[row["annotator"]].add(row["model_output_id"])

    n_outputs = max(len(v) for v in per_annotator_outputs.values())
    complete = sorted(
        label for label, seen in per_annotator_outputs.items() if len(seen) == n_outputs
    )
    excluded = {
        label: len(seen) for label, seen in per_annotator_outputs.items() if label not in complete
    }
    if excluded:
        print(f"excluding incomplete annotators (test accounts): {excluded}")

    note_counts: dict[str, int] = Counter()
    decisions: list[dict] = []
    for row in raw_rows:
        if row["annotator"] not in complete:
            continue
        if row["notes"].strip():
            note_counts[row["annotator"]] += 1
        parsed = json.loads(row["decisions_json"])
        for key, decision in parsed.items():
            decisions.append(
                {
                    "annotator": row["annotator"],
                    "record_id": row["record_id"],
                    "model_key": row["model_key"],
                    "model_output_id": row["model_output_id"],
                    "key": key,
                    "canonical": canonical_key(key),
                    "correct": decision == "correct",
                }
            )
    return decisions, complete, dict(note_counts)


# Fields whose value may legitimately be null or empty; conditional accuracy
# is computed on the subset of outputs where the model extracted an
# informative (non-null, non-empty) value, so sparse fields are not scored
# mostly on correct nulls.
CONDITIONAL_FIELDS = (
    "b_products_mentioned",
    "c_indication_subtype",
    "d_side_effect_pts",
    "f_sentiment_about_side_effect",
    "g_treatment_area_freetext",
    "h_dose_units",
    "i_provider_type",
    "j_first_time_or_repeat",
    "j_prior_treatment_count",
    "k_time_to_onset",
    "l_duration_status",
    "m_sought_medical_attention",
    "n_discontinuation_intent",
    "o_age_range",
    "o_sex",
)


def load_model_values(export_dir: Path) -> dict[int, dict[str, object]]:
    values: dict[int, dict[str, object]] = {}
    with (export_dir / "model_outputs_export.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            values[int(row["model_output_id"])] = json.loads(row["extraction_json"])
    return values


def is_informative(value: object) -> bool:
    if value is None:
        return False
    return not (isinstance(value, (list, str)) and len(value) == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    decisions, annotators, note_counts = load_annotations(args.export_dir)
    n_raters = len(annotators)
    print(f"complete annotators: {annotators}")
    print(f"decision rows: {len(decisions):,}")

    # Index ratings by unit: one unit is one field decision on one model output.
    unit_ratings: dict[tuple[int, str], dict[str, bool]] = defaultdict(dict)
    unit_meta: dict[tuple[int, str], tuple[str, str, str]] = {}
    for d in decisions:
        unit = (d["model_output_id"], d["key"])
        unit_ratings[unit][d["annotator"]] = d["correct"]
        unit_meta[unit] = (d["model_key"], d["canonical"], d["record_id"])

    incomplete_units = sum(1 for r in unit_ratings.values() if len(r) != n_raters)
    if incomplete_units:
        sys.exit(f"FATAL: {incomplete_units} units lack a rating from every annotator")

    # ---- per model x canonical field: pooled, per-annotator, majority-vote ----
    field_rows: list[dict] = []
    majority_by_unit: dict[tuple[int, str], bool] = {}
    for unit, ratings in unit_ratings.items():
        majority_by_unit[unit] = sum(ratings.values()) * 2 > n_raters

    for model in MODEL_ORDER:
        for canon in FIELD_ORDER:
            units = [u for u, meta in unit_meta.items() if meta[0] == model and meta[1] == canon]
            if not units:
                continue
            pooled_n = len(units) * n_raters
            pooled_correct = sum(sum(unit_ratings[u].values()) for u in units)
            maj_correct = sum(majority_by_unit[u] for u in units)
            lo, hi = wilson_interval(maj_correct, len(units))
            plo, phi = wilson_interval(pooled_correct, pooled_n)
            row = {
                "model": model,
                "field": canon,
                "label": FIELD_LABELS[canon],
                "n_units": len(units),
                "pooled_correct_rate": pooled_correct / pooled_n,
                "pooled_ci_low": plo,
                "pooled_ci_high": phi,
                "majority_correct_rate": maj_correct / len(units),
                "majority_ci_low": lo,
                "majority_ci_high": hi,
            }
            for annotator in annotators:
                ok = sum(unit_ratings[u][annotator] for u in units)
                row[f"rate_{annotator}"] = ok / len(units)
            field_rows.append(row)

    # ---- overall per model ----
    overall: dict[str, dict] = {}
    for model in MODEL_ORDER:
        units = [u for u, meta in unit_meta.items() if meta[0] == model]
        pooled_correct = sum(sum(unit_ratings[u].values()) for u in units)
        maj = sum(majority_by_unit[u] for u in units)
        counts = [sum(unit_ratings[u].values()) for u in units]
        overall[model] = {
            "name": MODEL_NAMES[model],
            "n_units": len(units),
            "pooled_correct_rate": pooled_correct / (len(units) * n_raters),
            "majority_correct_rate": maj / len(units),
            "majority_ci": wilson_interval(maj, len(units)),
            "unanimous_rate": sum(1 for c in counts if c in (0, n_raters)) / len(units),
            "fleiss_kappa": fleiss_kappa(counts, n_raters),
        }

    # ---- agreement per canonical field, pooled across both models ----
    agreement_rows: list[dict] = []
    for canon in FIELD_ORDER:
        units = [u for u, meta in unit_meta.items() if meta[1] == canon]
        if not units:
            continue
        counts = [sum(unit_ratings[u].values()) for u in units]
        agreement_rows.append(
            {
                "field": canon,
                "label": FIELD_LABELS[canon],
                "n_units": len(units),
                "unanimous_rate": sum(1 for c in counts if c in (0, len(annotators))) / len(units),
                "pairwise_agreement": pairwise_agreement(counts, n_raters),
                "fleiss_kappa": fleiss_kappa(counts, n_raters),
                "prevalence_correct": sum(counts) / (len(units) * n_raters),
            }
        )
    all_counts = [sum(r.values()) for r in unit_ratings.values()]
    agreement_overall = {
        "n_units": len(all_counts),
        "unanimous_rate": sum(1 for c in all_counts if c in (0, n_raters)) / len(all_counts),
        "pairwise_agreement": pairwise_agreement(all_counts, n_raters),
        "fleiss_kappa": fleiss_kappa(all_counts, n_raters),
    }

    # ---- paired model comparison on shared top-level fields (McNemar) ----
    comparison_rows: list[dict] = []
    top_level = [f for f in FIELD_ORDER if "[*]" not in f]
    maj_by_record: dict[str, dict[tuple[str, str], bool]] = {m: {} for m in MODEL_ORDER}
    for unit, meta in unit_meta.items():
        model, canon, record_id = meta
        if canon in top_level and unit[1] == canon:  # top-level keys are unindexed
            maj_by_record[model][(record_id, canon)] = majority_by_unit[unit]
    for canon in top_level:
        keys = [
            k for k in maj_by_record["stage1"] if k[1] == canon and k in maj_by_record["stage3"]
        ]
        b = sum(1 for k in keys if maj_by_record["stage1"][k] and not maj_by_record["stage3"][k])
        c = sum(1 for k in keys if not maj_by_record["stage1"][k] and maj_by_record["stage3"][k])
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue * 1.0 if (b + c) else float("nan")
        comparison_rows.append(
            {
                "field": canon,
                "label": FIELD_LABELS[canon],
                "n_paired": len(keys),
                "stage1_majority_rate": sum(maj_by_record["stage1"][k] for k in keys) / len(keys),
                "stage3_majority_rate": sum(maj_by_record["stage3"][k] for k in keys) / len(keys),
                "stage1_only_correct": b,
                "stage3_only_correct": c,
                "mcnemar_p": p,
            }
        )

    # ---- conditional accuracy: informative values only ----
    model_values = load_model_values(args.export_dir)
    conditional_rows: list[dict] = []
    for model in MODEL_ORDER:
        for field in CONDITIONAL_FIELDS:
            units = [
                u
                for u, meta in unit_meta.items()
                if meta[0] == model
                and meta[1] == field
                and u[1] == field  # top-level key only
                and is_informative(model_values[u[0]].get(field))
            ]
            if not units:
                continue
            correct = sum(majority_by_unit[u] for u in units)
            lo, hi = wilson_interval(correct, len(units))
            conditional_rows.append(
                {
                    "model": model,
                    "field": field,
                    "label": FIELD_LABELS[field],
                    "n_informative": len(units),
                    "majority_correct_rate": correct / len(units),
                    "majority_ci_low": lo,
                    "majority_ci_high": hi,
                }
            )

    # ---- write outputs ----
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(ANALYSIS_DIR / "05_field_correctness.csv", field_rows)
    write_csv(ANALYSIS_DIR / "05_annotator_agreement.csv", agreement_rows)
    write_csv(ANALYSIS_DIR / "05_model_comparison.csv", comparison_rows)
    write_csv(ANALYSIS_DIR / "05_conditional_correctness.csv", conditional_rows)

    summary = {
        "n_annotators": n_raters,
        "n_submissions": len({m[2] for m in unit_meta.values()}),
        "n_forms_per_annotator": len({u[0] for u in unit_ratings}),
        "n_decisions": len(decisions),
        "notes_with_text_per_annotator": note_counts,
        "overall": overall,
        "agreement_overall": agreement_overall,
        "field_correctness": field_rows,
        "agreement_by_field": agreement_rows,
        "model_comparison": comparison_rows,
        "conditional_correctness_informative_values": conditional_rows,
    }
    (ANALYSIS_DIR / "05_annotation_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    write_markdown(summary)
    make_figures(field_rows, agreement_rows, args.figures_dir)
    print("done")


def write_markdown(summary: dict) -> None:
    lines: list[str] = []
    lines.append("# Human Annotation Analysis")
    lines.append("")
    lines.append(
        f"Three annotators each rated every field of both blinded model outputs for "
        f"{summary['n_submissions']:,} submissions "
        f"({summary['n_forms_per_annotator']:,} forms per annotator, "
        f"{summary['n_decisions']:,} field decisions in total). "
        "Majority-vote accuracy treats a field as correct when at least two of three "
        "annotators marked it correct."
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| Model | Field decisions | Pooled marked correct | Majority-vote accuracy (95% CI) | Unanimous | Fleiss' kappa |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for o in summary["overall"].values():
        lo, hi = o["majority_ci"]
        lines.append(
            f"| {o['name']} | {o['n_units']:,} | {o['pooled_correct_rate']:.1%} | "
            f"{o['majority_correct_rate']:.1%} ({lo:.1%}-{hi:.1%}) | "
            f"{o['unanimous_rate']:.1%} | {o['fleiss_kappa']:.3f} |"
        )
    ag = summary["agreement_overall"]
    lines.append("")
    lines.append(
        f"Across all {ag['n_units']:,} rating units, the three annotators were unanimous on "
        f"{ag['unanimous_rate']:.1%}, pairwise agreement was {ag['pairwise_agreement']:.1%}, and "
        f"Fleiss' kappa was {ag['fleiss_kappa']:.3f}. Kappa is depressed by the high prevalence "
        "of correct ratings; raw agreement is the more interpretable quantity here."
    )
    lines.append("")
    lines.append("## Majority-vote accuracy by field")
    lines.append("")
    lines.append("| Field | DeepSeek n | DeepSeek accuracy | Luna n | Luna accuracy |")
    lines.append("|---|---:|---:|---:|---:|")
    by_field: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summary["field_correctness"]:
        by_field[row["field"]][row["model"]] = row
    for models in by_field.values():
        s1, s3 = models.get("stage1"), models.get("stage3")
        lines.append(
            f"| {s1['label'] if s1 else s3['label']} "
            f"| {s1['n_units']:,} | {s1['majority_correct_rate']:.1%} "
            f"| {s3['n_units']:,} | {s3['majority_correct_rate']:.1%} |"
        )
    lines.append("")
    lines.append("## Paired model comparison (McNemar on majority votes)")
    lines.append("")
    lines.append(
        "| Field | Paired n | DeepSeek | Luna | DeepSeek-only correct | Luna-only correct | McNemar p |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["model_comparison"]:
        lines.append(
            f"| {row['label']} | {row['n_paired']:,} | {row['stage1_majority_rate']:.1%} "
            f"| {row['stage3_majority_rate']:.1%} | {row['stage1_only_correct']} "
            f"| {row['stage3_only_correct']} | {row['mcnemar_p']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation cautions")
    lines.append("")
    lines.append(
        "- Annotators judged extraction outputs; they did not label submissions independently. These are verification accuracies, not sensitivity/PPV against an independent gold standard."
    )
    lines.append(
        "- Fleiss' kappa is prevalence-sensitive: fields judged nearly always correct can carry low kappa despite near-total raw agreement."
    )
    lines.append(
        "- Nested span/term rows are only defined where the model extracted a side effect, so their denominators differ between models."
    )
    (ANALYSIS_DIR / "05_annotation_analysis.md").write_text("\n".join(lines) + "\n")


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


def make_figures(field_rows: list[dict], agreement_rows: list[dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    by_field: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in field_rows:
        by_field[row["field"]][row["model"]] = row
    fields = [f for f in FIELD_ORDER if f in by_field]
    labels = [FIELD_LABELS[f] for f in fields]
    y = np.arange(len(fields))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    colors = {"stage1": "#1A3A6B", "stage3": "#C05B2E"}
    for model in MODEL_ORDER:
        rates = [by_field[f][model]["majority_correct_rate"] for f in fields]
        los = [by_field[f][model]["majority_ci_low"] for f in fields]
        his = [by_field[f][model]["majority_ci_high"] for f in fields]
        offset = 0.17 if model == "stage1" else -0.17
        ax.errorbar(
            rates,
            y + offset,
            xerr=[
                [r - lo for r, lo in zip(rates, los, strict=True)],
                [hi - r for r, hi in zip(rates, his, strict=True)],
            ],
            fmt="o",
            ms=4,
            lw=0,
            elinewidth=1.0,
            capsize=2,
            color=colors[model],
            label=MODEL_NAMES[model],
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Majority-vote accuracy (2 of 3 annotators), Wilson 95% CI")
    ax.set_xlim(0.5, 1.005)
    ax.legend(loc="lower left")
    ax.set_title("Human-judged field accuracy by model")
    save_figure(fig, figures_dir, "05_01_annotation_correctness")

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ag_fields = [r for r in agreement_rows]
    y2 = np.arange(len(ag_fields))[::-1]
    ax.barh(
        y2 + 0.18,
        [r["pairwise_agreement"] for r in ag_fields],
        height=0.34,
        color="#1A3A6B",
        label="Pairwise agreement",
    )
    ax.barh(
        y2 - 0.18,
        [max(0.0, r["fleiss_kappa"]) for r in ag_fields],
        height=0.34,
        color="#8A9BB8",
        label="Fleiss' kappa (floored at 0)",
    )
    ax.set_yticks(y2, [r["label"] for r in ag_fields])
    ax.set_xlabel("Inter-annotator reliability, both models pooled")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right")
    ax.set_title("Annotator agreement by field")
    save_figure(fig, figures_dir, "05_02_annotator_agreement")


if __name__ == "__main__":
    main()
