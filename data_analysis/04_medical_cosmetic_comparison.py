"""Compare candidate side effects for medical and cosmetic BoNT-A use.

The primary analysis uses the GPT-5.6 Luna batch extraction (``stage4``) over the
full submission corpus; ``--stage stage1`` reruns it on the DeepSeek V4 Flash
comparator (outputs for that run live in ``10_deepseek_comparison/``). The private extraction database is opened read-only and
only aggregate results are written.

The multi-label side-effect profile is compared with a record-level permutation
test of Jensen-Shannon divergence. This preserves correlation between multiple
candidate side effects extracted from the same submission. Per-term Fisher tests
are corrected with the Benjamini-Hochberg procedure.

Outputs:

* ``04_medical_cosmetic_comparison.json`` and ``.md``
* ``04_group_summary.csv``, ``04_omnibus_tests.csv``, and
  ``04_side_effect_comparison.csv``
* three publication figures in PNG and vector PDF under ``data_analysis/figures``

Example:

    uv run python data_analysis/04_medical_cosmetic_comparison.py
    uv run python data_analysis/04_medical_cosmetic_comparison.py --permutations 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.figure import Figure
from scipy import sparse, stats  # type: ignore[import-untyped]
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ANALYSIS_DIR / "02_llm_extraction_results"
DEFAULT_FIGURES_DIR = ANALYSIS_DIR / "figures"
DEFAULT_SEED = 20260715

GROUPS = ("medical", "cosmetic")
GROUP_LABELS = {"medical": "Medical", "cosmetic": "Cosmetic"}
GROUP_COLORS = {"medical": "#24557A", "cosmetic": "#C45A2A"}
STAGE_LABELS = {
    "stage1": "DeepSeek V4 Flash without reasoning",
    "stage2": "DeepSeek V4 Flash with high reasoning",
    "stage3": "GPT-5.6 Luna with low reasoning",
    "stage4": "GPT-5.6 Luna with low reasoning (full corpus, batch)",
}

Group = Literal["medical", "cosmetic"]


@dataclass(frozen=True)
class ComparisonRecord:
    indication: Group
    side_effects: frozenset[str]


def normalize_text(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def load_records(connection: sqlite3.Connection, stage: str) -> list[ComparisonRecord]:
    """Load only the non-verbatim fields needed for the comparison."""
    total = int(
        connection.execute(
            "SELECT COUNT(*) FROM stage_results WHERE stage = ?", (stage,)
        ).fetchone()[0]
    )
    records: list[ComparisonRecord] = []
    rows = connection.execute("SELECT extraction_json FROM stage_results WHERE stage = ?", (stage,))
    for (extraction_json,) in tqdm(
        rows, total=total, desc="load extractions", unit="rec", unit_scale=True
    ):
        extraction = json.loads(extraction_json)
        if not extraction.get("a_self_use"):
            continue
        indication = normalize_text(extraction.get("c_indication_category"))
        if indication not in GROUPS:
            continue
        side_effects = frozenset(
            normalize_text(effect.get("meddra_pt"))
            for effect in extraction.get("d_side_effect_pts") or []
            if isinstance(effect, Mapping) and effect.get("meddra_pt")
        )
        records.append(
            ComparisonRecord(
                indication="medical" if indication == "medical" else "cosmetic",
                side_effects=side_effects,
            )
        )
    return records


def divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


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


def newcombe_difference_interval(
    left_successes: int,
    left_n: int,
    right_successes: int,
    right_n: int,
) -> tuple[float, float]:
    """Newcombe-Wilson interval for p_left - p_right."""
    if left_n <= 0 or right_n <= 0:
        return (math.nan, math.nan)
    left = left_successes / left_n
    right = right_successes / right_n
    left_low, left_high = wilson_interval(left_successes, left_n)
    right_low, right_high = wilson_interval(right_successes, right_n)
    difference = left - right
    lower = difference - math.sqrt((left - left_low) ** 2 + (right_high - right) ** 2)
    upper = difference + math.sqrt((left_high - left) ** 2 + (right - right_low) ** 2)
    return lower, upper


def risk_ratio_interval(
    left_successes: int,
    left_n: int,
    right_successes: int,
    right_n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float, float, bool]:
    """Katz risk-ratio interval, with 0.5 correction for a zero cell."""
    corrected = left_successes == 0 or right_successes == 0
    if corrected:
        left_success = left_successes + 0.5
        right_success = right_successes + 0.5
        left_total = left_n + 1.0
        right_total = right_n + 1.0
    else:
        left_success = float(left_successes)
        right_success = float(right_successes)
        left_total = float(left_n)
        right_total = float(right_n)
    ratio = (left_success / left_total) / (right_success / right_total)
    standard_error = math.sqrt(
        1.0 / left_success - 1.0 / left_total + 1.0 / right_success - 1.0 / right_total
    )
    return (
        ratio,
        math.exp(math.log(ratio) - z * standard_error),
        math.exp(math.log(ratio) + z * standard_error),
        corrected,
    )


def group_summary_rows(records: Sequence[ComparisonRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        group_records = [record for record in records if record.indication == group]
        reporting = [record for record in group_records if record.side_effects]
        counts = np.asarray([len(record.side_effects) for record in reporting], dtype=float)
        side_effect_reports = len(reporting)
        low, high = wilson_interval(side_effect_reports, len(group_records))
        rows.append(
            {
                "indication": group,
                "indication_label": GROUP_LABELS[group],
                "self_use_records": len(group_records),
                "side_effect_reporting_records": side_effect_reports,
                "side_effect_report_prevalence": divide(side_effect_reports, len(group_records)),
                "prevalence_ci_low": low,
                "prevalence_ci_high": high,
                "candidate_pt_mentions": int(counts.sum()) if len(counts) else 0,
                "mean_distinct_pts_per_report": (float(np.mean(counts)) if len(counts) else None),
                "median_distinct_pts_per_report": (
                    float(np.median(counts)) if len(counts) else None
                ),
                "q1_distinct_pts_per_report": (
                    float(np.quantile(counts, 0.25)) if len(counts) else None
                ),
                "q3_distinct_pts_per_report": (
                    float(np.quantile(counts, 0.75)) if len(counts) else None
                ),
            }
        )
    return rows


def side_effect_prevalence_test(
    records: Sequence[ComparisonRecord],
) -> dict[str, Any]:
    totals = Counter(record.indication for record in records)
    reports = Counter(record.indication for record in records if record.side_effects)
    table = np.asarray(
        [
            [reports["medical"], totals["medical"] - reports["medical"]],
            [reports["cosmetic"], totals["cosmetic"] - reports["cosmetic"]],
        ],
        dtype=int,
    )
    chi_square, p_value, degrees_freedom, _expected = stats.chi2_contingency(
        table, correction=False
    )
    difference = reports["medical"] / totals["medical"] - reports["cosmetic"] / totals["cosmetic"]
    low, high = newcombe_difference_interval(
        reports["medical"],
        totals["medical"],
        reports["cosmetic"],
        totals["cosmetic"],
    )
    ratio, ratio_low, ratio_high, corrected = risk_ratio_interval(
        reports["medical"],
        totals["medical"],
        reports["cosmetic"],
        totals["cosmetic"],
    )
    return {
        "test": "pearson_chi_square_side_effect_presence",
        "comparison": "medical_vs_cosmetic",
        "analysis_population": "all_self_use_records_with_medical_or_cosmetic_indication",
        "statistic": float(chi_square),
        "degrees_of_freedom": int(degrees_freedom),
        "p_value": float(p_value),
        "effect_measure": "medical_minus_cosmetic_prevalence",
        "effect_estimate": difference,
        "effect_ci_low": low,
        "effect_ci_high": high,
        "risk_ratio_medical_vs_cosmetic": ratio,
        "risk_ratio_ci_low": ratio_low,
        "risk_ratio_ci_high": ratio_high,
        "risk_ratio_zero_cell_correction": corrected,
    }


def side_effect_burden_test(records: Sequence[ComparisonRecord]) -> dict[str, Any]:
    medical = np.asarray(
        [
            len(record.side_effects)
            for record in records
            if record.indication == "medical" and record.side_effects
        ],
        dtype=float,
    )
    cosmetic = np.asarray(
        [
            len(record.side_effects)
            for record in records
            if record.indication == "cosmetic" and record.side_effects
        ],
        dtype=float,
    )
    result = stats.mannwhitneyu(medical, cosmetic, alternative="two-sided", method="asymptotic")
    rank_biserial = 2.0 * float(result.statistic) / (len(medical) * len(cosmetic)) - 1.0
    return {
        "test": "mann_whitney_distinct_candidate_pts_per_report",
        "comparison": "medical_vs_cosmetic",
        "analysis_population": "side_effect_reporting_records_only",
        "statistic": float(result.statistic),
        "degrees_of_freedom": None,
        "p_value": float(result.pvalue),
        "effect_measure": "rank_biserial_correlation_medical_vs_cosmetic",
        "effect_estimate": rank_biserial,
        "effect_ci_low": None,
        "effect_ci_high": None,
        "medical_n": len(medical),
        "cosmetic_n": len(cosmetic),
    }


def jensen_shannon_divergence(left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray:
    """Base-2 Jensen-Shannon divergence for one or more count-vector pairs."""
    left = np.atleast_2d(np.asarray(left_counts, dtype=float))
    right = np.atleast_2d(np.asarray(right_counts, dtype=float))
    left_total = left.sum(axis=1, keepdims=True)
    right_total = right.sum(axis=1, keepdims=True)
    if np.any(left_total == 0) or np.any(right_total == 0):
        raise ValueError("each profile must contain at least one candidate term")
    left_probability = left / left_total
    right_probability = right / right_total
    midpoint = (left_probability + right_probability) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        left_component = np.where(
            left_probability > 0,
            left_probability * np.log2(left_probability / midpoint),
            0.0,
        )
        right_component = np.where(
            right_probability > 0,
            right_probability * np.log2(right_probability / midpoint),
            0.0,
        )
    return np.asarray(0.5 * (left_component.sum(axis=1) + right_component.sum(axis=1)), dtype=float)


def profile_permutation_tests(
    records: Sequence[ComparisonRecord],
    min_count: int,
    permutations: int,
    seed: int,
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    reporting = [record for record in records if record.side_effects]
    pooled_counts: Counter[str] = Counter()
    for record in reporting:
        pooled_counts.update(record.side_effects)
    terms = sorted(pooled_counts)
    term_index = {term: index for index, term in enumerate(terms)}
    row_index: list[int] = []
    column_index: list[int] = []
    for row, record in enumerate(reporting):
        for term in record.side_effects:
            row_index.append(row)
            column_index.append(term_index[term])
    matrix = sparse.csr_matrix(
        (np.ones(len(row_index), dtype=float), (row_index, column_index)),
        shape=(len(reporting), len(terms)),
    )
    medical_labels = np.asarray(
        [record.indication == "medical" for record in reporting], dtype=np.uint8
    )
    medical_counts = np.asarray(medical_labels.astype(float) @ matrix).ravel()
    total_counts = np.asarray(matrix.sum(axis=0)).ravel()
    cosmetic_counts = total_counts - medical_counts
    common_mask = np.asarray([pooled_counts[term] >= min_count for term in terms])
    masks = {"all_candidate_terms": np.ones(len(terms), dtype=bool), "common_terms": common_mask}

    observed = {
        key: float(jensen_shannon_divergence(medical_counts[mask], cosmetic_counts[mask])[0])
        for key, mask in masks.items()
    }
    exceedances = {key: 0 for key in masks}
    null_sums = {key: 0.0 for key in masks}
    null_squared_sums = {key: 0.0 for key in masks}
    rng = np.random.default_rng(seed)

    for start in tqdm(
        range(0, permutations, batch_size),
        desc="profile permutations",
        unit="batch",
    ):
        current_batch = min(batch_size, permutations - start)
        permuted_labels = rng.permuted(
            np.broadcast_to(medical_labels, (current_batch, len(medical_labels))), axis=1
        )
        permuted_medical = np.asarray(permuted_labels.astype(float) @ matrix)
        permuted_cosmetic = total_counts[np.newaxis, :] - permuted_medical
        for key, mask in masks.items():
            statistics = jensen_shannon_divergence(
                permuted_medical[:, mask], permuted_cosmetic[:, mask]
            )
            exceedances[key] += int(np.count_nonzero(statistics >= observed[key] - 1e-15))
            null_sums[key] += float(statistics.sum())
            null_squared_sums[key] += float(np.square(statistics).sum())

    results: list[dict[str, Any]] = []
    for key, mask in masks.items():
        null_mean = null_sums[key] / permutations if permutations else None
        null_sd = None
        if permutations > 1 and null_mean is not None:
            null_variance = (null_squared_sums[key] - permutations * null_mean * null_mean) / (
                permutations - 1
            )
            null_sd = math.sqrt(max(0.0, null_variance))
        results.append(
            {
                "test": f"record_label_permutation_jsd_{key}",
                "comparison": "medical_vs_cosmetic",
                "analysis_population": "side_effect_reporting_records_only",
                "statistic": observed[key],
                "degrees_of_freedom": None,
                "p_value": ((exceedances[key] + 1) / (permutations + 1) if permutations else None),
                "effect_measure": "jensen_shannon_divergence_base2",
                "effect_estimate": observed[key],
                "effect_ci_low": None,
                "effect_ci_high": None,
                "permutations": permutations,
                "permutation_exceedances": exceedances[key],
                "null_mean": null_mean,
                "null_sd": null_sd,
                "candidate_terms": int(np.count_nonzero(mask)),
                "minimum_pooled_record_count": min_count if key == "common_terms" else 1,
                "medical_records": int(medical_labels.sum()),
                "cosmetic_records": int(len(medical_labels) - medical_labels.sum()),
            }
        )
    return results


def side_effect_comparison_rows(
    records: Sequence[ComparisonRecord], min_count: int
) -> list[dict[str, Any]]:
    reporting = [record for record in records if record.side_effects]
    denominators = Counter(record.indication for record in reporting)
    counts: dict[str, Counter[str]] = {group: Counter() for group in GROUPS}
    for record in reporting:
        counts[record.indication].update(record.side_effects)
    pooled_counts = counts["medical"] + counts["cosmetic"]
    terms = [term for term, count in pooled_counts.items() if count >= min_count]
    terms.sort(key=lambda term: (-pooled_counts[term], term))

    p_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for rank, term in enumerate(terms, start=1):
        medical_count = counts["medical"][term]
        cosmetic_count = counts["cosmetic"][term]
        medical_n = denominators["medical"]
        cosmetic_n = denominators["cosmetic"]
        table = [
            [medical_count, medical_n - medical_count],
            [cosmetic_count, cosmetic_n - cosmetic_count],
        ]
        fisher = stats.fisher_exact(table, alternative="two-sided")
        p_value = float(fisher.pvalue)
        p_values.append(p_value)
        medical_prevalence = medical_count / medical_n
        cosmetic_prevalence = cosmetic_count / cosmetic_n
        difference = medical_prevalence - cosmetic_prevalence
        difference_low, difference_high = newcombe_difference_interval(
            medical_count, medical_n, cosmetic_count, cosmetic_n
        )
        ratio, ratio_low, ratio_high, corrected = risk_ratio_interval(
            medical_count, medical_n, cosmetic_count, cosmetic_n
        )
        rows.append(
            {
                "pooled_frequency_rank": rank,
                "candidate_pt": term,
                "pooled_records": pooled_counts[term],
                "medical_records": medical_count,
                "medical_denominator": medical_n,
                "medical_records_per_100_reports": 100.0 * medical_prevalence,
                "cosmetic_records": cosmetic_count,
                "cosmetic_denominator": cosmetic_n,
                "cosmetic_records_per_100_reports": 100.0 * cosmetic_prevalence,
                "risk_difference_percentage_points": 100.0 * difference,
                "risk_difference_ci_low": 100.0 * difference_low,
                "risk_difference_ci_high": 100.0 * difference_high,
                "risk_ratio_medical_vs_cosmetic": ratio,
                "risk_ratio_ci_low": ratio_low,
                "risk_ratio_ci_high": ratio_high,
                "risk_ratio_zero_cell_correction": corrected,
                "fisher_p_value": p_value,
                "bh_q_value": None,
                "significant_fdr_0_05": False,
                "higher_group": (
                    "medical" if difference > 0 else "cosmetic" if difference < 0 else "equal"
                ),
            }
        )
    if p_values:
        rejected, q_values, _sidak, _bonferroni = multipletests(
            p_values, alpha=0.05, method="fdr_bh"
        )
        for row, reject, q_value in zip(rows, rejected, q_values, strict=True):
            row["bh_q_value"] = float(q_value)
            row["significant_fdr_0_05"] = bool(reject)
    return rows


def side_effect_count_distribution_rows(
    records: Sequence[ComparisonRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        counts = Counter(
            "5+" if len(record.side_effects) >= 5 else str(len(record.side_effects))
            for record in records
            if record.indication == group and record.side_effects
        )
        denominator = sum(counts.values())
        for category in ("1", "2", "3", "4", "5+"):
            rows.append(
                {
                    "indication": group,
                    "distinct_candidate_pts": category,
                    "records": counts[category],
                    "side_effect_reporting_records": denominator,
                    "percent": 100.0 * counts[category] / denominator if denominator else None,
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


def figure_prevalence_and_burden(
    summaries: Sequence[Mapping[str, Any]],
    count_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stage_label: str,
) -> None:
    summary_by_group = {str(row["indication"]): row for row in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    positions = np.arange(len(GROUPS))
    prevalence = np.asarray(
        [
            100.0 * float(summary_by_group[group]["side_effect_report_prevalence"])
            for group in GROUPS
        ]
    )
    lower = np.asarray(
        [100.0 * float(summary_by_group[group]["prevalence_ci_low"]) for group in GROUPS]
    )
    upper = np.asarray(
        [100.0 * float(summary_by_group[group]["prevalence_ci_high"]) for group in GROUPS]
    )
    axes[0].bar(
        positions,
        prevalence,
        color=[GROUP_COLORS[group] for group in GROUPS],
        width=0.62,
    )
    axes[0].errorbar(
        positions,
        prevalence,
        yerr=np.vstack((prevalence - lower, upper - prevalence)),
        fmt="none",
        ecolor="#222222",
        capsize=3,
        linewidth=1,
    )
    axes[0].set_xticks(positions, [GROUP_LABELS[group] for group in GROUPS])
    axes[0].set_ylabel("Self-use records reporting a side effect (%)")
    axes[0].set_title("Side-effect reporting prevalence", loc="left", fontweight="bold")
    axes[0].grid(axis="x", visible=False)
    for position, group, value in zip(positions, GROUPS, prevalence, strict=True):
        summary = summary_by_group[group]
        axes[0].text(
            position,
            value + 1.0,
            f"{value:.1f}%\n({int(summary['side_effect_reporting_records']):,}/{int(summary['self_use_records']):,})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    count_lookup = {
        (str(row["indication"]), str(row["distinct_candidate_pts"])): row for row in count_rows
    }
    categories = ("1", "2", "3", "4", "5+")
    width = 0.36
    for index, group in enumerate(GROUPS):
        values = [
            float(count_lookup[(group, category)]["percent"] or 0.0) for category in categories
        ]
        axes[1].bar(
            np.arange(len(categories)) + (index - 0.5) * width,
            values,
            width=width,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    axes[1].set_xticks(np.arange(len(categories)), categories)
    axes[1].set_xlabel("Distinct candidate side effects in a report")
    axes[1].set_ylabel("Side-effect reporting records (%)")
    axes[1].set_title("Number of candidate terms per report", loc="left", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="x", visible=False)

    fig.suptitle(
        f"Medical versus cosmetic use: prevalence and reporting burden\n{stage_label}",
        x=0.055,
        y=1.04,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.055,
        -0.01,
        "Error bars are 95% Wilson confidence intervals. Candidate terms are unvalidated model outputs.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "04_01_side_effect_prevalence_and_burden")


def pretty_term(term: str) -> str:
    return term[:1].upper() + term[1:]


def figure_side_effect_profile(
    comparison_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    top_terms: int,
    stage_label: str,
) -> None:
    selected = list(comparison_rows[:top_terms])
    selected.reverse()
    labels = [pretty_term(str(row["candidate_pt"])) for row in selected]
    medical = np.asarray([float(row["medical_records_per_100_reports"]) for row in selected])
    cosmetic = np.asarray([float(row["cosmetic_records_per_100_reports"]) for row in selected])
    positions = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(9.2, max(6.4, 0.29 * len(selected) + 1.8)))
    ax.hlines(
        positions, np.minimum(medical, cosmetic), np.maximum(medical, cosmetic), color="#C5CBD0"
    )
    ax.scatter(medical, positions, color=GROUP_COLORS["medical"], s=30, label="Medical", zorder=3)
    ax.scatter(
        cosmetic, positions, color=GROUP_COLORS["cosmetic"], s=30, label="Cosmetic", zorder=3
    )
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlabel("Records with candidate term per 100 side-effect reports")
    ax.set_title(
        f"Most frequently extracted candidate side effects\n{stage_label}",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        -0.01,
        "Records are the unit of analysis; a submission can contribute to more than one term.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "04_02_side_effect_profile")


def figure_term_differences(
    comparison_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stage_label: str,
    maximum_terms: int = 20,
) -> None:
    significant = [row for row in comparison_rows if bool(row["significant_fdr_0_05"])]
    significant.sort(
        key=lambda row: abs(float(row["risk_difference_percentage_points"])), reverse=True
    )
    selected = significant[:maximum_terms]
    selected.reverse()

    fig, ax = plt.subplots(figsize=(9.2, max(5.2, 0.32 * max(len(selected), 10) + 1.8)))
    if selected:
        estimates = np.asarray(
            [float(row["risk_difference_percentage_points"]) for row in selected]
        )
        lower = np.asarray([float(row["risk_difference_ci_low"]) for row in selected])
        upper = np.asarray([float(row["risk_difference_ci_high"]) for row in selected])
        positions = np.arange(len(selected))
        colors = [
            GROUP_COLORS["medical"] if estimate > 0 else GROUP_COLORS["cosmetic"]
            for estimate in estimates
        ]
        for position, estimate, low, high, color in zip(
            positions, estimates, lower, upper, colors, strict=True
        ):
            ax.errorbar(
                estimate,
                position,
                xerr=np.asarray([[estimate - low], [high - estimate]]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=5,
            )
        ax.set_yticks(
            positions,
            [pretty_term(str(row["candidate_pt"])) for row in selected],
            fontsize=8,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No term met the FDR threshold",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_yticks([])
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_xlabel("Prevalence difference per 100 reports (medical minus cosmetic)")
    ax.set_title(
        f"Largest FDR-significant candidate-term differences\n{stage_label}",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        -0.01,
        "Points are prevalence differences with 95% Newcombe-Wilson confidence intervals; tests use Fisher's exact test with BH correction.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "04_03_candidate_term_differences")


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
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_p_value(value: float | None, permutations: int | None = None) -> str:
    if value is None:
        return "not calculated"
    if permutations and math.isclose(value, 1.0 / (permutations + 1)):
        return f"<{2.0 / (permutations + 1):.4g}"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def format_p_expression(value: float | None, permutations: int | None = None) -> str:
    formatted = format_p_value(value, permutations)
    return f"p{formatted}" if formatted.startswith("<") else f"p={formatted}"


def write_markdown_report(
    path: Path,
    stage_label: str,
    summaries: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    min_count: int,
) -> None:
    summary = {str(row["indication"]): row for row in summaries}
    prevalence_test = next(
        row for row in tests if row["test"] == "pearson_chi_square_side_effect_presence"
    )
    burden_test = next(
        row for row in tests if row["test"] == "mann_whitney_distinct_candidate_pts_per_report"
    )
    primary_profile = next(
        row for row in tests if row["test"] == "record_label_permutation_jsd_common_terms"
    )
    sensitivity_profile = next(
        row for row in tests if row["test"] == "record_label_permutation_jsd_all_candidate_terms"
    )
    significant = [row for row in comparisons if bool(row["significant_fdr_0_05"])]
    largest = sorted(
        significant,
        key=lambda row: abs(float(row["risk_difference_percentage_points"])),
        reverse=True,
    )[:15]

    lines = [
        "# Medical versus Cosmetic Side-Effect Comparison",
        "",
        f"Extraction approach: {stage_label}.",
        "",
        "The analysis includes self-use records classified as medical or cosmetic. Unknown "
        "indications are excluded. Candidate side-effect terms are unvalidated model outputs.",
        "",
        "## Group summary",
        "",
        "| Indication | Self-use records | Side-effect reports | Prevalence (95% CI) | Distinct candidate terms per report |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        row = summary[group]
        lines.append(
            f"| {GROUP_LABELS[group]} | {int(row['self_use_records']):,} | "
            f"{int(row['side_effect_reporting_records']):,} | "
            f"{100 * float(row['side_effect_report_prevalence']):.1f}% "
            f"({100 * float(row['prevalence_ci_low']):.1f}-"
            f"{100 * float(row['prevalence_ci_high']):.1f}%) | "
            f"{float(row['mean_distinct_pts_per_report']):.2f} |"
        )

    permutations = int(primary_profile["permutations"])
    lines.extend(
        [
            "",
            "## Statistical comparisons",
            "",
            f"Side-effect reporting prevalence differed by "
            f"{100 * float(prevalence_test['effect_estimate']):.2f} percentage points "
            f"(medical minus cosmetic; 95% CI "
            f"{100 * float(prevalence_test['effect_ci_low']):.2f} to "
            f"{100 * float(prevalence_test['effect_ci_high']):.2f}; "
            f"Pearson chi-square "
            f"{format_p_expression(float(prevalence_test['p_value']))}).",
            "",
            f"The conditional side-effect profile differed between indications in the primary "
            f"record-level permutation test of {int(primary_profile['candidate_terms']):,} terms "
            f"reported in at least {min_count} records (Jensen-Shannon divergence "
            f"{float(primary_profile['statistic']):.4f}; "
            f"{format_p_expression(float(primary_profile['p_value']), permutations)}; "
            f"{permutations:,} permutations). The all-term sensitivity analysis included "
            f"{int(sensitivity_profile['candidate_terms']):,} terms "
            f"(divergence {float(sensitivity_profile['statistic']):.4f}; "
            f"{format_p_expression(float(sensitivity_profile['p_value']), permutations)}).",
            "",
            f"Medical reports contained a median of "
            f"{float(summary['medical']['median_distinct_pts_per_report']):.0f} candidate terms "
            f"versus {float(summary['cosmetic']['median_distinct_pts_per_report']):.0f} in "
            f"cosmetic reports (Mann-Whitney "
            f"{format_p_expression(float(burden_test['p_value']))}; rank-biserial correlation "
            f"{float(burden_test['effect_estimate']):.3f}).",
            "",
            f"Per-term inference was limited to {len(comparisons):,} candidate terms found in "
            f"at least {min_count} records. {len(significant):,} remained significant at a "
            "Benjamini-Hochberg false-discovery rate of 5%.",
            "",
            "## Largest candidate-term differences",
            "",
            "| Candidate term | Medical per 100 | Cosmetic per 100 | Difference (95% CI) | BH q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in largest:
        lines.append(
            f"| {pretty_term(str(row['candidate_pt']))} | "
            f"{float(row['medical_records_per_100_reports']):.2f} | "
            f"{float(row['cosmetic_records_per_100_reports']):.2f} | "
            f"{float(row['risk_difference_percentage_points']):.2f} "
            f"({float(row['risk_difference_ci_low']):.2f} to "
            f"{float(row['risk_difference_ci_high']):.2f}) | "
            f"{format_p_value(float(row['bh_q_value']))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The permutation test shuffles indication labels at the submission level, "
            "preserving co-occurrence of multiple terms within a record.",
            "- Per-term denominators are records reporting at least one candidate side effect, "
            "so these results compare composition rather than absolute population risk.",
            "- Candidate terms may reflect underlying indications, ineffective treatment, or "
            "extraction error as well as adverse effects; this is not a causal comparison.",
            "- Statistical significance is amplified by the large corpus. Effect sizes and "
            "confidence intervals should guide interpretation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--stage", choices=tuple(STAGE_LABELS), default="stage4")
    parser.add_argument(
        "--permutations",
        type=int,
        default=10_000,
        help="Record-label permutations for the omnibus side-effect profile test",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum pooled record count for primary profile and per-term inference",
    )
    parser.add_argument("--top-terms", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    if args.min_count < 1:
        parser.error("--min-count must be positive")
    if args.top_terms < 1:
        parser.error("--top-terms must be positive")
    return args


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    database = results_dir / "private_extraction.sqlite3"
    if not database.is_file():
        raise SystemExit(f"Extraction database not found: {database}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        records = load_records(connection, args.stage)
    finally:
        connection.close()
    if not records or not all(
        any(record.indication == group for record in records) for group in GROUPS
    ):
        raise SystemExit("Both medical and cosmetic self-use records are required")

    summaries = group_summary_rows(records)
    prevalence_test = side_effect_prevalence_test(records)
    burden_test = side_effect_burden_test(records)
    profile_tests = profile_permutation_tests(
        records,
        min_count=args.min_count,
        permutations=args.permutations,
        seed=args.seed,
    )
    omnibus_tests = [prevalence_test, *profile_tests, burden_test]
    comparisons = side_effect_comparison_rows(records, args.min_count)
    count_distribution = side_effect_count_distribution_rows(records)

    configure_plotting()
    stage_label = STAGE_LABELS[args.stage]
    figure_prevalence_and_burden(summaries, count_distribution, figures_dir, stage_label)
    figure_side_effect_profile(comparisons, figures_dir, args.top_terms, stage_label)
    figure_term_differences(comparisons, figures_dir, stage_label)

    analysis = {
        "method": {
            "stage": args.stage,
            "approach": stage_label,
            "scope": "self-use records with medical or cosmetic indication",
            "profile_population": "records reporting at least one candidate side effect",
            "profile_test": "record-label permutation test of base-2 Jensen-Shannon divergence",
            "permutations": args.permutations,
            "minimum_pooled_record_count": args.min_count,
            "seed": args.seed,
            "per_term_test": "two-sided Fisher exact test with Benjamini-Hochberg correction",
        },
        "group_summary": summaries,
        "omnibus_tests": omnibus_tests,
        "side_effect_count_distribution": count_distribution,
        "candidate_term_comparisons": comparisons,
    }
    (output_dir / "04_medical_cosmetic_comparison.json").write_text(
        json.dumps(json_ready(analysis), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "04_group_summary.csv", summaries)
    write_csv(output_dir / "04_omnibus_tests.csv", omnibus_tests)
    write_csv(output_dir / "04_side_effect_comparison.csv", comparisons)
    write_csv(output_dir / "04_side_effect_count_distribution.csv", count_distribution)
    write_markdown_report(
        output_dir / "04_medical_cosmetic_comparison.md",
        stage_label,
        summaries,
        omnibus_tests,
        comparisons,
        args.min_count,
    )

    summary_by_group = {str(row["indication"]): row for row in summaries}
    primary_profile = next(
        row for row in profile_tests if row["test"] == "record_label_permutation_jsd_common_terms"
    )
    print(
        f"medical={int(summary_by_group['medical']['self_use_records']):,} "
        f"({int(summary_by_group['medical']['side_effect_reporting_records']):,} reports); "
        f"cosmetic={int(summary_by_group['cosmetic']['self_use_records']):,} "
        f"({int(summary_by_group['cosmetic']['side_effect_reporting_records']):,} reports)"
    )
    print(
        f"side-effect prevalence chi-square p={float(prevalence_test['p_value']):.6g}; "
        f"profile permutation p={float(primary_profile['p_value']):.6g}; "
        f"{sum(bool(row['significant_fdr_0_05']) for row in comparisons):,}/"
        f"{len(comparisons):,} tested terms significant at BH q<0.05"
    )
    print(f"wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
