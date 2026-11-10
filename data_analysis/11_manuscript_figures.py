"""Manuscript figures drawn at final print size from aggregate outputs only.

The earlier figures were drawn at 9-12 inches wide and shrunk to the text
width, which left their labels at 3-5 pt. These are drawn at the printed width
(130.7 mm, the text width of the Springer Nature template) with 6-7.5 pt text, so
LaTeX does not rescale them and every font prints at the size drawn here.
Nothing here reads row-level data: every number comes from the aggregate
outputs of scripts 03, 04, 05, 07, 08, 09 and 10 and from the two count manifests.

* Figure 1 -- study overview: corpus flow, extraction, model selection and the
  headline comparison, at 180 mm (Nature double-column width). Nature
  Communications does not allow a graphical abstract, so this replaces it.
* Figure 2 -- full-corpus extraction overview (``03_stage4_descriptive_counts``).
* Figure 3 -- the 25 most frequent candidate terms by indication
  (``04_side_effect_comparison``).
* Figure 4 -- robustness of the prevalence difference and the profile
  divergence (``08``, ``09``, ``10``, ``04_group_summary``).

Medical and cosmetic colours were checked with the dataviz palette validator
(lightness band, chroma floor, colour-vision-deficiency separation, contrast).

Example:

    uv run python data_analysis/11_manuscript_figures.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_DIR / "cache" / "matplotlib"))

import matplotlib as mpl  # noqa: E402
import matplotlib.patches  # noqa: E402  (mpl.patches.Patch)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIGURES_DIR = ANALYSIS_DIR / "figures"
MANUSCRIPT_FIGURES = ANALYSIS_DIR.parent / "69e4e2dfc6719da1d8363cc0" / "figures"

WIDTH = 372.0 / 72.27  # inches: the text width of sn-jnl.cls (372 pt, 130.7 mm)
MINUS = "\u2212"  # the typographic minus sign used for excluded counts in Figure 1a
MEDICAL = "#1F5C99"
COSMETIC = "#C45A2A"
UNKNOWN = "#9AA3AA"
NEUTRAL = "#5B6770"
INK = "#33404A"
MUTED = "#6B7780"
GRID = "#E3E6E8"
RULE = "#C9CFD4"
PALE = "#F4F6F7"
MEDICAL_PALE = "#E6EDF5"
COSMETIC_PALE = "#F7E9E2"


def _load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


a04 = _load_module("a04", "04_medical_cosmetic_comparison.py")


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # Mathtext only sets the odd italic word (bonta) in the body font.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig: plt.Figure, stem: str, manuscript_name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    if MANUSCRIPT_FIGURES.is_dir():
        shutil.copyfile(pdf, MANUSCRIPT_FIGURES / manuscript_name)


def panel_label(ax: plt.Axes, letter: str) -> None:
    # Same baseline as a panel title drawn with pad=4, so the letter and the heading read as one line.
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-4, 4),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="baseline",
    )


def hbars(
    ax: plt.Axes,
    labels: list[str],
    counts: list[int],
    denominator: int,
    colors: list[str] | str,
    title: str,
    xlabel: str,
) -> None:
    """One panel of Figure 2: horizontal bars with the count and share printed beside each bar."""
    shares = [100 * c / denominator for c in counts]
    y = list(range(len(labels)))[::-1]
    ax.barh(y, shares, height=0.6, color=colors, edgecolor="white", linewidth=0.8)
    for yi, share, count in zip(y, shares, counts, strict=True):
        ax.text(
            share + max(shares) * 0.04,
            yi,
            f"{count:,} ({share:.1f}%)",
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(shares) * 1.8)
    step = 40 if max(shares) > 70 else 20 if max(shares) > 40 else 10
    ax.set_xticks(np.arange(0, max(shares) + 0.1, step))
    ax.set_xlabel(xlabel, color=MUTED, fontsize=6.5)
    ax.set_title(title, loc="left", fontweight="bold", pad=4, fontsize=7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def figure_overview() -> None:
    counts: dict[tuple[str, str], int] = {}
    denominators: dict[str, int] = {}
    for r in rows(ANALYSIS_DIR / "03_stage4_descriptive_counts.csv"):
        counts[(r["field"], r["value"])] = int(r["count"])
        denominators[r["field"]] = int(r["denominator"])
    n_all = denominators["a_self_use"]
    self_use = counts[("a_self_use", "true")]
    per_record = {k: counts[("d_side_effect_count", k)] for k in ("0", "1", "2", "3", "4", "5+")}
    any_se = sum(v for k, v in per_record.items() if k != "0")
    multiple = any_se - per_record["1"]

    fig, axes = plt.subplots(
        2, 2, figsize=(WIDTH, 3.6), gridspec_kw={"hspace": 0.85, "wspace": 0.75}
    )
    hbars(
        axes[0, 0],
        ["Self-use", "Any side effect", "Multiple\nside effects"],
        [self_use, any_se, multiple],
        n_all,
        NEUTRAL,
        "Nested record classifications",
        f"% of {n_all:,} extracted submissions",
    )
    hbars(
        axes[0, 1],
        ["Medical", "Cosmetic", "Unknown"],
        [counts[("c_indication_category", g)] for g in ("medical", "cosmetic", "unknown")],
        self_use,
        [MEDICAL, COSMETIC, UNKNOWN],
        "Indication category",
        f"% of {self_use:,} self-use submissions",
    )
    se_den = denominators["f_sentiment_about_side_effect"]
    hbars(
        axes[1, 0],
        ["Distressed", "Accepting", "Dismissive"],
        [
            counts[("f_sentiment_about_side_effect", v)]
            for v in ("distressed", "accepting", "dismissive")
        ],
        se_den,
        NEUTRAL,
        "Attitude toward the side effect",
        f"% of {se_den:,} side-effect reports",
    )
    hbars(
        axes[1, 1],
        ["None", "1", "2", "3", "4", "5 or more"],
        list(per_record.values()),
        self_use,
        NEUTRAL,
        "Side effects per submission",
        f"% of {self_use:,} self-use submissions",
    )
    for ax, letter in zip(axes.flat, "abcd", strict=True):
        panel_label(ax, letter)
    save(fig, "11_fig1_extraction_overview", "extraction_overview.pdf")


def group_colour(medical_rate: float, cosmetic_rate: float) -> str:
    """Colour of the group that reports a term more often."""
    return MEDICAL if medical_rate > cosmetic_rate else COSMETIC


def figure_profile(top: int = 20) -> None:
    """Figure 3: the most frequent terms, medical bars to the left and cosmetic bars to the right.

    No title above the plot (the caption carries it), a single legend at 7.5 pt, and term names
    coloured by the group that reports the term more often.
    """
    table = sorted(
        rows(ANALYSIS_DIR / "04_side_effect_comparison.csv"),
        key=lambda r: int(r["pooled_frequency_rank"]),
    )[:top]
    terms = [r["candidate_pt"] for r in table]
    med = [float(r["medical_records_per_100_reports"]) for r in table]
    cos = [float(r["cosmetic_records_per_100_reports"]) for r in table]
    limit = max(med + cos) * 1.06
    y = list(range(len(terms)))[::-1]

    fig = plt.figure(figsize=(WIDTH, 2.5))
    grid = GridSpec(
        1,
        3,
        width_ratios=[1, 0.72, 1],
        wspace=0.03,
        left=0.0,
        right=1.0,
        top=0.9,
        bottom=0.14,
        figure=fig,
    )
    left, mid, right = (fig.add_subplot(grid[0, i]) for i in range(3))
    left.barh(y, med, height=0.66, color=MEDICAL, edgecolor="white", linewidth=0.4)
    right.barh(y, cos, height=0.66, color=COSMETIC, edgecolor="white", linewidth=0.4)
    left.set_xlim(limit, 0)
    right.set_xlim(0, limit)
    for ax in (left, right):
        ax.set_ylim(-0.7, len(terms) - 0.3)
        ax.set_yticks([])
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6.3)
        ax.set_xlabel("Per 100 side-effect reports", color=MUTED, fontsize=6.5)
    left.spines["left"].set_visible(False)
    left.spines["right"].set_visible(True)
    right.spines["left"].set_visible(True)
    mid.set_ylim(-0.7, len(terms) - 0.3)
    mid.axis("off")
    for yi, term, m, c in zip(y, terms, med, cos, strict=True):
        mid.text(
            0.5,
            yi,
            term[0].upper() + term[1:],
            ha="center",
            va="center",
            fontsize=6.2,
            color=group_colour(m, c),
        )
    handles = [
        mpl.patches.Patch(color=MEDICAL, label="Medical"),
        mpl.patches.Patch(color=COSMETIC, label="Cosmetic"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.6,
    )
    save(fig, "11_fig2_term_profile", "medical_cosmetic_profile.pdf")


def figure_robustness() -> None:
    r08 = json.loads((ANALYSIS_DIR / "08_robustness_analyses.json").read_text(encoding="utf-8"))
    review = json.loads((ANALYSIS_DIR / "13_review_analyses.json").read_text(encoding="utf-8"))
    r13 = review["unknown_stratum"]
    deepseek_prev = next(
        r
        for r in rows(ANALYSIS_DIR / "10_deepseek_comparison" / "04_omnibus_tests.csv")
        if r["test"].startswith("pearson")
    )
    deepseek_jsd = next(
        r
        for r in rows(ANALYSIS_DIR / "10_deepseek_comparison" / "04_omnibus_tests.csv")
        if r["test"].endswith("common_terms")
    )
    groups = {r["indication"]: r for r in rows(ANALYSIS_DIR / "04_group_summary.csv")}
    by_indication = {
        r["indication_category"]: r
        for r in rows(ANALYSIS_DIR / "03_stage4_side_effects_by_indication.csv")
    }
    unknown = {
        "n": int(by_indication["unknown"]["self_use_records"]),
        "se": int(by_indication["unknown"]["side_effect_report_records"]),
    }
    med_n, med_se = (
        int(groups["medical"]["self_use_records"]),
        int(groups["medical"]["side_effect_reporting_records"]),
    )
    cos_n, cos_se = (
        int(groups["cosmetic"]["self_use_records"]),
        int(groups["cosmetic"]["side_effect_reporting_records"]),
    )

    def bound(to_medical: bool) -> tuple[float, float, float]:
        m_se, m_n = (
            (med_se + unknown["se"], med_n + unknown["n"]) if to_medical else (med_se, med_n)
        )
        c_se, c_n = (
            (cos_se, cos_n) if to_medical else (cos_se + unknown["se"], cos_n + unknown["n"])
        )
        lo, hi = a04.newcombe_difference_interval(m_se, m_n, c_se, c_n)
        return 100 * (m_se / m_n - c_se / c_n), 100 * lo, 100 * hi

    loco = {r["variant"]: r for r in r08["loco"]}
    named = {"without r/migraine", "without r/noburp", "without r/botoxsupportcommunity"}
    others = [r for r in r08["loco"] if r["kind"] == "community" and r["variant"] not in named]
    periods = {r["variant"]: r for r in r08["periods"]}
    vocab = {r["variant"]: r for r in r08["vocabulary"]}
    c = r08["cluster"]
    d = r08["author_draws"]
    b = r08["baseline"]
    # Post-level side-effect presence, which counts the two re-read false positives that describe a
    # genuine side effect as true positives; 09's strict-rule correction is in Supplementary Table 7.
    corr = review["adjudication"]["correction"]

    def row(
        label: str, r: dict[str, Any]
    ) -> tuple[str, float, float, float, float | None, float | None, float | None]:
        return (
            label,
            100 * r["prevalence_difference"],
            100 * r["prevalence_ci_low"],
            100 * r["prevalence_ci_high"],
            r["jsd"],
            None,
            None,
        )

    items: list[Any] = [
        "Primary analysis",
        (
            "Submissions treated as independent",
            100 * b["prevalence_difference"],
            100 * b["prevalence_ci_low"],
            100 * b["prevalence_ci_high"],
            b["jsd"],
            None,
            None,
        ),
        (
            "Author-clustered bootstrap",
            100 * c["prevalence_difference"],
            100 * c["prevalence_ci_low"],
            100 * c["prevalence_ci_high"],
            c["jsd"],
            c["jsd_ci_low"],
            c["jsd_ci_high"],
        ),
        (
            "One submission per author (200 draws)",
            100 * d["prevalence_difference_median"],
            100 * d["prevalence_difference_p2_5"],
            100 * d["prevalence_difference_p97_5"],
            d["jsd_median"],
            d["jsd_p2_5"],
            d["jsd_p97_5"],
        ),
        "Communities left out",
        row("Without r/migraine", loco["without r/migraine"]),
        row("Without r/BotoxSupportCommunity", loco["without r/botoxsupportcommunity"]),
        row("Without r/noburp", loco["without r/noburp"]),
        row(
            "Without r/noburp and r/BotoxSupportCommunity",
            loco["without r/noburp and r/BotoxSupportCommunity together"],
        ),
        (
            "Without any other of the 20 largest (range)",
            100 * min(r["prevalence_difference"] for r in others),
            100 * min(r["prevalence_difference"] for r in others),
            100 * max(r["prevalence_difference"] for r in others),
            None,
            min(r["jsd"] for r in others),
            max(r["jsd"] for r in others),
        ),
        "Calendar period",
        row("2015-2019", periods["2015-2019"]),
        row("2020-2022", periods["2020-2022"]),
        row("2023-2025", periods["2023-2025"]),
        "Vocabulary",
        row("Synonyms merged", vocab["synonym merges"]),
        row("Indication terms also excluded", vocab["synonym merges and symmetric exclusions"]),
        "Measurement",
        (
            "Comparator model (DeepSeek V4 Flash)",
            100 * float(deepseek_prev["effect_estimate"]),
            100 * float(deepseek_prev["effect_ci_low"]),
            100 * float(deepseek_prev["effect_ci_high"]),
            float(deepseek_jsd["statistic"]),
            None,
            None,
        ),
        (
            "Corrected for extraction error by group",
            100 * corr["corrected_difference"],
            100 * corr["corrected_difference_ci_low"],
            100 * corr["corrected_difference_ci_high"],
            None,
            None,
            None,
        ),
        "Unknown-indication submissions",
        ("All assigned to medical", *bound(True), None, None, None),
        ("All assigned to cosmetic", *bound(False), None, None, None),
        (
            "Those in r/noburp assigned to medical",
            100 * r13["noburp_to_medical"]["difference"],
            100 * r13["noburp_to_medical"]["ci_low"],
            100 * r13["noburp_to_medical"]["ci_high"],
            None,
            None,
            None,
        ),
        (
            "Reallocated by verified proportions",
            100 * r13["verified_reallocation"]["median"],
            100 * r13["verified_reallocation"]["p2_5"],
            100 * r13["verified_reallocation"]["p97_5"],
            None,
            None,
            None,
        ),
    ]

    height = 0.15 * len(items) + 0.6
    fig = plt.figure(figsize=(WIDTH, height))
    grid = GridSpec(
        1,
        3,
        width_ratios=[1.45, 1.15, 0.85],
        wspace=0.13,
        left=0.0,
        right=0.985,
        top=0.96,
        bottom=0.11,
        figure=fig,
    )
    labels_ax, prev_ax, jsd_ax = (fig.add_subplot(grid[0, i]) for i in range(3))
    n = len(items)
    for ax in (labels_ax, prev_ax, jsd_ax):
        ax.set_ylim(n - 0.5, -0.8)
    labels_ax.axis("off")
    for i, item in enumerate(items):
        if isinstance(item, str):
            labels_ax.text(0, i, item, fontweight="bold", va="center", fontsize=6.8)
            continue
        label, est, lo, hi, jsd, jlo, jhi = item
        labels_ax.text(0.04, i, label, va="center", fontsize=6.4)
        is_range = label.endswith("(range)")
        prev_ax.plot([lo, hi], [i, i], color=INK, linewidth=1.0, solid_capstyle="round")
        if not is_range:
            prev_ax.plot(
                est, i, "o", color=INK, markersize=3.4, markeredgecolor="white", markeredgewidth=0.6
            )
        if jlo is not None and jhi is not None:
            jsd_ax.plot([jlo, jhi], [i, i], color=INK, linewidth=1.0, solid_capstyle="round")
        if jsd is not None:
            jsd_ax.plot(
                jsd, i, "o", color=INK, markersize=3.4, markeredgecolor="white", markeredgewidth=0.6
            )
    prev_ax.axvline(0, color=MUTED, linewidth=0.6)
    prev_ax.set_xlabel(
        "Medical minus cosmetic side-effect\nprevalence (percentage points, 95% interval)",
        color=MUTED,
        fontsize=6.5,
    )
    prev_ax.set_xlim(-12.5, 6.5)
    prev_ax.text(-0.5, -0.7, "cosmetic higher", fontsize=6, color=MUTED, va="center", ha="right")
    prev_ax.text(0.5, -0.7, "medical higher", fontsize=6, color=MUTED, va="center", ha="left")
    null_mean = next(
        r for r in rows(ANALYSIS_DIR / "04_omnibus_tests.csv") if r["test"].endswith("common_terms")
    )["null_mean"]
    jsd_ax.axvline(float(null_mean), color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    jsd_ax.text(
        float(null_mean) + 0.012, -0.7, "permutation null", fontsize=6, color=MUTED, va="center"
    )
    jsd_ax.set_xlim(0, 0.45)
    jsd_ax.set_xlabel("Jensen-Shannon divergence\nof term profiles", color=MUTED, fontsize=6.5)
    for ax in (prev_ax, jsd_ax):
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.tick_params(axis="x", labelsize=6.3)
        ax.set_axisbelow(True)
    panel_label(prev_ax, "a")
    panel_label(jsd_ax, "b")
    save(fig, "11_fig3_robustness", "robustness.pdf")


def figure_study_overview() -> None:
    """Figure 1: corpus flow, extraction, model selection and the headline comparison.

    Drawn on a millimetre grid at the manuscript's text width, so every position and font size
    below is the printed one. White backgrounds; one palette (ink and greys, plus the medical blue
    and cosmetic orange used in every figure); a two-by-two grid whose panels share their left
    edges, top lines and bottom lines. All numbers come from aggregate outputs.
    """
    W, H = WIDTH * 25.4, 125.5
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, W)
    canvas.set_ylim(0, H)
    canvas.axis("off")
    LEFT, RIGHT = 0.0, 67.0  # left edges of the two columns (mm)
    COL = W - RIGHT  # column width (mm)
    ROW1, ROW2 = 0.0, 77.0  # top edges of the two rows (mm)

    def y(top: float) -> float:
        """Convert a distance from the top edge (mm) to the canvas y coordinate."""
        return H - top

    def axes_mm(x: float, top: float, w: float, h: float) -> plt.Axes:
        """Add axes at a position given in millimetres from the top-left corner."""
        return fig.add_axes((x / W, (H - top - h) / H, w / W, h / H))

    def box(x: float, top: float, w: float, h: float, edge: str = RULE, width: float = 0.5) -> None:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y(top + h)),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.7",
                linewidth=width,
                edgecolor=edge,
                facecolor="white",
            )
        )

    def arrow(x0: float, top0: float, x1: float, top1: float) -> None:
        canvas.add_patch(
            FancyArrowPatch(
                (x0, y(top0)),
                (x1, y(top1)),
                arrowstyle="-|>",
                mutation_scale=4.5,
                linewidth=0.5,
                color=MUTED,
                shrinkA=0,
                shrinkB=0,
            )
        )

    def heading(x: float, top: float, letter: str, title: str) -> None:
        canvas.text(x, y(top), letter, fontsize=8.5, fontweight="bold", va="center")
        canvas.text(x + 4.2, y(top), title, fontsize=7.3, fontweight="bold", va="center")

    def note(
        x: float, top: float, text: str, size: float = 6.3, color: str = MUTED, **kw: Any
    ) -> None:
        canvas.text(x, y(top), text, fontsize=size, color=color, va="center", **kw)

    # ----------------------------------------------------------------- inputs
    corpus = json.loads(
        (ANALYSIS_DIR.parent / "filtered_dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (ANALYSIS_DIR / "02_llm_extraction_results" / "experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    no_text = corpus["submissions_dropped_no_body"] + corpus["context_submissions"]
    matched = corpus["submissions_kept"] + no_text
    counts = {
        (r["field"], r["value"]): int(r["count"])
        for r in rows(ANALYSIS_DIR / "03_stage4_descriptive_counts.csv")
    }
    extracted = counts[("a_self_use", "true")] + counts[("a_self_use", "false")]
    self_use = counts[("a_self_use", "true")]
    eligible = int(experiment["eligible_records"])
    annotation = json.loads(
        (ANALYSIS_DIR / "05_annotation_analysis.json").read_text(encoding="utf-8")
    )
    comparison = {r["field"]: r for r in rows(ANALYSIS_DIR / "05_model_comparison.csv")}
    conditional = {
        (r["model"], r["field"]): float(r["majority_correct_rate"])
        for r in rows(ANALYSIS_DIR / "05_conditional_correctness.csv")
    }
    groups = {r["indication"]: r for r in rows(ANALYSIS_DIR / "04_group_summary.csv")}
    omnibus = next(
        r for r in rows(ANALYSIS_DIR / "04_omnibus_tests.csv") if r["test"].endswith("common_terms")
    )
    terms = rows(ANALYSIS_DIR / "04_side_effect_comparison.csv")

    # ------------------------------------------------- a: corpus and analysis set
    heading(LEFT, ROW1 + 3.0, "a", "Corpus and analysis set")
    note(LEFT, ROW1 + 8.0, "Reddit archive, 2015–2025, 132 monthly files")  # noqa: RUF001 (en dash intended)
    bw, bh, gap = 37.0, 6.9, 3.6
    x0 = LEFT + 0.3
    cx = x0 + bw / 2
    flow = [
        (matched, "submissions naming the drug", f"{MINUS}{no_text:,} without body text"),
        (
            corpus["submissions_kept"],
            "with body text",
            f"{MINUS}{int(experiment['bonta_only_excluded']):,} matched only by $\\mathit{{bonta}}$",
        ),
        (
            eligible,
            "submitted for extraction",
            f"{MINUS}{eligible - extracted:,} rejected by the provider",
        ),
        (extracted, "extracted", None),
        (self_use, "describe their own treatment", None),
    ]
    top = ROW1 + 12.0
    for i, (number, label, exclusion) in enumerate(flow):
        box(x0, top, bw, bh)
        canvas.text(
            cx, y(top + 2.6), f"{number:,}", fontsize=7, fontweight="bold", ha="center", va="center"
        )
        canvas.text(cx, y(top + 5.2), label, fontsize=6.3, ha="center", va="center")
        if i < len(flow) - 1:
            arrow(cx, top + bh, cx, top + bh + gap)
            if exclusion:
                note(cx + 1.8, top + bh + gap / 2, exclusion, size=6.0)
        top += bh + gap
    last_bottom = top - gap
    split_top = last_bottom + 4.4
    sw = (bw - 2 * 1.3) / 3
    arms = [
        ("Medical", groups["medical"]["self_use_records"], MEDICAL),
        ("Cosmetic", groups["cosmetic"]["self_use_records"], COSMETIC),
        ("Unknown", counts[("c_indication_category", "unknown")], UNKNOWN),
    ]
    for k, (label, n, edge) in enumerate(arms):
        bx = x0 + k * (sw + 1.3)
        arrow(cx, last_bottom, bx + sw / 2, split_top)
        box(bx, split_top, sw, bh, edge=edge, width=0.8)
        canvas.text(
            bx + sw / 2,
            y(split_top + 2.6),
            f"{int(n):,}",
            fontsize=6.8,
            fontweight="bold",
            ha="center",
            va="center",
        )
        canvas.text(bx + sw / 2, y(split_top + 5.2), label, fontsize=6.2, ha="center", va="center")
    note(x0 + 2 * (sw + 1.3) + sw + 1.8, split_top + bh / 2, "not compared", size=6.0)

    # ---------------------------------------------------- b: structured extraction
    heading(RIGHT, ROW1 + 3.0, "b", "Structured extraction")
    note(RIGHT, ROW1 + 8.0, "A language model reads the title and body of each submission")
    note(RIGHT, ROW1 + 11.2, "and returns one schema-validated record.")
    card_top, card_h = ROW1 + 14.6, 20.4
    box(RIGHT, card_top, COL - 0.3, card_h)
    note(RIGHT + 2.2, card_top + 3.2, "CONSTRUCTED EXAMPLE, NOT FROM THE CORPUS", size=5.4)
    canvas.text(
        RIGHT + 2.2,
        y(card_top + 7.4),
        "Botox for chronic migraine, week 3",
        fontsize=6.6,
        fontweight="bold",
        va="center",
    )
    for k, line in enumerate(
        (
            "My neurologist injected 155 units.",
            "The headaches are better, but swallowing",
            "has been hard since.",
        )
    ):
        canvas.text(
            RIGHT + 2.2,
            y(card_top + 11.4 + 3.3 * k),
            line,
            fontsize=6.3,
            style="italic",
            va="center",
        )
    arrow(RIGHT + COL / 2, card_top + card_h, RIGHT + COL / 2, card_top + card_h + 3.6)
    fields = [
        ("Self-use", "yes"),
        ("Indication", "medical: chronic migraine"),
        ("Product", "Botox"),
        ("Dose", "155 units"),
        ("Provider", "neurologist"),
        ("Side effect", "“swallowing has been hard”"),
        ("Candidate term", "dysphagia"),
        ("Attitude to side effect", "distressed"),
    ]
    table_top, pitch = card_top + card_h + 5.8, 3.85
    for k, (label, value) in enumerate(fields):
        row_top = table_top + pitch * k
        note(RIGHT, row_top, label, size=6.3)
        canvas.text(
            RIGHT + 25.5,
            y(row_top),
            value,
            fontsize=6.3,
            va="center",
            fontweight="bold" if label == "Candidate term" else "normal",
        )
        canvas.plot([RIGHT, W - 0.3], [y(row_top + 1.92)] * 2, color=GRID, linewidth=0.45)
    note(
        RIGHT,
        table_top + pitch * len(fields) + 0.6,
        "and 11 further fields, such as onset, duration and age",
        size=6.0,
    )

    # ------------------------------------------ c: model selection by verification
    heading(LEFT, ROW2 + 3.0, "c", "Model selection by blinded verification")
    note(
        LEFT,
        ROW2 + 8.0,
        f"Three annotators judged {annotation['n_decisions']:,} field decisions on",
    )
    note(
        LEFT,
        ROW2 + 11.2,
        f"{annotation['n_submissions']:,} submissions, blind to which model produced each.",
    )
    legend_top = ROW2 + 15.6
    canvas.plot(
        LEFT + 1.2,
        y(legend_top),
        "o",
        markersize=3.4,
        markerfacecolor="white",
        markeredgecolor=NEUTRAL,
        markeredgewidth=0.8,
    )
    note(LEFT + 3.0, legend_top, "DeepSeek V4 Flash", size=6.3, color=INK)
    canvas.plot(LEFT + 26.5, y(legend_top), "o", markersize=3.4, color=INK)
    note(LEFT + 28.3, legend_top, "GPT-5.6 Luna (selected)", size=6.3, color=INK)
    accuracy = [
        (
            "All fields",
            annotation["overall"]["stage1"]["majority_correct_rate"],
            annotation["overall"]["stage3"]["majority_correct_rate"],
        ),
        (
            "Self-use",
            float(comparison["a_self_use"]["stage1_majority_rate"]),
            float(comparison["a_self_use"]["stage3_majority_rate"]),
        ),
        (
            "Product list",
            float(comparison["b_products_mentioned"]["stage1_majority_rate"]),
            float(comparison["b_products_mentioned"]["stage3_majority_rate"]),
        ),
        (
            "Non-empty\nside-effect lists",
            conditional[("stage1", "d_side_effect_pts")],
            conditional[("stage3", "d_side_effect_pts")],
        ),
    ]
    ax = axes_mm(LEFT + 21.0, ROW2 + 19.5, COL - 23.0, 19.0)
    for k, (_label, deepseek, luna) in enumerate(accuracy):
        yy = len(accuracy) - 1 - k
        ax.plot([100 * deepseek, 100 * luna], [yy, yy], color=RULE, linewidth=1.0, zorder=1)
        ax.plot(
            100 * deepseek,
            yy,
            "o",
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor=NEUTRAL,
            markeredgewidth=0.8,
            zorder=2,
        )
        ax.plot(100 * luna, yy, "o", markersize=3.4, color=INK, zorder=3)
    ax.set_yticks(range(len(accuracy)), [a[0] for a in accuracy][::-1], fontsize=6.3)
    ax.set_ylim(-0.6, len(accuracy) - 0.4)
    ax.set_xlim(65, 100)
    ax.set_xticks([70, 80, 90, 100])
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_xlabel("Judged correct (%)", fontsize=6.5, color=MUTED, labelpad=2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)

    # ------------------------------------------- d: medical versus cosmetic reports
    heading(RIGHT, ROW2 + 3.0, "d", "Medical versus cosmetic reports")
    n_med = int(groups["medical"]["side_effect_reporting_records"])
    n_cos = int(groups["cosmetic"]["side_effect_reporting_records"])
    note(RIGHT, ROW2 + 8.0, "Submissions per 100 side-effect reports")
    note(RIGHT, ROW2 + 11.2, f"({n_med:,} medical, {n_cos:,} cosmetic)")
    for x0_, colour, label in (
        (RIGHT + 0.3, MEDICAL, "Medical"),
        (RIGHT + 18.0, COSMETIC, "Cosmetic"),
    ):
        canvas.add_patch(
            plt.Rectangle((x0_, y(ROW2 + 15.6) - 0.95), 1.9, 1.9, color=colour, linewidth=0)
        )
        note(x0_ + 2.8, ROW2 + 15.6, label, size=6.3, color=INK)
    # The three largest differences in each direction; migraine restates an indication and eructation
    # is the intended outcome of cricopharyngeal injection, so neither is shown here.
    shown = [r for r in terms if r["candidate_pt"] not in {"migraine", "eructation"}]
    shown.sort(key=lambda r: float(r["risk_difference_percentage_points"]))
    selected = sorted(
        shown[-3:], key=lambda r: -float(r["risk_difference_percentage_points"])
    ) + sorted(shown[:3], key=lambda r: float(r["risk_difference_percentage_points"]))
    ax = axes_mm(RIGHT + 21.5, ROW2 + 19.0, COL - 22.0, 18.5)
    ypos = list(range(len(selected)))[::-1]
    med = [float(r["medical_records_per_100_reports"]) for r in selected]
    cos = [float(r["cosmetic_records_per_100_reports"]) for r in selected]
    ax.barh(ypos, [-m for m in med], height=0.62, color=MEDICAL, edgecolor="white", linewidth=0.5)
    ax.barh(ypos, cos, height=0.62, color=COSMETIC, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=MUTED, linewidth=0.5)
    ax.set_yticks(
        ypos, [r["candidate_pt"][0].upper() + r["candidate_pt"][1:] for r in selected], fontsize=6.3
    )
    for tick, m, c in zip(ax.get_yticklabels(), med, cos, strict=True):
        tick.set_color(group_colour(m, c))
    limit = math.ceil(max(med + cos) / 5) * 5
    ax.set_xlim(-limit, limit * 0.75)
    ticks = list(range(-limit, int(limit * 0.75) + 1, 5))
    ax.set_xticks(ticks, [str(abs(t)) for t in ticks])
    ax.tick_params(axis="x", labelsize=6.3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(selected) - 0.4)
    med_prev = 100 * float(groups["medical"]["side_effect_report_prevalence"])
    cos_prev = 100 * float(groups["cosmetic"]["side_effect_report_prevalence"])
    note(
        RIGHT,
        ROW2 + 44.6,
        f"Any side effect: medical {med_prev:.1f}%, cosmetic {cos_prev:.1f}%",
        color=INK,
    )
    note(
        RIGHT,
        ROW2 + 47.8,
        f"Profile divergence {float(omnibus['statistic']):.3f} "
        f"(permutation null {float(omnibus['null_mean']):.3f})",
        color=INK,
    )

    save(fig, "11_fig0_study_overview", "study_overview.pdf")


def main() -> None:
    style()
    figure_study_overview()
    figure_overview()
    figure_profile()
    figure_robustness()
    print(f"wrote figures to {FIGURES_DIR} and {MANUSCRIPT_FIGURES}")


if __name__ == "__main__":
    main()
