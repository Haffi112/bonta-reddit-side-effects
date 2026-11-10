"""First look at the filtered corpus: what is in it, and what shape is it.

Reads `filtered_dataset/` read-only, writes figures to `data_analysis/figures/`
and a summary to `data_analysis/01_overview_stats.json`. Nothing here interprets
clinical content; that is Phase 3. This is the sanity pass that tells us whether
the corpus is the one we think we built.

The parse of 3.3M comments takes a couple of minutes, so slim per-record fields
are cached to `data_analysis/cache/*.parquet` for the later scripts in this
sequence. The cache carries `author` and `subreddit`, so it is treated as
identifiable and stays gitignored and unsynced, like the corpus itself.

    uv run python data_analysis/01_data_overview.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "filtered_dataset"
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figures"
CACHE = OUT / "cache"

EMPTY_BODY = {"", "[removed]", "[deleted]"}

# The term list mixes a brand, its competitors, and the scientific names. Group
# them, because "which product" is a study variable and "botulinum toxin type a"
# and "onabotulinumtoxina" are the same answer as "botox" for that purpose.
BRAND = {
    "botox": "Botox",
    "onabotulinumtoxina": "Botox",
    "dysport": "Dysport",
    "abobotulinumtoxina": "Dysport",
    "xeomin": "Xeomin",
    "incobotulinumtoxina": "Xeomin",
    "jeuveau": "Jeuveau",
    "prabotulinumtoxina": "Jeuveau",
    "daxxify": "Daxxify",
    "daxibotulinumtoxina": "Daxxify",
    "newtox": "Jeuveau",
}


def brand_of(terms: list[str]) -> str:
    """A named brand if one appears, else the generic scientific vocabulary."""
    for t in terms:
        if t in BRAND:
            return BRAND[t]
    return "Generic / unbranded" if terms else "None (thread reply)"


def utc(v) -> int:
    """created_utc arrives as an int in some months and a string in others."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)
INK = "#1b3a5c"
WARM = "#c2482d"
MUTED = "#8a9bab"


def guard_cache() -> None:
    """The cache holds authors and subreddits. Same two channels as the corpus."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ok = (
        subprocess.run(
            ["git", "check-ignore", "-q", str(CACHE / "comments.parquet")], capture_output=True
        ).returncode
        == 0
    )
    if not ok:
        sys.exit(f"REFUSING: {CACHE}/ is not gitignored; it holds authors and subreddits.")
    if "Dropbox" in str(CACHE):
        subprocess.run(["xattr", "-w", "com.dropbox.ignored", "1", str(CACHE)], check=False)


def count_lines(p: Path) -> int:
    with p.open("rb") as fh:
        return sum(1 for _ in fh)


def load_comments() -> pl.DataFrame:
    if (CACHE / "comments.parquet").exists():
        print("comments: using cache")
        return pl.read_parquet(CACHE / "comments.parquet")
    path = DATA / "comments.jsonl"
    n = count_lines(path)
    rows = {
        k: []
        for k in (
            "subreddit",
            "created_utc",
            "score",
            "body_len",
            "match_reason",
            "parent_status",
            "brand",
            "author",
            "deleted",
        )
    }
    with path.open() as fh:
        for line in tqdm(fh, total=n, desc="comments   ", unit="rec", unit_scale=True):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = (d.get("body") or "").strip()
            rows["subreddit"].append(d.get("subreddit") or "?")
            rows["created_utc"].append(utc(d.get("created_utc")))
            rows["score"].append(int(d.get("score") or 0))
            rows["body_len"].append(0 if body in EMPTY_BODY else len(body))
            rows["match_reason"].append(d.get("_match_reason") or "?")
            rows["parent_status"].append(d.get("_parent_status") or "?")
            rows["brand"].append(brand_of(d.get("_matched_terms") or []))
            rows["author"].append(d.get("author") or "[unknown]")
            rows["deleted"].append(body in EMPTY_BODY)
    df = pl.DataFrame(rows)
    df.write_parquet(CACHE / "comments.parquet")
    return df


def load_submissions() -> pl.DataFrame:
    if (CACHE / "submissions.parquet").exists():
        print("submissions: using cache")
        return pl.read_parquet(CACHE / "submissions.parquet")
    path = DATA / "submissions.jsonl"
    n = count_lines(path)
    rows = {
        k: []
        for k in (
            "subreddit",
            "created_utc",
            "score",
            "num_comments",
            "title_len",
            "selftext_len",
            "brand",
            "author",
            "over_18",
            "is_self",
        )
    }
    with path.open() as fh:
        for line in tqdm(fh, total=n, desc="submissions", unit="rec", unit_scale=True):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = (d.get("selftext") or "").strip()
            rows["subreddit"].append(d.get("subreddit") or "?")
            rows["created_utc"].append(utc(d.get("created_utc")))
            rows["score"].append(int(d.get("score") or 0))
            rows["num_comments"].append(int(d.get("num_comments") or 0))
            rows["title_len"].append(len(d.get("title") or ""))
            rows["selftext_len"].append(len(st))
            rows["brand"].append(brand_of(d.get("_matched_terms") or []))
            rows["author"].append(d.get("author") or "[unknown]")
            rows["over_18"].append(bool(d.get("over_18")))
            rows["is_self"].append(bool(d.get("is_self")))
    df = pl.DataFrame(rows)
    df.write_parquet(CACHE / "submissions.parquet")
    return df


def month_col(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.from_epoch("created_utc", time_unit="s").dt.truncate("1mo").alias("month")
    )


# --- figures -----------------------------------------------------------------


def fig_lengths(subs: pl.DataFrame, coms: pl.DataFrame) -> None:
    """How much text is there per record? This bounds what extraction can see."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, vals, title, colour in (
        (axes[0], subs["selftext_len"].to_numpy(), "Submission body", INK),
        (axes[1], coms.filter(pl.col("body_len") > 0)["body_len"].to_numpy(), "Comment body", WARM),
    ):
        v = vals[vals > 0]
        bins = np.logspace(0, np.log10(max(v.max(), 10)), 60)
        ax.hist(v, bins=bins, color=colour, alpha=0.85, edgecolor="none")
        ax.set_xscale("log")
        med = float(np.median(v))
        ax.axvline(med, color="black", lw=1, ls="--")
        ax.annotate(
            f"median {med:,.0f} chars\nn = {len(v):,}",
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("characters (log scale)")
        ax.set_ylabel("records")
    fig.suptitle("Text length per record", x=0.005, y=1.06, ha="left", fontweight="bold")
    fig.savefig(FIGS / "01_length_distributions.png")
    plt.close(fig)


def _barh(ax, labels, values, colour, xlabel):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colour, alpha=0.9)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=6.5, color="#444")


def fig_top_subreddits(subs: pl.DataFrame, coms: pl.DataFrame, k: int = 25) -> None:
    """Where the discussion lives, and where it hides.

    Left: subreddits whose *posts* are about BoNT-A. Right: subreddits where a
    comment names the drug under a post that never mentioned it. The right-hand
    list is the one a hand-written subreddit list would never have found: the
    drug comes up inside conversations that are ostensibly about something else.
    """
    top_s = subs.group_by("subreddit").len().sort("len", descending=True).head(k)
    orphan = coms.filter(pl.col("parent_status") == "missing")
    top_c = orphan.group_by("subreddit").len().sort("len", descending=True).head(k)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7))
    _barh(axes[0], top_s["subreddit"].to_list(), top_s["len"].to_list(), INK, "submissions")
    axes[0].set_title(
        f"Top {k} subreddits by submission\n(the post itself names a BoNT-A product)",
        loc="left",
        fontweight="bold",
        fontsize=9,
    )
    _barh(axes[1], top_c["subreddit"].to_list(), top_c["len"].to_list(), WARM, "comments")
    axes[1].set_title(
        f"Top {k} subreddits by comment whose parent post\ndoes NOT mention BoNT-A "
        f"(n = {len(orphan):,})",
        loc="left",
        fontweight="bold",
        fontsize=9,
    )
    fig.savefig(FIGS / "02_top_subreddits.png")
    plt.close(fig)

    overlap = set(top_s["subreddit"]) & set(top_c["subreddit"])
    return {
        "top_submission_subs": top_s["subreddit"].to_list(),
        "top_orphan_comment_subs": top_c["subreddit"].to_list(),
        "overlap_in_top_k": sorted(overlap),
    }


def fig_volume(subs: pl.DataFrame, coms: pl.DataFrame) -> None:
    """Volume over time. A sharp bend usually means an artefact, not a trend."""
    s = month_col(subs).group_by("month").len().sort("month")
    c = month_col(coms).group_by("month").len().sort("month")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(c["month"], c["len"], color=WARM, lw=1.6, label="comments")
    ax.plot(s["month"], s["len"], color=INK, lw=1.6, label="submissions")
    ax.set_yscale("log")
    ax.set_ylabel("records per month (log)")
    ax.legend(frameon=False)
    ax.set_title("Corpus volume over time", loc="left", fontweight="bold")
    fig.savefig(FIGS / "03_volume_over_time.png")
    plt.close(fig)


def fig_brand_share(subs: pl.DataFrame, coms: pl.DataFrame) -> None:
    """Which product is named, over time. Botox is a genericised trademark, so
    this is a lower bound on competitor use, not a market share."""
    named = pl.concat(
        [
            month_col(subs).select("month", "brand"),
            month_col(coms)
            .filter(pl.col("brand") != "None (thread reply)")
            .select("month", "brand"),
        ]
    )
    piv = (
        named.group_by(["month", "brand"])
        .len()
        .pivot(values="len", index="month", on="brand")
        .fill_null(0)
        .sort("month")
    )
    brands = [c for c in piv.columns if c != "month"]
    share = piv.select(
        [pl.col("month")]
        + [
            (pl.col(b) / pl.sum_horizontal([pl.col(x) for x in brands]) * 100).alias(b)
            for b in brands
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for b in sorted(brands):
        axes[0].plot(share["month"], share[b], lw=1.5, label=b)
    axes[0].set_ylabel("% of records naming a product")
    axes[0].set_title("Product named, share over time", loc="left", fontweight="bold", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, ncols=2)

    tot = {b: int(piv[b].sum()) for b in brands}
    non_botox = {b: v for b, v in tot.items() if b != "Botox"}
    axes[1].bar(list(non_botox), list(non_botox.values()), color=MUTED)
    axes[1].set_title(
        f"Competitors only (Botox = {tot.get('Botox', 0):,}, excluded for scale)",
        loc="left",
        fontweight="bold",
        fontsize=9,
    )
    axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    axes[1].set_ylabel("records")
    fig.savefig(FIGS / "04_brand_share.png")
    plt.close(fig)
    return tot


def fig_thread_capture(coms: pl.DataFrame) -> None:
    """What thread capture bought us, over time.

    A `thread` comment named no product; we kept it because its parent post did.
    These are the replies that carry the adverse event without repeating the drug
    name, and they are the majority of the corpus.
    """
    m = (
        month_col(coms)
        .group_by(["month", "match_reason"])
        .len()
        .pivot(values="len", index="month", on="match_reason")
        .fill_null(0)
        .sort("month")
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    axes[0].stackplot(
        m["month"],
        m["term"],
        m["thread"],
        labels=["term (names a product)", "thread (parent named it)"],
        colors=[INK, WARM],
        alpha=0.9,
    )
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("comments per month")
    axes[0].set_title(
        "How each comment entered the corpus", loc="left", fontweight="bold", fontsize=9
    )

    ps = coms.group_by("parent_status").len().sort("len", descending=True)
    lbl = {
        "text_post": "parent is a text post",
        "context_post": "parent kept for context\n(no body)",
        "missing": "parent never harvested\n(post never named the drug)",
    }
    axes[1].bar([lbl.get(x, x) for x in ps["parent_status"]], ps["len"], color=[INK, MUTED, WARM])
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].set_ylabel("comments")
    tot = int(ps["len"].sum())
    for i, v in enumerate(ps["len"]):
        axes[1].text(i, v, f"{v:,}\n{100 * v / tot:.1f}%", ha="center", va="bottom", fontsize=7)
    axes[1].set_ylim(0, max(ps["len"]) * 1.22)
    axes[1].set_title(
        "Where each comment's parent ended up", loc="left", fontweight="bold", fontsize=9
    )
    fig.savefig(FIGS / "05_thread_capture.png")
    plt.close(fig)


def fig_deletion(coms: pl.DataFrame) -> None:
    """Deleted and removed comments over time.

    This is the missingness that matters for an adverse-event study: people who
    have a bad outcome are exactly the people who later delete the post. It is
    also why the PRAW-based ingestion we abandoned was unusable.
    """
    m = (
        month_col(coms)
        .group_by("month")
        .agg(pl.col("deleted").mean().alias("rate"), pl.len().alias("n"))
        .sort("month")
    )
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(m["month"], m["rate"] * 100, color=WARM, lw=1.6)
    ax.set_ylabel("% of comments with body\n[deleted] or [removed]")
    ax.set_title(
        "Deleted and removed content over time (the missingness that matters)",
        loc="left",
        fontweight="bold",
        fontsize=9,
    )
    overall = float(coms["deleted"].mean()) * 100
    ax.axhline(overall, color=MUTED, ls="--", lw=1)
    ax.annotate(
        f"overall {overall:.1f}%",
        xy=(0.99, overall),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555",
    )
    fig.savefig(FIGS / "06_deletion_over_time.png")
    plt.close(fig)
    return overall


def fig_engagement(subs: pl.DataFrame, coms: pl.DataFrame) -> None:
    """Thread size and score. Tells us how much context a typical post carries."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    nc = subs["num_comments"].to_numpy()
    nc = nc[nc > 0]
    axes[0].hist(
        nc,
        bins=np.logspace(0, np.log10(max(nc.max(), 10)), 50),
        color=INK,
        alpha=0.85,
        edgecolor="none",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("comments on the post (log)")
    axes[0].set_ylabel("submissions")
    axes[0].axvline(float(np.median(nc)), color="black", lw=1, ls="--")
    axes[0].set_title(
        f"Thread size (median {np.median(nc):,.0f})", loc="left", fontweight="bold", fontsize=9
    )

    for vals, lab, colour in (
        (subs["score"].to_numpy(), "submissions", INK),
        (coms["score"].to_numpy(), "comments", WARM),
    ):
        v = np.clip(vals, -5, 200)
        axes[1].hist(v, bins=60, color=colour, alpha=0.6, label=lab, edgecolor="none", density=True)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("score (clipped to [-5, 200])")
    axes[1].set_ylabel("density (log)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Score distribution", loc="left", fontweight="bold", fontsize=9)
    fig.savefig(FIGS / "07_engagement.png")
    plt.close(fig)


def main() -> int:
    if not DATA.exists():
        sys.exit(f"no dataset at {DATA}")
    guard_cache()
    FIGS.mkdir(parents=True, exist_ok=True)

    subs = load_submissions()
    coms = load_comments()
    print(f"\nsubmissions {len(subs):,} | comments {len(coms):,}\n")

    sub_stats = fig_top_subreddits(subs, coms)
    fig_lengths(subs, coms)
    fig_volume(subs, coms)
    brands = fig_brand_share(subs, coms)
    fig_thread_capture(coms)
    del_rate = fig_deletion(coms)
    fig_engagement(subs, coms)

    orphan = coms.filter(pl.col("parent_status") == "missing")
    stats = {
        "submissions": len(subs),
        "comments": len(coms),
        "unique_subreddits_submissions": subs["subreddit"].n_unique(),
        "unique_subreddits_comments": coms["subreddit"].n_unique(),
        "unique_authors": pl.concat([subs["author"], coms["author"]]).n_unique(),
        "median_submission_body_chars": int(subs["selftext_len"].median()),
        "median_comment_body_chars": int(coms.filter(pl.col("body_len") > 0)["body_len"].median()),
        "comment_parent_status": dict(Counter(coms["parent_status"].to_list())),
        "comment_match_reason": dict(Counter(coms["match_reason"].to_list())),
        "brand_mentions": brands,
        "deleted_or_removed_pct": round(del_rate, 2),
        "orphan_comment_subreddits": orphan["subreddit"].n_unique(),
        **sub_stats,
    }
    (OUT / "01_overview_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print("=== overview ===")
    print(
        f"  subreddits: {stats['unique_subreddits_submissions']:,} with submissions, "
        f"{stats['unique_subreddits_comments']:,} with comments"
    )
    print(f"  unique authors           : {stats['unique_authors']:,}")
    print(
        f"  median body: submission {stats['median_submission_body_chars']:,} chars, "
        f"comment {stats['median_comment_body_chars']:,} chars"
    )
    print(f"  deleted/removed comments : {del_rate:.1f}%")
    print(f"  orphan-comment subreddits: {stats['orphan_comment_subreddits']:,}")
    print(f"  brands: {brands}")
    print(f"\n  {len(list(FIGS.glob('*.png')))} figures -> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
