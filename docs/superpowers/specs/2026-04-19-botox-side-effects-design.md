# Botox (BoNT-A) Self-Reported Side Effects on Reddit: Medical vs Cosmetic Indication Analysis with FAERS Triangulation

**Status:** Draft design — awaiting implementation plan
**Author:** Haffi Einarsson (haffi.e@gmail.com)
**Date:** 2026-04-19
**Template paper:** Sehgal NKR, Tronieri JS, Ungar L, Guntuku SC. *Self-Reported Side Effects of Semaglutide and Tirzepatide in Online Communities.* medRxiv 2026.03.12.26348253v1 (2026-03-13).
**Template code:** https://github.com/sehgal-neil/glp1-side-effects-analysis

---

## 1. Research question and hypotheses

### 1.1 Primary question

How does the profile of self-reported side effects from botulinum toxin type A (BoNT-A) products differ between medical and cosmetic indications as reported on Reddit, and how do both profiles compare against FDA product labeling and against FAERS spontaneous reports?

### 1.2 Hypotheses

**H1 (within-Reddit indication contrast).** The distribution of MedDRA Preferred Terms (PTs) reported on Reddit by users of BoNT-A products differs between cosmetic-indication and medical-indication users.

**H2 (label gap — cosmetic).** A subset of cosmetic-indication PTs frequently reported on Reddit are not enumerated in the FDA labels for Botox Cosmetic, Dysport, Xeomin, Jeuveau, or Daxxify, consistent with the hypothesis that cosmetic AEs are under-characterized in formal pharmacovigilance.

**H3 (Reddit–FAERS divergence).** Reddit surfaces AE signals (PTs) for cosmetic indications at a higher relative frequency than FAERS spontaneous reports stratified to cosmetic indications, consistent with under-reporting of cosmetic AEs through formal channels.

### 1.3 Secondary questions

- Does sentiment about the side effect (distressed / accepting / dismissive) predict discontinuation intent, and does the relationship differ by indication?
- Do product-stratified PT profiles (Botox vs Dysport vs Xeomin vs Jeuveau vs Daxxify) diverge within the cosmetic or medical arm?
- Do temporal trends in specific PTs (e.g., "frozen look", ptosis) correlate with product launches, regulatory events, or media coverage?

---

## 2. Data sources

### 2.1 Reddit

- **Collection tools:** Pushshift + Arctic Shift (the same tooling stack Sehgal et al. used).
- **Window:** 2015-01-01 through 2026-03-31. In practice, nearly all eligible posts are expected to be from 2019 onward, matching the original paper's empirical window.
- **Target subreddits** (15–20, finalized by data-inventory script at §5.1):
  - Medical-leaning: `r/migraine`, `r/ChronicMigraine`, `r/dystonia`, `r/Hyperhidrosis`, `r/OveractiveBladder`, `r/Blepharospasm`, `r/TMJ`, `r/Strabismus`, `r/ChronicPain`, `r/Stroke`.
  - Cosmetic-leaning: `r/Botox`, `r/30PlusSkinCare`, `r/SkincareAddiction`, `r/PlasticSurgery`, `r/CosmeticSurgery`.
  - Mixed: `r/AskDocs`.
- **Search terms:** brand names + generic names + slang + common misspellings — `botox`, `boto`, `botx`, `tox`, `neurotoxin`, `dysport`, `xeomin`, `jeuveau`, `daxxify`, `onabotulinumtoxina`, `abobotulinumtoxina`, `incobotulinumtoxina`, `prabotulinumtoxina`, `daxibotulinumtoxina`.
- **Deduplication:** by post/comment ID; author-level uniqueness enforced at analysis time.

Indication is **not inferred from the subreddit**; subreddit selection is a sampling frame only. Indication is extracted at the post level (§4, field `c`).

### 2.2 FDA product labels

- **Source:** DailyMed SPL XML.
- **Labels to pull:** Botox (medical indications), Botox Cosmetic, Dysport (medical), Dysport Cosmetic, Xeomin (medical), Xeomin Cosmetic, Jeuveau (cosmetic only), Daxxify (medical + cosmetic where applicable), Myobloc (type B — pulled for reference only, not analyzed).
- **Parsed fields:** approved indications, adverse reactions section (§6 of the SPL), post-marketing experience section.
- **Output:** a mapping `(product, indication_arm) → set[MedDRA PT]` where PTs are derived by mapping the label's adverse-reaction terms to MedDRA vocabulary.

### 2.3 FAERS

- **Source:** openFDA `/drug/event.json` endpoint.
- **Window:** 2015-01-01 through most recent quarter available.
- **Query:** by generic and brand name for each of the five BoNT-A products.
- **Indication stratification:** best-effort via the FAERS `drug.drugindication` field. The field is sparse and inconsistent; coverage rate is reported alongside all stratified results.
- **Deduplication:** by `safetyreportid`; follow-up reports collapsed.

---

## 3. Subreddit selection strategy

The 15–20 target subreddits above are the starting set. A data-inventory script (§5.1) enumerates post volume per subreddit matching the search terms; any subreddit with fewer than 50 matching posts is dropped, and the inventory step may add adjacent subreddits surfaced through Reddit's related-community metadata.

Selection does not determine indication labeling. Every post's indication (medical / cosmetic / unknown) is produced by the LLM extractor per §4.

---

## 4. Extraction schema

One JSON schema, one LLM call per post, returned via structured outputs.

```
PostExtraction {
  # Gate (field, not separate stage)
  a_self_use: bool                                   # user personally uses/used a BoNT-A product
  
  # Product
  b_products_mentioned: list[enum]                   # {botox, dysport, xeomin, jeuveau, daxxify, unspecified_bont_a}
                                                      # empty list if a_self_use = false or none identified
  
  # Indication
  c_indication_category: enum                        # {medical, cosmetic, unknown}
  c_indication_subtype: enum | null
    # medical values: {chronic_migraine, cervical_dystonia, blepharospasm, spasticity,
    #                  hyperhidrosis, overactive_bladder, strabismus, tmj_medical, other_medical}
    # cosmetic values: {glabellar, forehead, crows_feet, masseter_jawline,
    #                   neck_platysma, lip_flip, baby_botox, other_cosmetic}
    # null when c_indication_category = unknown
  
  # Core outcome
  d_side_effect_pts: list[{raw_span: str, meddra_pt: str}]   # post-processed via MedDRA RAG map
  
  # Sentiment
  e1_post_overall_sentiment: enum                    # {positive, negative, mixed, neutral}
  e2_sentiment_about_botox: enum                     # {positive, negative, mixed, neutral}
  f_sentiment_about_side_effect: enum | null         # {distressed, accepting, dismissive}
                                                      # null if no side effect mentioned
  
  # Procedure / dose
  g_treatment_area_freetext: str | null              # e.g., "left masseter", "between eyebrows"
  h_dose_units: int | null                           # if disclosed, integer units
  
  # Context
  i_provider_type: enum | null                       # {medspa, dermatologist, plastic_surgeon,
                                                      #  neurologist, other_md, nurse_injector,
                                                      #  diy, unspecified}
  j_first_time_or_repeat: enum | null                # {first_time, repeat}
  j_prior_treatment_count: int | null                # if disclosed
  k_time_to_onset: enum | null                       # {<24h, 1_3d, 4_7d, 1_2w, 2_4w, >4w, unspecified}
  l_duration_status: enum | null                     # {ongoing, resolved_weeks, resolved_months,
                                                      #  permanent, unspecified}
  m_sought_medical_attention: bool | null
  n_discontinuation_intent: enum | null              # {will_continue, considering_stopping,
                                                      #  stopped, unclear}
  
  # Demographics (voluntarily disclosed only)
  o_age_range: enum | null                           # {<20, 20-29, 30-39, 40-49, 50-59, 60-69, 70+}
  o_sex: enum | null                                 # {female, male, nonbinary}
}
```

**Hard rules encoded in the prompt and enforced by post-processing:**

- If `a_self_use = false`, all downstream fields must be null / empty list before analysis. The prompt instructs the model accordingly; a post-processing pass enforces this regardless of model output to guard against hallucinated content on non-self-use posts.
- `d_side_effect_pts` raw spans are mapped to MedDRA PTs by a separate RAG step (§5.3), not by the LLM directly producing PT strings. The LLM produces a candidate PT; the RAG step validates and canonicalizes it against the MedDRA dictionary.
- Null-allowing fields are always reported with a **coverage rate** in addition to extraction accuracy.

---

## 5. Extraction pipeline

### 5.1 Ingestion

1. Data-inventory script: query Pushshift + Arctic Shift for each candidate subreddit and search term, produce a per-subreddit volume table. Drop subreddits with <50 matching posts.
2. Full ingestion: pull posts and top-level comments matching the search terms from the finalized subreddit set for the window 2015-01-01 to 2026-03-31.
3. Raw data is written to `data/raw/` (git-ignored, local only).
4. Deduplication (by ID), language filter (English only for v1), length filter (drop <20 characters), and a basic spam filter (bot posts, repeated template text) produce `data/interim/posts.parquet`.

### 5.2 LLM extraction

- **Model:** `gpt-5.3-chat-latest` (non-reasoning, structured outputs enabled).
- **Prompt design:** single system prompt + user message containing the post body. Structured-output JSON schema matches §4 exactly. Schema is enforced by the API — no response parser needed beyond JSON deserialization.
- **Batching:** OpenAI Batch API where possible to cut cost by ~50% for the main extraction pass. Synchronous endpoint used only for validation annotation and retries.
- **Retries:** 3 attempts with exponential backoff on 5xx / rate-limit errors. Posts that fail three times are flagged and reported as extraction-failure counts in the final manuscript.
- **Prompts versioned** under `config/llm_prompts/` with semantic version tags; every output record stores the prompt version used.

### 5.3 MedDRA mapping (RAG)

1. MedDRA PT dictionary loaded into a vector index (e.g., `sentence-transformers/all-MiniLM-L6-v2` embeddings, FAISS index).
2. For each `d_side_effect_pts[i].raw_span`:
   - Exact-match against PT strings and common synonyms first.
   - If no exact match, retrieve top-5 nearest PTs from the vector index.
   - LLM reranker (same `gpt-5.3-chat-latest`, separate lightweight prompt) picks the best match or returns "no_acceptable_match".
3. "No_acceptable_match" spans are collected into a review file; a human reviewer maps them in a final manual pass, producing a project-specific `raw_span → PT` override table for reproducibility.

### 5.4 Single-stage extraction rationale

The original paper used two stages (self-use classifier → side-effect extractor). Collapsing to one stage:

- Same full schema is requested on every eligible post; non-self-use posts return `a_self_use = false` with other fields defaulted.
- Cost implication: the full extraction prompt runs on *all* matched posts instead of only self-use posts. With `gpt-5.3-chat-latest` pricing and structured outputs (no thinking trace), this is the cheaper overall path when output tokens on non-self-use posts are minimal (all nulls).
- Reliability implication: one prompt to maintain, one validation statistic per field, no inter-stage coupling bugs.

---

## 6. Analysis plan

### 6.1 Cohort construction

After extraction, partition extracted records:

- **Person-indication record:** a (user, indication_category) pair where the user has at least one self-use post with that indication. A user whose self-use posts span both medical and cosmetic contributes one record to each arm; a user with only one indication contributes one record. Posts tagged `c_indication_category = unknown` do not create a record on their own — such a user appears in no arm unless they also have at least one non-unknown post.
- **User-level rollup (secondary):** unique Reddit author IDs with the set of arms they appear in. Reported for context, not used as the primary denominator.
- **Primary denominator for PT prevalence and ROR:** person-indication records. Secondary analyses at the user level are reported for sensitivity.

### 6.2 Descriptive analysis (H0)

For each arm (Reddit-Medical, Reddit-Cosmetic, Reddit-Unknown):

- Arm size (users, person-indications, posts).
- PT prevalence: `#_users_reporting_PT / #_users_in_arm`.
- PT grouped by MedDRA System Organ Class (SOC), matching Sehgal Table 1 structure.
- Top-20 PTs per arm.
- Product-stratified sub-tables (five products + `unspecified_bont_a`).

### 6.3 Disproportionality — within-Reddit (H1)

For each PT with ≥20 reports across both arms:

- Compute Reporting Odds Ratio (ROR) with Reddit-Cosmetic as exposure group and Reddit-Medical as reference.
- 95% CIs via log-normal approximation on ln(ROR).
- Benjamini–Hochberg FDR correction at q = 0.05 across the PT set.
- Results labeled explicitly as **reporting-pattern** ROR, not incidence RR.

### 6.4 Label concordance (H2)

For each arm and each product's relevant label:

- Classify each top-N PT as **in label** or **not in label** (label arm-matched: cosmetic Reddit PTs checked against cosmetic labels; medical Reddit PTs against medical labels).
- Output a concordance matrix: rows = PTs, columns = (product × indication-arm), cells = in-label / not-in-label / not-reported-in-arm.
- The headline result for H2: top-N cosmetic PTs not appearing in any cosmetic label.

### 6.5 FAERS triangulation (H3)

- For each BoNT-A product, pull FAERS reports with product as suspect (`drugcharacterization = 1`) or concomitant (`drugcharacterization = 2`).
- Indication stratification via `drug.drugindication`; coverage rate reported.
- For each PT with sufficient volume: Reddit-arm prevalence vs FAERS-arm prevalence. Plot concordance; tabulate divergences.
- Pre-specified test: Spearman rank correlation of PT ranks between Reddit-Cosmetic and FAERS-Cosmetic, and the same between Reddit-Medical and FAERS-Medical. H3 predicts the cosmetic correlation is weaker than the medical correlation.

### 6.6 Temporal analysis

- Monthly PT report counts per arm for top-15 PTs per arm.
- Normalized by total arm volume per month (rate, not raw count) to separate signal from discussion-volume growth.
- Visual annotation of step-changes coincident with regulatory events (e.g., Daxxify FDA approval 2022-09-07), major product launches, or viral news coverage.

### 6.7 Sentiment analysis

- Distribution of `e1_post_overall_sentiment`, `e2_sentiment_about_botox`, `f_sentiment_about_side_effect` per arm.
- Cross-tab: `f_sentiment_about_side_effect` × PT presence × arm. Tests whether cosmetic users reporting serious-appearing PTs (dysphagia, systemic weakness) skew dismissive relative to medical users reporting the same PTs.
- Relationship: `n_discontinuation_intent` as a function of `f_sentiment_about_side_effect`, stratified by arm.

---

## 7. Deliverables

### 7.1 Repository layout

```
botox-side-effects-analysis/
├── README.md
├── LICENSE                                   # MIT for code, CC-BY-4.0 for docs
├── pyproject.toml                            # Python 3.12, uv-managed
├── config/
│   ├── subreddits.yaml
│   ├── search_terms.yaml
│   └── llm_prompts/                          # versioned prompt files
├── src/botox_se/
│   ├── __init__.py
│   ├── ingest/                               # Pushshift + Arctic Shift collection
│   ├── extract/                              # single-stage LLM extractor
│   ├── meddra/                               # PT dictionary + RAG mapping
│   ├── faers/                                # openFDA ingestion + normalization
│   ├── labels/                               # DailyMed SPL parsing
│   ├── analysis/                             # descriptive, ROR, concordance, temporal, sentiment
│   └── reporting/                            # table & figure generation
├── data/
│   ├── raw/                                  # gitignored
│   ├── interim/
│   └── processed/
├── notebooks/                                # exploration + final figure generation
├── validation/
│   ├── annotation_protocol.md
│   ├── annotated_sample.csv                  # 200 posts, dual-annotated
│   └── reliability_report.ipynb
├── docs/
│   ├── superpowers/specs/                    # this spec + plan
│   ├── methods.md
│   └── paper/                                # manuscript drafts
└── tests/                                    # unit + integration
```

### 7.2 Publication targets

- medRxiv preprint.
- Target journals (primary → secondary): *JMIR Public Health and Surveillance*, *Drug Safety*, *Pharmacoepidemiology and Drug Safety*.

### 7.3 Code & data release

- Code: public GitHub repo, MIT license.
- Prompts: versioned in-repo, fully disclosed.
- Derived data (MedDRA PT counts per arm, FAERS summary stats, label concordance matrix): released as CSV artifacts.
- Raw Reddit data: not redistributed (Reddit ToS); repository documents the exact collection procedure for re-derivation.

---

## 8. Validation and reliability

### 8.1 Annotation

- **Pilot:** 20 posts, two annotators, consensus discussion on every disagreement. Produces annotation-guide v1.
- **Main:** 200 posts, stratified-random sample across subreddits and indication types, dual-annotated. Disagreements resolved by a third adjudicator.
- Annotators are blinded to model outputs during annotation.

### 8.2 Reliability metrics

- **Cohen's κ per field** (categorical/ordinal fields): `c_indication_category`, `c_indication_subtype`, `e1`, `e2`, `f`, `i`, `j`, `k`, `l`, `n`, `o_age_range`, `o_sex`.
- **Sensitivity / PPV** for `a_self_use`, `b_products_mentioned` (per product), `d_side_effect_pts` (span-level and PT-level), `h_dose_units` (presence + value), `m_sought_medical_attention`.
- **Coverage rate** for every null-allowing field: `% of posts where the field is non-null in the human gold set`.

### 8.3 Reliability thresholds

- Fields with Cohen's κ < 0.6 on the subjective set (sentiment, severity) are reported with an explicit caveat and are **excluded from inferential analyses** (ROR, rank correlations). They are retained for descriptive reporting only.
- Fields with extraction precision < 0.8 are flagged in the manuscript's methods section and sensitivity analyses are run with those fields excluded.

---

## 9. Ethics, Terms of Service, MedDRA licensing

### 9.1 IRB / ethics review

- Public, non-identifiable Reddit content, consistent with Sehgal et al. and established precedent in social-media pharmacovigilance research.
- Confirm with the University of Iceland IRB (Vísindasiðanefnd) before data collection. Expected outcome: a no-review / exemption letter. This step is a plan milestone, not an assumption.

### 9.2 Reddit Terms of Service

- Raw posts / comments are not redistributed in any form.
- Aggregated extraction outputs (PT counts per arm, MedDRA mappings) are distributable.
- Code, prompts, and config are fully public.
- No attempts to access private, deleted, or restricted content.

### 9.3 Author / poster protection

- No contact with posters under any circumstance.
- No attempts to de-pseudonymize.
- No verbatim post quotes in the manuscript; any illustrative examples are paraphrased.
- Demographics (`o_age_range`, `o_sex`) are extracted only when voluntarily disclosed; no inference from name, subreddit, or writing style.

### 9.4 MedDRA license

- MedDRA is proprietary; a non-commercial academic subscription via MSSO is required before §5.3 can run.
- License application is a plan milestone with expected 2–6 week turnaround.

---

## 10. Risks and limitations

Stated up front so the manuscript abstract can state them too:

1. **Population confounding.** Reddit-Cosmetic and Reddit-Medical user populations differ on age, sex, reason-for-posting, and willingness-to-complain. All inter-arm ROR estimates are *reporting-pattern* comparisons, not incidence estimates. The paper states this prominently in abstract + discussion.
2. **Self-selection to Reddit.** Reddit users skew younger, more male, more US-based than general Botox-using populations. This is a limitation inherited from the source paper and noted identically.
3. **Product identification.** Patients often say "Botox" regardless of actual product used. The `unspecified_bont_a` bucket is expected to be the largest single product category, and product-stratified claims are interpreted accordingly.
4. **Indication mis-classification.** LLM may misread a post whose author mentions both cosmetic and medical uses. Handled by: `c_indication_category = unknown` when ambiguous, and person-indication records for split users.
5. **FAERS limitations.** FAERS indication stratification is sparse; FAERS itself has known reporting biases (clinician-driven for medical, rare for cosmetic). The triangulation is descriptive, not inferential.
6. **MedDRA mapping error.** Adjudicated by human reviewer for unmatched spans; the review file is part of the reproducibility artifact.
7. **Sentiment reliability.** If Cohen's κ on sentiment fields falls below 0.6, those fields are descriptive-only (§8.3).
8. **LLM drift.** `gpt-5.3-chat-latest` can be updated by OpenAI without notice. All extractions tag the model snapshot version (`model_fingerprint` from the OpenAI response); re-extraction on a locked model snapshot is required if the paper is revised after a model update.

---

## 11. Open items (defaults chosen; revisit any before plan writing)

| Item | Default | Status |
|------|---------|--------|
| Collection window end | 2026-03-31 | Locked unless user revises |
| Primary LLM provider & model | OpenAI `gpt-5.3-chat-latest`, structured outputs, no thinking trace | **Confirmed by user** |
| Extraction pipeline stages | Single stage with `a_self_use` as a field | **Confirmed by user** |
| Language / runtime | Python 3.12, `uv`-managed | Default |
| Validation sample size | 200 posts dual-annotated | Default |
| FAERS window | 2015-present | Default |
| Subreddit count | 15–20, data-inventory-filtered | Default |
| Language filter | English only for v1 | Default |
| Primary analysis unit | Person-indication (user × arm) | Default |
| Repo license | MIT (code), CC-BY-4.0 (docs) | Default |

---

## 12. What this spec does NOT cover

- Detailed implementation steps, file-by-file build order, test specifications — those belong in the implementation plan produced by the writing-plans skill after this spec is approved.
- Manuscript outline, figure drafts, author list — handled separately after analysis results are in.
- Grant applications, funding — assumed covered by the user's institutional resources.
