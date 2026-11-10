# Botulinum toxin side effects on Reddit: code and aggregate results

Code and aggregate output tables for the article *Self-reported side effects of botulinum toxin
type A in medical and cosmetic use on Reddit* (Einarsson, Santos, Örnólfsson, Sverrisdóttir and
Benónísdóttir; under review). The archived version of this repository is on Zenodo; the DOI is
given in the article's Code availability statement.

## What is here

| Path | Contents |
|---|---|
| `data_analysis/02_llm_extraction_experiment.py`, `06_luna_batch_extraction.py` | Extraction: prompt, JSON schema, Pydantic validation, model-selection runs and the full-corpus batch run |
| `data_analysis/02_llm_extraction_output_form.json` | The extraction schema as a readable form |
| `data_analysis/03_*.py` to `13_*.py` | Numbered analysis scripts, in the order they are run |
| `data_analysis/12_reproduce_paper.ipynb` | Notebook that re-implements every analysis and checks the printed numbers (outputs cleared) |
| `data_analysis/*.csv`, `*.json`, `*.md` | Aggregate output tables behind the tables and figures in the article |
| `data_analysis/12_notebook_outputs/reproduction_checks.csv` | Each number in the article, the value the notebook recomputed, and the result |
| `data_analysis/figures/11_fig*.pdf` | The article's figures, drawn by `11_manuscript_figures.py` from the aggregate tables |
| `evaluation_interface/` | The web application used for the blinded human verification (no data) |
| `src/`, `scripts/`, `config/`, `tests/` | Shared package, search terms and extraction enumerations, and tests |
| `docs/superpowers/specs/` | The study design document written before data collection |

The collection and filtering code that scans the archive is a separate repository:
<https://github.com/Haffi112/reddit_torrent_filtering>.

## Data policy

The study was approved by the National Bioethics Committee of Iceland (Vísindasiðanefnd, reference
VSN2608019). The approved research plan limits publication to non-identifiable aggregate results.
This repository therefore contains no Reddit post text, usernames or record identifiers, and no
row-level extraction records.

- **Small cells.** A count of one to four submissions is shown as `<5`. Any value from which such a
  count could be recomputed (the other group's count, rates, differences, intervals and P values on
  the same row, and the matching clustered estimates in `08_cluster_terms.csv`) is shown as `---`.
  This affects 47 of 221 terms in `04_side_effect_comparison.csv` and 38 of 189 in the DeepSeek
  comparison.
- **Verification counts** (true and false positives and negatives, agreement counts) describe
  model errors rather than groups of authors and are reported in full.
- **Annotators** appear as `annotator_1` to `annotator_3`.
- **Single-record values** (minimum and maximum of dose and similar fields) are omitted.
- Reddit user-profile pages are omitted from every community-level table.

## Reproducing the analysis

1. Obtain the Pushshift and Arctic Shift monthly dumps from Academic Torrents (infohash
   `3d426c47c767d40f82c7ef0f47c3acacedd2bf44`) and filter them with the collection code above.
2. Install the environment with [uv](https://docs.astral.sh/uv/): `uv sync`. Copy `.env.example` to
   `.env` and add API keys for OpenAI and OpenRouter.
3. Run the extraction (`02`, then `06`) and the analysis scripts in numerical order, for example
   `uv run python data_analysis/04_medical_cosmetic_comparison.py`.
4. Run `data_analysis/12_reproduce_paper.ipynb` to recompute every number in the article.

The aggregate tables are enough to recompute the prevalence comparisons and the per-term tests for
terms without a suppressed cell. The permutation tests, the bootstrap and the other resampling
analyses need the row-level extraction records, which steps 1 to 3 re-derive from the public archive.
Re-running the extraction calls models with provider-default sampling, so individual outputs can
differ slightly from those analysed in the article.

## Licence

MIT; see `LICENSE`.
