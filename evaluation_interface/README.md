# BoNT-A extraction evaluation interface

A private, blinded web interface comparing DeepSeek V4 Flash without reasoning against GPT-5.6 Luna. Annotators see one Reddit submission above both structured outputs, then mark every field **correct** or **incorrect**. Model identities are hidden and the left/right response positions are reshuffled whenever a comparison is opened.

The default import takes the successful intersection of `stage1` and `stage3`. With the current experiment checkpoint this is **1,168 submissions and 2,336 blinded response forms per annotator**. The DeepSeek high-reasoning `stage2` run is intentionally excluded.

## What is included

- key-only annotator login, with access keys generated and rotated by an administrator;
- independent binary decisions for all 18 top-level extraction fields;
- separate decisions for every nested `raw_span` and `meddra_pt` value;
- click-to-highlight for `raw_span` and `g_treatment_area_freetext` values when exact wording occurs in the title or body;
- a sticky source panel above two side-by-side, randomly ordered response columns;
- a separate optional review note for each response;
- editable review history for each annotator;
- admin progress by annotator, model-level correctness, field-level unanimous and pairwise agreement, and Fleiss’ kappa;
- private long-format CSV export of decisions, identifiers, notes, submission titles, and submission bodies;
- PostgreSQL for Heroku and SQLite for local development.

## Privacy boundary

This application contains private Reddit title/body text and model-produced verbatim spans. It must not be deployed until the Heroku app, database, administrator access, data-processing terms, backup policy, and institutional approvals have been confirmed for this research data.

The importer copies only `record_id`, `title`, and `selftext` from the source corpus. It does **not** copy author names, permalinks, subreddit names, or other source metadata. Annotators see only a random blinded item ID. The private admin CSV includes the title and body on each decision row, but still omits model values, including verbatim spans. Potential spreadsheet-formula prefixes in untrusted text are neutralized when exported.

For a local SQLite import inside Dropbox, the import command checks that the database is gitignored and sets/verifies `com.dropbox.ignored` on the `instance/` directory before copying any source text. Remote PostgreSQL imports require the explicit `--confirm-private-data` flag.

## Local setup

Run commands from this folder:

```bash
uv sync
cp .env.example .env
```

Generate separate random values for `SECRET_KEY` and `ADMIN_KEY` and place them in `.env`. The included development environment loader makes these values available to the Flask CLI. You may also export them directly, for example:

```bash
export SECRET_KEY="..."
export ADMIN_KEY="..."
uv run flask --app eval_app:create_app init-db
uv run flask --app eval_app:create_app run --debug
```

Open `http://127.0.0.1:5000/admin/login`, sign in with `ADMIN_KEY`, and generate the three annotator keys. Each annotator key is shown once; only its SHA-256 digest is retained.

## Import the evaluation set

First confirm the expected intersection without writing:

```bash
uv run flask --app eval_app:create_app import-data \
  --source-jsonl ../filtered_dataset/submissions.jsonl \
  --results-sqlite ../data_analysis/02_llm_extraction_results/private_extraction.sqlite3 \
  --dry-run
```

Then import into the configured database:

```bash
uv run flask --app eval_app:create_app import-data \
  --source-jsonl ../filtered_dataset/submissions.jsonl \
  --results-sqlite ../data_analysis/02_llm_extraction_results/private_extraction.sqlite3 \
  --confirm-private-data
```

The import is idempotent. It refuses to overwrite changed source text or model output once an affected item has annotations. Use `--limit N` for a small local test. Repeating `--stage NAME` replaces the default two-stage selection.

## Heroku deployment

The app is self-contained in this directory. Deploy it as the repository root (for example, from a branch created with `git subtree split --prefix evaluation_interface`) so Heroku detects this `Procfile` and `pyproject.toml`.

1. Create a Heroku app and attach Heroku Postgres.
2. Configure strong, independent `SECRET_KEY` and `ADMIN_KEY` values.
3. Deploy this directory. The release process runs `init-db`; the web process runs Gunicorn.
4. Import the private data from the approved workstation directly into the attached PostgreSQL database; do not add the corpus or extraction results to the Git slug.

One way to perform step 4 is to expose the app’s `DATABASE_URL` only to the local import process:

```bash
export DATABASE_URL="$(heroku config:get DATABASE_URL --app YOUR_APP)"
uv run flask --app eval_app:create_app import-data \
  --source-jsonl ../filtered_dataset/submissions.jsonl \
  --results-sqlite ../data_analysis/02_llm_extraction_results/private_extraction.sqlite3 \
  --confirm-private-data
unset DATABASE_URL
```

After import, sign in at `/admin/login`, create exactly the intended annotator accounts, and deactivate any test accounts. The admin dashboard is `/admin`; the annotation CSV is `/admin/export/annotations.csv`.

## Agreement definitions

- **Marked correct:** correct decisions divided by all submitted decisions for that canonical field.
- **Unanimous agreement:** percentage of complete rating units on which every active annotator chose the same decision.
- **Pairwise agreement:** agreeing annotator pairs divided by all annotator pairs across complete units.
- **Fleiss’ kappa:** chance-corrected binary agreement, calculated only on units completed by every active annotator.

Nested side-effect fields are grouped across array positions under `d_side_effect_pts[*].raw_span` and `d_side_effect_pts[*].meddra_pt`. Deactivating an annotator removes them from the live agreement denominator but does not delete their annotations or export rows.

## Tests

```bash
uv run pytest
uv run ruff check .
```
