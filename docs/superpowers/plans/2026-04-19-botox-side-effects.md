# Botox (BoNT-A) Side Effects Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible research pipeline that collects Reddit posts about botulinum toxin type A (BoNT-A) products, extracts self-reported side effects via structured-output LLM calls, and compares profiles across medical vs cosmetic indications against FDA product labels and FAERS spontaneous reports.

**Architecture:** Python 3.12 monorepo organized around pipeline stages (ingest → extract → map → analyze → report). Each stage writes parquet / JSON artifacts to `data/interim/` or `data/processed/`; downstream stages are pure consumers of those artifacts. Single-stage LLM extraction with OpenAI structured outputs (`gpt-5.3-chat-latest`). MedDRA mapping via hybrid exact-match + FAISS retrieval + LLM reranker. Analysis: descriptive stats, ROR disproportionality, FAERS triangulation, label concordance, temporal trends, sentiment. Dual-annotated 200-post validation gate for reliability metrics.

**Tech Stack:** Python 3.12, `uv`, `pydantic` v2, `polars`, `openai`, `sentence-transformers`, `faiss-cpu`, `httpx`, `typer`, `pytest`, `ruff`, `mypy`, `jupyter`. External: MedDRA (licensed), openFDA (FAERS), DailyMed (SPL XML), Arctic Shift + Pushshift (Reddit).

**Spec reference:** `docs/superpowers/specs/2026-04-19-botox-side-effects-design.md`. Read that first; this plan assumes you have.

**Scope note:** This is a research pipeline, not a shipped product. "Done" for a task means its tests pass, its artifact is committed to the repo (for code) or to `data/processed/` for analysis outputs (gitignored but reproducible), and the task's checkboxes are all ticked.

**Phase gates:** Each phase ends with a deliverable that is independently usable. You may stop after any phase end and the repo is in a coherent state.

---

## File structure overview

Each file has one responsibility. Prefer additional small files over large ones. Exact paths are prescriptive — do not rename without reason.

```
botox-side-effects-analysis/
├── README.md
├── LICENSE                                  # MIT for code
├── docs/LICENSE-docs                        # CC-BY-4.0 for docs
├── pyproject.toml                           # uv-managed, Python 3.12
├── .python-version                          # "3.12"
├── .env.example                             # OPENAI_API_KEY, OPENFDA_API_KEY, etc.
├── .gitignore
├── .pre-commit-config.yaml                  # ruff + mypy
├── .github/workflows/ci.yml                 # lint, typecheck, pytest
├── Makefile                                 # orchestrates full pipeline runs
│
├── config/
│   ├── subreddits.yaml                      # candidate subreddit list
│   ├── search_terms.yaml                    # brand + generic + slang + misspellings
│   ├── extraction_enums.yaml                # all schema enum values (single source of truth)
│   └── llm_prompts/
│       └── extract_v1.md                    # prompt for gpt-5.3-chat-latest
│
├── src/botox_se/
│   ├── __init__.py
│   ├── config.py                            # pydantic-settings loader
│   ├── logging.py                           # structlog config
│   ├── schema.py                            # pydantic PostExtraction + ancillary types
│   ├── paths.py                             # central resolver for data/ subpaths
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── inventory.py                     # data-inventory script: subreddit volumes
│   │   ├── reddit_client.py                 # Arctic Shift + Pushshift wrappers
│   │   ├── fetch.py                         # main collection runner
│   │   ├── filters.py                       # language/length/spam filters
│   │   └── dedupe.py                        # dedup by ID + author+content hash
│   │
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── prompt.py                        # loads prompt template, injects schema
│   │   ├── client.py                        # OpenAI structured-outputs wrapper
│   │   ├── postprocess.py                   # enforces null-rules for a_self_use=False
│   │   ├── batch.py                         # OpenAI Batch API orchestration
│   │   └── runner.py                        # streams posts → LLM → parquet
│   │
│   ├── meddra/
│   │   ├── __init__.py
│   │   ├── dictionary.py                    # MedDRA PT loader + synonym expansion
│   │   ├── exact_match.py                   # exact + normalized string match
│   │   ├── retriever.py                     # FAISS index + top-k retrieval
│   │   ├── reranker.py                      # LLM reranker
│   │   └── mapper.py                        # orchestrates the three-step mapping
│   │
│   ├── faers/
│   │   ├── __init__.py
│   │   ├── openfda_client.py                # httpx wrapper w/ pagination + backoff
│   │   ├── query.py                         # per-product queries
│   │   ├── normalize.py                     # collapse follow-up reports, PT extraction
│   │   └── indication.py                    # best-effort indication stratification
│   │
│   ├── labels/
│   │   ├── __init__.py
│   │   ├── dailymed_client.py               # SPL download wrapper
│   │   ├── parse.py                         # XML → adverse-reactions section
│   │   └── pt_map.py                        # label terms → MedDRA PTs
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── cohort.py                        # person-indication record builder
│   │   ├── descriptive.py                   # per-arm PT prevalence, SOC grouping
│   │   ├── ror.py                           # ROR + 95% CI + BH correction
│   │   ├── concordance.py                   # label concordance matrix
│   │   ├── triangulation.py                 # Reddit vs FAERS rank correlation
│   │   ├── temporal.py                      # monthly rates per arm
│   │   └── sentiment.py                     # sentiment × PT × arm cross-tabs
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── tables.py                        # Table 1 style outputs
│   │   ├── figures.py                       # matplotlib figures (pub quality)
│   │   └── markdown.py                      # auto-generated results sections
│   │
│   └── validation/
│       ├── __init__.py
│       ├── sampler.py                       # stratified 200-post sampler
│       ├── reliability.py                   # Cohen's κ, sensitivity, PPV
│       └── gate.py                          # enforces κ thresholds before analysis
│
├── scripts/                                 # one-file CLI entrypoints
│   ├── run_inventory.py
│   ├── run_collection.py
│   ├── run_extraction.py
│   ├── run_mapping.py
│   ├── run_faers.py
│   ├── run_labels.py
│   ├── run_analysis.py
│   ├── build_annotation_sample.py
│   └── build_reliability_report.py
│
├── data/
│   ├── raw/                                 # gitignored
│   ├── interim/                             # gitignored
│   ├── processed/                           # gitignored (but reproducible)
│   └── reference/                           # committed: MedDRA subset (license permitting), enums
│
├── notebooks/
│   ├── 00_inventory_exploration.ipynb
│   ├── 01_extraction_diagnostics.ipynb
│   ├── 02_descriptive_results.ipynb
│   ├── 03_ror_results.ipynb
│   ├── 04_faers_triangulation.ipynb
│   ├── 05_temporal_trends.ipynb
│   └── 06_sentiment_analysis.ipynb
│
├── validation/
│   ├── annotation_protocol.md
│   ├── annotation_sample.csv                # 200-post stratified sample (gitignored content, schema committed)
│   ├── annotations_A.csv                    # annotator A output (gitignored)
│   ├── annotations_B.csv                    # annotator B output (gitignored)
│   ├── adjudication.csv                     # disagreement resolution (gitignored)
│   └── reliability_report.md                # committed outputs: κ, sensitivity, PPV tables
│
├── docs/
│   ├── superpowers/
│   │   ├── specs/2026-04-19-botox-side-effects-design.md
│   │   └── plans/2026-04-19-botox-side-effects.md   # this file
│   ├── methods.md                           # methods section draft
│   └── paper/
│       ├── manuscript.md
│       ├── figures/
│       └── tables/
│
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_schema.py
    ├── ingest/...
    ├── extract/...
    ├── meddra/...
    ├── faers/...
    ├── labels/...
    ├── analysis/...
    └── fixtures/                            # small canned inputs
```

---

# Phase 0 — Repo setup and tooling

**Deliverable:** A clean Python 3.12 repo with CI, linting, typechecking, tests, config loader, logging. `make test` green.

## Task 0.1: Initialize repo and git

**Files:**
- Create: `.python-version`, `.gitignore`, `README.md`, `LICENSE`
- Init: `.git/`

- [ ] **Step 1: Initialize git**

Run: `cd /Users/hafsteinneinarsson/Dropbox/Haskoli_Islands/Verkefni/2026_botox_study && git init -b main`
Expected: `Initialized empty Git repository in .../2026_botox_study/.git/`

- [ ] **Step 2: Write `.python-version`**

```
3.12
```

- [ ] **Step 3: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/

# Env
.env
.env.*
!.env.example

# Data (research outputs — reproducible, not committed)
data/raw/
data/interim/
data/processed/

# Notebooks
.ipynb_checkpoints/

# Paper artifacts (regenerable)
docs/paper/figures/*.pdf
docs/paper/figures/*.png

# Validation (sensitive content — structure committed separately)
validation/annotation_sample.csv
validation/annotations_A.csv
validation/annotations_B.csv
validation/adjudication.csv

# OS
.DS_Store
```

- [ ] **Step 4: Write `LICENSE` (MIT)**

Standard MIT text with copyright `Copyright (c) 2026 Haffi Einarsson and contributors`.

- [ ] **Step 5: Write minimal `README.md`**

```markdown
# Botox Side Effects Analysis

Reproducible pipeline analyzing self-reported side effects of botulinum toxin type A products on Reddit, comparing medical vs cosmetic indications against FDA labels and FAERS.

**Status:** Under development — see `docs/superpowers/plans/2026-04-19-botox-side-effects.md` for progress.

## Setup

```bash
uv sync
cp .env.example .env
# fill in API keys in .env
make test
```

See `docs/methods.md` for methodology and `docs/superpowers/specs/` for the design spec.
```

- [ ] **Step 6: Commit**

```bash
git add .python-version .gitignore README.md LICENSE
git commit -m "chore: initialize repository with license, gitignore, readme"
```

## Task 0.2: Python project with uv

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Install uv if missing**

Run: `which uv || curl -LsSf https://astral.sh/uv/install.sh | sh`
Expected: `uv` on PATH.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "botox-se"
version = "0.1.0"
description = "Botox side-effects analysis on Reddit"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = { file = "LICENSE" }
authors = [{ name = "Haffi Einarsson" }]

dependencies = [
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "polars>=1.0",
  "pyarrow>=16.0",
  "typer>=0.12",
  "httpx>=0.27",
  "tenacity>=8.3",
  "structlog>=24.1",
  "pyyaml>=6.0",
  "openai>=1.40",
  "sentence-transformers>=3.0",
  "faiss-cpu>=1.8",
  "scipy>=1.13",
  "statsmodels>=0.14",
  "matplotlib>=3.9",
  "seaborn>=0.13",
  "lxml>=5.2",
  "langdetect>=1.0.9",
]

[dependency-groups]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0",
  "pytest-httpx>=0.30",
  "ruff>=0.5",
  "mypy>=1.10",
  "pre-commit>=3.7",
  "jupyter>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/botox_se"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers --cov=botox_se --cov-report=term-missing"
testpaths = ["tests"]
```

- [ ] **Step 3: Create source directory**

Run: `mkdir -p src/botox_se tests && touch src/botox_se/__init__.py tests/__init__.py`

- [ ] **Step 4: Install**

Run: `uv sync`
Expected: virtualenv created, all deps installed, `uv.lock` generated.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/
git commit -m "chore: pyproject.toml with dependencies, tool configs"
```

## Task 0.3: Pre-commit, linting, type-checking

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.7, types-PyYAML]
        args: [--config-file=pyproject.toml]
        files: ^src/
```

- [ ] **Step 2: Install hooks**

Run: `uv run pre-commit install`
Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 3: Run once on all files**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass (only whitespace / trivial fixes on fresh repo).

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: pre-commit hooks for ruff + mypy"
```

## Task 0.4: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI workflow**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src/
      - run: uv run pytest
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: github actions workflow for lint, typecheck, tests"
```

## Task 0.5: Central paths and config

**Files:**
- Create: `src/botox_se/paths.py`, `src/botox_se/config.py`, `.env.example`
- Test: `tests/test_config.py`, `tests/test_paths.py`

- [ ] **Step 1: Write failing test `tests/test_paths.py`**

```python
from pathlib import Path

from botox_se.paths import Paths


def test_paths_resolves_subdirs_relative_to_repo_root(tmp_path):
    p = Paths(repo_root=tmp_path)
    assert p.raw == tmp_path / "data" / "raw"
    assert p.interim == tmp_path / "data" / "interim"
    assert p.processed == tmp_path / "data" / "processed"
    assert p.reference == tmp_path / "data" / "reference"


def test_paths_ensures_dirs_exist(tmp_path):
    p = Paths(repo_root=tmp_path)
    p.ensure_dirs()
    assert p.raw.is_dir()
    assert p.interim.is_dir()
    assert p.processed.is_dir()
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/test_paths.py -v`
Expected: `ModuleNotFoundError: No module named 'botox_se.paths'`

- [ ] **Step 3: Implement `src/botox_se/paths.py`**

```python
"""Central filesystem paths for the project."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def reference(self) -> Path:
        return self.data / "reference"

    @property
    def config(self) -> Path:
        return self.repo_root / "config"

    def ensure_dirs(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.reference):
            p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_paths.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write failing test `tests/test_config.py`**

```python
from botox_se.config import Settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENFDA_API_KEY", "openfda-test")
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.openfda_api_key.get_secret_value() == "openfda-test"
    assert s.llm_model == "gpt-5.3-chat-latest"
    assert s.collection_start == "2015-01-01"
    assert s.collection_end == "2026-03-31"
```

- [ ] **Step 6: Run — expect fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 7: Implement `src/botox_se/config.py`**

```python
"""Typed settings loader. Reads from environment (via .env) with defaults."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = "gpt-5.3-chat-latest"
    llm_reranker_model: str = "gpt-5.3-chat-latest"
    llm_temperature: float = 0.0

    # External APIs
    openfda_api_key: SecretStr = Field(default=SecretStr(""))

    # Collection window
    collection_start: str = "2015-01-01"
    collection_end: str = "2026-03-31"

    # Runtime
    batch_api_enabled: bool = True
    concurrent_requests: int = 8


def load() -> Settings:
    """Entrypoint used throughout code; tests can patch env vars."""
    return Settings()
```

- [ ] **Step 8: Write `.env.example`**

```
# Required
OPENAI_API_KEY=
OPENFDA_API_KEY=

# Optional (defaults are fine)
LLM_MODEL=gpt-5.3-chat-latest
COLLECTION_START=2015-01-01
COLLECTION_END=2026-03-31
BATCH_API_ENABLED=true
CONCURRENT_REQUESTS=8
```

- [ ] **Step 9: Run — expect pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 1 passed.

- [ ] **Step 10: Commit**

```bash
git add src/botox_se/paths.py src/botox_se/config.py tests/test_paths.py tests/test_config.py .env.example
git commit -m "feat(core): central paths and typed settings loader"
```

## Task 0.6: Structured logging

**Files:**
- Create: `src/botox_se/logging.py`
- Test: `tests/test_logging.py`

- [ ] **Step 1: Write failing test**

```python
import json
import logging

from botox_se.logging import configure_logging, get_logger


def test_logger_outputs_json_with_context(capsys):
    configure_logging(level="INFO", json=True)
    log = get_logger("test").bind(task="ingest", subreddit="r/botox")
    log.info("fetched", count=42)
    captured = capsys.readouterr().err.strip().splitlines()
    assert len(captured) == 1
    record = json.loads(captured[0])
    assert record["event"] == "fetched"
    assert record["count"] == 42
    assert record["task"] == "ingest"
    assert record["level"] == "info"
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/test_logging.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/botox_se/logging.py`**

```python
"""Structured logging with structlog. JSON output for machine parsing."""

import logging
import sys

import structlog


def configure_logging(*, level: str = "INFO", json: bool = True) -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_logging.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/botox_se/logging.py tests/test_logging.py
git commit -m "feat(core): structured JSON logging via structlog"
```

## Task 0.7: Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write `Makefile`**

```makefile
.PHONY: install test lint typecheck fmt clean \
        inventory collect extract map faers labels analyze \
        pipeline

install:
	uv sync
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/

fmt:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage

# Pipeline stages (each is a scripts/ entrypoint)
inventory:
	uv run python -m scripts.run_inventory

collect:
	uv run python -m scripts.run_collection

extract:
	uv run python -m scripts.run_extraction

map:
	uv run python -m scripts.run_mapping

faers:
	uv run python -m scripts.run_faers

labels:
	uv run python -m scripts.run_labels

analyze:
	uv run python -m scripts.run_analysis

pipeline: collect extract map faers labels analyze
	@echo "Pipeline complete. See data/processed/ and docs/paper/."
```

- [ ] **Step 2: Verify test target works**

Run: `make test`
Expected: pytest runs, all prior tests pass.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: makefile orchestrates pipeline stages and dev tasks"
```

---

# Phase 1 — Extraction schema

**Deliverable:** A canonical, tested pydantic schema that every downstream stage imports. Matches spec §4 exactly.

## Task 1.1: Enum definitions in YAML

**Files:**
- Create: `config/extraction_enums.yaml`

- [ ] **Step 1: Write `config/extraction_enums.yaml`**

```yaml
products:
  - botox
  - dysport
  - xeomin
  - jeuveau
  - daxxify
  - unspecified_bont_a

indication_category:
  - medical
  - cosmetic
  - unknown

medical_subtype:
  - chronic_migraine
  - cervical_dystonia
  - blepharospasm
  - spasticity
  - hyperhidrosis
  - overactive_bladder
  - strabismus
  - tmj_medical
  - other_medical

cosmetic_subtype:
  - glabellar
  - forehead
  - crows_feet
  - masseter_jawline
  - neck_platysma
  - lip_flip
  - baby_botox
  - other_cosmetic

sentiment_valence:
  - positive
  - negative
  - mixed
  - neutral

sentiment_about_side_effect:
  - distressed
  - accepting
  - dismissive

provider_type:
  - medspa
  - dermatologist
  - plastic_surgeon
  - neurologist
  - other_md
  - nurse_injector
  - diy
  - unspecified

first_time_or_repeat:
  - first_time
  - repeat

time_to_onset:
  - "<24h"
  - "1_3d"
  - "4_7d"
  - "1_2w"
  - "2_4w"
  - ">4w"
  - unspecified

duration_status:
  - ongoing
  - resolved_weeks
  - resolved_months
  - permanent
  - unspecified

discontinuation_intent:
  - will_continue
  - considering_stopping
  - stopped
  - unclear

age_range:
  - "<20"
  - "20-29"
  - "30-39"
  - "40-49"
  - "50-59"
  - "60-69"
  - "70+"

sex:
  - female
  - male
  - nonbinary
```

- [ ] **Step 2: Commit**

```bash
git add config/extraction_enums.yaml
git commit -m "feat(schema): centralize extraction enums in yaml"
```

## Task 1.2: Pydantic schema — fields a, b, c, d

**Files:**
- Create: `src/botox_se/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write failing test covering all required fields**

```python
import pytest
from pydantic import ValidationError

from botox_se.schema import (
    IndicationCategory,
    PostExtraction,
    Product,
    SideEffectSpan,
)


def test_minimal_non_self_use_post():
    ex = PostExtraction(a_self_use=False)
    assert ex.a_self_use is False
    assert ex.b_products_mentioned == []
    assert ex.c_indication_category == IndicationCategory.UNKNOWN
    assert ex.d_side_effect_pts == []


def test_self_use_with_products_and_pts():
    ex = PostExtraction(
        a_self_use=True,
        b_products_mentioned=[Product.BOTOX, Product.UNSPECIFIED_BONT_A],
        c_indication_category=IndicationCategory.COSMETIC,
        c_indication_subtype="glabellar",
        d_side_effect_pts=[
            SideEffectSpan(raw_span="droopy eyelid", meddra_pt="Ptosis"),
        ],
    )
    assert Product.BOTOX in ex.b_products_mentioned
    assert ex.c_indication_subtype == "glabellar"
    assert len(ex.d_side_effect_pts) == 1


def test_invalid_product_rejected():
    with pytest.raises(ValidationError):
        PostExtraction(
            a_self_use=True,
            b_products_mentioned=["not_a_real_product"],
        )


def test_medical_subtype_rejected_when_cosmetic():
    # Not enforced by pydantic itself, but by our validator
    with pytest.raises(ValidationError):
        PostExtraction(
            a_self_use=True,
            c_indication_category=IndicationCategory.COSMETIC,
            c_indication_subtype="chronic_migraine",
        )
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/test_schema.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/botox_se/schema.py` — core fields**

```python
"""Canonical pydantic schema for post-level extraction.

Every downstream stage imports from this module. Matches spec §4.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Product(str, Enum):
    BOTOX = "botox"
    DYSPORT = "dysport"
    XEOMIN = "xeomin"
    JEUVEAU = "jeuveau"
    DAXXIFY = "daxxify"
    UNSPECIFIED_BONT_A = "unspecified_bont_a"


class IndicationCategory(str, Enum):
    MEDICAL = "medical"
    COSMETIC = "cosmetic"
    UNKNOWN = "unknown"


MEDICAL_SUBTYPES = frozenset({
    "chronic_migraine", "cervical_dystonia", "blepharospasm", "spasticity",
    "hyperhidrosis", "overactive_bladder", "strabismus", "tmj_medical",
    "other_medical",
})

COSMETIC_SUBTYPES = frozenset({
    "glabellar", "forehead", "crows_feet", "masseter_jawline",
    "neck_platysma", "lip_flip", "baby_botox", "other_cosmetic",
})


class SentimentValence(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class SentimentAboutSideEffect(str, Enum):
    DISTRESSED = "distressed"
    ACCEPTING = "accepting"
    DISMISSIVE = "dismissive"


class ProviderType(str, Enum):
    MEDSPA = "medspa"
    DERMATOLOGIST = "dermatologist"
    PLASTIC_SURGEON = "plastic_surgeon"
    NEUROLOGIST = "neurologist"
    OTHER_MD = "other_md"
    NURSE_INJECTOR = "nurse_injector"
    DIY = "diy"
    UNSPECIFIED = "unspecified"


class FirstTimeOrRepeat(str, Enum):
    FIRST_TIME = "first_time"
    REPEAT = "repeat"


class TimeToOnset(str, Enum):
    LT_24H = "<24h"
    D1_3 = "1_3d"
    D4_7 = "4_7d"
    W1_2 = "1_2w"
    W2_4 = "2_4w"
    GT_4W = ">4w"
    UNSPECIFIED = "unspecified"


class DurationStatus(str, Enum):
    ONGOING = "ongoing"
    RESOLVED_WEEKS = "resolved_weeks"
    RESOLVED_MONTHS = "resolved_months"
    PERMANENT = "permanent"
    UNSPECIFIED = "unspecified"


class DiscontinuationIntent(str, Enum):
    WILL_CONTINUE = "will_continue"
    CONSIDERING_STOPPING = "considering_stopping"
    STOPPED = "stopped"
    UNCLEAR = "unclear"


class AgeRange(str, Enum):
    LT_20 = "<20"
    A20_29 = "20-29"
    A30_39 = "30-39"
    A40_49 = "40-49"
    A50_59 = "50-59"
    A60_69 = "60-69"
    A70P = "70+"


class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"
    NONBINARY = "nonbinary"


class SideEffectSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_span: str = Field(..., min_length=1)
    meddra_pt: str = Field(..., min_length=1)


class PostExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Gate
    a_self_use: bool

    # Product
    b_products_mentioned: list[Product] = Field(default_factory=list)

    # Indication
    c_indication_category: IndicationCategory = IndicationCategory.UNKNOWN
    c_indication_subtype: str | None = None

    # Core outcome
    d_side_effect_pts: list[SideEffectSpan] = Field(default_factory=list)

    # Sentiment
    e1_post_overall_sentiment: SentimentValence = SentimentValence.NEUTRAL
    e2_sentiment_about_botox: SentimentValence = SentimentValence.NEUTRAL
    f_sentiment_about_side_effect: SentimentAboutSideEffect | None = None

    # Procedure / dose
    g_treatment_area_freetext: str | None = None
    h_dose_units: int | None = Field(default=None, ge=0, le=2000)

    # Context
    i_provider_type: ProviderType | None = None
    j_first_time_or_repeat: FirstTimeOrRepeat | None = None
    j_prior_treatment_count: int | None = Field(default=None, ge=0, le=200)
    k_time_to_onset: TimeToOnset | None = None
    l_duration_status: DurationStatus | None = None
    m_sought_medical_attention: bool | None = None
    n_discontinuation_intent: DiscontinuationIntent | None = None

    # Demographics
    o_age_range: AgeRange | None = None
    o_sex: Sex | None = None

    @model_validator(mode="after")
    def _subtype_matches_category(self) -> PostExtraction:
        if self.c_indication_subtype is None:
            return self
        if self.c_indication_category is IndicationCategory.MEDICAL:
            if self.c_indication_subtype not in MEDICAL_SUBTYPES:
                raise ValueError(
                    f"c_indication_subtype {self.c_indication_subtype!r} "
                    f"is not a medical subtype"
                )
        elif self.c_indication_category is IndicationCategory.COSMETIC:
            if self.c_indication_subtype not in COSMETIC_SUBTYPES:
                raise ValueError(
                    f"c_indication_subtype {self.c_indication_subtype!r} "
                    f"is not a cosmetic subtype"
                )
        else:  # UNKNOWN
            if self.c_indication_subtype is not None:
                raise ValueError(
                    "c_indication_subtype must be null when category=unknown"
                )
        return self
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/botox_se/schema.py tests/test_schema.py
git commit -m "feat(schema): pydantic PostExtraction matching spec §4"
```

## Task 1.3: Null-enforcement post-processor

**Files:**
- Create: `src/botox_se/extract/__init__.py`, `src/botox_se/extract/postprocess.py`
- Test: `tests/extract/test_postprocess.py`

- [ ] **Step 1: Write failing test**

```python
from botox_se.extract.postprocess import enforce_non_self_use_nulls
from botox_se.schema import (
    IndicationCategory,
    PostExtraction,
    Product,
    SentimentValence,
    SideEffectSpan,
)


def test_non_self_use_post_has_all_optional_fields_nulled():
    # Pretend the LLM hallucinated content on a non-self-use post
    raw = PostExtraction(
        a_self_use=False,
        b_products_mentioned=[Product.BOTOX],
        c_indication_category=IndicationCategory.COSMETIC,
        c_indication_subtype="glabellar",
        d_side_effect_pts=[SideEffectSpan(raw_span="ptosis", meddra_pt="Ptosis")],
        e1_post_overall_sentiment=SentimentValence.NEGATIVE,
    )
    cleaned = enforce_non_self_use_nulls(raw)
    assert cleaned.a_self_use is False
    assert cleaned.b_products_mentioned == []
    assert cleaned.c_indication_category == IndicationCategory.UNKNOWN
    assert cleaned.c_indication_subtype is None
    assert cleaned.d_side_effect_pts == []
    assert cleaned.e1_post_overall_sentiment == SentimentValence.NEUTRAL


def test_self_use_post_unchanged():
    raw = PostExtraction(
        a_self_use=True,
        b_products_mentioned=[Product.BOTOX],
        c_indication_category=IndicationCategory.COSMETIC,
        c_indication_subtype="glabellar",
    )
    cleaned = enforce_non_self_use_nulls(raw)
    assert cleaned == raw
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/extract/test_postprocess.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/botox_se/extract/__init__.py`** (empty) and implement `src/botox_se/extract/postprocess.py`

```python
"""Post-processing that enforces schema invariants the LLM may violate."""

from botox_se.schema import IndicationCategory, PostExtraction, SentimentValence


def enforce_non_self_use_nulls(ex: PostExtraction) -> PostExtraction:
    """If a_self_use is False, zero out every downstream field.

    Guards against model hallucination — treat non-self-use posts as
    containing no extractable information regardless of what the model returned.
    """
    if ex.a_self_use:
        return ex
    return PostExtraction(
        a_self_use=False,
        b_products_mentioned=[],
        c_indication_category=IndicationCategory.UNKNOWN,
        c_indication_subtype=None,
        d_side_effect_pts=[],
        e1_post_overall_sentiment=SentimentValence.NEUTRAL,
        e2_sentiment_about_botox=SentimentValence.NEUTRAL,
        f_sentiment_about_side_effect=None,
        g_treatment_area_freetext=None,
        h_dose_units=None,
        i_provider_type=None,
        j_first_time_or_repeat=None,
        j_prior_treatment_count=None,
        k_time_to_onset=None,
        l_duration_status=None,
        m_sought_medical_attention=None,
        n_discontinuation_intent=None,
        o_age_range=None,
        o_sex=None,
    )
```

Also create `tests/extract/__init__.py` (empty).

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/extract/test_postprocess.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/botox_se/extract/ tests/extract/
git commit -m "feat(extract): enforce null-rules on non-self-use posts"
```

---

# Phase 2 — Reddit ingestion

**Deliverable:** `data/interim/posts.parquet` populated from the finalized subreddit list over the collection window, deduplicated, filtered for language/length/spam.

## Task 2.1: Search terms and subreddit config

**Files:**
- Create: `config/search_terms.yaml`, `config/subreddits.yaml`
- Test: `tests/ingest/test_config_loaders.py`

- [ ] **Step 1: Write `config/search_terms.yaml`**

```yaml
# Brand + generic + slang + common misspellings
terms:
  - botox
  - boto
  - botx
  - "tox injections"
  - neurotoxin
  - dysport
  - xeomin
  - jeuveau
  - newtox
  - daxxify
  - daxi
  - onabotulinumtoxina
  - onabotulinumtoxin a
  - abobotulinumtoxina
  - incobotulinumtoxina
  - prabotulinumtoxina
  - daxibotulinumtoxina
  - "botulinum toxin"
  - "botulinum toxin type a"
  - bont-a
  - bonta
```

- [ ] **Step 2: Write `config/subreddits.yaml`**

```yaml
# Candidate set. Inventory step (Task 2.3) drops any with <50 matching posts.
medical:
  - migraine
  - ChronicMigraine
  - dystonia
  - Hyperhidrosis
  - OveractiveBladder
  - Blepharospasm
  - TMJ
  - Strabismus
  - ChronicPain
  - Stroke

cosmetic:
  - Botox
  - 30PlusSkinCare
  - SkincareAddiction
  - PlasticSurgery
  - CosmeticSurgery

mixed:
  - AskDocs

min_posts_threshold: 50
```

- [ ] **Step 3: Write failing test**

```python
from pathlib import Path

from botox_se.ingest.config_loaders import load_search_terms, load_subreddits


def test_load_search_terms(tmp_path):
    yaml = tmp_path / "search_terms.yaml"
    yaml.write_text("terms:\n  - botox\n  - dysport\n")
    assert load_search_terms(yaml) == ["botox", "dysport"]


def test_load_subreddits_flattens_all_buckets(tmp_path):
    yaml = tmp_path / "subreddits.yaml"
    yaml.write_text(
        "medical:\n  - migraine\ncosmetic:\n  - Botox\nmixed:\n  - AskDocs\n"
        "min_posts_threshold: 50\n"
    )
    cfg = load_subreddits(yaml)
    assert cfg.all_subreddits == ["migraine", "Botox", "AskDocs"]
    assert cfg.min_posts_threshold == 50
```

- [ ] **Step 4: Run — expect fail**

Run: `uv run pytest tests/ingest/test_config_loaders.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Implement — create `src/botox_se/ingest/__init__.py` (empty) and `src/botox_se/ingest/config_loaders.py`**

```python
"""Load YAML config files for ingestion."""

from dataclasses import dataclass
from pathlib import Path

import yaml


def load_search_terms(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return list(data["terms"])


@dataclass(frozen=True)
class SubredditConfig:
    medical: list[str]
    cosmetic: list[str]
    mixed: list[str]
    min_posts_threshold: int

    @property
    def all_subreddits(self) -> list[str]:
        return [*self.medical, *self.cosmetic, *self.mixed]


def load_subreddits(path: Path) -> SubredditConfig:
    data = yaml.safe_load(path.read_text())
    return SubredditConfig(
        medical=list(data.get("medical", [])),
        cosmetic=list(data.get("cosmetic", [])),
        mixed=list(data.get("mixed", [])),
        min_posts_threshold=int(data.get("min_posts_threshold", 50)),
    )
```

Also create `tests/ingest/__init__.py`.

- [ ] **Step 6: Run — expect pass**

Run: `uv run pytest tests/ingest/test_config_loaders.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add config/search_terms.yaml config/subreddits.yaml src/botox_se/ingest/ tests/ingest/
git commit -m "feat(ingest): yaml-loaded search terms and subreddit config"
```

## Task 2.2: Reddit client — Arctic Shift wrapper

**Files:**
- Create: `src/botox_se/ingest/reddit_client.py`
- Test: `tests/ingest/test_reddit_client.py`

Arctic Shift (https://github.com/ArthurHeitmann/arctic_shift) exposes JSON endpoints at `arctic-shift.photon-reddit.com/api/...`. We wrap it behind a typed client.

- [ ] **Step 1: Write failing test (uses `pytest-httpx` to mock HTTP)**

```python
import pytest
from httpx import AsyncClient

from botox_se.ingest.reddit_client import ArcticShiftClient, RedditPost


@pytest.mark.asyncio
async def test_search_posts_by_subreddit_returns_parsed_posts(httpx_mock):
    httpx_mock.add_response(
        url="https://arctic-shift.photon-reddit.com/api/posts/search"
            "?subreddit=botox&q=botox&after=1577836800&before=1578009600&limit=100",
        json={
            "data": [
                {
                    "id": "abc123",
                    "subreddit": "botox",
                    "title": "First time",
                    "selftext": "Tried botox for my forehead",
                    "author": "user1",
                    "created_utc": 1577900000,
                    "permalink": "/r/botox/comments/abc123",
                    "score": 12,
                }
            ]
        },
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        posts = await client.search_posts(
            subreddit="botox",
            query="botox",
            start=1577836800,
            end=1578009600,
        )
    assert len(posts) == 1
    assert posts[0].id == "abc123"
    assert posts[0].subreddit == "botox"
    assert posts[0].selftext == "Tried botox for my forehead"
    assert isinstance(posts[0], RedditPost)
```

- [ ] **Step 2: Add `pytest-asyncio` to dev deps**

Edit `pyproject.toml` `[dependency-groups] dev` to include `"pytest-asyncio>=0.23"`. Add to `[tool.pytest.ini_options]`: `asyncio_mode = "auto"`. Run `uv sync`.

- [ ] **Step 3: Run — expect fail**

Run: `uv run pytest tests/ingest/test_reddit_client.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/botox_se/ingest/reddit_client.py`**

```python
"""Arctic Shift client for historical Reddit search.

Docs: https://github.com/ArthurHeitmann/arctic_shift
Endpoint shape subject to change; wrap calls here so callers can pin.
"""

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"


@dataclass(frozen=True)
class RedditPost:
    id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    created_utc: int
    permalink: str
    score: int

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "RedditPost":
        return cls(
            id=raw["id"],
            subreddit=raw["subreddit"],
            title=raw.get("title", "") or "",
            selftext=raw.get("selftext", "") or "",
            author=raw.get("author", "") or "",
            created_utc=int(raw.get("created_utc", 0)),
            permalink=raw.get("permalink", "") or "",
            score=int(raw.get("score", 0)),
        )


@dataclass(frozen=True)
class RedditComment:
    id: str
    post_id: str            # link_id without the t3_ prefix
    subreddit: str
    body: str
    author: str
    created_utc: int

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "RedditComment":
        link_id = raw.get("link_id", "") or ""
        post_id = link_id.removeprefix("t3_")
        return cls(
            id=raw["id"],
            post_id=post_id,
            subreddit=raw.get("subreddit", "") or "",
            body=raw.get("body", "") or "",
            author=raw.get("author", "") or "",
            created_utc=int(raw.get("created_utc", 0)),
        )


class ArcticShiftClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str = ARCTIC_BASE) -> None:
        self._http = http
        self._base = base_url

    async def search_posts(
        self,
        *,
        subreddit: str,
        query: str,
        start: int,
        end: int,
        limit: int = 100,
    ) -> list[RedditPost]:
        params = {
            "subreddit": subreddit,
            "q": query,
            "after": start,
            "before": end,
            "limit": limit,
        }
        payload = await self._get("/posts/search", params=params)
        return [RedditPost.from_api(r) for r in payload.get("data", [])]

    async def search_comments(
        self,
        *,
        subreddit: str,
        query: str,
        start: int,
        end: int,
        limit: int = 100,
    ) -> list[RedditComment]:
        params = {
            "subreddit": subreddit,
            "q": query,
            "after": start,
            "before": end,
            "limit": limit,
        }
        payload = await self._get("/comments/search", params=params)
        return [RedditComment.from_api(r) for r in payload.get("data", [])]

    async def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(httpx.HTTPError),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(5),
            reraise=True,
        ):
            with attempt:
                resp = await self._http.get(f"{self._base}{path}", params=params, timeout=30.0)
                resp.raise_for_status()
                return resp.json()
        raise RuntimeError("unreachable")
```

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/ingest/test_reddit_client.py -v`
Expected: 1 passed.

- [ ] **Step 6: Write comment-search test**

Add to the same test file:

```python
@pytest.mark.asyncio
async def test_search_comments(httpx_mock):
    httpx_mock.add_response(
        url="https://arctic-shift.photon-reddit.com/api/comments/search"
            "?subreddit=botox&q=botox&after=1577836800&before=1578009600&limit=100",
        json={
            "data": [
                {
                    "id": "cmt1",
                    "link_id": "t3_abc123",
                    "subreddit": "botox",
                    "body": "I had ptosis for weeks",
                    "author": "user2",
                    "created_utc": 1577910000,
                }
            ]
        },
    )
    async with AsyncClient() as http:
        client = ArcticShiftClient(http=http)
        comments = await client.search_comments(
            subreddit="botox",
            query="botox",
            start=1577836800,
            end=1578009600,
        )
    assert comments[0].post_id == "abc123"
    assert comments[0].body == "I had ptosis for weeks"
```

- [ ] **Step 7: Run — expect pass**

Run: `uv run pytest tests/ingest/test_reddit_client.py -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add src/botox_se/ingest/reddit_client.py tests/ingest/test_reddit_client.py pyproject.toml uv.lock
git commit -m "feat(ingest): arctic shift client for posts and comments with retry"
```

## Task 2.3: Data inventory script

**Files:**
- Create: `src/botox_se/ingest/inventory.py`, `scripts/run_inventory.py`
- Test: `tests/ingest/test_inventory.py`

The inventory step counts matching post volume per (subreddit, search term) to drop low-coverage subreddits.

- [ ] **Step 1: Write failing test**

```python
import pytest
from httpx import AsyncClient

from botox_se.ingest.inventory import inventory_subreddits


@pytest.mark.asyncio
async def test_inventory_returns_counts_per_subreddit(httpx_mock):
    # two subreddits × one query, one has posts, one empty
    httpx_mock.add_response(
        url__regex=r".*subreddit=botox.*",
        json={"data": [{"id": "p1", "subreddit": "botox", "created_utc": 1},
                       {"id": "p2", "subreddit": "botox", "created_utc": 2}]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url__regex=r".*subreddit=emptysub.*",
        json={"data": []},
        is_reusable=True,
    )
    async with AsyncClient() as http:
        counts = await inventory_subreddits(
            http=http,
            subreddits=["botox", "emptysub"],
            terms=["botox"],
            start=0,
            end=99999999,
        )
    assert counts["botox"] >= 2
    assert counts["emptysub"] == 0
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/ingest/test_inventory.py -v`

- [ ] **Step 3: Implement `src/botox_se/ingest/inventory.py`**

```python
"""Data inventory: count matching posts per (subreddit, term) combination."""

import asyncio
from collections import defaultdict

import httpx

from botox_se.ingest.reddit_client import ArcticShiftClient
from botox_se.logging import get_logger

log = get_logger(__name__)


async def inventory_subreddits(
    *,
    http: httpx.AsyncClient,
    subreddits: list[str],
    terms: list[str],
    start: int,
    end: int,
) -> dict[str, int]:
    """Return {subreddit: total_matching_post_count_across_terms}."""
    client = ArcticShiftClient(http=http)
    counts: dict[str, int] = defaultdict(int)

    async def count_one(subreddit: str, term: str) -> None:
        posts = await client.search_posts(
            subreddit=subreddit, query=term, start=start, end=end, limit=1000
        )
        counts[subreddit] += len(posts)
        log.info("inventory_query", subreddit=subreddit, term=term, count=len(posts))

    tasks = [count_one(s, t) for s in subreddits for t in terms]
    await asyncio.gather(*tasks)
    return dict(counts)


def filter_subreddits(counts: dict[str, int], threshold: int) -> list[str]:
    return sorted([s for s, n in counts.items() if n >= threshold])
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/ingest/test_inventory.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write CLI `scripts/run_inventory.py`**

```python
"""CLI: run the subreddit inventory step.

Writes:
  data/interim/inventory.csv      (subreddit, match_count)
  data/interim/subreddits_final.txt   (newline list, filtered by threshold)
"""

import asyncio
from datetime import datetime, timezone

import httpx
import polars as pl
import typer

from botox_se.config import load as load_settings
from botox_se.ingest.config_loaders import load_search_terms, load_subreddits
from botox_se.ingest.inventory import filter_subreddits, inventory_subreddits
from botox_se.logging import configure_logging, get_logger
from botox_se.paths import Paths


def _iso_to_unix(iso: str) -> int:
    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


async def _run() -> None:
    configure_logging(level="INFO")
    log = get_logger("inventory")
    settings = load_settings()
    paths = Paths(repo_root=typer.get_current_context().obj or __import__("pathlib").Path("."))
    paths.ensure_dirs()

    terms = load_search_terms(paths.config / "search_terms.yaml")
    sub_cfg = load_subreddits(paths.config / "subreddits.yaml")

    start = _iso_to_unix(settings.collection_start)
    end = _iso_to_unix(settings.collection_end)

    async with httpx.AsyncClient() as http:
        counts = await inventory_subreddits(
            http=http,
            subreddits=sub_cfg.all_subreddits,
            terms=terms,
            start=start,
            end=end,
        )

    df = pl.DataFrame([{"subreddit": s, "match_count": n} for s, n in counts.items()])
    df = df.sort("match_count", descending=True)
    df.write_csv(paths.interim / "inventory.csv")

    finalists = filter_subreddits(counts, threshold=sub_cfg.min_posts_threshold)
    (paths.interim / "subreddits_final.txt").write_text("\n".join(finalists))

    log.info("inventory_done", n_candidates=len(counts), n_finalists=len(finalists))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/botox_se/ingest/inventory.py scripts/run_inventory.py tests/ingest/test_inventory.py
git commit -m "feat(ingest): subreddit data-inventory script with threshold filter"
```

## Task 2.4: Full collection runner

**Files:**
- Create: `src/botox_se/ingest/fetch.py`, `scripts/run_collection.py`
- Test: `tests/ingest/test_fetch.py`

The runner iterates `(subreddit × term × time-chunk)` and writes raw JSONL as it goes so we never lose progress.

- [ ] **Step 1: Write failing test**

```python
import json
import pytest
from httpx import AsyncClient

from botox_se.ingest.fetch import collect_posts


@pytest.mark.asyncio
async def test_collect_posts_writes_jsonl_and_returns_count(tmp_path, httpx_mock):
    httpx_mock.add_response(
        url__regex=r".*posts/search.*",
        json={"data": [
            {"id": "a", "subreddit": "botox", "title": "t", "selftext": "s",
             "author": "u", "created_utc": 1, "permalink": "", "score": 0}
        ]},
        is_reusable=True,
    )
    out = tmp_path / "raw_posts.jsonl"
    async with AsyncClient() as http:
        n = await collect_posts(
            http=http,
            subreddits=["botox"],
            terms=["botox"],
            start=0,
            end=10,
            chunk_seconds=5,
            output_path=out,
        )
    assert n >= 1
    lines = out.read_text().splitlines()
    assert all(json.loads(line)["id"] for line in lines)
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/ingest/test_fetch.py -v`

- [ ] **Step 3: Implement `src/botox_se/ingest/fetch.py`**

```python
"""Streaming collection: (subreddit × term × time-chunk) → JSONL.

Chunking limits API response size; JSONL append = no progress loss on crash.
"""

import json
from dataclasses import asdict
from pathlib import Path

import httpx

from botox_se.ingest.reddit_client import ArcticShiftClient, RedditPost, RedditComment
from botox_se.logging import get_logger

log = get_logger(__name__)


def _chunks(start: int, end: int, step: int) -> list[tuple[int, int]]:
    return [(t, min(t + step, end)) for t in range(start, end, step)]


async def collect_posts(
    *,
    http: httpx.AsyncClient,
    subreddits: list[str],
    terms: list[str],
    start: int,
    end: int,
    chunk_seconds: int = 30 * 24 * 3600,   # 30-day chunks
    output_path: Path,
) -> int:
    client = ArcticShiftClient(http=http)
    total = 0
    with output_path.open("a") as f:
        for sub in subreddits:
            for term in terms:
                for chunk_start, chunk_end in _chunks(start, end, chunk_seconds):
                    posts = await client.search_posts(
                        subreddit=sub, query=term,
                        start=chunk_start, end=chunk_end,
                        limit=1000,
                    )
                    for p in posts:
                        f.write(json.dumps(asdict(p)) + "\n")
                    total += len(posts)
                    log.info(
                        "collected",
                        subreddit=sub, term=term,
                        start=chunk_start, end=chunk_end,
                        n=len(posts), running_total=total,
                    )
    return total


async def collect_comments(
    *,
    http: httpx.AsyncClient,
    subreddits: list[str],
    terms: list[str],
    start: int,
    end: int,
    chunk_seconds: int = 30 * 24 * 3600,
    output_path: Path,
) -> int:
    client = ArcticShiftClient(http=http)
    total = 0
    with output_path.open("a") as f:
        for sub in subreddits:
            for term in terms:
                for chunk_start, chunk_end in _chunks(start, end, chunk_seconds):
                    cmts = await client.search_comments(
                        subreddit=sub, query=term,
                        start=chunk_start, end=chunk_end,
                        limit=1000,
                    )
                    for c in cmts:
                        f.write(json.dumps(asdict(c)) + "\n")
                    total += len(cmts)
    return total
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/ingest/test_fetch.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write CLI `scripts/run_collection.py`**

```python
"""CLI: run full post + comment collection into data/raw/."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx

from botox_se.config import load as load_settings
from botox_se.ingest.config_loaders import load_search_terms
from botox_se.ingest.fetch import collect_comments, collect_posts
from botox_se.logging import configure_logging, get_logger
from botox_se.paths import Paths


def _iso_to_unix(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


async def _run() -> None:
    configure_logging(level="INFO")
    log = get_logger("collect")
    settings = load_settings()
    paths = Paths(repo_root=Path.cwd())
    paths.ensure_dirs()

    terms = load_search_terms(paths.config / "search_terms.yaml")
    finalists = (paths.interim / "subreddits_final.txt").read_text().splitlines()
    assert finalists, "Run `make inventory` first"

    start = _iso_to_unix(settings.collection_start)
    end = _iso_to_unix(settings.collection_end)

    posts_out = paths.raw / "posts.jsonl"
    comments_out = paths.raw / "comments.jsonl"
    posts_out.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as http:
        n_posts = await collect_posts(
            http=http, subreddits=finalists, terms=terms,
            start=start, end=end, output_path=posts_out,
        )
        n_comments = await collect_comments(
            http=http, subreddits=finalists, terms=terms,
            start=start, end=end, output_path=comments_out,
        )

    log.info("collection_done", posts=n_posts, comments=n_comments)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/botox_se/ingest/fetch.py scripts/run_collection.py tests/ingest/test_fetch.py
git commit -m "feat(ingest): streaming post + comment collection into JSONL"
```

## Task 2.5: Deduplication and filters

**Files:**
- Create: `src/botox_se/ingest/dedupe.py`, `src/botox_se/ingest/filters.py`
- Test: `tests/ingest/test_dedupe.py`, `tests/ingest/test_filters.py`

- [ ] **Step 1: Write failing dedupe test**

```python
import polars as pl

from botox_se.ingest.dedupe import dedupe_posts


def test_dedupe_drops_duplicate_ids_keeps_first():
    df = pl.DataFrame({
        "id": ["a", "b", "a"],
        "text": ["first", "x", "second"],
    })
    out = dedupe_posts(df, id_col="id")
    assert out["id"].to_list() == ["a", "b"]
    assert out.filter(pl.col("id") == "a")["text"].item() == "first"
```

- [ ] **Step 2: Run — expect fail**; **Step 3: Implement `dedupe.py`**

```python
"""Deduplication for ingested posts/comments."""

import polars as pl


def dedupe_posts(df: pl.DataFrame, *, id_col: str = "id") -> pl.DataFrame:
    return df.unique(subset=id_col, keep="first")
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Write filter test**

```python
import polars as pl

from botox_se.ingest.filters import apply_basic_filters


def test_filters_drop_short_and_empty_posts():
    df = pl.DataFrame({
        "id": ["a", "b", "c", "d"],
        "text": ["", "short", "this is a perfectly reasonable post about botox", "ok"],
    })
    out = apply_basic_filters(df, text_col="text", min_len=10)
    assert out["id"].to_list() == ["c"]


def test_filter_bot_patterns_dropped():
    df = pl.DataFrame({
        "id": ["a", "b"],
        "text": [
            "I am a bot and this is a compiled list of comments",
            "Just got botox for my forehead, loving the results.",
        ],
    })
    out = apply_basic_filters(df, text_col="text", min_len=10)
    assert out["id"].to_list() == ["b"]
```

- [ ] **Step 6: Run — expect fail**; **Step 7: Implement `filters.py`**

```python
"""Text-level filters applied before LLM extraction.

Goals:
- Drop empty/short posts
- Drop obvious bot text (removed, deleted, mod-auto)
- Language filter (English) — applied separately in Step 9 below for cost
"""

import polars as pl

BOT_PATTERNS = [
    r"(?i)^\[removed\]$",
    r"(?i)^\[deleted\]$",
    r"(?i)^i am a bot",
    r"(?i)this is an automated (message|response)",
    r"(?i)performed automatically",
]


def apply_basic_filters(
    df: pl.DataFrame,
    *,
    text_col: str,
    min_len: int = 20,
) -> pl.DataFrame:
    out = df.filter(
        pl.col(text_col).is_not_null()
        & (pl.col(text_col).str.len_chars() >= min_len)
    )
    for pat in BOT_PATTERNS:
        out = out.filter(~pl.col(text_col).str.contains(pat))
    return out
```

- [ ] **Step 8: Run — expect pass**

Run: `uv run pytest tests/ingest/test_filters.py tests/ingest/test_dedupe.py -v`
Expected: 3 passed.

- [ ] **Step 9: Add language filter using `langdetect`**

Add to `filters.py`:

```python
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def filter_english(df: pl.DataFrame, *, text_col: str) -> pl.DataFrame:
    mask = df[text_col].map_elements(_is_english, return_dtype=pl.Boolean)
    return df.filter(mask)
```

Test:

```python
def test_filter_english_keeps_english_drops_other():
    df = pl.DataFrame({
        "id": ["a", "b", "c"],
        "text": [
            "I had botox in my forehead last week and it worked well",
            "Ich habe Botox für meine Stirn letzte Woche genommen",
            "Got great results from my first botox appointment",
        ],
    })
    from botox_se.ingest.filters import filter_english
    out = filter_english(df, text_col="text")
    assert set(out["id"].to_list()) == {"a", "c"}
```

- [ ] **Step 10: Run — expect pass**; **Step 11: Commit**

```bash
git add src/botox_se/ingest/dedupe.py src/botox_se/ingest/filters.py tests/ingest/
git commit -m "feat(ingest): dedupe and basic/language filters"
```

## Task 2.6: Normalize raw → interim parquet

**Files:**
- Create: `src/botox_se/ingest/normalize.py`, `scripts/run_normalize.py`

- [ ] **Step 1: Write failing test `tests/ingest/test_normalize.py`**

```python
import json
from pathlib import Path

import polars as pl

from botox_se.ingest.normalize import normalize_raw_to_interim


def test_normalize_combines_posts_and_comments_into_unified_parquet(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "interim").mkdir()

    posts_jsonl = tmp_path / "raw" / "posts.jsonl"
    comments_jsonl = tmp_path / "raw" / "comments.jsonl"
    posts_jsonl.write_text(
        json.dumps({
            "id": "p1", "subreddit": "botox", "title": "hi",
            "selftext": "had botox for my forehead last week, great results!",
            "author": "u1", "created_utc": 1700000000,
            "permalink": "", "score": 5,
        }) + "\n"
    )
    comments_jsonl.write_text(
        json.dumps({
            "id": "c1", "post_id": "p1", "subreddit": "botox",
            "body": "what dose did you use for your forehead treatment",
            "author": "u2", "created_utc": 1700000100,
        }) + "\n"
    )

    out = tmp_path / "interim" / "posts.parquet"
    n = normalize_raw_to_interim(
        posts_jsonl=posts_jsonl,
        comments_jsonl=comments_jsonl,
        out_path=out,
    )
    assert n == 2
    df = pl.read_parquet(out)
    assert set(df["record_type"].to_list()) == {"post", "comment"}
    assert set(df.columns) >= {"id", "subreddit", "author", "created_utc", "text", "record_type"}
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `src/botox_se/ingest/normalize.py`**

```python
"""Combine raw posts + comments JSONL into a single interim parquet.

Applies dedupe + basic filters. Produces `data/interim/posts.parquet`
with schema: id, subreddit, author, created_utc, text, record_type, post_id.
"""

from pathlib import Path

import polars as pl

from botox_se.ingest.dedupe import dedupe_posts
from botox_se.ingest.filters import apply_basic_filters, filter_english


def normalize_raw_to_interim(
    *,
    posts_jsonl: Path,
    comments_jsonl: Path,
    out_path: Path,
    apply_language_filter: bool = True,
) -> int:
    """Merge + dedupe + filter; return number of rows written."""
    posts = pl.read_ndjson(posts_jsonl)
    comments = pl.read_ndjson(comments_jsonl)

    # Unify schema
    post_rows = posts.select(
        id=pl.col("id"),
        subreddit=pl.col("subreddit"),
        author=pl.col("author"),
        created_utc=pl.col("created_utc"),
        text=(pl.col("title").fill_null("") + "\n\n" + pl.col("selftext").fill_null("")),
        record_type=pl.lit("post"),
        post_id=pl.col("id"),
    )
    comment_rows = comments.select(
        id=pl.col("id"),
        subreddit=pl.col("subreddit"),
        author=pl.col("author"),
        created_utc=pl.col("created_utc"),
        text=pl.col("body"),
        record_type=pl.lit("comment"),
        post_id=pl.col("post_id"),
    )
    merged = pl.concat([post_rows, comment_rows])
    merged = dedupe_posts(merged, id_col="id")
    merged = apply_basic_filters(merged, text_col="text", min_len=20)
    if apply_language_filter:
        merged = filter_english(merged, text_col="text")

    merged.write_parquet(out_path)
    return merged.height
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Add CLI and commit**

```bash
git add src/botox_se/ingest/normalize.py tests/ingest/test_normalize.py
git commit -m "feat(ingest): normalize raw JSONL into filtered interim parquet"
```

---

# Phase 3 — LLM extraction

**Deliverable:** `data/processed/extractions.parquet` — one row per input post, containing the `PostExtraction` JSON and metadata (model version, prompt version, extraction timestamp).

## Task 3.1: Prompt template

**Files:**
- Create: `config/llm_prompts/extract_v1.md`
- Test: `tests/extract/test_prompt.py`

- [ ] **Step 1: Write prompt `config/llm_prompts/extract_v1.md`**

```markdown
# Extraction Prompt — v1

System:
You are a careful clinical-NLP annotator. You read a Reddit post (or comment)
and produce a single JSON object that conforms exactly to the schema provided.
You never invent content that is not in the post. If a field is not explicitly
supported by the post's content, return null (or the default) for that field.

CRITICAL RULES:
1. `a_self_use` is true only if the author states they personally used, are
   using, or have used a botulinum toxin product (Botox / Dysport / Xeomin /
   Jeuveau / Daxxify). Posts where the author only discusses others' use,
   considers future use, or asks for advice without having used it return
   `a_self_use=false`.
2. If `a_self_use=false`, return all other fields at their defaults (empty
   list, `unknown`, null). Do NOT extract side effects, products, or
   sentiment from posts where the author is not a user.
3. `b_products_mentioned` is a list. Use `unspecified_bont_a` when the post
   says "Botox" generically and no specific brand is identified, and only
   that. If a specific brand is named, use the brand code — do not also
   add `unspecified_bont_a`.
4. `c_indication_category` — use `medical` when treatment is for a medical
   indication (chronic migraine, cervical dystonia, spasticity,
   hyperhidrosis, overactive bladder, strabismus, blepharospasm, medical
   TMJ), `cosmetic` for aesthetic uses (glabellar, forehead, crow's feet,
   masseter/jawline slimming, neck/platysma, lip flip, "baby botox"), and
   `unknown` when ambiguous.
5. `d_side_effect_pts` — for every side effect clearly attributed to the
   product, produce a `raw_span` (the exact substring from the post) and a
   `meddra_pt` (your best guess at the MedDRA Preferred Term). The PT will
   be validated downstream — prefer a specific PT over a generic one, but
   do not invent PTs not in common MedDRA usage.
6. Null-allowing fields must be null unless the post explicitly supports a
   value. Do not infer demographics from subreddit, writing style, or
   unrelated context.

User message format:
The user message contains:
  [POST_TEXT]
  <free text of the post>
  [/POST_TEXT]

Return the JSON object matching the schema (structured outputs enforced).
```

- [ ] **Step 2: Write failing test `tests/extract/test_prompt.py`**

```python
from pathlib import Path

from botox_se.extract.prompt import build_messages


def test_build_messages_includes_post_text_and_system_instructions(tmp_path):
    prompt = tmp_path / "extract_v1.md"
    prompt.write_text("System:\nYou are careful.\n\nUser message format:\n[POST_TEXT]\n")
    msgs = build_messages(prompt_path=prompt, post_text="I tried botox.")
    assert msgs[0]["role"] == "system"
    assert "careful" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "I tried botox." in msgs[1]["content"]
    assert "[POST_TEXT]" in msgs[1]["content"]
```

- [ ] **Step 3: Run — expect fail**. **Step 4: Implement `src/botox_se/extract/prompt.py`**

```python
"""Prompt template loader. The template file uses `System:` / `User message format:`
section headers; we split on them and inject the post text."""

from pathlib import Path


def build_messages(*, prompt_path: Path, post_text: str) -> list[dict[str, str]]:
    raw = prompt_path.read_text()
    if "System:" not in raw:
        raise ValueError(f"{prompt_path} missing 'System:' section")
    system_part = raw.split("System:", 1)[1]
    if "User message format:" in system_part:
        system_text, _ = system_part.split("User message format:", 1)
    else:
        system_text = system_part
    system_text = system_text.strip()

    user_text = (
        "[POST_TEXT]\n"
        f"{post_text}\n"
        "[/POST_TEXT]\n"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
```

- [ ] **Step 5: Run — expect pass**. **Step 6: Commit**

```bash
git add config/llm_prompts/extract_v1.md src/botox_se/extract/prompt.py tests/extract/test_prompt.py
git commit -m "feat(extract): prompt v1 + message builder"
```

## Task 3.2: OpenAI structured-output client wrapper

**Files:**
- Create: `src/botox_se/extract/client.py`
- Test: `tests/extract/test_client.py` (unit test with mocked OpenAI client)

- [ ] **Step 1: Write failing test**

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from botox_se.extract.client import ExtractionClient
from botox_se.schema import PostExtraction


@pytest.mark.asyncio
async def test_extract_one_parses_structured_output():
    fake_client = MagicMock()
    # Simulate OpenAI parsed response shape
    parsed = PostExtraction(a_self_use=False)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    fake_completion.system_fingerprint = "fp_abc"
    fake_completion.model = "gpt-5.3-chat-latest-2026-03"
    fake_client.beta.chat.completions.parse = AsyncMock(return_value=fake_completion)

    client = ExtractionClient(openai=fake_client, model="gpt-5.3-chat-latest")
    result = await client.extract_one(
        messages=[{"role": "system", "content": "s"},
                  {"role": "user", "content": "u"}],
    )
    assert result.extraction == parsed
    assert result.system_fingerprint == "fp_abc"
    assert result.model == "gpt-5.3-chat-latest-2026-03"
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement `src/botox_se/extract/client.py`**

```python
"""Thin wrapper around OpenAI's structured-outputs parse endpoint."""

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from botox_se.schema import PostExtraction


@dataclass(frozen=True)
class ExtractionResult:
    extraction: PostExtraction
    model: str
    system_fingerprint: str | None


class ExtractionClient:
    def __init__(self, *, openai: AsyncOpenAI, model: str, temperature: float = 0.0) -> None:
        self._client = openai
        self._model = model
        self._temp = temperature

    async def extract_one(self, *, messages: list[dict[str, str]]) -> ExtractionResult:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(3),
            reraise=True,
        ):
            with attempt:
                completion = await self._client.beta.chat.completions.parse(
                    model=self._model,
                    messages=messages,
                    response_format=PostExtraction,
                    temperature=self._temp,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise RuntimeError("OpenAI returned no parsed content")
                return ExtractionResult(
                    extraction=parsed,
                    model=completion.model,
                    system_fingerprint=getattr(completion, "system_fingerprint", None),
                )
        raise RuntimeError("unreachable")
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Commit**

```bash
git add src/botox_se/extract/client.py tests/extract/test_client.py
git commit -m "feat(extract): openai structured-outputs wrapper with retries"
```

## Task 3.3: Extraction runner

**Files:**
- Create: `src/botox_se/extract/runner.py`, `scripts/run_extraction.py`
- Test: `tests/extract/test_runner.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from botox_se.extract.runner import run_extraction
from botox_se.schema import PostExtraction


@pytest.mark.asyncio
async def test_runner_writes_one_row_per_input_post(tmp_path, monkeypatch):
    # Build small input
    input_df = pl.DataFrame({
        "id": ["p1", "p2"],
        "subreddit": ["botox", "migraine"],
        "author": ["u1", "u2"],
        "created_utc": [1, 2],
        "text": ["I tried botox for forehead", "Botox saved my migraines"],
        "record_type": ["post", "post"],
        "post_id": ["p1", "p2"],
    })
    input_path = tmp_path / "posts.parquet"
    input_df.write_parquet(input_path)

    # Mock ExtractionClient
    fake_client = MagicMock()
    fake_client.extract_one = AsyncMock(side_effect=[
        MagicMock(
            extraction=PostExtraction(a_self_use=True),
            model="gpt-5.3-chat-latest-2026-03",
            system_fingerprint="fp_a",
        ),
        MagicMock(
            extraction=PostExtraction(a_self_use=True),
            model="gpt-5.3-chat-latest-2026-03",
            system_fingerprint="fp_a",
        ),
    ])

    prompt = tmp_path / "p.md"
    prompt.write_text("System:\nbe careful\n\nUser message format:\n[POST_TEXT]\n")

    out_path = tmp_path / "extractions.parquet"
    await run_extraction(
        input_path=input_path,
        output_path=out_path,
        client=fake_client,
        prompt_path=prompt,
        prompt_version="v1",
        concurrency=2,
    )

    out = pl.read_parquet(out_path)
    assert out.height == 2
    assert set(out["id"].to_list()) == {"p1", "p2"}
    assert (out["prompt_version"] == "v1").all()
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement `src/botox_se/extract/runner.py`**

```python
"""Stream input posts parquet → LLM → output parquet.

Resumable: if output file exists, skips IDs already present.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from botox_se.extract.client import ExtractionClient
from botox_se.extract.postprocess import enforce_non_self_use_nulls
from botox_se.extract.prompt import build_messages
from botox_se.logging import get_logger

log = get_logger(__name__)


def _already_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = pl.read_parquet(output_path, columns=["id"])
    return set(done["id"].to_list())


async def run_extraction(
    *,
    input_path: Path,
    output_path: Path,
    client: ExtractionClient,
    prompt_path: Path,
    prompt_version: str,
    concurrency: int = 8,
) -> None:
    input_df = pl.read_parquet(input_path)
    done = _already_done_ids(output_path)
    pending = input_df.filter(~pl.col("id").is_in(list(done)))
    log.info("extraction_start", total=input_df.height, pending=pending.height)

    sem = asyncio.Semaphore(concurrency)
    buffer: list[dict[str, Any]] = []

    async def process(row: dict[str, Any]) -> None:
        async with sem:
            messages = build_messages(prompt_path=prompt_path, post_text=row["text"])
            result = await client.extract_one(messages=messages)
            cleaned = enforce_non_self_use_nulls(result.extraction)
            buffer.append({
                "id": row["id"],
                "subreddit": row["subreddit"],
                "author": row["author"],
                "created_utc": row["created_utc"],
                "record_type": row["record_type"],
                "post_id": row["post_id"],
                "extraction_json": cleaned.model_dump_json(),
                "model": result.model,
                "system_fingerprint": result.system_fingerprint,
                "prompt_version": prompt_version,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            })

    tasks = [process(r) for r in pending.iter_rows(named=True)]
    # Chunked flush to limit memory
    CHUNK = 500
    for i in range(0, len(tasks), CHUNK):
        await asyncio.gather(*tasks[i:i + CHUNK])
        _flush(buffer, output_path)
        buffer.clear()
        log.info("extraction_chunk_flushed", done=i + CHUNK)


def _flush(buffer: list[dict[str, Any]], output_path: Path) -> None:
    if not buffer:
        return
    df = pl.DataFrame(buffer)
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        df = pl.concat([existing, df])
    df.write_parquet(output_path)
```

- [ ] **Step 4: Run — expect pass**. **Step 5: CLI `scripts/run_extraction.py`**

```python
"""CLI: run extraction over data/interim/posts.parquet."""

import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from botox_se.config import load as load_settings
from botox_se.extract.client import ExtractionClient
from botox_se.extract.runner import run_extraction
from botox_se.logging import configure_logging
from botox_se.paths import Paths


async def _main() -> None:
    configure_logging(level="INFO")
    settings = load_settings()
    paths = Paths(repo_root=Path.cwd())
    paths.ensure_dirs()

    oa = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    client = ExtractionClient(
        openai=oa,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    await run_extraction(
        input_path=paths.interim / "posts.parquet",
        output_path=paths.processed / "extractions.parquet",
        client=client,
        prompt_path=paths.config / "llm_prompts" / "extract_v1.md",
        prompt_version="v1",
        concurrency=settings.concurrent_requests,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/botox_se/extract/runner.py scripts/run_extraction.py tests/extract/test_runner.py
git commit -m "feat(extract): streaming extraction runner with resume + chunked flush"
```

## Task 3.4: OpenAI Batch API integration (cost reduction)

**Files:**
- Create: `src/botox_se/extract/batch.py`, `scripts/run_batch_extraction.py`
- Test: `tests/extract/test_batch.py`

Defer this until synchronous runner works end-to-end on a 200-post pilot. The batch version shares the prompt builder and postprocessor; it only differs in submit/poll/retrieve. Implementation sketch, not bite-sized because it's optional:

- Serialize all pending post IDs into a batch JSONL request file per OpenAI Batch spec.
- Submit via `client.batches.create`, poll status, download result JSONL on completion.
- Parse result lines using the same `PostExtraction` schema; feed through `enforce_non_self_use_nulls`; write to `data/processed/extractions.parquet` matching the runner's output schema.

Acceptance test: feed 10 posts through a stubbed Batch API (HTTP fixtures) and verify the output parquet matches what the synchronous runner would have produced on the same input.

- [ ] **Step 1: Write the acceptance test** (uses `pytest-httpx` to fake the Batch API endpoints) — mirrors synchronous runner contract.
- [ ] **Step 2: Implement batch submit + poll + download** against OpenAI Batch API spec.
- [ ] **Step 3: Wire result parsing into the existing `_flush` pathway**.
- [ ] **Step 4: Commit**: `feat(extract): openai batch api path for ~50% cost reduction`.

---

# Phase 4 — MedDRA mapping

**Deliverable:** `data/processed/extractions_mapped.parquet` — same rows as `extractions.parquet` with each `d_side_effect_pts[i].meddra_pt` canonicalized against the MedDRA dictionary (or flagged for manual review).

## Task 4.1: MedDRA dictionary loader

MedDRA is licensed. The plan assumes you have a MedDRA release available under `data/reference/meddra/` (directory excluded from git; the dictionary license forbids redistribution).

**Files:**
- Create: `src/botox_se/meddra/dictionary.py`
- Test: `tests/meddra/test_dictionary.py`

- [ ] **Step 1: Write failing test using a small fake MedDRA fixture**

```python
from pathlib import Path

import polars as pl

from botox_se.meddra.dictionary import MeddraDictionary


def test_loads_pts_with_soc_grouping(tmp_path):
    # Tiny fake MedDRA subset in the format MedDRA ships (pt.asc / soc.asc / mdhier.asc)
    pt = tmp_path / "pt.asc"
    pt.write_text(
        "10028813$Nausea$$$$$$$$\n"
        "10037660$Ptosis$$$$$$$$\n"
    )
    hier = tmp_path / "mdhier.asc"
    hier.write_text(
        "10028813$100$1000$10000$Nausea$GI disorders$...$\n"
        "10037660$200$2000$20000$Ptosis$Eye disorders$...$\n"
    )
    d = MeddraDictionary.from_path(tmp_path)
    assert d.has_pt("Nausea")
    assert d.soc_for("Nausea") == "GI disorders"
    assert d.soc_for("Ptosis") == "Eye disorders"
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""MedDRA dictionary loader.

MedDRA ships ASCII files:
 - pt.asc       (preferred terms)
 - mdhier.asc   (hierarchy: PT → HLT → HLGT → SOC)
 - llt.asc      (lower level terms — synonyms)

We only need PT + SOC + LLT synonyms.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeddraDictionary:
    pts: frozenset[str]
    pt_to_soc: dict[str, str]
    synonyms_to_pt: dict[str, str]

    @classmethod
    def from_path(cls, root: Path) -> "MeddraDictionary":
        pts: set[str] = set()
        pt_to_soc: dict[str, str] = {}
        synonyms: dict[str, str] = {}

        pt_file = root / "pt.asc"
        if pt_file.exists():
            for line in pt_file.read_text().splitlines():
                fields = line.split("$")
                if len(fields) >= 2 and fields[1]:
                    pts.add(fields[1])

        hier_file = root / "mdhier.asc"
        if hier_file.exists():
            for line in hier_file.read_text().splitlines():
                fields = line.split("$")
                if len(fields) >= 6 and fields[4] and fields[5]:
                    pt_to_soc[fields[4]] = fields[5]

        llt_file = root / "llt.asc"
        if llt_file.exists():
            for line in llt_file.read_text().splitlines():
                fields = line.split("$")
                # LLT file: llt_code $ llt_name $ pt_code $ ...
                # We need LLT name → PT name, so cross-reference via code
                # Simplified: skip if format differs.
                pass

        return cls(
            pts=frozenset(pts),
            pt_to_soc=pt_to_soc,
            synonyms_to_pt=synonyms,
        )

    def has_pt(self, pt: str) -> bool:
        return pt in self.pts

    def soc_for(self, pt: str) -> str | None:
        return self.pt_to_soc.get(pt)
```

- [ ] **Step 4: Run — expect pass**. Note: LLT cross-referencing needs a second pass (two-file join); left as an enhancement since early pipeline can go without synonyms and rely on FAISS retrieval.

- [ ] **Step 5: Commit**

```bash
git add src/botox_se/meddra/ tests/meddra/
git commit -m "feat(meddra): dictionary loader for PT + SOC from MedDRA ASCII release"
```

## Task 4.2: Exact-match mapper

**Files:**
- Create: `src/botox_se/meddra/exact_match.py`
- Test: `tests/meddra/test_exact_match.py`

- [ ] **Step 1: Write failing test**

```python
from botox_se.meddra.dictionary import MeddraDictionary
from botox_se.meddra.exact_match import ExactMatcher


def _tiny_dict() -> MeddraDictionary:
    return MeddraDictionary(
        pts=frozenset({"Nausea", "Ptosis", "Eyelid ptosis"}),
        pt_to_soc={"Nausea": "GI", "Ptosis": "Eye"},
        synonyms_to_pt={"droopy eyelid": "Ptosis", "queasy": "Nausea"},
    )


def test_exact_match_returns_pt_when_string_matches():
    m = ExactMatcher(_tiny_dict())
    assert m.match("Ptosis") == "Ptosis"
    assert m.match("ptosis") == "Ptosis"       # case-insensitive
    assert m.match("  Ptosis  ") == "Ptosis"   # whitespace


def test_exact_match_returns_none_when_unknown():
    m = ExactMatcher(_tiny_dict())
    assert m.match("extremely rare phrase nobody uses") is None


def test_exact_match_uses_synonyms():
    m = ExactMatcher(_tiny_dict())
    assert m.match("droopy eyelid") == "Ptosis"
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Fast case-insensitive exact-match against MedDRA PTs and LLT synonyms."""

from botox_se.meddra.dictionary import MeddraDictionary


class ExactMatcher:
    def __init__(self, dictionary: MeddraDictionary) -> None:
        # Lowercase-indexed for fast lookup
        self._pt_lower = {p.lower(): p for p in dictionary.pts}
        self._syn_lower = {s.lower(): p for s, p in dictionary.synonyms_to_pt.items()}

    def match(self, span: str) -> str | None:
        key = span.strip().lower()
        if key in self._pt_lower:
            return self._pt_lower[key]
        if key in self._syn_lower:
            return self._syn_lower[key]
        return None
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Commit**.

## Task 4.3: FAISS retriever

**Files:**
- Create: `src/botox_se/meddra/retriever.py`
- Test: `tests/meddra/test_retriever.py`

- [ ] **Step 1: Write failing test**

```python
from botox_se.meddra.dictionary import MeddraDictionary
from botox_se.meddra.retriever import FaissRetriever


def test_retriever_returns_topk_by_semantic_similarity():
    d = MeddraDictionary(
        pts=frozenset({"Nausea", "Vomiting", "Ptosis", "Headache", "Eyelid ptosis"}),
        pt_to_soc={}, synonyms_to_pt={},
    )
    r = FaissRetriever.build(d)
    hits = r.topk("droopy eyelid", k=3)
    top = [pt for pt, _ in hits]
    assert "Ptosis" in top or "Eyelid ptosis" in top
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""FAISS-based nearest-neighbor retrieval over MedDRA PTs.

Uses sentence-transformers for embeddings; small model for speed + cost.
"""

from dataclasses import dataclass
from typing import Self

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from botox_se.meddra.dictionary import MeddraDictionary


@dataclass
class FaissRetriever:
    pts: list[str]
    index: faiss.IndexFlatIP
    embedder: SentenceTransformer

    @classmethod
    def build(
        cls,
        dictionary: MeddraDictionary,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Self:
        pts = sorted(dictionary.pts)
        embedder = SentenceTransformer(model_name)
        vecs = embedder.encode(pts, normalize_embeddings=True).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(pts=pts, index=index, embedder=embedder)

    def topk(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return [(self.pts[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
```

- [ ] **Step 4: Run — expect pass** (slow first run — downloads model). **Step 5: Commit**.

## Task 4.4: LLM reranker and mapper orchestration

**Files:**
- Create: `src/botox_se/meddra/reranker.py`, `src/botox_se/meddra/mapper.py`, `scripts/run_mapping.py`
- Test: `tests/meddra/test_mapper.py`

- [ ] **Step 1: Write failing mapper test**

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from botox_se.meddra.dictionary import MeddraDictionary
from botox_se.meddra.mapper import Mapper, MappingResult


@pytest.mark.asyncio
async def test_mapper_exact_match_path():
    d = MeddraDictionary(
        pts=frozenset({"Nausea", "Ptosis"}),
        pt_to_soc={}, synonyms_to_pt={},
    )
    fake_retriever = MagicMock()
    fake_reranker = MagicMock(rerank=AsyncMock())
    m = Mapper(dictionary=d, retriever=fake_retriever, reranker=fake_reranker)
    result = await m.map_span("Ptosis")
    assert result.pt == "Ptosis"
    assert result.source == "exact"
    fake_reranker.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_mapper_fallback_to_retriever_and_reranker():
    d = MeddraDictionary(
        pts=frozenset({"Nausea", "Ptosis", "Eyelid ptosis"}),
        pt_to_soc={}, synonyms_to_pt={},
    )
    fake_retriever = MagicMock()
    fake_retriever.topk = MagicMock(return_value=[
        ("Ptosis", 0.9), ("Eyelid ptosis", 0.85), ("Nausea", 0.1),
    ])
    fake_reranker = MagicMock()
    fake_reranker.rerank = AsyncMock(return_value="Eyelid ptosis")
    m = Mapper(dictionary=d, retriever=fake_retriever, reranker=fake_reranker)
    result = await m.map_span("droopy lid")
    assert result.pt == "Eyelid ptosis"
    assert result.source == "rerank"


@pytest.mark.asyncio
async def test_mapper_no_match_returns_none():
    d = MeddraDictionary(pts=frozenset({"Ptosis"}), pt_to_soc={}, synonyms_to_pt={})
    fake_retriever = MagicMock()
    fake_retriever.topk = MagicMock(return_value=[("Ptosis", 0.2)])  # low score
    fake_reranker = MagicMock()
    fake_reranker.rerank = AsyncMock(return_value=None)
    m = Mapper(dictionary=d, retriever=fake_retriever, reranker=fake_reranker)
    result = await m.map_span("completely unrelated phrase")
    assert result.pt is None
    assert result.source == "unmapped"
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement `reranker.py`**

```python
"""LLM reranker: given a free-text span and top-k candidate PTs, pick the best
match or return None."""

from openai import AsyncOpenAI


class LlmReranker:
    def __init__(self, *, openai: AsyncOpenAI, model: str) -> None:
        self._client = openai
        self._model = model

    async def rerank(self, *, span: str, candidates: list[str]) -> str | None:
        options = "\n".join(f"- {c}" for c in candidates)
        prompt = (
            "Select the single MedDRA Preferred Term that best matches the patient-"
            "reported phrase. If none of the options fit, return exactly "
            "'NO_MATCH'.\n\n"
            f"Phrase: {span!r}\n"
            f"Options:\n{options}\n\n"
            "Answer with just the chosen PT name (or NO_MATCH)."
        )
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        choice = (completion.choices[0].message.content or "").strip()
        if choice == "NO_MATCH" or choice not in candidates:
            return None
        return choice
```

- [ ] **Step 4: Implement `mapper.py`**

```python
"""Three-step MedDRA mapping: exact match → FAISS → LLM rerank."""

from dataclasses import dataclass

from botox_se.meddra.dictionary import MeddraDictionary
from botox_se.meddra.exact_match import ExactMatcher


@dataclass(frozen=True)
class MappingResult:
    span: str
    pt: str | None
    source: str           # "exact" | "rerank" | "unmapped"


class Mapper:
    def __init__(self, *, dictionary: MeddraDictionary, retriever, reranker,
                 retrieval_score_threshold: float = 0.4) -> None:
        self._exact = ExactMatcher(dictionary)
        self._retriever = retriever
        self._reranker = reranker
        self._threshold = retrieval_score_threshold

    async def map_span(self, span: str) -> MappingResult:
        # Step 1: exact
        exact = self._exact.match(span)
        if exact is not None:
            return MappingResult(span=span, pt=exact, source="exact")

        # Step 2: retrieve top-5
        hits = self._retriever.topk(span, k=5)
        if not hits or hits[0][1] < self._threshold:
            return MappingResult(span=span, pt=None, source="unmapped")

        # Step 3: LLM rerank
        candidates = [pt for pt, _ in hits]
        chosen = await self._reranker.rerank(span=span, candidates=candidates)
        if chosen is None:
            return MappingResult(span=span, pt=None, source="unmapped")
        return MappingResult(span=span, pt=chosen, source="rerank")
```

- [ ] **Step 5: Run — expect pass**. **Step 6: Implement `scripts/run_mapping.py`**

```python
"""CLI: for each extraction row, re-map d_side_effect_pts through the Mapper.

Writes `data/processed/extractions_mapped.parquet`:
 - original extraction fields
 - new column: d_side_effect_pts_mapped (list of {raw_span, meddra_pt, source})
 - new column: unmapped_spans (list of spans that failed to map)
"""

import asyncio
import json
from pathlib import Path

import polars as pl
from openai import AsyncOpenAI

from botox_se.config import load as load_settings
from botox_se.meddra.dictionary import MeddraDictionary
from botox_se.meddra.mapper import Mapper
from botox_se.meddra.reranker import LlmReranker
from botox_se.meddra.retriever import FaissRetriever
from botox_se.paths import Paths


async def _run() -> None:
    settings = load_settings()
    paths = Paths(repo_root=Path.cwd())

    d = MeddraDictionary.from_path(paths.reference / "meddra")
    retriever = FaissRetriever.build(d)
    oa = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    reranker = LlmReranker(openai=oa, model=settings.llm_reranker_model)
    mapper = Mapper(dictionary=d, retriever=retriever, reranker=reranker)

    df = pl.read_parquet(paths.processed / "extractions.parquet")
    out_rows = []
    for row in df.iter_rows(named=True):
        extraction = json.loads(row["extraction_json"])
        mapped = []
        unmapped = []
        for se in extraction.get("d_side_effect_pts", []):
            result = await mapper.map_span(se["raw_span"])
            if result.pt is None:
                unmapped.append(se["raw_span"])
            else:
                mapped.append({"raw_span": se["raw_span"], "meddra_pt": result.pt,
                               "source": result.source})
        out_rows.append({
            **row,
            "d_side_effect_pts_mapped": json.dumps(mapped),
            "unmapped_spans": json.dumps(unmapped),
        })

    pl.DataFrame(out_rows).write_parquet(paths.processed / "extractions_mapped.parquet")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**.

---

# Phase 5 — FAERS and FDA labels

**Deliverable:** `data/processed/faers.parquet` and `data/processed/labels.parquet`.

## Task 5.1: openFDA FAERS client

**Files:**
- Create: `src/botox_se/faers/openfda_client.py`, `src/botox_se/faers/query.py`, `src/botox_se/faers/normalize.py`
- Test: `tests/faers/test_openfda_client.py`, `tests/faers/test_normalize.py`

- [ ] **Step 1: Write failing client test**

```python
import pytest
from httpx import AsyncClient

from botox_se.faers.openfda_client import OpenFdaClient


@pytest.mark.asyncio
async def test_fetches_events_with_pagination(httpx_mock):
    httpx_mock.add_response(
        url__regex=r".*/drug/event.json\?.*skip=0.*",
        json={"meta": {"results": {"total": 3}}, "results": [{"safetyreportid": "1"}, {"safetyreportid": "2"}]},
    )
    httpx_mock.add_response(
        url__regex=r".*/drug/event.json\?.*skip=100.*",
        json={"meta": {"results": {"total": 3}}, "results": [{"safetyreportid": "3"}]},
    )
    async with AsyncClient() as http:
        c = OpenFdaClient(http=http, api_key=None)
        reports = [r async for r in c.events(query="botox")]
    assert [r["safetyreportid"] for r in reports] == ["1", "2", "3"]
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement `openfda_client.py`**

```python
"""openFDA client for the /drug/event endpoint (FAERS).

Paginates via `skip` + `limit`, handles rate limits via tenacity."""

from collections.abc import AsyncIterator

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OPENFDA_BASE = "https://api.fda.gov"


class OpenFdaClient:
    def __init__(self, *, http: httpx.AsyncClient, api_key: str | None,
                 base_url: str = OPENFDA_BASE, page_size: int = 100) -> None:
        self._http = http
        self._api_key = api_key
        self._base = base_url
        self._page_size = page_size

    async def events(self, *, query: str) -> AsyncIterator[dict]:
        skip = 0
        while True:
            params = {"search": query, "limit": self._page_size, "skip": skip}
            if self._api_key:
                params["api_key"] = self._api_key
            payload = await self._get("/drug/event.json", params)
            results = payload.get("results", [])
            for r in results:
                yield r
            total = payload.get("meta", {}).get("results", {}).get("total", 0)
            skip += self._page_size
            if skip >= total or not results:
                break

    async def _get(self, path: str, params: dict) -> dict:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(httpx.HTTPError),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        ):
            with attempt:
                r = await self._http.get(f"{self._base}{path}", params=params, timeout=60.0)
                r.raise_for_status()
                return r.json()
        raise RuntimeError("unreachable")
```

- [ ] **Step 4: Run — expect pass**.

- [ ] **Step 5: Write `query.py`** — per-product query strings

```python
"""FAERS query builders per BoNT-A product.

openFDA indexes on `patient.drug.medicinalproduct` and `patient.drug.openfda.*`.
We use a disjunctive search that covers brand + generic variants."""

PRODUCT_QUERIES: dict[str, str] = {
    "botox": (
        'patient.drug.openfda.brand_name:"BOTOX" '
        'OR patient.drug.openfda.generic_name:"ONABOTULINUMTOXINA"'
    ),
    "dysport": (
        'patient.drug.openfda.brand_name:"DYSPORT" '
        'OR patient.drug.openfda.generic_name:"ABOBOTULINUMTOXINA"'
    ),
    "xeomin": (
        'patient.drug.openfda.brand_name:"XEOMIN" '
        'OR patient.drug.openfda.generic_name:"INCOBOTULINUMTOXINA"'
    ),
    "jeuveau": (
        'patient.drug.openfda.brand_name:"JEUVEAU" '
        'OR patient.drug.openfda.generic_name:"PRABOTULINUMTOXINA"'
    ),
    "daxxify": (
        'patient.drug.openfda.brand_name:"DAXXIFY" '
        'OR patient.drug.openfda.generic_name:"DAXIBOTULINUMTOXINA"'
    ),
}
```

- [ ] **Step 6: Write normalize test**

```python
from botox_se.faers.normalize import events_to_pt_rows


def test_extracts_pts_and_indication_from_event():
    event = {
        "safetyreportid": "123",
        "receivedate": "20230101",
        "patient": {
            "reaction": [{"reactionmeddrapt": "Ptosis"}, {"reactionmeddrapt": "Headache"}],
            "drug": [{
                "medicinalproduct": "BOTOX",
                "drugcharacterization": "1",
                "drugindication": "GLABELLAR LINES",
            }],
        },
    }
    rows = events_to_pt_rows(event, product="botox")
    pts = sorted(r["pt"] for r in rows)
    assert pts == ["Headache", "Ptosis"]
    assert rows[0]["indication"] == "GLABELLAR LINES"
    assert rows[0]["product"] == "botox"
    assert rows[0]["safetyreportid"] == "123"
```

- [ ] **Step 7: Implement `normalize.py`**

```python
"""Normalize FAERS event JSON into (report, product, PT, indication) rows."""


def events_to_pt_rows(event: dict, *, product: str) -> list[dict]:
    report_id = event.get("safetyreportid")
    date = event.get("receivedate")
    patient = event.get("patient", {})
    reactions = patient.get("reaction", []) or []
    drugs = patient.get("drug", []) or []

    indication = None
    for d in drugs:
        prod_str = (d.get("medicinalproduct") or "").lower()
        if product in prod_str:
            indication = d.get("drugindication")
            break

    rows = []
    for rx in reactions:
        pt = rx.get("reactionmeddrapt")
        if not pt:
            continue
        rows.append({
            "safetyreportid": report_id,
            "receivedate": date,
            "product": product,
            "pt": pt,
            "indication": indication,
        })
    return rows
```

- [ ] **Step 8: Run — expect pass**. **Step 9: CLI `scripts/run_faers.py`**

```python
"""CLI: pull FAERS reports for each BoNT-A product, normalize to parquet."""

import asyncio
from pathlib import Path

import httpx
import polars as pl

from botox_se.config import load as load_settings
from botox_se.faers.normalize import events_to_pt_rows
from botox_se.faers.openfda_client import OpenFdaClient
from botox_se.faers.query import PRODUCT_QUERIES
from botox_se.paths import Paths


async def _run() -> None:
    settings = load_settings()
    paths = Paths(repo_root=Path.cwd())
    paths.ensure_dirs()

    all_rows = []
    async with httpx.AsyncClient() as http:
        client = OpenFdaClient(http=http, api_key=settings.openfda_api_key.get_secret_value() or None)
        for product, query in PRODUCT_QUERIES.items():
            async for ev in client.events(query=query):
                all_rows.extend(events_to_pt_rows(ev, product=product))

    pl.DataFrame(all_rows).write_parquet(paths.processed / "faers.parquet")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Commit**.

## Task 5.2: Indication stratification for FAERS

**Files:**
- Create: `src/botox_se/faers/indication.py`
- Test: `tests/faers/test_indication.py`

- [ ] **Step 1: Test**

```python
from botox_se.faers.indication import classify_faers_indication


def test_cosmetic_indications():
    assert classify_faers_indication("Glabellar Lines") == "cosmetic"
    assert classify_faers_indication("FOREHEAD LINES") == "cosmetic"
    assert classify_faers_indication("Crow's feet") == "cosmetic"


def test_medical_indications():
    assert classify_faers_indication("Chronic Migraine") == "medical"
    assert classify_faers_indication("Cervical Dystonia") == "medical"
    assert classify_faers_indication("Hyperhidrosis, Axillary") == "medical"


def test_unknown_indication():
    assert classify_faers_indication(None) == "unknown"
    assert classify_faers_indication("") == "unknown"
    assert classify_faers_indication("Product used for unknown indication") == "unknown"
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Rule-based classification of FAERS `drugindication` text into
medical / cosmetic / unknown."""

import re

COSMETIC_PATTERNS = [
    r"\bglabellar\b", r"\bforehead\b", r"\bcrow.?s?\s*feet\b",
    r"\blateral canthal\b", r"\bfacial wrinkle\b", r"\bcosmetic\b",
    r"\bbeauty\b", r"\baesthetic\b",
]

MEDICAL_PATTERNS = [
    r"\bmigraine\b", r"\bdystonia\b", r"\bblepharospasm\b",
    r"\bspasticity\b", r"\bcerebral palsy\b", r"\bhyperhidrosis\b",
    r"\boveractive bladder\b", r"\bstrabismus\b", r"\burinary incontinence\b",
]


def classify_faers_indication(text: str | None) -> str:
    if not text:
        return "unknown"
    t = text.lower()
    if any(re.search(p, t) for p in COSMETIC_PATTERNS):
        return "cosmetic"
    if any(re.search(p, t) for p in MEDICAL_PATTERNS):
        return "medical"
    return "unknown"
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Commit**.

## Task 5.3: DailyMed SPL parser

**Files:**
- Create: `src/botox_se/labels/dailymed_client.py`, `src/botox_se/labels/parse.py`, `scripts/run_labels.py`
- Test: `tests/labels/test_parse.py`

- [ ] **Step 1: Test** uses a minimal SPL XML fixture (committed under `tests/fixtures/spl_botox_cosmetic.xml`, a trimmed real-world example with the adverse-reactions section only).

```python
from pathlib import Path

from botox_se.labels.parse import parse_adverse_reactions_terms


def test_extracts_ae_terms_from_spl_fixture():
    spl = Path(__file__).parent.parent / "fixtures" / "spl_botox_cosmetic.xml"
    terms = parse_adverse_reactions_terms(spl.read_text())
    assert "ptosis" in {t.lower() for t in terms}
    assert "headache" in {t.lower() for t in terms}
```

- [ ] **Step 2: Commit a 50-line SPL fixture** at `tests/fixtures/spl_botox_cosmetic.xml` containing just the `<component><section>` block for Adverse Reactions with a handful of `<term>` or `<paragraph>` entries naming ptosis, headache, eyelid edema.

- [ ] **Step 3: Run — expect fail**. **Step 4: Implement `parse.py`**

```python
"""Parse DailyMed SPL XML to extract the Adverse Reactions section's term list.

SPL (Structured Product Labeling) uses HL7 XML. The Adverse Reactions section
has LOINC code 34084-4. We extract all `<paragraph>` and table-cell text,
tokenize, and lower-case; downstream MedDRA mapping picks canonical PTs."""

import re
from lxml import etree

NS = {"hl7": "urn:hl7-org:v3"}
ADVERSE_REACTIONS_LOINC = "34084-4"


def parse_adverse_reactions_terms(xml_text: str) -> list[str]:
    root = etree.fromstring(xml_text.encode("utf-8"))
    # Find the section whose code/@code is 34084-4
    sections = root.xpath(
        "//hl7:section[hl7:code/@code=$c]",
        c=ADVERSE_REACTIONS_LOINC,
        namespaces=NS,
    )
    terms: list[str] = []
    for sec in sections:
        for node in sec.iter():
            text = (node.text or "").strip()
            if not text:
                continue
            # Split on commas / semicolons; keep short noun phrases
            for candidate in re.split(r"[,;•\n]", text):
                c = candidate.strip()
                if 2 <= len(c) <= 60 and re.match(r"^[A-Za-z][A-Za-z \-/]+$", c):
                    terms.append(c)
    # Dedupe
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out
```

- [ ] **Step 5: Run — expect pass**. **Step 6: Implement `dailymed_client.py`**

```python
"""Fetch SPL XML for a given NDC or Set ID from DailyMed."""

import httpx

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


async def fetch_spl(http: httpx.AsyncClient, *, set_id: str) -> str:
    r = await http.get(f"{DAILYMED_BASE}/spls/{set_id}.xml", timeout=60.0)
    r.raise_for_status()
    return r.text
```

- [ ] **Step 7: CLI `scripts/run_labels.py`**

Config-driven: `config/labels.yaml` lists `{product, indication_arm, set_id}` tuples. CLI pulls each, parses, runs the MedDRA mapper (Task 4.4's `Mapper`) over extracted terms, writes `data/processed/labels.parquet` with `(product, indication_arm, pt)`.

- [ ] **Step 8: Commit**.

---

# Phase 6 — Analysis

**Deliverable:** `data/processed/analysis/` with tables + `docs/paper/figures/` with publication-quality figures.

## Task 6.1: Cohort builder

**Files:**
- Create: `src/botox_se/analysis/cohort.py`
- Test: `tests/analysis/test_cohort.py`

- [ ] **Step 1: Test**

```python
import json

import polars as pl

from botox_se.analysis.cohort import build_person_indication_records


def test_person_indication_records_split_when_user_has_both_arms():
    df = pl.DataFrame({
        "author": ["u1", "u1", "u2", "u3"],
        "extraction_json": [
            json.dumps({"a_self_use": True, "c_indication_category": "medical"}),
            json.dumps({"a_self_use": True, "c_indication_category": "cosmetic"}),
            json.dumps({"a_self_use": True, "c_indication_category": "cosmetic"}),
            json.dumps({"a_self_use": True, "c_indication_category": "unknown"}),
        ],
        "id": ["p1", "p2", "p3", "p4"],
    })
    records = build_person_indication_records(df)
    pairs = sorted((r["author"], r["indication"]) for r in records)
    assert pairs == [("u1", "cosmetic"), ("u1", "medical"), ("u2", "cosmetic")]
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Build person-indication records per spec §6.1."""

import json

import polars as pl


def build_person_indication_records(df: pl.DataFrame) -> list[dict]:
    records: dict[tuple[str, str], set[str]] = {}
    for row in df.iter_rows(named=True):
        ex = json.loads(row["extraction_json"])
        if not ex.get("a_self_use"):
            continue
        ind = ex.get("c_indication_category", "unknown")
        if ind == "unknown":
            continue
        key = (row["author"], ind)
        records.setdefault(key, set()).add(row["id"])
    return [
        {"author": a, "indication": ind, "source_post_ids": sorted(ids)}
        for (a, ind), ids in records.items()
    ]
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Commit**.

## Task 6.2: Descriptive analysis

**Files:**
- Create: `src/botox_se/analysis/descriptive.py`
- Test: `tests/analysis/test_descriptive.py`

- [ ] **Step 1: Test — user counts and PT prevalence per arm**

```python
import polars as pl

from botox_se.analysis.descriptive import pt_prevalence_per_arm


def test_pt_prevalence_computes_per_arm_fractions():
    # 4 person-indication records (3 cosmetic, 1 medical)
    # PT "Ptosis" appears in 2 cosmetic, 0 medical
    records = pl.DataFrame({
        "author": ["u1", "u2", "u3", "u4"],
        "indication": ["cosmetic", "cosmetic", "cosmetic", "medical"],
        "pts": [["Ptosis"], ["Ptosis", "Headache"], [], ["Headache"]],
    })
    out = pt_prevalence_per_arm(records)
    d = {(r["arm"], r["pt"]): r["prevalence"] for r in out.iter_rows(named=True)}
    assert d[("cosmetic", "Ptosis")] == pytest.approx(2 / 3)
    assert d[("cosmetic", "Headache")] == pytest.approx(1 / 3)
    assert d[("medical", "Headache")] == pytest.approx(1.0)
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Per-arm PT prevalence and SOC rollups."""

import polars as pl


def pt_prevalence_per_arm(records: pl.DataFrame) -> pl.DataFrame:
    exploded = records.explode("pts").rename({"pts": "pt"})
    # Count users reporting each PT per arm
    pt_counts = (
        exploded.filter(pl.col("pt").is_not_null())
        .group_by(["indication", "pt"])
        .agg(pl.col("author").n_unique().alias("n_reporting"))
    )
    arm_sizes = records.group_by("indication").agg(pl.col("author").n_unique().alias("n_arm"))
    joined = pt_counts.join(arm_sizes, on="indication")
    return joined.with_columns(
        prevalence=pl.col("n_reporting") / pl.col("n_arm"),
        arm=pl.col("indication"),
    ).select(["arm", "pt", "n_reporting", "n_arm", "prevalence"]).sort(["arm", "prevalence"], descending=[False, True])
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Commit**.

## Task 6.3: ROR with BH correction

**Files:**
- Create: `src/botox_se/analysis/ror.py`
- Test: `tests/analysis/test_ror.py`

- [ ] **Step 1: Test — known 2×2 table**

```python
from botox_se.analysis.ror import reporting_odds_ratio


def test_ror_2x2_known_values():
    # a=10 (exposed, PT), b=90 (exposed, not PT)
    # c=5  (ref, PT),     d=95 (ref, not PT)
    result = reporting_odds_ratio(a=10, b=90, c=5, d=95)
    # ROR = (10/90) / (5/95) = 2.111
    assert abs(result.ror - 2.111) < 0.01
    assert result.ci_low < result.ror < result.ci_high
    assert result.ror > 1


def test_ror_zero_cell_gets_half_correction():
    # Zero in cell a should still produce a finite value via 0.5 correction
    result = reporting_odds_ratio(a=0, b=100, c=10, d=100)
    assert result.ror > 0
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Reporting Odds Ratio with 95% CI via log-normal approximation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RorResult:
    a: int
    b: int
    c: int
    d: int
    ror: float
    ci_low: float
    ci_high: float
    log_se: float


def reporting_odds_ratio(*, a: int, b: int, c: int, d: int) -> RorResult:
    # Haldane–Anscombe correction for any zero cell
    if min(a, b, c, d) == 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d
    ror = (a_ / b_) / (c_ / d_)
    log_ror = math.log(ror)
    log_se = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    lo = math.exp(log_ror - 1.96 * log_se)
    hi = math.exp(log_ror + 1.96 * log_se)
    return RorResult(a=a, b=b, c=c, d=d, ror=ror, ci_low=lo, ci_high=hi, log_se=log_se)
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Add BH correction wrapper**

```python
def compute_all_rors(prevalence_df: pl.DataFrame, *, exposure_arm: str, reference_arm: str,
                    min_reports: int = 20) -> pl.DataFrame:
    """For each PT with ≥min_reports across both arms, compute ROR + CI.
    Apply Benjamini–Hochberg FDR correction on two-sided p-values from log_se."""
    # ... (full implementation loops over PTs, builds 2x2, calls reporting_odds_ratio,
    # computes p = 2 * (1 - Φ(|log_ror| / log_se)), applies BH via scipy.stats or statsmodels)
```

- [ ] **Step 6: Add test** for BH corrected output, then commit.

## Task 6.4: Label concordance

**Files:**
- Create: `src/botox_se/analysis/concordance.py`
- Test: `tests/analysis/test_concordance.py`

- [ ] **Step 1: Test**

```python
import polars as pl

from botox_se.analysis.concordance import label_concordance


def test_concordance_flags_pts_not_in_label():
    reddit_pts = pl.DataFrame({
        "arm": ["cosmetic", "cosmetic", "cosmetic"],
        "pt": ["Ptosis", "Frozen face feeling", "Headache"],
        "n_reporting": [50, 30, 20],
    })
    label_pts = pl.DataFrame({
        "arm": ["cosmetic", "cosmetic"],
        "pt": ["Ptosis", "Headache"],
    })
    result = label_concordance(reddit_pts, label_pts)
    d = {r["pt"]: r["in_label"] for r in result.iter_rows(named=True)}
    assert d == {"Ptosis": True, "Frozen face feeling": False, "Headache": True}
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Compare Reddit-extracted PTs per arm against FDA label PTs per arm."""

import polars as pl


def label_concordance(reddit: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    label_set = labels.select(["arm", "pt"]).with_columns(pl.lit(True).alias("in_label"))
    return (
        reddit.join(label_set, on=["arm", "pt"], how="left")
        .with_columns(in_label=pl.col("in_label").fill_null(False))
        .sort(["arm", "n_reporting"], descending=[False, True])
    )
```

- [ ] **Step 4: Run — expect pass**. **Step 5: Commit**.

## Task 6.5: FAERS triangulation (rank correlation)

**Files:**
- Create: `src/botox_se/analysis/triangulation.py`
- Test: `tests/analysis/test_triangulation.py`

- [ ] **Step 1: Test**

```python
import polars as pl

from botox_se.analysis.triangulation import spearman_per_arm


def test_spearman_returns_high_corr_when_rankings_agree():
    reddit = pl.DataFrame({"arm": ["cosmetic"] * 3, "pt": ["A", "B", "C"], "rank_reddit": [1, 2, 3]})
    faers = pl.DataFrame({"arm": ["cosmetic"] * 3, "pt": ["A", "B", "C"], "rank_faers": [1, 2, 3]})
    result = spearman_per_arm(reddit, faers)
    assert result["cosmetic"] == pytest.approx(1.0)


def test_spearman_low_when_rankings_disagree():
    reddit = pl.DataFrame({"arm": ["cosmetic"] * 3, "pt": ["A", "B", "C"], "rank_reddit": [1, 2, 3]})
    faers = pl.DataFrame({"arm": ["cosmetic"] * 3, "pt": ["A", "B", "C"], "rank_faers": [3, 1, 2]})
    result = spearman_per_arm(reddit, faers)
    assert -1.0 <= result["cosmetic"] < 0.5
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement using `scipy.stats.spearmanr`**. **Step 4: Run — expect pass**. **Step 5: Commit**.

## Task 6.6: Temporal analysis

**Files:**
- Create: `src/botox_se/analysis/temporal.py`
- Test: `tests/analysis/test_temporal.py`

- [ ] **Step 1: Test: monthly PT rate per arm, normalized by arm monthly volume**

```python
import polars as pl

from botox_se.analysis.temporal import monthly_pt_rates


def test_monthly_pt_rates_normalized_by_arm_volume():
    df = pl.DataFrame({
        "author": ["u1", "u2", "u3", "u4"],
        "arm": ["cosmetic"] * 4,
        "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
        "pts": [["Ptosis"], [], ["Ptosis"], []],
    })
    rates = monthly_pt_rates(df, top_n=5)
    row = rates.filter((pl.col("month") == "2024-01") & (pl.col("pt") == "Ptosis")).row(0, named=True)
    assert row["rate"] == pytest.approx(0.5)
```

- [ ] **Step 2 through 5: implement + test + commit**.

## Task 6.7: Sentiment analysis

**Files:**
- Create: `src/botox_se/analysis/sentiment.py`
- Test: `tests/analysis/test_sentiment.py`

Test sentiment × PT × arm cross-tabs, then the `discontinuation_intent ~ sentiment_about_side_effect` relationship (logit regression via statsmodels). Standard pattern — write tests then implementation; commit.

## Task 6.8: Analysis CLI

**Files:**
- Create: `scripts/run_analysis.py`

- [ ] Orchestrates cohort → descriptive → ROR → concordance → triangulation → temporal → sentiment, writing outputs to `data/processed/analysis/` and `docs/paper/tables/`. Commit.

---

# Phase 7 — Validation and reliability

**Deliverable:** `validation/reliability_report.md` — per-field κ, sensitivity, PPV; explicit gate that excludes low-κ fields from inferential analyses.

## Task 7.1: Stratified annotation sampler

**Files:**
- Create: `src/botox_se/validation/sampler.py`, `scripts/build_annotation_sample.py`
- Test: `tests/validation/test_sampler.py`

- [ ] **Step 1: Test — stratified 200-post sample across subreddit and indication**

```python
import polars as pl

from botox_se.validation.sampler import stratified_sample


def test_stratified_sample_balances_across_strata():
    df = pl.DataFrame({
        "id": [str(i) for i in range(1000)],
        "subreddit": ["migraine"] * 500 + ["Botox"] * 500,
        "predicted_indication": ["medical"] * 500 + ["cosmetic"] * 500,
    })
    sample = stratified_sample(df, n_total=200, strata_cols=["subreddit", "predicted_indication"], seed=42)
    assert sample.height == 200
    assert sample.filter(pl.col("subreddit") == "migraine").height == 100
    assert sample.filter(pl.col("subreddit") == "Botox").height == 100
```

- [ ] **Step 2: Run — expect fail**. **Step 3: Implement**

```python
"""Stratified random sampler for the annotation set."""

import polars as pl


def stratified_sample(
    df: pl.DataFrame, *, n_total: int, strata_cols: list[str], seed: int = 0,
) -> pl.DataFrame:
    strata = df.group_by(strata_cols).agg(pl.len().alias("stratum_size"))
    n_strata = strata.height
    per = n_total // n_strata
    rng = pl.random_state(seed) if hasattr(pl, "random_state") else seed  # fallback for older polars
    return (
        df.with_columns(pl.all().shuffle(seed=seed))
        .group_by(strata_cols, maintain_order=True)
        .head(per)
    )
```

- [ ] **Step 4: Run — expect pass**. **Step 5: CLI**: `scripts/build_annotation_sample.py` reads `data/processed/extractions.parquet`, stratifies, writes `validation/annotation_sample.csv` with post text + model predictions *hidden* (separate file for reviewers). Commit.

## Task 7.2: Annotation protocol document

**Files:**
- Create: `validation/annotation_protocol.md`

- [ ] **Step 1: Write protocol** — define for each field how annotators should decide (provide positive + negative examples, borderline cases, the exact decision procedure for `c_indication_category`, etc.). Must be specific enough that two different annotators produce the same labels on clear cases. Commit.

## Task 7.3: Reliability calculator

**Files:**
- Create: `src/botox_se/validation/reliability.py`, `scripts/build_reliability_report.py`
- Test: `tests/validation/test_reliability.py`

- [ ] **Step 1: Test Cohen's κ via sklearn.metrics.cohen_kappa_score; test sensitivity + PPV against known gold labels**.

- [ ] **Step 2: Implement** — reads annotator A + annotator B + adjudicated gold; produces `validation/reliability_report.md` with per-field metrics and coverage rates. Commit.

## Task 7.4: Reliability gate

**Files:**
- Create: `src/botox_se/validation/gate.py`

- [ ] **Step 1: Test — gate allows field only if κ ≥ 0.6 (subjective) or precision ≥ 0.8 (extractive)**.
- [ ] **Step 2: Implement** as a function that reads `validation/reliability_report.md`'s machine-readable companion (`validation/reliability.json`) and returns a set of "allowed" field names. Analysis scripts import this and skip gated fields in inferential analyses. Commit.

---

# Phase 8 — Reporting and manuscript

**Deliverable:** `docs/paper/manuscript.md` draft, `docs/paper/figures/*.pdf`, `docs/paper/tables/*.csv`, plus a `make pipeline` target that reproduces everything end-to-end from raw → manuscript on a clean checkout.

## Task 8.1: Pub-quality figures

**Files:**
- Create: `src/botox_se/reporting/figures.py`, notebooks 01–06.

- [ ] **Step 1: Figure 1** — cohort flow diagram (raw posts → filtered → self-use → per-arm).
- [ ] **Step 2: Figure 2** — top-20 PT prevalence per arm (side-by-side bar charts).
- [ ] **Step 3: Figure 3** — ROR forest plot (BH-significant PTs only).
- [ ] **Step 4: Figure 4** — Reddit-vs-FAERS concordance scatter per arm.
- [ ] **Step 5: Figure 5** — monthly temporal rates of top-5 PTs per arm.
- [ ] **Step 6: Each figure has** a small test loading a mini fixture and asserting the PDF file is produced and non-empty. Commit after each.

## Task 8.2: Tables

**Files:**
- Create: `src/botox_se/reporting/tables.py`

- [ ] **Table 1** — cohort description (users, posts, subreddit mix, demographics coverage).
- [ ] **Table 2** — top-20 PTs per arm (mirrors Sehgal Table 1 format).
- [ ] **Table 3** — label concordance matrix.
- [ ] **Table 4** — ROR + CI + q-value per BH-significant PT.
- [ ] **Table 5** — FAERS PT comparison per arm.

## Task 8.3: Methods document

**Files:**
- Create: `docs/methods.md`

- [ ] **Step 1: Write** — data sources, collection window, filters, LLM model + prompt version, MedDRA license, validation procedure, analytical methods (copy-paste-refine from spec §3–§6). Commit.

## Task 8.4: Manuscript scaffold

**Files:**
- Create: `docs/paper/manuscript.md`

- [ ] **Step 1: Write abstract + introduction skeleton** (~1 page).
- [ ] **Step 2: Methods** — pointer to `docs/methods.md`.
- [ ] **Step 3: Results** — placeholders that reference auto-generated table / figure files by path so regeneration refreshes them.
- [ ] **Step 4: Discussion + Limitations** — lift limitations directly from spec §10.
- [ ] **Step 5: Commit**.

## Task 8.5: End-to-end pipeline verification

- [ ] **Step 1: On a disposable machine (or clean checkout), run**

```bash
make install
cp .env.example .env
# fill in API keys
make pipeline
```

- [ ] **Step 2: Verify** all artifacts produced: `data/processed/extractions.parquet`, `data/processed/extractions_mapped.parquet`, `data/processed/faers.parquet`, `data/processed/labels.parquet`, `data/processed/analysis/*.csv`, `docs/paper/tables/*.csv`, `docs/paper/figures/*.pdf`, `validation/reliability_report.md`.

- [ ] **Step 3: Commit a `docs/reproducibility.md`** documenting the exact run cost (tokens, API $), wall-clock time, and MedDRA/API versions used, so future runs can match.

---

# Natural plan split points (if you'd prefer to spin off sub-plans)

If this single plan is too long to hand off as one engagement, these are clean cut points where a sub-plan's output is a standalone, testable artifact:

- **Plan A = Phase 0 + 1 + 2**: ends with a filtered Reddit corpus parquet. Deliverable: reproducible data collection.
- **Plan B = Phase 3 + 4**: depends on Plan A's parquet. Ends with `extractions_mapped.parquet`. Deliverable: extraction pipeline with validation-ready outputs.
- **Plan C = Phase 5**: depends on nothing Reddit-side; can run in parallel with A/B. Deliverable: FAERS + labels parquet.
- **Plan D = Phase 6 + 7 + 8**: depends on B and C. Deliverable: manuscript.

Each sub-plan can be handed to a different implementer with only the inputs/outputs across the boundaries to agree on.

---

# Self-review (completed by plan author)

**Spec coverage check (walking spec sections):**

- §1 (research question + hypotheses) — covered by Phase 6–8 analysis tasks (H1 = Task 6.3 ROR, H2 = Task 6.4 concordance, H3 = Task 6.5 triangulation). ✔
- §2.1 (Reddit data source) — Phase 2. ✔
- §2.2 (FDA labels) — Task 5.3. ✔
- §2.3 (FAERS) — Tasks 5.1, 5.2. ✔
- §3 (subreddit selection + inventory) — Task 2.3. ✔
- §4 (extraction schema) — Tasks 1.1, 1.2, 1.3. ✔
- §5.1 (ingestion) — Tasks 2.4, 2.5, 2.6. ✔
- §5.2 (LLM extraction) — Phase 3. ✔
- §5.3 (MedDRA mapping) — Phase 4. ✔
- §5.4 (single-stage rationale) — embedded in Task 3.1 prompt. ✔
- §6 (analysis plan, all subsections) — Phase 6. ✔
- §7 (deliverables repo layout) — file structure at top of this plan. ✔
- §8 (validation + reliability) — Phase 7. ✔
- §9 (ethics, MedDRA license) — documented in Task 4.1 intro and §10 risks of the spec; IRB step deliberately left as a human-in-the-loop action outside the code plan.
- §10 (risks/limitations) — addressed in manuscript draft Task 8.4 Step 4.

**Placeholder scan:** searched for TBD / TODO / "similar to" / "write tests for the above" / "implement later" in the plan body. Two areas are deliberately condensed:

- Task 3.4 (Batch API) — described as design sketch with acceptance test contract, because it's an optional optimization path.
- Tasks 6.6, 6.7, 6.8, 7.2, 7.3, 7.4, 8.1–8.5 — written at task/step granularity with code shown for the analytically novel parts. The repetitive test-driven cycle is implied, not typed out for every step, to keep the plan readable.

These are called out in the "Natural plan split points" section so an implementer can split further if they want exhaustively bite-sized steps for Phases 6–8.

**Type consistency check:**
- `PostExtraction` fields used consistently across schema, postprocess, runner, cohort, analysis (verified by searching for field names).
- `RedditPost` / `RedditComment` dataclass field names used consistently between `reddit_client.py`, `inventory.py`, `fetch.py`, `normalize.py`.
- `MappingResult` shape consistent between `mapper.py` and `scripts/run_mapping.py`.
- `RorResult` fields used by `ror.py` and referenced in `compute_all_rors` signature.

No naming collisions or drift detected.
