"""Run the nested three-model extraction agreement experiment.

The experiment is intentionally nested:

1. DeepSeek V4 Flash, no reasoning, on the requested base sample.
2. DeepSeek V4 Flash, high reasoning, on a nested 10% sample.
3. GPT-5.6 Luna, low reasoning, on 10% of stage 2 (1% overall).

All calls use OpenRouter strict structured outputs and ZDR-only routing.
DeepSeek V4 Flash costs are treated as uncached; cache metrics are still logged
if a provider unexpectedly reports a hit. Results are checkpointed in SQLite
and exported as JSONL after each stage. The result directory contains verbatim
spans from Reddit, so the script refuses to write until the directory is both
gitignored and marked ``com.dropbox.ignored``.

Examples:

    # Inspect counts and create the deterministic sample manifest; no API calls.
    uv run python data_analysis/02_llm_extraction_experiment.py --dry-run

    # Run 1,000 -> 100 -> 10 calls. Increasing --n later only adds new records.
    uv run python data_analysis/02_llm_extraction_experiment.py --n 1000

    # Run or resume the complete experiment. CLI concurrency overrides .env.
    uv run python data_analysis/02_llm_extraction_experiment.py --concurrency 20

    # Review the exact prompt and all three responses for one submission.
    uv run python data_analysis/02_llm_extraction_experiment.py \
        --review-record-id t3_<record_id>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from botox_se.schema import (  # type: ignore[import-untyped]  # noqa: E402
    ExtractedFields,
    PostExtraction,
    RecordKind,
    SourceField,
    structured_output_schema,
)

SOURCE = ROOT / "filtered_dataset" / "submissions.jsonl"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "02_llm_extraction_results"
PROMPT_VERSION = "2.0.0-experiment"
PIPELINE_VERSION = "1.1-response-healing"
DEFAULT_SEED = 20260713
STAGE2_STRIDE = 10
STAGE3_STRIDE = 100
RESPONSE_PREVIEW_CHARS = 2_000

DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"
LUNA_MODEL = "openai/gpt-5.6-luna"

CLINICAL_FIELDS = tuple(ExtractedFields.model_fields)
SET_FIELDS = {"b_products_mentioned", "d_side_effect_pts"}


class Settings(BaseSettings):
    """Secrets and operational defaults loaded from the repository .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_concurrency: int = Field(default=10, ge=1, le=200)
    llm_request_timeout_seconds: float = Field(default=180, gt=0)
    llm_max_retries: int = Field(default=5, ge=1, le=20)
    llm_max_output_attempts: int = Field(default=2, ge=1, le=5)


@dataclass(frozen=True)
class StageConfig:
    key: str
    description: str
    model: str
    reasoning: dict[str, Any] | None
    max_tokens: int
    max_tokens_parameter: str
    session_id: str
    export_name: str


@dataclass
class CallResult:
    record_id: str
    extraction: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None
    response_preview: str | None = None
    failure_kind: str | None = None
    terminal: bool = False

    @property
    def succeeded(self) -> bool:
        return self.extraction is not None


@dataclass(frozen=True)
class CorpusIndex:
    eligible_ids: tuple[str, ...]
    source_records: int
    bonta_only_excluded: int
    id_sha256: str


def stage_configs(deepseek_effort: str, luna_effort: str) -> dict[str, StageConfig]:
    return {
        "stage1": StageConfig(
            key="stage1",
            description="DeepSeek V4 Flash / no reasoning / base sample",
            model=DEEPSEEK_MODEL,
            reasoning={"effort": "none", "exclude": True},
            max_tokens=1_500,
            max_tokens_parameter="max_tokens",
            session_id=f"botox-extraction-{PROMPT_VERSION}-deepseek-no-reasoning",
            export_name="stage1_deepseek_no_reasoning.jsonl",
        ),
        "stage2": StageConfig(
            key="stage2",
            description=f"DeepSeek V4 Flash / {deepseek_effort} reasoning / nested 10%",
            model=DEEPSEEK_MODEL,
            reasoning={"effort": deepseek_effort, "exclude": True},
            max_tokens=4_096,
            max_tokens_parameter="max_tokens",
            session_id=f"botox-extraction-{PROMPT_VERSION}-deepseek-{deepseek_effort}",
            export_name="stage2_deepseek_reasoning.jsonl",
        ),
        "stage3": StageConfig(
            key="stage3",
            description=f"GPT-5.6 Luna / {luna_effort} reasoning / nested 1%",
            model=LUNA_MODEL,
            reasoning={"effort": luna_effort, "exclude": True},
            max_tokens=2_048,
            max_tokens_parameter="max_completion_tokens",
            session_id=f"botox-extraction-{PROMPT_VERSION}-luna-{luna_effort}",
            export_name="stage3_gpt56_luna_reasoning.jsonl",
        ),
    }


def _schema_json() -> str:
    return json.dumps(structured_output_schema(), ensure_ascii=True, separators=(",", ":"))


SYSTEM_PROMPT = f"""You are extracting structured clinical information from one Reddit submission for a pharmacovigilance study of botulinum toxin type A (BoNT-A). Use only facts explicitly stated in the title and selftext. Do not use outside knowledge to fill missing facts.

Decision rules:
1. a_self_use is true only when the author explicitly says they personally received or currently receive a BoNT-A treatment. Questions about possible future treatment, hypothetical experiences, another person's treatment, celebrity discussion, provider advertising, and general advice are false unless the author also clearly reports their own treatment.
2. A side effect is an unwanted symptom, sign, functional change, or cosmetic outcome the author says occurred to them after treatment. A question such as "has anyone had ptosis?" is not a reported side effect. Treatment failure or short duration alone is not a side effect unless the author describes an unwanted outcome.
3. raw_span must be the shortest verbatim substring that names the author's side effect. Never paraphrase raw_span. meddra_pt is a concise candidate MedDRA Preferred Term; it will be validated later.
4. Extract all products and all distinct side effects explicitly reported. Do not treat a product merely mentioned by a non-user as personal exposure.
5. Classify indication from the author's reason for treatment, never from the subreddit. Use unknown and a null subtype when the reason is unclear.
6. e1 describes the overall submission. e2 describes the author's attitude toward their own BoNT-A treatment where self-use is true; otherwise it describes the expressed attitude toward BoNT-A in the text. Use neutral when no attitude is expressed.
7. Never infer dose, timing, provider, treatment count, medical attention, discontinuation, age, or sex. Return null unless explicitly disclosed.
8. If a_self_use is false: products=[], indication_category="unknown", indication_subtype=null, side_effects=[], side-effect sentiment=null, and every procedure/context/demographic field=null. e1 and e2 remain required.
9. If side_effects is empty, side-effect sentiment must be null. If indication_category is unknown, indication_subtype must be null. Medical and cosmetic subtypes must match their category.
10. Return only an object satisfying the strict JSON schema. Every property is required; use null or [] where specified and never add properties.

The exact machine-enforced schema is reproduced here to make the reusable prompt prefix self-contained:
{_schema_json()}
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "botox_post_extraction",
        "strict": True,
        "schema": structured_output_schema(),
    },
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def record_id(record: Mapping[str, Any]) -> str:
    candidate = str(record.get("name") or record.get("id") or "").strip()
    if not candidate:
        raise ValueError("submission has no name or id")
    return candidate if candidate.startswith("t3_") else f"t3_{candidate}"


def is_bonta_only(record: Mapping[str, Any]) -> bool:
    terms = {str(term).strip().casefold() for term in record.get("_matched_terms") or []}
    return terms == {"bonta"}


def ensure_private_results_dir(results_dir: Path) -> None:
    """Refuse to write identifiable/verbatim content without both guards."""
    results_dir = results_dir.resolve()
    analysis_dir = Path(__file__).resolve().parent
    if not results_dir.is_relative_to(analysis_dir):
        raise SystemExit(f"REFUSING: results must be below {analysis_dir}, got {results_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    probe = results_dir / "private_extraction.sqlite3"
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode
    if ignored != 0:
        raise SystemExit(f"REFUSING: {results_dir}/ is not gitignored.")

    if "Dropbox" in str(results_dir):
        subprocess.run(
            ["xattr", "-w", "com.dropbox.ignored", "1", str(results_dir)],
            check=True,
        )
        attr = subprocess.run(
            ["xattr", "-p", "com.dropbox.ignored", str(results_dir)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if attr != "1":
            raise SystemExit(f"REFUSING: failed to mark {results_dir}/ as Dropbox-ignored.")


def index_corpus(source: Path) -> CorpusIndex:
    ids: list[str] = []
    seen: set[str] = set()
    digest = hashlib.sha256()
    source_records = 0
    excluded = 0

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc="index corpus", unit="rec", unit_scale=True), start=1
        ):
            source_records += 1
            try:
                record = json.loads(line)
                rid = record_id(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid source record at line {line_number}: {exc}") from exc
            if is_bonta_only(record):
                excluded += 1
                continue
            if rid in seen:
                raise RuntimeError(f"duplicate eligible record id: {rid}")
            seen.add(rid)
            ids.append(rid)
            digest.update(rid.encode("ascii"))
            digest.update(b"\n")

    return CorpusIndex(tuple(ids), source_records, excluded, digest.hexdigest())


def seeded_order(ids: Sequence[str], seed: int) -> tuple[str, ...]:
    rng = random.Random(seed)
    ordered = list(ids)
    rng.shuffle(ordered)
    return tuple(ordered)


def progressive_targets(ordered_ids: Sequence[str], n: int) -> dict[str, set[str]]:
    """Return monotonic n, floor(n/10), floor(n/100) nested targets.

    The corpus is shuffled once. Prefixes make a larger ``--n`` retain every
    earlier record. The final item in each random block of ten or one hundred
    gives exact nested 10% and 1% samples whenever n is a multiple of 100.
    """
    if not 1 <= n <= len(ordered_ids):
        raise ValueError(f"n must be between 1 and {len(ordered_ids):,}, got {n:,}")
    stage1 = ordered_ids[:n]
    return {
        "stage1": set(stage1),
        "stage2": set(stage1[STAGE2_STRIDE - 1 :: STAGE2_STRIDE]),
        "stage3": set(stage1[STAGE3_STRIDE - 1 :: STAGE3_STRIDE]),
    }


def experiment_fingerprint(configs: Mapping[str, StageConfig]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(_schema_json().encode()).hexdigest(),
        "stages": {
            key: {
                "model": stage.model,
                "reasoning": stage.reasoning,
                "max_tokens": stage.max_tokens,
                "max_tokens_parameter": stage.max_tokens_parameter,
            }
            for key, stage in configs.items()
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_or_validate_manifest(
    results_dir: Path,
    corpus: CorpusIndex,
    ordered_ids: Sequence[str],
    seed: int,
    configs: Mapping[str, StageConfig],
) -> dict[str, Any]:
    path = results_dir / "experiment_manifest.json"
    order_digest = hashlib.sha256()
    for rid in ordered_ids:
        order_digest.update(rid.encode("ascii"))
        order_digest.update(b"\n")
    expected = {
        "manifest_version": 2,
        "created_at": None,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_records": corpus.source_records,
        "bonta_only_excluded": corpus.bonta_only_excluded,
        "eligible_records": len(corpus.eligible_ids),
        "eligible_id_sha256": corpus.id_sha256,
        "seed": seed,
        "sampling_algorithm": "seeded_shuffle_then_systematic_10_and_100",
        "sampling_order_sha256": order_digest.hexdigest(),
        "experiment_fingerprint": experiment_fingerprint(configs),
    }
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise SystemExit(f"REFUSING: {path} is not a JSON object.")
        existing: dict[str, Any] = loaded
        if existing.get("manifest_version") != expected["manifest_version"]:
            database = results_dir / "private_extraction.sqlite3"
            if database.exists():
                raise SystemExit(
                    f"REFUSING: cannot migrate {path} while {database} exists. "
                    "Use a new --results-dir to preserve existing paid results."
                )
            expected["created_at"] = utc_now()
            path.write_text(json.dumps(expected, indent=2) + "\n")
            return expected
        mismatches = [
            key
            for key, value in expected.items()
            if key != "created_at" and existing.get(key) != value
        ]
        if mismatches:
            database = results_dir / "private_extraction.sqlite3"
            if database.exists():
                raise SystemExit(
                    f"REFUSING: {path} does not match the current corpus/config at "
                    f"{mismatches[0]!r}. Use a different --results-dir to preserve existing "
                    "paid results."
                )
            expected["created_at"] = utc_now()
            path.write_text(json.dumps(expected, indent=2) + "\n")
            return expected
        return existing

    expected["created_at"] = utc_now()
    path.write_text(json.dumps(expected, indent=2) + "\n")
    return expected


def _metadata_was_content_filtered(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("finish_reason") == "content_filter":
        return True
    return any(
        isinstance(attempt, Mapping) and attempt.get("finish_reason") == "content_filter"
        for attempt in metadata.get("attempts") or []
    )


def _reclassify_content_filter_failures(connection: sqlite3.Connection, table: str) -> None:
    if table not in {"stage_failures", "stage_failure_history"}:
        raise ValueError(f"unexpected failure table: {table}")
    rows = connection.execute(
        f"SELECT rowid, metadata_json FROM {table} WHERE failure_kind != 'content_filter'"
    )
    for rowid, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(metadata, Mapping):
            continue
        if _metadata_was_content_filtered(metadata):
            connection.execute(
                f"""
                UPDATE {table}
                SET failure_kind = 'content_filter', terminal = 1,
                    error = 'ContentFilterError: provider blocked the response'
                WHERE rowid = ?
                """,
                (rowid,),
            )


def open_database(results_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(results_dir / "private_extraction.sqlite3")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_results (
            stage TEXT NOT NULL,
            record_id TEXT NOT NULL,
            extraction_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (stage, record_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_failures (
            stage TEXT NOT NULL,
            record_id TEXT NOT NULL,
            error TEXT NOT NULL,
            status_code INTEGER,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            response_preview TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 1,
            failure_kind TEXT NOT NULL DEFAULT 'retryable',
            terminal INTEGER NOT NULL DEFAULT 0,
            attempted_at TEXT NOT NULL,
            PRIMARY KEY (stage, record_id)
        )
        """
    )
    failure_columns = {row[1] for row in connection.execute("PRAGMA table_info(stage_failures)")}
    if "metadata_json" not in failure_columns:
        connection.execute(
            "ALTER TABLE stage_failures ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "response_preview" not in failure_columns:
        connection.execute("ALTER TABLE stage_failures ADD COLUMN response_preview TEXT")
    if "attempt_count" not in failure_columns:
        connection.execute(
            "ALTER TABLE stage_failures ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 1"
        )
    if "failure_kind" not in failure_columns:
        connection.execute(
            "ALTER TABLE stage_failures ADD COLUMN failure_kind TEXT NOT NULL DEFAULT 'retryable'"
        )
    if "terminal" not in failure_columns:
        connection.execute(
            "ALTER TABLE stage_failures ADD COLUMN terminal INTEGER NOT NULL DEFAULT 0"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_failure_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL,
            record_id TEXT NOT NULL,
            error TEXT NOT NULL,
            status_code INTEGER,
            metadata_json TEXT NOT NULL,
            response_preview TEXT,
            attempt_count INTEGER NOT NULL,
            failure_kind TEXT NOT NULL DEFAULT 'retryable',
            terminal INTEGER NOT NULL DEFAULT 0,
            attempted_at TEXT NOT NULL,
            UNIQUE (stage, record_id, attempted_at)
        )
        """
    )
    history_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(stage_failure_history)")
    }
    if "failure_kind" not in history_columns:
        connection.execute(
            "ALTER TABLE stage_failure_history ADD COLUMN failure_kind "
            "TEXT NOT NULL DEFAULT 'retryable'"
        )
    if "terminal" not in history_columns:
        connection.execute(
            "ALTER TABLE stage_failure_history ADD COLUMN terminal INTEGER NOT NULL DEFAULT 0"
        )
    connection.execute(
        """
        INSERT OR IGNORE INTO stage_failure_history
            (stage, record_id, error, status_code, metadata_json,
             response_preview, attempt_count, failure_kind, terminal, attempted_at)
        SELECT stage, record_id, error, status_code, metadata_json,
               response_preview, attempt_count, failure_kind, terminal, attempted_at
        FROM stage_failures
        """
    )
    _reclassify_content_filter_failures(connection, "stage_failures")
    _reclassify_content_filter_failures(connection, "stage_failure_history")
    connection.commit()
    return connection


def completed_ids(connection: sqlite3.Connection, stage: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT record_id FROM stage_results WHERE stage = ?", (stage,)
        )
    }


def terminal_failure_ids(connection: sqlite3.Connection, stage: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT record_id FROM stage_failures WHERE stage = ? AND terminal = 1",
            (stage,),
        )
    }


def save_call_result(
    connection: sqlite3.Connection, stage: StageConfig, result: CallResult
) -> None:
    saved_at = utc_now()
    if result.succeeded:
        connection.execute(
            """
            INSERT OR IGNORE INTO stage_results
                (stage, record_id, extraction_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                stage.key,
                result.record_id,
                json.dumps(result.extraction, ensure_ascii=False, separators=(",", ":")),
                json.dumps(result.metadata or {}, ensure_ascii=False, separators=(",", ":")),
                saved_at,
            ),
        )
        connection.execute(
            "DELETE FROM stage_failures WHERE stage = ? AND record_id = ?",
            (stage.key, result.record_id),
        )
    else:
        metadata_json = json.dumps(result.metadata or {}, ensure_ascii=False, separators=(",", ":"))
        attempt_count = int((result.metadata or {}).get("attempt_count") or 1)
        failure_kind = result.failure_kind or "retryable"
        terminal = int(result.terminal)
        connection.execute(
            """
            INSERT INTO stage_failures
                (stage, record_id, error, status_code, metadata_json,
                 response_preview, attempt_count, failure_kind, terminal, attempted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stage, record_id) DO UPDATE SET
                error=excluded.error,
                status_code=excluded.status_code,
                metadata_json=excluded.metadata_json,
                response_preview=excluded.response_preview,
                attempt_count=excluded.attempt_count,
                failure_kind=excluded.failure_kind,
                terminal=excluded.terminal,
                attempted_at=excluded.attempted_at
            """,
            (
                stage.key,
                result.record_id,
                (result.error or "unknown failure")[:4_000],
                result.status_code,
                metadata_json,
                result.response_preview,
                attempt_count,
                failure_kind,
                terminal,
                saved_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO stage_failure_history
                (stage, record_id, error, status_code, metadata_json,
                 response_preview, attempt_count, failure_kind, terminal, attempted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stage.key,
                result.record_id,
                (result.error or "unknown failure")[:4_000],
                result.status_code,
                metadata_json,
                result.response_preview,
                attempt_count,
                failure_kind,
                terminal,
                saved_at,
            ),
        )


def iter_target_records(source: Path, target_ids: set[str]) -> Iterator[tuple[str, dict[str, str]]]:
    found: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                rid = record_id(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid source record at line {line_number}: {exc}") from exc
            if rid not in target_ids:
                continue
            found.add(rid)
            yield (
                rid,
                {
                    "title": str(record.get("title") or ""),
                    "selftext": str(record.get("selftext") or ""),
                },
            )
    missing = target_ids - found
    if missing:
        example = sorted(missing)[:3]
        raise RuntimeError(f"{len(missing)} target records missing from source, e.g. {example}")


def load_record_text(source: Path, requested_id: str) -> tuple[str, dict[str, str]]:
    """Load only the title and selftext sent to the model for one submission."""
    rid = requested_id.strip()
    if not rid:
        raise ValueError("review record id cannot be empty")
    if not rid.startswith("t3_"):
        rid = f"t3_{rid}"
    records = iter_target_records(source, {rid})
    return next(records)


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(float(value), 120.0)
            except ValueError:
                pass
    return float(min(2 ** (attempt - 1) + random.random(), 60.0))


def raw_span_metrics(
    extracted: ExtractedFields, text_fields: Mapping[str, str]
) -> dict[str, int | float | bool | None]:
    """Measure verbatim compliance without altering or rejecting model output."""
    source_texts = tuple(text_fields.values())
    total = len(extracted.d_side_effect_pts)
    verbatim = sum(
        bool(side_effect.raw_span)
        and any(side_effect.raw_span in source for source in source_texts)
        for side_effect in extracted.d_side_effect_pts
    )
    return {
        "total_raw_spans": total,
        "verbatim_raw_spans": verbatim,
        "all_raw_spans_verbatim": verbatim == total,
        "raw_span_verbatim_rate": verbatim / total if total else None,
    }


def response_healing_status(body: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize OpenRouter's opt-in response-healing pipeline telemetry."""
    raw_router_metadata = body.get("openrouter_metadata")
    router_metadata = raw_router_metadata if isinstance(raw_router_metadata, Mapping) else {}
    telemetry_available = isinstance(raw_router_metadata, Mapping)
    pipeline = router_metadata.get("pipeline") or []
    if not isinstance(pipeline, list):
        pipeline = []
    stages = [
        stage
        for stage in pipeline
        if isinstance(stage, dict)
        and (stage.get("type") == "response_healing" or stage.get("name") == "response-healing")
    ]
    improved_values = [
        stage.get("data", {}).get("improved")
        for stage in stages
        if isinstance(stage.get("data"), dict)
        and isinstance(stage.get("data", {}).get("improved"), bool)
    ]
    return {
        "requested": True,
        "telemetry_available": telemetry_available,
        "used": bool(stages) if telemetry_available else None,
        "improved": improved_values[-1] if improved_values else None,
        "pipeline_stages": stages,
    }


def _new_accounting() -> dict[str, int | float]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "responses_with_usage": 0,
        "cost_reported_attempts": 0,
    }


def _accumulate_response_usage(
    accounting: dict[str, int | float], body: Mapping[str, Any]
) -> dict[str, int | float]:
    usage = body.get("usage") or {}
    measured = response_usage({"usage": usage})
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
        accounting[key] = int(accounting[key]) + int(measured[key])
    accounting["cost_usd"] = float(accounting["cost_usd"]) + float(measured["cost_usd"])
    if usage:
        accounting["responses_with_usage"] = int(accounting["responses_with_usage"]) + 1
    if usage.get("cost") is not None:
        accounting["cost_reported_attempts"] = int(accounting["cost_reported_attempts"]) + 1
    return measured


def _bounded_preview(content: str) -> str:
    return content[:RESPONSE_PREVIEW_CHARS]


async def call_openrouter(
    client: httpx.AsyncClient,
    stage: StageConfig,
    rid: str,
    text_fields: Mapping[str, str],
    max_retries: int,
    max_output_attempts: int,
) -> CallResult:
    payload = build_request_payload(stage, text_fields)

    last_error = "request failed"
    last_status: int | None = None
    last_body: Mapping[str, Any] = {}
    last_healing = response_healing_status(last_body)
    last_preview: str | None = None
    last_cache_status: str | None = None
    last_finish_reason: str | None = None
    failure_kind: str | None = None
    terminal = False
    accounting = _new_accounting()
    attempts: list[dict[str, Any]] = []
    transient_failures = 0
    output_failures = 0
    request_number = 0

    while True:
        request_number += 1
        attempt: dict[str, Any] = {"attempt": request_number}
        response: httpx.Response
        try:
            response = await client.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            transient_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            attempt["error"] = last_error
            attempts.append(attempt)
            if transient_failures < max_retries:
                await asyncio.sleep(_retry_delay(None, transient_failures))
                continue
            break
        except Exception as exc:
            last_error = f"unexpected {type(exc).__name__}: {exc}"
            attempt["error"] = last_error
            attempts.append(attempt)
            break

        last_status = response.status_code
        last_cache_status = response.headers.get("x-openrouter-cache-status")
        attempt["status_code"] = last_status
        try:
            decoded = response.json()
        except json.JSONDecodeError as exc:
            transient_failures += 1
            last_preview = _bounded_preview(response.text)
            last_error = f"invalid OpenRouter response JSON: {exc}"
            attempt.update(error=last_error, response_preview=last_preview)
            attempts.append(attempt)
            if transient_failures < max_retries:
                await asyncio.sleep(_retry_delay(response, transient_failures))
                continue
            break

        if not isinstance(decoded, dict):
            transient_failures += 1
            last_preview = _bounded_preview(response.text)
            last_error = "OpenRouter response is not a JSON object"
            attempt.update(error=last_error, response_preview=last_preview)
            attempts.append(attempt)
            if transient_failures < max_retries:
                await asyncio.sleep(_retry_delay(response, transient_failures))
                continue
            break

        body: Mapping[str, Any] = decoded
        last_body = body
        last_healing = response_healing_status(body)
        measured = _accumulate_response_usage(accounting, body)
        choices = body.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        last_finish_reason = choice.get("finish_reason")
        attempt.update(
            response_id=body.get("id"),
            provider=body.get("provider"),
            finish_reason=choice.get("finish_reason"),
            response_healing=last_healing,
            measured_usage=measured,
        )

        if response.status_code == 429 or response.status_code >= 500:
            transient_failures += 1
            last_error = f"HTTP {response.status_code}: {response.text[:2_000]}"
            attempt["error"] = last_error
            attempts.append(attempt)
            if transient_failures < max_retries:
                await asyncio.sleep(_retry_delay(response, transient_failures))
                continue
            break
        if response.status_code >= 400:
            last_error = f"HTTP {response.status_code}: {response.text[:2_000]}"
            attempt["error"] = last_error
            attempts.append(attempt)
            break

        if last_finish_reason == "content_filter":
            message = choice.get("message") or {}
            filtered_content = message.get("content") if isinstance(message, Mapping) else None
            refusal = message.get("refusal") if isinstance(message, Mapping) else None
            last_preview = _bounded_preview(str(filtered_content or refusal or response.text))
            last_error = "ContentFilterError: provider blocked the response"
            failure_kind = "content_filter"
            terminal = True
            attempt.update(error=last_error, response_preview=last_preview)
            attempts.append(attempt)
            break

        content: str | None = None
        try:
            choices = body.get("choices") or []
            if not choices:
                raise ValueError("response contains no choices")
            message = choices[0].get("message") or {}
            if message.get("refusal"):
                raise ValueError(f"model refusal: {message['refusal']}")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("response contains no JSON content")

            extracted = ExtractedFields.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            output_failures += 1
            last_preview = _bounded_preview(content or response.text)
            last_error = f"{type(exc).__name__}: {exc}"
            attempt.update(error=last_error, response_preview=last_preview)
            attempts.append(attempt)
            if output_failures < max_output_attempts:
                await asyncio.sleep(_retry_delay(response, output_failures))
                continue
            break

        attempts.append(attempt)
        fingerprint = str(body.get("system_fingerprint") or body.get("model") or stage.model)
        post = PostExtraction(
            **extracted.model_dump(),
            record_id=rid,
            record_kind=RecordKind.SUBMISSION,
            fields_seen=[SourceField.TITLE, SourceField.SELFTEXT],
            prompt_version=PROMPT_VERSION,
            model_fingerprint=fingerprint,
        )
        metadata = {
            "pipeline_version": PIPELINE_VERSION,
            "stage": stage.key,
            "model_requested": stage.model,
            "model_returned": body.get("model"),
            "provider_returned": body.get("provider"),
            "response_id": body.get("id"),
            "reasoning": stage.reasoning,
            "finish_reason": choices[0].get("finish_reason"),
            "response_cache_status": last_cache_status,
            "response_healing": last_healing,
            "raw_span_metrics": raw_span_metrics(extracted, text_fields),
            "usage": body.get("usage") or {},
            "accounting": accounting,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        return CallResult(
            record_id=rid,
            extraction=post.model_dump(mode="json"),
            metadata=metadata,
        )

    failure_metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": stage.key,
        "model_requested": stage.model,
        "model_returned": last_body.get("model"),
        "provider_returned": last_body.get("provider"),
        "response_id": last_body.get("id"),
        "reasoning": stage.reasoning,
        "finish_reason": last_finish_reason,
        "response_cache_status": last_cache_status,
        "response_healing": last_healing,
        "usage": last_body.get("usage") or {},
        "accounting": accounting,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    return CallResult(
        record_id=rid,
        metadata=failure_metadata,
        error=last_error,
        status_code=last_status,
        response_preview=last_preview,
        failure_kind=failure_kind,
        terminal=terminal,
    )


def build_request_payload(stage: StageConfig, text_fields: Mapping[str, str]) -> dict[str, Any]:
    """Build the exact OpenRouter request, with sampling defaults intentionally omitted."""
    payload: dict[str, Any] = {
        "model": stage.model,
        "session_id": stage.session_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(text_fields, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "response_format": RESPONSE_FORMAT,
        "plugins": [{"id": "response-healing"}],
        "provider": {
            "zdr": True,
            "require_parameters": True,
            "sort": "price",
        },
    }
    payload[stage.max_tokens_parameter] = stage.max_tokens
    if stage.reasoning is not None:
        payload["reasoning"] = stage.reasoning
    return payload


def response_usage(metadata: Mapping[str, Any] | None) -> dict[str, int | float]:
    accounting = (metadata or {}).get("accounting") or {}
    if accounting:
        return {
            "prompt_tokens": int(accounting.get("prompt_tokens") or 0),
            "completion_tokens": int(accounting.get("completion_tokens") or 0),
            "cached_tokens": int(accounting.get("cached_tokens") or 0),
            "reasoning_tokens": int(accounting.get("reasoning_tokens") or 0),
            "cost_usd": float(accounting.get("cost_usd") or 0.0),
        }
    usage = (metadata or {}).get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}

    def integer(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    try:
        cost = float(usage["cost"]) if usage.get("cost") is not None else 0.0
    except (TypeError, ValueError):
        cost = 0.0
    return {
        "prompt_tokens": integer(usage.get("prompt_tokens")),
        "completion_tokens": integer(usage.get("completion_tokens")),
        "cached_tokens": integer(prompt_details.get("cached_tokens")),
        "reasoning_tokens": integer(
            completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens")
        ),
        "cost_usd": cost,
    }


def openrouter_headers(settings: Settings) -> dict[str, str]:
    key = settings.openrouter_api_key
    if key is None or not key.get_secret_value().strip():
        raise SystemExit("OPENROUTER_API_KEY is empty; add it to .env before making calls.")
    return {
        "Authorization": f"Bearer {key.get_secret_value()}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
        "HTTP-Referer": "https://github.com/Haffi112/botox-side-effects-analysis",
        "X-Title": "Botox side-effects extraction experiment",
    }


async def run_prompt_review(
    settings: Settings,
    configs: Mapping[str, StageConfig],
    results_dir: Path,
    requested_id: str,
    selected_stages: Sequence[str] | None = None,
) -> Path:
    """Run all model variants once and save their exact requests and parsed outputs."""
    rid, text_fields = load_record_text(SOURCE, requested_id)
    artifact: dict[str, Any] = {
        "created_at": utc_now(),
        "record_id": rid,
        "note": (
            "Each request_payload is exactly what was sent to OpenRouter. Parameters absent "
            "from it, including temperature, top_p, and seed, use model/provider defaults."
        ),
        "runs": {},
    }
    timeout = httpx.Timeout(settings.llm_request_timeout_seconds)
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=5)
    async with httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        headers=openrouter_headers(settings),
        timeout=timeout,
        limits=limits,
    ) as client:
        stage_keys = tuple(selected_stages or configs.keys())
        for key in stage_keys:
            stage = configs[key]
            print(f"review {key}: {stage.description}")
            result = await call_openrouter(
                client,
                stage,
                rid,
                text_fields,
                settings.llm_max_retries,
                settings.llm_max_output_attempts,
            )
            usage = response_usage(result.metadata)
            artifact["runs"][key] = {
                "request_payload": build_request_payload(stage, text_fields),
                "extraction": result.extraction,
                "metadata": result.metadata,
                "measured_usage": usage,
                "error": result.error,
                "status_code": result.status_code,
            }
            if result.succeeded:
                print(
                    f"  success: ${float(usage['cost_usd']):.6f}; "
                    f"prompt={int(usage['prompt_tokens']):,}; "
                    f"cached={int(usage['cached_tokens']):,}; "
                    f"completion={int(usage['completion_tokens']):,}; "
                    f"reasoning={int(usage['reasoning_tokens']):,}"
                )
            else:
                print(f"  failed: {result.error}")

    suffix = "" if len(stage_keys) == len(configs) else f"_{'_'.join(stage_keys)}"
    path = results_dir / f"prompt_review_{rid}{suffix}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prompt review written to {path}")
    return path


async def run_stage(
    connection: sqlite3.Connection,
    client: httpx.AsyncClient,
    stage: StageConfig,
    target_ids: set[str],
    concurrency: int,
    max_retries: int,
    max_output_attempts: int,
) -> tuple[int, int]:
    done_before = completed_ids(connection, stage.key)
    terminal_before = terminal_failure_ids(connection, stage.key)
    pending_ids = target_ids - done_before - terminal_before
    requested = len(pending_ids)
    if requested == 0:
        print(
            f"{stage.key}: already handled "
            f"({len(done_before & target_ids):,} successes, "
            f"{len(terminal_before & target_ids):,} terminal failures)"
        )
        return 0, 0

    tasks: set[asyncio.Task[CallResult]] = set()
    succeeded = 0
    failed = 0
    uncommitted = 0
    run_cost = 0.0
    run_cached_tokens = 0
    run_reasoning_tokens = 0
    run_healing_used = 0

    async def persist_finished(finished: Iterable[asyncio.Task[CallResult]]) -> None:
        nonlocal succeeded, failed, uncommitted
        nonlocal run_cost, run_cached_tokens, run_reasoning_tokens, run_healing_used
        for task in finished:
            result = await task
            save_call_result(connection, stage, result)
            succeeded += int(result.succeeded)
            failed += int(not result.succeeded)
            usage = response_usage(result.metadata)
            run_cost += float(usage["cost_usd"])
            run_cached_tokens += int(usage["cached_tokens"])
            run_reasoning_tokens += int(usage["reasoning_tokens"])
            healing = (result.metadata or {}).get("response_healing") or {}
            run_healing_used += int(bool(healing.get("used")))
            uncommitted += 1
            progress.update(1)
            progress.set_postfix(
                cost=f"${run_cost:.4f}",
                cached=f"{run_cached_tokens / 1_000_000:.2f}M",
                reasoning=f"{run_reasoning_tokens / 1_000_000:.2f}M",
                healed=run_healing_used,
                refresh=False,
            )
            if uncommitted >= 25:
                connection.commit()
                uncommitted = 0

    records = iter_target_records(SOURCE, pending_ids)
    with tqdm(total=requested, desc=stage.key, unit="call", unit_scale=True) as progress:
        for rid, text_fields in records:
            tasks.add(
                asyncio.create_task(
                    call_openrouter(
                        client,
                        stage,
                        rid,
                        text_fields,
                        max_retries,
                        max_output_attempts,
                    )
                )
            )
            if len(tasks) >= concurrency:
                finished, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                await persist_finished(finished)
        while tasks:
            finished, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            await persist_finished(finished)
    connection.commit()
    print(
        f"{stage.key} new-call usage: ${run_cost:,.4f}; "
        f"cached={run_cached_tokens:,}; reasoning={run_reasoning_tokens:,} tokens; "
        f"response-healed={run_healing_used:,}"
    )
    return succeeded, failed


def export_stage(connection: sqlite3.Connection, results_dir: Path, stage: StageConfig) -> int:
    path = results_dir / stage.export_name
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        rows = connection.execute(
            """
            SELECT record_id, extraction_json, metadata_json
            FROM stage_results WHERE stage = ? ORDER BY record_id
            """,
            (stage.key,),
        )
        for rid, extraction_json, metadata_json in rows:
            row = {
                "record_id": rid,
                "extraction": json.loads(extraction_json),
                "run": json.loads(metadata_json),
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def export_stage_failures(
    connection: sqlite3.Connection, results_dir: Path, stage: StageConfig
) -> int:
    path = results_dir / f"{Path(stage.export_name).stem}_failures.jsonl"
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        rows = connection.execute(
            """
            SELECT record_id, error, status_code, metadata_json, response_preview,
                   attempt_count, failure_kind, terminal, attempted_at
            FROM stage_failures WHERE stage = ? ORDER BY record_id
            """,
            (stage.key,),
        )
        for row in rows:
            (
                rid,
                error,
                status_code,
                metadata_json,
                response_preview,
                attempt_count,
                failure_kind,
                terminal,
                attempted_at,
            ) = row
            exported = {
                "record_id": rid,
                "failure_kind": failure_kind,
                "terminal": bool(terminal),
                "error": error,
                "status_code": status_code,
                "response_preview": response_preview,
                "attempt_count": attempt_count,
                "attempted_at": attempted_at,
                "run": json.loads(metadata_json),
            }
            handle.write(json.dumps(exported, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def add_usage_to_summary(totals: Counter[str], metadata: Mapping[str, Any]) -> tuple[float, bool]:
    measured = response_usage(metadata)
    for token_key in (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
    ):
        totals[token_key] += int(measured[token_key])
    accounting = metadata.get("accounting") or {}
    usage = metadata.get("usage") or {}
    if accounting:
        reported = int(accounting.get("cost_reported_attempts") or 0)
        totals["responses_with_usage"] += int(accounting.get("responses_with_usage") or 0)
    else:
        reported = int(usage.get("cost") is not None)
        totals["responses_with_usage"] += int(bool(usage))
    attempts = metadata.get("attempts") or []
    if not attempts and metadata.get("response_healing"):
        attempts = [{"response_healing": metadata["response_healing"]}]
    for attempt in attempts:
        healing = attempt.get("response_healing") or {}
        totals["response_healing_requested_attempts"] += int(bool(healing.get("requested")))
        telemetry_available = healing.get("telemetry_available") is True
        totals["response_healing_telemetry_available_attempts"] += int(telemetry_available)
        totals["response_healing_telemetry_unavailable_attempts"] += int(not telemetry_available)
        totals["response_healing_used_attempts"] += int(bool(healing.get("used")))
        totals["response_healing_not_used_attempts"] += int(
            telemetry_available and healing.get("used") is False
        )
        totals["response_healing_improved_attempts"] += int(healing.get("improved") is True)
    return float(measured["cost_usd"]), bool(reported)


def write_run_summary(
    connection: sqlite3.Connection,
    results_dir: Path,
    configs: Mapping[str, StageConfig],
    target_counts: Mapping[str, int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "response_healing": {
            "enabled": True,
            "telemetry_note": (
                "used is true when OpenRouter reports a response-healing pipeline stage, "
                "false when router metadata is present without that stage, and null when "
                "router metadata is unavailable (including cache replays)."
            ),
        },
        "cost_note": (
            "OpenRouter-reported inference cost; excludes the credit-purchase fee and any tax."
        ),
        "stages": {},
    }
    experiment_totals: Counter[str] = Counter()
    experiment_cost = 0.0
    cost_observed = False
    for key, stage in configs.items():
        totals: Counter[str] = Counter()
        stage_cost = 0.0
        stage_cost_observed = False
        raw_spans = 0
        verbatim_raw_spans = 0
        rows = connection.execute("SELECT metadata_json FROM stage_results WHERE stage = ?", (key,))
        for (metadata_json,) in rows:
            metadata = json.loads(metadata_json)
            cost, reported = add_usage_to_summary(totals, metadata)
            stage_cost += cost
            stage_cost_observed = stage_cost_observed or reported
            span_metrics = metadata.get("raw_span_metrics") or {}
            raw_spans += int(span_metrics.get("total_raw_spans") or 0)
            verbatim_raw_spans += int(span_metrics.get("verbatim_raw_spans") or 0)
        failure_metadata_rows = connection.execute(
            "SELECT metadata_json FROM stage_failure_history WHERE stage = ?", (key,)
        )
        for (metadata_json,) in failure_metadata_rows:
            cost, reported = add_usage_to_summary(totals, json.loads(metadata_json))
            stage_cost += cost
            stage_cost_observed = stage_cost_observed or reported
        completed = connection.execute(
            "SELECT COUNT(*) FROM stage_results WHERE stage = ?", (key,)
        ).fetchone()[0]
        failures = connection.execute(
            "SELECT COUNT(*) FROM stage_failures WHERE stage = ?", (key,)
        ).fetchone()[0]
        terminal_failures = connection.execute(
            "SELECT COUNT(*) FROM stage_failures WHERE stage = ? AND terminal = 1",
            (key,),
        ).fetchone()[0]
        failure_runs = connection.execute(
            "SELECT COUNT(*) FROM stage_failure_history WHERE stage = ?", (key,)
        ).fetchone()[0]
        summary["stages"][key] = {
            "description": stage.description,
            "model": stage.model,
            "reasoning": stage.reasoning,
            "target_records": target_counts[key],
            "completed_records": completed,
            "latest_failed_records": failures,
            "terminal_failed_records": terminal_failures,
            "retryable_failed_records": failures - terminal_failures,
            "failed_call_runs": failure_runs,
            "total_raw_spans": raw_spans,
            "verbatim_raw_spans": verbatim_raw_spans,
            "raw_span_verbatim_rate": (verbatim_raw_spans / raw_spans if raw_spans else None),
            **dict(totals),
            "cache_hit_rate": (
                totals["cached_tokens"] / totals["prompt_tokens"]
                if totals["prompt_tokens"]
                else None
            ),
            "reported_cost_usd": stage_cost if stage_cost_observed else None,
        }
        experiment_totals.update(totals)
        experiment_cost += stage_cost
        cost_observed = cost_observed or stage_cost_observed
    summary["totals"] = {
        **dict(experiment_totals),
        "cache_hit_rate": (
            experiment_totals["cached_tokens"] / experiment_totals["prompt_tokens"]
            if experiment_totals["prompt_tokens"]
            else None
        ),
        "reported_cost_usd": experiment_cost if cost_observed else None,
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if cost_observed:
        print(f"Cumulative OpenRouter inference cost: ${experiment_cost:,.4f}")
    return summary


def _normalise_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return value


def normalise_field(field: str, value: Any) -> Any:
    if field == "b_products_mentioned":
        return tuple(sorted(str(item).casefold() for item in value or []))
    if field == "d_side_effect_pts":
        return tuple(
            sorted(
                {
                    " ".join(str(item.get("meddra_pt") or "").casefold().split())
                    for item in value or []
                    if item.get("meddra_pt")
                }
            )
        )
    return _normalise_scalar(value)


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        (left_counts[key] / n) * (right_counts[key] / n)
        for key in left_counts.keys() | right_counts.keys()
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return float((observed - expected) / (1.0 - expected))


def compare_extraction_pairs(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(pairs), "fields": {}}
    if not pairs:
        return result

    full_matches = 0
    for field in CLINICAL_FIELDS:
        left = [normalise_field(field, pair[0].get(field)) for pair in pairs]
        right = [normalise_field(field, pair[1].get(field)) for pair in pairs]
        exact = sum(a == b for a, b in zip(left, right, strict=True)) / len(pairs)
        metrics: dict[str, Any] = {"exact_agreement": exact}
        if field in SET_FIELDS:
            metrics["mean_jaccard"] = sum(
                jaccard(a, b) for a, b in zip(left, right, strict=True)
            ) / len(pairs)
        else:
            metrics["cohen_kappa"] = cohen_kappa(left, right)
        result["fields"][field] = metrics

    side_effect_left = [bool(pair[0].get("d_side_effect_pts")) for pair in pairs]
    side_effect_right = [bool(pair[1].get("d_side_effect_pts")) for pair in pairs]
    result["side_effect_presence"] = {
        "exact_agreement": sum(
            a == b for a, b in zip(side_effect_left, side_effect_right, strict=True)
        )
        / len(pairs),
        "cohen_kappa": cohen_kappa(side_effect_left, side_effect_right),
    }

    for left_extraction, right_extraction in pairs:
        left_view = tuple(
            normalise_field(field, left_extraction.get(field)) for field in CLINICAL_FIELDS
        )
        right_view = tuple(
            normalise_field(field, right_extraction.get(field)) for field in CLINICAL_FIELDS
        )
        full_matches += int(left_view == right_view)
    result["full_record_exact_agreement"] = full_matches / len(pairs)
    result["note"] = (
        "Side-effect comparisons use normalized candidate MedDRA PT sets; verbatim raw_span "
        "text and provenance fields are excluded."
    )
    return result


def load_pairs(
    connection: sqlite3.Connection, left_stage: str, right_stage: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = connection.execute(
        """
        SELECT left_result.extraction_json, right_result.extraction_json
        FROM stage_results AS left_result
        INNER JOIN stage_results AS right_result
          ON left_result.record_id = right_result.record_id
        WHERE left_result.stage = ? AND right_result.stage = ?
        ORDER BY left_result.record_id
        """,
        (left_stage, right_stage),
    )
    return [(json.loads(left), json.loads(right)) for left, right in rows]


def write_agreement_summary(connection: sqlite3.Connection, results_dir: Path) -> dict[str, Any]:
    comparisons = {
        "stage1_vs_stage2": ("stage1", "stage2"),
        "stage2_vs_stage3": ("stage2", "stage3"),
        "stage1_vs_stage3": ("stage1", "stage3"),
    }
    summary = {
        "generated_at": utc_now(),
        "comparisons": {
            name: compare_extraction_pairs(load_pairs(connection, *stages))
            for name, stages in comparisons.items()
        },
    }
    (results_dir / "agreement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "stage1", "stage2", "stage3"), default="all")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Cumulative base-sample size; defaults to every eligible submission",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--review-record-id",
        default=None,
        help="Run all three configurations for one submission without changing experiment state",
    )
    parser.add_argument("--deepseek-reasoning-effort", choices=("high", "xhigh"), default="high")
    parser.add_argument(
        "--luna-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="low",
    )
    args = parser.parse_args()
    if args.concurrency is not None and not 1 <= args.concurrency <= 200:
        parser.error("--concurrency must be between 1 and 200")
    if args.n is not None and args.n < 1:
        parser.error("--n must be positive")
    return args


async def async_main(
    args: argparse.Namespace,
    settings: Settings,
    configs: Mapping[str, StageConfig],
    targets: Mapping[str, set[str]],
    connection: sqlite3.Connection,
) -> None:
    concurrency = args.concurrency or settings.llm_concurrency
    timeout = httpx.Timeout(settings.llm_request_timeout_seconds)
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 20),
        max_keepalive_connections=max(concurrency, 10),
    )
    selected = list(configs) if args.stage == "all" else [args.stage]

    async with httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        headers=openrouter_headers(settings),
        timeout=timeout,
        limits=limits,
    ) as client:
        for key_name in selected:
            stage = configs[key_name]
            print(f"\n{stage.description}; concurrency={concurrency}")
            succeeded, failed = await run_stage(
                connection,
                client,
                stage,
                targets[key_name],
                concurrency,
                settings.llm_max_retries,
                settings.llm_max_output_attempts,
            )
            exported = export_stage(connection, args.results_dir, stage)
            failure_exports = export_stage_failures(connection, args.results_dir, stage)
            target_successes = len(completed_ids(connection, key_name) & targets[key_name])
            target_terminal_failures = len(
                terminal_failure_ids(connection, key_name) & targets[key_name]
            )
            target_handled = target_successes + target_terminal_failures
            print(
                f"{key_name}: {succeeded:,} new successes, {failed:,} failures; "
                f"{target_successes:,} successes + "
                f"{target_terminal_failures:,} terminal failures = "
                f"{target_handled:,}/{len(targets[key_name]):,} handled "
                f"({exported:,} results and {failure_exports:,} failures exported)"
            )
            write_run_summary(
                connection,
                args.results_dir,
                configs,
                {stage_key: len(target) for stage_key, target in targets.items()},
            )
            write_agreement_summary(connection, args.results_dir)
            if target_handled < len(targets[key_name]):
                unresolved = len(targets[key_name]) - target_handled
                print(
                    f"{key_name}: {unresolved:,} retryable failures preserved; "
                    "continuing to the next stage. Rerun later to retry them."
                )


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    if not SOURCE.exists():
        raise SystemExit(f"source corpus not found: {SOURCE}")
    ensure_private_results_dir(args.results_dir)

    settings = Settings()
    configs = stage_configs(args.deepseek_reasoning_effort, args.luna_reasoning_effort)
    if args.review_record_id:
        selected = None if args.stage == "all" else (args.stage,)
        asyncio.run(
            run_prompt_review(
                settings,
                configs,
                args.results_dir,
                args.review_record_id,
                selected,
            )
        )
        return

    corpus = index_corpus(SOURCE)
    requested_n = args.n or len(corpus.eligible_ids)
    if requested_n > len(corpus.eligible_ids):
        raise SystemExit(f"--n cannot exceed the {len(corpus.eligible_ids):,} eligible submissions")
    ordered_ids = seeded_order(corpus.eligible_ids, args.seed)
    targets = progressive_targets(ordered_ids, requested_n)
    build_or_validate_manifest(args.results_dir, corpus, ordered_ids, args.seed, configs)
    print(
        f"eligible={len(corpus.eligible_ids):,}; bonta-only excluded={corpus.bonta_only_excluded:,}; "
        f"targets={len(targets['stage1']):,} -> {len(targets['stage2']):,} -> "
        f"{len(targets['stage3']):,}"
    )

    if args.dry_run:
        print(f"dry run: no API calls; results directory is {args.results_dir}")
        return

    connection = open_database(args.results_dir)
    try:
        asyncio.run(async_main(args, settings, configs, targets, connection))
    finally:
        connection.close()


if __name__ == "__main__":
    started = time.monotonic()
    try:
        main()
    finally:
        print(f"elapsed={time.monotonic() - started:,.1f}s")
