"""Full-corpus GPT-5.6 Luna extraction through the OpenAI Batch API.

Runs the byte-identical stage-3 prompt (system prompt with embedded schema,
strict structured output, reasoning effort low) over every submission that
stage 1 extracted, as a new checkpoint stage ``stage4``. Requests are chunked
into JSONL files bounded by the Batch API limits (50,000 requests, 200 MB per
file), submitted with a 24 h completion window, monitored, and ingested into
the same private SQLite checkpoint the synchronous runs used, so downstream
analysis scripts read stage 4 exactly like any other stage.

Costs are computed at the published Batch API prices for gpt-5.6-luna
(USD 0.10 / 0.01 / 0.60 per million input / cached input / output tokens)
because OpenAI usage objects carry no cost field.

Subcommands:

    uv run python data_analysis/06_luna_batch_extraction.py build
    uv run python data_analysis/06_luna_batch_extraction.py submit
    uv run python data_analysis/06_luna_batch_extraction.py status
    uv run python data_analysis/06_luna_batch_extraction.py monitor   # loops
    uv run python data_analysis/06_luna_batch_extraction.py ingest    # re-run

``monitor`` is resumable and idempotent: it submits pending chunks whenever
the queue accepts them, downloads and ingests each batch as it completes,
retries retryable per-request failures once in a final sweep batch, and exits
0 when every record has either a stage_results row or a terminal failure row.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_experiment_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp02", Path(__file__).resolve().parent / "02_llm_extraction_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp02"] = module
    spec.loader.exec_module(module)
    return module


exp = _load_experiment_module()

STAGE_KEY = "stage4"
MODEL = "gpt-5.6-luna"
PIPELINE_VERSION = "2.0-openai-batch"
MAX_REQUESTS_PER_CHUNK = 50_000
MAX_CHUNK_BYTES = 190 * 1024 * 1024
COMPLETION_WINDOW = "24h"
POLL_SECONDS = 300
PRICE_INPUT = 0.10 / 1e6
PRICE_CACHED = 0.01 / 1e6
PRICE_OUTPUT = 0.60 / 1e6

RESULTS_DIR = Path(__file__).resolve().parent / "02_llm_extraction_results"
BATCH_DIR = RESULTS_DIR / "batch_luna"
STATE_PATH = BATCH_DIR / "state.json"

STAGE = exp.StageConfig(
    key=STAGE_KEY,
    description="GPT-5.6 Luna / low reasoning / full corpus / OpenAI Batch API",
    model=MODEL,
    reasoning={"effort": "low", "exclude": True},
    max_tokens=2_048,
    max_tokens_parameter="max_completion_tokens",
    session_id=f"botox-extraction-{exp.PROMPT_VERSION}-luna-low-batch",
    export_name="stage4_gpt56_luna_batch.jsonl",
)


def openai_client() -> Any:
    import os

    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
            os.environ.setdefault("OPENAI_API_KEY", line.split("=", 1)[1].strip())
    from openai import OpenAI

    return OpenAI(max_retries=5)


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"chunks": []}


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def request_line(rid: str, text_fields: dict[str, str]) -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": exp.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(text_fields, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "response_format": exp.RESPONSE_FORMAT,
        "reasoning_effort": "low",
        "max_completion_tokens": STAGE.max_tokens,
    }
    return json.dumps(
        {"custom_id": rid, "method": "POST", "url": "/v1/chat/completions", "body": body},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def stage1_record_ids(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT record_id FROM stage_results WHERE stage = 'stage1' ORDER BY record_id"
        )
    ]


def ingested_ids(connection: sqlite3.Connection) -> set[str]:
    done = {
        row[0]
        for row in connection.execute(
            "SELECT record_id FROM stage_results WHERE stage = ?", (STAGE_KEY,)
        )
    }
    done |= {
        row[0]
        for row in connection.execute(
            "SELECT record_id FROM stage_failures WHERE stage = ? AND terminal = 1",
            (STAGE_KEY,),
        )
    }
    return done


def build_chunks(target_ids: list[str], label: str) -> list[dict[str, Any]]:
    """Write chunked JSONL request files; returns chunk state entries."""
    chunks: list[dict[str, Any]] = []
    writer = None
    written_bytes = 0
    written_lines = 0
    index = 0

    def open_next() -> None:
        nonlocal writer, written_bytes, written_lines, index
        if writer is not None:
            writer.close()
        index += 1
        path = BATCH_DIR / f"requests_{label}_{index:03d}.jsonl"
        chunks.append({"path": str(path), "n_requests": 0, "status": "built"})
        writer = path.open("w", encoding="utf-8")
        written_bytes = 0
        written_lines = 0

    open_next()
    source = ROOT / "filtered_dataset" / "submissions.jsonl"
    remaining = set(target_ids)
    for rid, text_fields in exp.iter_target_records(source, remaining):
        line = request_line(rid, text_fields) + "\n"
        encoded = len(line.encode("utf-8"))
        if written_lines >= MAX_REQUESTS_PER_CHUNK or written_bytes + encoded > MAX_CHUNK_BYTES:
            open_next()
        writer.write(line)
        written_bytes += encoded
        written_lines += 1
        chunks[-1]["n_requests"] += 1
    writer.close()
    return [c for c in chunks if c["n_requests"] > 0]


def cmd_build(args: argparse.Namespace) -> None:
    exp.ensure_private_results_dir(RESULTS_DIR)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if state["chunks"]:
        raise SystemExit("state.json already has chunks; delete batch_luna/ to rebuild.")
    connection = exp.open_database(RESULTS_DIR)
    ids = stage1_record_ids(connection)
    done = ingested_ids(connection)
    todo = [rid for rid in ids if rid not in done]
    print(
        f"stage1 records: {len(ids):,}; already in {STAGE_KEY}: {len(done):,}; to run: {len(todo):,}"
    )
    chunks = build_chunks(todo, "main")
    state["chunks"] = chunks
    state["target_count"] = len(todo)
    save_state(state)
    total_mb = sum(Path(c["path"]).stat().st_size for c in chunks) / 1e6
    print(f"built {len(chunks)} chunk files, {total_mb:,.0f} MB total")


def _submit_chunk(client: Any, chunk: dict[str, Any]) -> bool:
    """Upload and create one batch. Returns False when the queue rejects it."""
    path = Path(chunk["path"])
    if not chunk.get("file_id"):
        with path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        chunk["file_id"] = uploaded.id
    try:
        batch = client.batches.create(
            input_file_id=chunk["file_id"],
            endpoint="/v1/chat/completions",
            completion_window=COMPLETION_WINDOW,
            metadata={"project": "botox_se", "stage": STAGE_KEY, "chunk": path.name},
        )
    except Exception as exc:  # queue/token limits are transient at this scale
        message = str(exc)
        if any(
            token in message.lower() for token in ("enqueued", "token_limit", "queue", "pending")
        ):
            print(f"  queue rejected {path.name}: {message[:160]}")
            return False
        raise
    chunk["batch_id"] = batch.id
    chunk["status"] = "submitted"
    print(f"  submitted {path.name} -> {batch.id} ({chunk['n_requests']:,} requests)")
    return True


def cmd_submit(args: argparse.Namespace) -> None:
    client = openai_client()
    state = load_state()
    if not state["chunks"]:
        raise SystemExit("no chunks; run build first")
    for chunk in state["chunks"]:
        if chunk["status"] == "built":
            if not _submit_chunk(client, chunk):
                break
            save_state(state)
    save_state(state)


TERMINAL_BATCH = {"completed", "failed", "expired", "cancelled"}


def usage_cost(usage: dict[str, Any]) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return (prompt - cached) * PRICE_INPUT + cached * PRICE_CACHED + completion * PRICE_OUTPUT


def text_fields_index(chunk_path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    with chunk_path.open(encoding="utf-8") as fh:
        for line in fh:
            request = json.loads(line)
            index[request["custom_id"]] = json.loads(request["body"]["messages"][1]["content"])
    return index


def ingest_response_line(
    connection: sqlite3.Connection,
    parsed: dict[str, Any],
    texts: dict[str, dict[str, str]],
) -> tuple[bool, str | None]:
    """Store one batch output line. Returns (succeeded, retryable_record_id)."""
    rid = parsed["custom_id"]
    response = parsed.get("response") or {}
    error = parsed.get("error")
    body = response.get("body") or {}
    status = response.get("status_code")
    usage = body.get("usage") or {}
    if usage:
        usage = dict(usage)
        usage["cost"] = usage_cost(usage)
        usage["cost_source"] = "computed_openai_batch_pricing"

    choices = body.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    finish = choice.get("finish_reason")
    message = choice.get("message") or {}

    base_metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": STAGE_KEY,
        "model_requested": MODEL,
        "model_returned": body.get("model"),
        "provider_returned": "openai_batch",
        "response_id": body.get("id"),
        "batch_request_id": parsed.get("id"),
        "reasoning": STAGE.reasoning,
        "finish_reason": finish,
        "response_cache_status": None,
        "response_healing": {"requested": False},
        "usage": usage,
        "attempt_count": 1,
    }

    def failure(
        err: str, kind: str, terminal: bool, preview: str | None = None
    ) -> tuple[bool, str | None]:
        result = exp.CallResult(
            record_id=rid,
            metadata=base_metadata,
            error=err,
            status_code=status,
            response_preview=preview,
            failure_kind=kind,
            terminal=terminal,
        )
        exp.save_call_result(connection, STAGE, result)
        return False, (None if terminal else rid)

    if error:
        return failure(f"batch error: {json.dumps(error)[:1000]}", "retryable", False)
    if status and status != 200:
        return failure(f"HTTP {status}: {json.dumps(body)[:1000]}", "retryable", False)
    if finish == "content_filter":
        preview = str(message.get("content") or message.get("refusal") or "")[:2000]
        return failure(
            "ContentFilterError: provider blocked the response", "content_filter", True, preview
        )
    if message.get("refusal"):
        return failure(f"model refusal: {message['refusal']}"[:1000], "refusal", True)

    content = message.get("content")
    try:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("response contains no JSON content")
        extracted = exp.ExtractedFields.model_validate(json.loads(content))
    except Exception as exc:
        return failure(
            f"{type(exc).__name__}: {exc}"[:1000], "retryable", False, str(content)[:2000]
        )

    fingerprint = str(body.get("system_fingerprint") or body.get("model") or MODEL)
    post = exp.PostExtraction(
        **extracted.model_dump(),
        record_id=rid,
        record_kind=exp.RecordKind.SUBMISSION,
        fields_seen=[exp.SourceField.TITLE, exp.SourceField.SELFTEXT],
        prompt_version=exp.PROMPT_VERSION,
        model_fingerprint=fingerprint,
    )
    metadata = dict(base_metadata)
    metadata["raw_span_metrics"] = exp.raw_span_metrics(extracted, texts.get(rid) or {})
    exp.save_call_result(
        connection,
        STAGE,
        exp.CallResult(record_id=rid, extraction=post.model_dump(mode="json"), metadata=metadata),
    )
    return True, None


def ingest_chunk(client: Any, connection: sqlite3.Connection, chunk: dict[str, Any]) -> list[str]:
    """Download and store a completed batch. Returns retryable record ids."""
    batch = client.batches.retrieve(chunk["batch_id"])
    texts = text_fields_index(Path(chunk["path"]))
    retryable: list[str] = []
    n_ok = 0
    for file_key in ("output_file_id", "error_file_id"):
        file_id = getattr(batch, file_key, None)
        if not file_id:
            continue
        content = client.files.content(file_id)
        for line in content.text.splitlines():
            if not line.strip():
                continue
            ok, retry_id = ingest_response_line(connection, json.loads(line), texts)
            n_ok += ok
            if retry_id:
                retryable.append(retry_id)
    # Requests missing from both files (expired batches) are retryable too.
    seen_ok = ingested_ids(connection)
    missing = [rid for rid in texts if rid not in seen_ok and rid not in set(retryable)]
    # A record already ingested from a previous round is fine; only count truly absent ones.
    absent = [
        rid
        for rid in missing
        if not connection.execute(
            "SELECT 1 FROM stage_failures WHERE stage=? AND record_id=?", (STAGE_KEY, rid)
        ).fetchone()
    ]
    retryable.extend(absent)
    connection.commit()
    chunk["status"] = "ingested"
    chunk["batch_status"] = batch.status
    chunk["n_succeeded"] = n_ok
    chunk["n_retryable"] = len(retryable)
    print(
        f"  ingested {Path(chunk['path']).name}: batch {batch.status}, "
        f"{n_ok:,} ok, {len(retryable):,} retryable"
    )
    return retryable


def cmd_monitor(args: argparse.Namespace) -> None:
    exp.ensure_private_results_dir(RESULTS_DIR)
    client = openai_client()
    connection = exp.open_database(RESULTS_DIR)
    state = load_state()
    if not state["chunks"]:
        raise SystemExit("no chunks; run build first")
    retry_round = state.get("retry_round", 0)

    while True:
        submitted_now = 0
        for chunk in state["chunks"]:
            if chunk["status"] == "built":
                if _submit_chunk(client, chunk):
                    submitted_now += 1
                    save_state(state)
                else:
                    break

        counts: dict[str, int] = {}
        for chunk in state["chunks"]:
            if chunk["status"] == "submitted":
                batch = client.batches.retrieve(chunk["batch_id"])
                if batch.status in TERMINAL_BATCH:
                    retryable = ingest_chunk(client, connection, chunk)
                    chunk["retryable_ids"] = retryable
                    save_state(state)
                else:
                    rc = getattr(batch, "request_counts", None)
                    done = getattr(rc, "completed", 0) if rc else 0
                    total = getattr(rc, "total", 0) if rc else 0
                    counts[batch.status] = counts.get(batch.status, 0) + 1
                    print(f"  {Path(chunk['path']).name}: {batch.status} {done:,}/{total:,}")
        pending = [c for c in state["chunks"] if c["status"] in ("built", "submitted")]
        if not pending:
            all_retryable = sorted(
                {rid for c in state["chunks"] for rid in c.get("retryable_ids", [])}
            )
            still_missing = [rid for rid in all_retryable if rid not in ingested_ids(connection)]
            if still_missing and retry_round < 2:
                retry_round += 1
                state["retry_round"] = retry_round
                print(f"retry round {retry_round}: {len(still_missing):,} records")
                new_chunks = build_chunks(still_missing, f"retry{retry_round}")
                state["chunks"].extend(new_chunks)
                save_state(state)
                continue
            break
        time.sleep(POLL_SECONDS)

    connection = exp.open_database(RESULTS_DIR)
    n_results = connection.execute(
        "SELECT count(*) FROM stage_results WHERE stage=?", (STAGE_KEY,)
    ).fetchone()[0]
    n_terminal = connection.execute(
        "SELECT count(*) FROM stage_failures WHERE stage=? AND terminal=1", (STAGE_KEY,)
    ).fetchone()[0]
    total_cost = connection.execute(
        "SELECT sum(json_extract(metadata_json,'$.usage.cost')) FROM stage_results WHERE stage=?",
        (STAGE_KEY,),
    ).fetchone()[0]
    exp.export_stage(connection, RESULTS_DIR, STAGE)
    print(
        f"DONE: {n_results:,} successful extractions, {n_terminal:,} terminal failures, "
        f"computed cost US${total_cost:,.2f}"
    )


def cmd_status(args: argparse.Namespace) -> None:
    client = openai_client()
    state = load_state()
    for chunk in state["chunks"]:
        line = f"{Path(chunk['path']).name}: {chunk['status']}"
        if chunk.get("batch_id") and chunk["status"] == "submitted":
            batch = client.batches.retrieve(chunk["batch_id"])
            rc = getattr(batch, "request_counts", None)
            line += f" | batch {batch.status}"
            if rc:
                line += f" {rc.completed:,}/{rc.total:,} (failed {rc.failed:,})"
        print(line)
    connection = exp.open_database(RESULTS_DIR)
    n = connection.execute(
        "SELECT count(*) FROM stage_results WHERE stage=?", (STAGE_KEY,)
    ).fetchone()[0]
    print(f"stage4 rows ingested: {n:,}")


def cmd_ingest(args: argparse.Namespace) -> None:
    client = openai_client()
    connection = exp.open_database(RESULTS_DIR)
    state = load_state()
    for chunk in state["chunks"]:
        if chunk.get("batch_id") and chunk["status"] != "ingested":
            batch = client.batches.retrieve(chunk["batch_id"])
            if batch.status in TERMINAL_BATCH:
                chunk["retryable_ids"] = ingest_chunk(client, connection, chunk)
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (
        ("build", cmd_build),
        ("submit", cmd_submit),
        ("status", cmd_status),
        ("monitor", cmd_monitor),
        ("ingest", cmd_ingest),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
