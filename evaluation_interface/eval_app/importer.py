from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from .fields import model_fields
from .models import Annotation, ModelOutput, Submission, db

DEFAULT_STAGES = ("stage1", "stage3")
STAGE_LABELS = {
    "stage1": "DeepSeek V4 Flash — no reasoning",
    "stage3": "GPT-5.6 Luna — low reasoning",
}


def _sha256_json(value: Any) -> str:
    serialised = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _source_hash(title: str, selftext: str) -> str:
    return hashlib.sha256(f"{title}\0{selftext}".encode()).hexdigest()


def _record_id(record: dict[str, Any]) -> str:
    value = str(record.get("name") or record.get("id") or "").strip()
    if not value:
        raise ValueError("source record has no name or id")
    return value if value.startswith("t3_") else f"t3_{value}"


def load_intersection(
    results_sqlite: Path, stages: tuple[str, ...], limit: int | None = None
) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
    if not stages or len(set(stages)) != len(stages):
        raise ValueError("stages must be a non-empty set of unique names")
    connection = sqlite3.connect(f"file:{results_sqlite.resolve().as_posix()}?mode=ro", uri=True)
    try:
        available = {
            row[0]
            for row in connection.execute("SELECT DISTINCT stage FROM stage_results")
        }
        missing_stages = set(stages) - available
        if missing_stages:
            raise ValueError(f"result database is missing stages: {sorted(missing_stages)}")
        placeholders = ",".join("?" for _ in stages)
        query = f"""
            SELECT record_id
            FROM stage_results
            WHERE stage IN ({placeholders})
            GROUP BY record_id
            HAVING COUNT(DISTINCT stage) = ?
            ORDER BY record_id
        """
        record_ids = [
            row[0] for row in connection.execute(query, (*stages, len(stages)))
        ]
        if limit is not None:
            record_ids = record_ids[:limit]
        target_ids = set(record_ids)
        outputs: dict[tuple[str, str], dict[str, Any]] = {}
        for stage in stages:
            for record_id, extraction_json in connection.execute(
                "SELECT record_id, extraction_json FROM stage_results WHERE stage = ?",
                (stage,),
            ):
                if record_id in target_ids:
                    extraction = json.loads(extraction_json)
                    if not isinstance(extraction, dict):
                        raise ValueError(f"{stage}/{record_id} extraction is not an object")
                    outputs[(stage, record_id)] = extraction
    finally:
        connection.close()
    expected = len(record_ids) * len(stages)
    if len(outputs) != expected:
        raise ValueError(f"expected {expected} outputs for the intersection, found {len(outputs)}")
    return record_ids, outputs


def load_source_records(source_jsonl: Path, record_ids: list[str]) -> dict[str, tuple[str, str]]:
    targets = set(record_ids)
    found: dict[str, tuple[str, str]] = {}
    with source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                record_id = _record_id(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid source record at line {line_number}: {exc}") from exc
            if record_id not in targets:
                continue
            found[record_id] = (
                str(record.get("title") or ""),
                str(record.get("selftext") or ""),
            )
            if len(found) == len(targets):
                break
    missing = targets - set(found)
    if missing:
        raise ValueError(f"source JSONL is missing {len(missing)} target records")
    return found


def import_evaluation_data(
    source_jsonl: Path,
    results_sqlite: Path,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    limit: int | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    record_ids, raw_outputs = load_intersection(results_sqlite, stages, limit)
    source_records = load_source_records(source_jsonl, record_ids)
    if dry_run:
        return {
            "intersection_records": len(record_ids),
            "model_outputs": len(raw_outputs),
            "new_submissions": 0,
            "new_outputs": 0,
        }

    new_submissions = 0
    new_outputs = 0
    existing_submissions = {
        submission.record_id: submission
        for submission in db.session.scalars(
            select(Submission).where(Submission.record_id.in_(record_ids))
        ).all()
    }
    existing_outputs = {
        (output.submission.record_id, output.model_key): output
        for output in db.session.scalars(
            select(ModelOutput)
            .join(ModelOutput.submission)
            .where(Submission.record_id.in_(record_ids))
        ).all()
    }
    for record_id in record_ids:
        title, selftext = source_records[record_id]
        source_sha256 = _source_hash(title, selftext)
        submission = existing_submissions.get(record_id)
        if submission is None:
            submission = Submission(
                record_id=record_id,
                title=title,
                selftext=selftext,
                source_sha256=source_sha256,
            )
            db.session.add(submission)
            db.session.flush()
            existing_submissions[record_id] = submission
            new_submissions += 1
        elif submission.source_sha256 != source_sha256:
            annotation_count = db.session.scalar(
                select(func.count(Annotation.id))
                .join(ModelOutput)
                .where(ModelOutput.submission_id == submission.id)
            )
            if annotation_count:
                raise ValueError(
                    f"source text changed for annotated record {record_id}; refusing to overwrite"
                )
            submission.title = title
            submission.selftext = selftext
            submission.source_sha256 = source_sha256

        for stage in stages:
            raw = raw_outputs[(stage, record_id)]
            extraction = model_fields(raw)
            extraction_sha256 = _sha256_json(extraction)
            output = existing_outputs.get((record_id, stage))
            label = STAGE_LABELS.get(stage, str(raw.get("model_fingerprint") or stage))
            if output is None:
                output = ModelOutput(
                    submission_id=submission.id,
                    model_key=stage,
                    model_label=label,
                    blinded_id="item_" + secrets.token_urlsafe(9),
                    random_order=secrets.token_hex(32),
                    extraction=extraction,
                    extraction_sha256=extraction_sha256,
                )
                db.session.add(output)
                existing_outputs[(record_id, stage)] = output
                new_outputs += 1
            elif output.extraction_sha256 != extraction_sha256:
                annotation_count = db.session.scalar(
                    select(func.count(Annotation.id)).where(
                        Annotation.model_output_id == output.id
                    )
                )
                if annotation_count:
                    raise ValueError(
                        f"model output changed for annotated item {record_id}/{stage}; refusing to overwrite"
                    )
                output.extraction = extraction
                output.extraction_sha256 = extraction_sha256
                output.model_label = label

    db.session.commit()
    return {
        "intersection_records": len(record_ids),
        "model_outputs": len(raw_outputs),
        "new_submissions": new_submissions,
        "new_outputs": new_outputs,
    }
