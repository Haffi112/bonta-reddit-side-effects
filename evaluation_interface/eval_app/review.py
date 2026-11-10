from __future__ import annotations

import secrets
from typing import Any

from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for
from sqlalchemy import and_, func, select
from sqlalchemy.orm import joinedload, selectinload

from .fields import review_rows, validate_decisions
from .importer import DEFAULT_STAGES
from .models import Annotation, Annotator, ModelOutput, Submission, db, utc_now
from .security import annotator_required

bp = Blueprint("review", __name__, url_prefix="/review")


def _current_annotator() -> Annotator:
    annotator = db.session.get(Annotator, session.get("annotator_id"))
    if annotator is None or not annotator.is_active:
        session.clear()
        abort(401)
    return annotator


def _eligible_comparisons_query():  # type: ignore[no-untyped-def]
    return (
        select(ModelOutput.submission_id)
        .where(ModelOutput.model_key.in_(DEFAULT_STAGES))
        .group_by(ModelOutput.submission_id)
        .having(func.count(ModelOutput.id) == len(DEFAULT_STAGES))
    )


def _progress(annotator_id: int) -> tuple[int, int]:
    eligible = _eligible_comparisons_query().subquery()
    total = int(db.session.scalar(select(func.count()).select_from(eligible)) or 0)
    annotation_join = and_(
        Annotation.model_output_id == ModelOutput.id,
        Annotation.annotator_id == annotator_id,
    )
    completed = (
        select(ModelOutput.submission_id)
        .outerjoin(Annotation, annotation_join)
        .where(ModelOutput.model_key.in_(DEFAULT_STAGES))
        .group_by(ModelOutput.submission_id)
        .having(func.count(ModelOutput.id) == len(DEFAULT_STAGES))
        .having(func.count(Annotation.id) == len(DEFAULT_STAGES))
        .subquery()
    )
    completed_count = int(
        db.session.scalar(select(func.count()).select_from(completed)) or 0
    )
    return completed_count, total


def _load_submission(blinded_id: str) -> Submission:
    handle = db.session.scalar(
        select(ModelOutput).where(ModelOutput.blinded_id == blinded_id)
    )
    if handle is None:
        abort(404)
    submission = db.session.scalar(
        select(Submission)
        .options(selectinload(Submission.outputs))
        .where(Submission.id == handle.submission_id)
    )
    if submission is None:
        abort(404)
    return submission


def _ordered_outputs(
    submission: Submission, posted_order: list[str] | None = None
) -> list[ModelOutput]:
    outputs = [
        output for output in submission.outputs if output.model_key in DEFAULT_STAGES
    ]
    if len(outputs) != len(DEFAULT_STAGES):
        abort(409, "This submission does not have both evaluation responses.")
    by_blinded_id = {output.blinded_id: output for output in outputs}
    if posted_order:
        if len(posted_order) != len(outputs) or set(posted_order) != set(by_blinded_id):
            abort(400, "The blinded response order is invalid.")
        return [by_blinded_id[blinded_id] for blinded_id in posted_order]
    secrets.SystemRandom().shuffle(outputs)
    return outputs


def _existing_annotations(
    annotator_id: int, outputs: list[ModelOutput]
) -> dict[int, Annotation]:
    return {
        annotation.model_output_id: annotation
        for annotation in db.session.scalars(
            select(Annotation).where(
                Annotation.annotator_id == annotator_id,
                Annotation.model_output_id.in_([output.id for output in outputs]),
            )
        ).all()
    }


def _comparison_cards(
    outputs: list[ModelOutput],
    annotations: dict[int, Annotation],
    submitted_decisions: dict[str, dict[str, str]] | None = None,
    submitted_notes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        annotation = annotations.get(output.id)
        cards.append(
            {
                "label": f"Response {chr(ord('A') + index)}",
                "output": output,
                "rows": review_rows(output.extraction),
                "decisions": (
                    (submitted_decisions or {}).get(output.blinded_id)
                    if submitted_decisions is not None
                    else annotation.decisions if annotation else {}
                )
                or {},
                "notes": (
                    (submitted_notes or {}).get(output.blinded_id)
                    if submitted_notes is not None
                    else annotation.notes if annotation else ""
                )
                or "",
            }
        )
    return cards


@bp.get("/next")
@annotator_required
def next_item():  # type: ignore[no-untyped-def]
    annotator = _current_annotator()
    annotation_join = and_(
        Annotation.model_output_id == ModelOutput.id,
        Annotation.annotator_id == annotator.id,
    )
    pending_submission_id = db.session.scalar(
        select(ModelOutput.submission_id)
        .outerjoin(Annotation, annotation_join)
        .where(ModelOutput.model_key.in_(DEFAULT_STAGES))
        .group_by(ModelOutput.submission_id)
        .having(func.count(ModelOutput.id) == len(DEFAULT_STAGES))
        .having(func.count(Annotation.id) < len(DEFAULT_STAGES))
        .order_by(func.min(ModelOutput.random_order))
        .limit(1)
    )
    if pending_submission_id is None:
        completed, total = _progress(annotator.id)
        return render_template(
            "complete.html", annotator=annotator, completed=completed, total=total
        )
    handle = db.session.scalar(
        select(ModelOutput.blinded_id)
        .where(
            ModelOutput.submission_id == pending_submission_id,
            ModelOutput.model_key.in_(DEFAULT_STAGES),
        )
        .order_by(ModelOutput.blinded_id)
        .limit(1)
    )
    return redirect(url_for("review.item", blinded_id=handle))


@bp.route("/<blinded_id>", methods=["GET", "POST"])
@annotator_required
def item(blinded_id: str):  # type: ignore[no-untyped-def]
    annotator = _current_annotator()
    submission = _load_submission(blinded_id)
    posted_order = request.form.getlist("output_order") if request.method == "POST" else None
    outputs = _ordered_outputs(submission, posted_order)
    annotations = _existing_annotations(annotator.id, outputs)

    if request.method == "POST":
        submitted_decisions: dict[str, dict[str, str]] = {}
        submitted_notes: dict[str, str] = {}
        missing_count = 0
        for output in outputs:
            rows = review_rows(output.extraction)
            submitted = {
                row.key: request.form.get(
                    f"decision::{output.blinded_id}::{row.key}", ""
                )
                for row in rows
            }
            decisions, missing = validate_decisions(output.extraction, submitted)
            submitted_decisions[output.blinded_id] = decisions
            submitted_notes[output.blinded_id] = request.form.get(
                f"notes::{output.blinded_id}", ""
            ).strip()[:4_000]
            missing_count += len(missing)

        if missing_count:
            completed, total = _progress(annotator.id)
            flash(
                f"Please mark every field in both responses. {missing_count} decision{'s are' if missing_count != 1 else ' is'} still missing.",
                "error",
            )
            return (
                render_template(
                    "review.html",
                    annotator=annotator,
                    submission=submission,
                    cards=_comparison_cards(
                        outputs,
                        annotations,
                        submitted_decisions,
                        submitted_notes,
                    ),
                    completed=completed,
                    total=total,
                ),
                422,
            )

        for output in outputs:
            annotation = annotations.get(output.id)
            if annotation is None:
                db.session.add(
                    Annotation(
                        annotator_id=annotator.id,
                        model_output_id=output.id,
                        decisions=submitted_decisions[output.blinded_id],
                        notes=submitted_notes[output.blinded_id],
                    )
                )
            else:
                annotation.decisions = submitted_decisions[output.blinded_id]
                annotation.notes = submitted_notes[output.blinded_id]
                annotation.submitted_at = utc_now()
        db.session.commit()
        flash("Both response reviews were saved.", "success")
        return redirect(url_for("review.next_item"))

    completed, total = _progress(annotator.id)
    return render_template(
        "review.html",
        annotator=annotator,
        submission=submission,
        cards=_comparison_cards(outputs, annotations),
        completed=completed,
        total=total,
    )


@bp.get("/history")
@annotator_required
def history():  # type: ignore[no-untyped-def]
    annotator = _current_annotator()
    annotations = db.session.scalars(
        select(Annotation)
        .join(Annotation.model_output)
        .options(
            joinedload(Annotation.model_output)
            .joinedload(ModelOutput.submission)
            .selectinload(Submission.outputs)
        )
        .where(
            Annotation.annotator_id == annotator.id,
            ModelOutput.model_key.in_(DEFAULT_STAGES),
        )
        .order_by(Annotation.submitted_at.desc())
        .limit(400)
    ).all()
    grouped: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        submission = annotation.model_output.submission
        item = grouped.setdefault(
            submission.id,
            {
                "blinded_id": next(
                    output.blinded_id
                    for output in submission.outputs
                    if output.model_key in DEFAULT_STAGES
                ),
                "submitted_at": annotation.submitted_at,
                "correct": 0,
                "decisions": 0,
                "responses": 0,
            },
        )
        item["submitted_at"] = max(item["submitted_at"], annotation.submitted_at)
        item["correct"] += sum(
            decision == "correct" for decision in annotation.decisions.values()
        )
        item["decisions"] += len(annotation.decisions)
        item["responses"] += 1
    history_items = sorted(
        grouped.values(), key=lambda item: item["submitted_at"], reverse=True
    )[:100]
    completed, total = _progress(annotator.id)
    return render_template(
        "history.html",
        annotator=annotator,
        history_items=history_items,
        completed=completed,
        total=total,
    )
