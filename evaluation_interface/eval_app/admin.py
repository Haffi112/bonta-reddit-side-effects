from __future__ import annotations

import csv
import io
import secrets
from collections.abc import Iterator

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from .fields import canonical_decision_key
from .importer import DEFAULT_STAGES
from .models import Annotation, Annotator, ModelOutput, db
from .security import (
    access_key_digest,
    admin_required,
    generate_access_key,
    safe_equal,
)
from .stats import dashboard_statistics

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _spreadsheet_safe(value: object) -> object:
    """Prevent untrusted text from becoming a formula when a CSV is opened."""
    if not isinstance(value, str):
        return value
    first_visible = value.lstrip()[:1]
    if first_visible in {"=", "+", "-", "@"} or value.startswith(("\t", "\r")):
        return "'" + value
    return value


@bp.route("/login", methods=["GET", "POST"])
def login():  # type: ignore[no-untyped-def]
    configured_key = str(current_app.config.get("ADMIN_KEY") or "")
    if request.method == "POST":
        supplied = request.form.get("admin_key", "")
        if not configured_key:
            flash("ADMIN_KEY has not been configured on this deployment.", "error")
        elif safe_equal(supplied, configured_key):
            session.clear()
            session["admin_authenticated"] = True
            return redirect(url_for("admin.dashboard"))
        else:
            flash("Invalid admin key.", "error")
    return render_template("admin/login.html", configured=bool(configured_key))


@bp.post("/logout")
def logout():  # type: ignore[no-untyped-def]
    session.clear()
    return redirect(url_for("admin.login"))


@bp.get("")
@admin_required
def dashboard():  # type: ignore[no-untyped-def]
    return render_template("admin/dashboard.html", stats=dashboard_statistics())


@bp.post("/annotators")
@admin_required
def create_annotator():  # type: ignore[no-untyped-def]
    label = " ".join(request.form.get("label", "").split())[:100]
    if not label:
        flash("Enter a label for the annotator.", "error")
        return redirect(url_for("admin.dashboard"))
    key = generate_access_key()
    annotator = Annotator(label=label, key_digest=access_key_digest(key))
    db.session.add(annotator)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        flash("That annotator label already exists.", "error")
        return redirect(url_for("admin.dashboard"))
    return render_template("admin/key_created.html", annotator=annotator, access_key=key)


@bp.post("/annotators/<int:annotator_id>/toggle")
@admin_required
def toggle_annotator(annotator_id: int):  # type: ignore[no-untyped-def]
    annotator = db.get_or_404(Annotator, annotator_id)
    annotator.is_active = not annotator.is_active
    db.session.commit()
    action = "reactivated" if annotator.is_active else "deactivated"
    flash(f"{annotator.label} was {action}.", "success")
    return redirect(url_for("admin.dashboard"))


@bp.post("/annotators/<int:annotator_id>/rotate")
@admin_required
def rotate_key(annotator_id: int):  # type: ignore[no-untyped-def]
    annotator = db.get_or_404(Annotator, annotator_id)
    key = generate_access_key()
    annotator.key_digest = access_key_digest(key)
    annotator.is_active = True
    db.session.commit()
    return render_template("admin/key_created.html", annotator=annotator, access_key=key)


def _csv_rows() -> Iterator[str]:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "record_id",
            "blinded_id",
            "model_key",
            "model_label",
            "submission_title",
            "submission_body",
            "annotator",
            "decision_key",
            "canonical_field",
            "decision",
            "notes",
            "submitted_at",
            "updated_at",
        ]
    )
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    annotations = db.session.scalars(
        select(Annotation)
        .join(Annotation.model_output)
        .options(
            joinedload(Annotation.annotator),
            joinedload(Annotation.model_output).joinedload(ModelOutput.submission),
        )
        .where(ModelOutput.model_key.in_(DEFAULT_STAGES))
        .order_by(Annotation.id)
    ).all()
    for annotation in annotations:
        output = annotation.model_output
        for decision_key, decision in annotation.decisions.items():
            writer.writerow(
                [
                    output.submission.record_id,
                    output.blinded_id,
                    output.model_key,
                    output.model_label,
                    _spreadsheet_safe(output.submission.title),
                    _spreadsheet_safe(output.submission.selftext),
                    annotation.annotator.label,
                    decision_key,
                    canonical_decision_key(decision_key),
                    decision,
                    _spreadsheet_safe(annotation.notes),
                    annotation.submitted_at.isoformat(),
                    annotation.updated_at.isoformat(),
                ]
            )
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)


@bp.get("/export/annotations.csv")
@admin_required
def export_annotations():  # type: ignore[no-untyped-def]
    filename = f"botox_annotations_{secrets.token_hex(4)}.csv"
    return Response(
        stream_with_context(_csv_rows()),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
