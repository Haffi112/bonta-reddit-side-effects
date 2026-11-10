from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import select, text

from .models import Annotator, db, utc_now
from .security import access_key_digest

bp = Blueprint("auth", __name__)


@bp.get("/")
def index():  # type: ignore[no-untyped-def]
    if session.get("admin_authenticated"):
        return redirect(url_for("admin.dashboard"))
    if session.get("annotator_id"):
        return redirect(url_for("review.next_item"))
    return redirect(url_for("auth.login"))


@bp.get("/healthz")
def health():  # type: ignore[no-untyped-def]
    db.session.execute(text("SELECT 1"))
    return {"status": "ok"}


@bp.route("/login", methods=["GET", "POST"])
def login():  # type: ignore[no-untyped-def]
    if request.method == "POST":
        key = request.form.get("access_key", "").strip()
        annotator = db.session.scalar(
            select(Annotator).where(Annotator.key_digest == access_key_digest(key))
        )
        if annotator is None or not annotator.is_active:
            flash("That key is not valid or has been deactivated.", "error")
        else:
            session.clear()
            session["annotator_id"] = annotator.id
            annotator.last_login_at = utc_now()
            db.session.commit()
            return redirect(url_for("review.next_item"))
    return render_template("login.html")


@bp.post("/logout")
def logout():  # type: ignore[no-untyped-def]
    session.clear()
    return redirect(url_for("auth.login"))
