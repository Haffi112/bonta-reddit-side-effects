from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable
from functools import wraps
from typing import cast

from flask import flash, redirect, session, url_for


def access_key_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_access_key() -> str:
    return "ann_" + secrets.token_urlsafe(24)


def safe_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def annotator_required[**P, R](view: Callable[P, R]) -> Callable[P, R]:
    @wraps(view)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        from .models import Annotator, db

        annotator_id = session.get("annotator_id")
        annotator = db.session.get(Annotator, annotator_id) if annotator_id else None
        if annotator is None or not annotator.is_active:
            session.clear()
            flash("Enter your annotator key to continue.", "info")
            return cast(R, redirect(url_for("auth.login")))
        return view(*args, **kwargs)

    return wrapped


def admin_required[**P, R](view: Callable[P, R]) -> Callable[P, R]:
    @wraps(view)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if not session.get("admin_authenticated"):
            flash("Admin authentication is required.", "info")
            return cast(R, redirect(url_for("admin.login")))
        return view(*args, **kwargs)

    return wrapped
