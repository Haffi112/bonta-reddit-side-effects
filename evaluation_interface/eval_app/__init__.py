from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from flask_wtf.csrf import CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix

from .models import db

csrf = CSRFProtect()


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return "sqlite:///evaluation.sqlite3"
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://") and "+psycopg" not in url.split(":", 1)[0]:
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def create_app(test_config: dict[str, object] | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    production = bool(os.environ.get("DYNO"))
    secret_key = os.environ.get("SECRET_KEY", "").strip()
    if production and not secret_key:
        raise RuntimeError("SECRET_KEY must be configured on Heroku")

    app.config.from_mapping(
        SECRET_KEY=secret_key or "local-development-only",
        SQLALCHEMY_DATABASE_URI=_database_url(),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        ADMIN_KEY=os.environ.get("ADMIN_KEY", ""),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=production,
        # Comparison forms can remain open for hours. The token stays bound to
        # the signed browser session and same-origin Referer, but does not expire.
        WTF_CSRF_TIME_LIMIT=None,
        MAX_CONTENT_LENGTH=1 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    csrf.init_app(app)
    if production:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    from .admin import bp as admin_bp
    from .auth import bp as auth_bp
    from .commands import register_commands
    from .review import bp as review_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(admin_bp)
    register_commands(app)

    @app.after_request
    def secure_private_responses(response):  # type: ignore[no-untyped-def]
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        # Flask-WTF's HTTPS CSRF check requires a same-origin Referer on POST.
        # Keep it for this origin while still suppressing referrers to other sites.
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    return app
