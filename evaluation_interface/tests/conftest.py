from __future__ import annotations

import pytest

from eval_app import create_app
from eval_app.models import db


@pytest.fixture
def app(tmp_path):
    database = tmp_path / "test.sqlite3"
    app = create_app(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{database}",
            "ADMIN_KEY": "admin-test-key",
            "SECRET_KEY": "test-secret",
        }
    )
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()

