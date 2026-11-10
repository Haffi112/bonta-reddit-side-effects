from __future__ import annotations

import re

from sqlalchemy import select

from eval_app.admin import _spreadsheet_safe
from eval_app.fields import model_fields, required_decision_keys
from eval_app.models import Annotation, Annotator, ModelOutput, Submission, db
from eval_app.security import access_key_digest

from .helpers import extraction


def seed_review(
    app, key: str = "ann-test-secret"
) -> tuple[dict[str, str], dict[str, object]]:
    with app.app_context():
        annotator = Annotator(label="Annotator 1", key_digest=access_key_digest(key))
        submission = Submission(
            record_id="t3_test",
            title="Botox problem",
            selftext="I had Botox in my forehead and my eyelid drooped.",
            source_sha256="a" * 64,
        )
        db.session.add_all([annotator, submission])
        db.session.flush()
        extracted_fields = model_fields(extraction())
        outputs = [
            ModelOutput(
                submission_id=submission.id,
                model_key=stage,
                model_label=label,
                blinded_id=blinded_id,
                random_order=str(index) * 64,
                extraction=extracted_fields,
                extraction_sha256=str(index + 3) * 64,
            )
            for index, (stage, label, blinded_id) in enumerate(
                (
                    ("stage1", "Secret no-reasoning model", "item_stage1"),
                    ("stage2", "Excluded high-reasoning model", "item_stage2"),
                    ("stage3", "Secret Luna model", "item_stage3"),
                ),
                start=1,
            )
        ]
        db.session.add_all(outputs)
        db.session.commit()
        return {output.model_key: output.blinded_id for output in outputs}, extracted_fields


def test_annotator_reviews_both_shuffled_responses_with_separate_notes(
    app, client, monkeypatch
):
    blinded_ids, extracted_fields = seed_review(app)

    class AlternatingShuffle:
        calls = 0

        def shuffle(self, values):
            if self.__class__.calls % 2:
                values.reverse()
            self.__class__.calls += 1

    monkeypatch.setattr("eval_app.review.secrets.SystemRandom", AlternatingShuffle)
    response = client.post("/login", data={"access_key": "ann-test-secret"})
    assert response.status_code == 302
    response = client.get("/review/next", follow_redirects=True)
    assert response.status_code == 200
    assert b"Botox problem" in response.data
    assert b"Locate wording in post" in response.data
    assert b"Response A" in response.data
    assert b"Response B" in response.data
    assert b"Secret no-reasoning model" not in response.data
    assert b"Secret Luna model" not in response.data
    assert b"item_stage2" not in response.data
    self_use_signatures = re.findall(
        rb'data-field-key="a_self_use" data-value-signature="([a-f0-9]{64})"',
        response.data,
    )
    assert len(self_use_signatures) == 2
    assert len(set(self_use_signatures)) == 1
    first_order = re.findall(rb'name="output_order" value="([^"]+)"', response.data)

    reopened = client.get(f"/review/{blinded_ids['stage1']}")
    second_order = re.findall(rb'name="output_order" value="([^"]+)"', reopened.data)
    assert first_order == list(reversed(second_order))

    incomplete = client.post(f"/review/{blinded_ids['stage1']}", data={})
    assert incomplete.status_code == 422

    decisions: dict[str, object] = {
        "output_order": [blinded_ids["stage1"], blinded_ids["stage3"]],
        f"notes::{blinded_ids['stage1']}": "Note for the first response",
        f"notes::{blinded_ids['stage3']}": "Different note for the second response",
    }
    for output_id in (blinded_ids["stage1"], blinded_ids["stage3"]):
        decisions.update(
            {
                f"decision::{output_id}::{key}": "correct"
                for key in required_decision_keys(extracted_fields)
            }
        )
    saved = client.post(
        f"/review/{blinded_ids['stage1']}", data=decisions, follow_redirects=True
    )
    assert saved.status_code == 200
    assert b"All 1 submissions are reviewed" in saved.data

    with app.app_context():
        annotations = db.session.scalars(select(Annotation).order_by(Annotation.id)).all()
        assert len(annotations) == 2
        assert {annotation.notes for annotation in annotations} == {
            "Note for the first response",
            "Different note for the second response",
        }

    history = client.get("/review/history")
    assert b"2 / 2" in history.data
    assert b"42 correct / 42" in history.data


def test_health_check_reaches_database(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}
    assert response.headers["Referrer-Policy"] == "same-origin"


def test_csv_export_neutralizes_untrusted_spreadsheet_formulas():
    assert _spreadsheet_safe("=HYPERLINK(\"https://example.test\")") == (
        "'=HYPERLINK(\"https://example.test\")"
    )
    assert _spreadsheet_safe("Ordinary Reddit text") == "Ordinary Reddit text"


def test_long_running_review_forms_do_not_expire_csrf_tokens(app):
    assert app.config["WTF_CSRF_TIME_LIMIT"] is None


def test_admin_creates_rotatable_key_and_exports_decisions(app, client):
    blinded_ids, _ = seed_review(app)
    with app.app_context():
        annotator = db.session.query(Annotator).filter_by(label="Annotator 1").one()
        outputs = db.session.scalars(
            select(ModelOutput).where(
                ModelOutput.blinded_id.in_(blinded_ids.values())
            )
        ).all()
        for output in outputs:
            db.session.add(
                Annotation(
                    annotator_id=annotator.id,
                    model_output_id=output.id,
                    decisions={"a_self_use": "correct"},
                    notes=f"Checked {output.model_key}",
                )
            )
        db.session.commit()
    client.post("/admin/login", data={"admin_key": "admin-test-key"})
    created = client.post("/admin/annotators", data={"label": "Annotator 2"})
    assert created.status_code == 200
    match = re.search(rb"ann_[A-Za-z0-9_-]+", created.data)
    assert match is not None
    assert created.data.count(match.group()) == 1

    dashboard = client.get("/admin")
    assert dashboard.status_code == 200
    assert b"Secret no-reasoning model" in dashboard.data
    assert b"Secret Luna model" in dashboard.data
    assert b"Excluded high-reasoning model" not in dashboard.data
    assert b"1 / 1 comparisons" in dashboard.data
    assert b"100.0%" in dashboard.data

    export = client.get("/admin/export/annotations.csv")
    assert export.status_code == 200
    assert b"t3_test" in export.data
    assert b"a_self_use" in export.data
    assert b"Checked stage1" in export.data
    assert b"Checked stage3" in export.data
    assert b"Checked stage2" not in export.data
    assert b"my eyelid drooped" in export.data
    assert b"Botox problem" in export.data
