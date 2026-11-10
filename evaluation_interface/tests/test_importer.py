from __future__ import annotations

import json
import sqlite3

from eval_app.importer import import_evaluation_data
from eval_app.models import ModelOutput, Submission, db

from .helpers import extraction


def test_imports_two_way_intersection_without_high_reasoning_stage(app, tmp_path):
    source = tmp_path / "submissions.jsonl"
    source.write_text(
        json.dumps(
            {
                "name": "t3_test",
                "title": "A title",
                "selftext": "After forehead Botox my eyelid drooped.",
                "author": "must-not-be-imported",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.sqlite3"
    connection = sqlite3.connect(results)
    connection.execute(
        "CREATE TABLE stage_results (stage TEXT, record_id TEXT, extraction_json TEXT)"
    )
    for stage in ("stage1", "stage2", "stage3"):
        connection.execute(
            "INSERT INTO stage_results VALUES (?, ?, ?)",
            (stage, "t3_test", json.dumps(extraction())),
        )
    connection.commit()
    connection.close()

    with app.app_context():
        result = import_evaluation_data(source, results)
        assert result == {
            "intersection_records": 1,
            "model_outputs": 2,
            "new_submissions": 1,
            "new_outputs": 2,
        }
        submission = db.session.query(Submission).one()
        assert submission.title == "A title"
        assert not hasattr(submission, "author")
        outputs = db.session.query(ModelOutput).all()
        assert len(outputs) == 2
        assert {output.model_key for output in outputs} == {"stage1", "stage3"}
        assert all("model_fingerprint" not in output.extraction for output in outputs)
        assert len({output.blinded_id for output in outputs}) == 2

        repeated = import_evaluation_data(source, results)
        assert repeated["new_submissions"] == 0
        assert repeated["new_outputs"] == 0
