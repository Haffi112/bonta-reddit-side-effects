"""Focused tests for the staged LLM extraction experiment."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

SCRIPT = Path(__file__).resolve().parents[1] / "data_analysis" / "02_llm_extraction_experiment.py"
SPEC = importlib.util.spec_from_file_location("llm_extraction_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def non_user_extraction() -> dict[str, object]:
    return {
        "a_self_use": False,
        "b_products_mentioned": [],
        "c_indication_category": "unknown",
        "c_indication_subtype": None,
        "d_side_effect_pts": [],
        "e1_post_overall_sentiment": "neutral",
        "e2_sentiment_about_botox": "neutral",
        "f_sentiment_about_side_effect": None,
        "g_treatment_area_freetext": None,
        "h_dose_units": None,
        "i_provider_type": None,
        "j_first_time_or_repeat": None,
        "j_prior_treatment_count": None,
        "k_time_to_onset": None,
        "l_duration_status": None,
        "m_sought_medical_attention": None,
        "n_discontinuation_intent": None,
        "o_age_range": None,
        "o_sex": None,
    }


def test_bonta_only_filter_requires_no_other_term():
    assert experiment.is_bonta_only({"_matched_terms": ["bonta"]})
    assert experiment.is_bonta_only({"_matched_terms": ["BONTA", "bonta"]})
    assert not experiment.is_bonta_only({"_matched_terms": ["bonta", "botox"]})
    assert not experiment.is_bonta_only({"_matched_terms": []})


def test_progressive_targets_are_exact_nested_deterministic_and_cumulative():
    ids = [f"t3_{number:04d}" for number in range(3_000)]
    order_a = experiment.seeded_order(ids, seed=17)
    order_b = experiment.seeded_order(ids, seed=17)
    first = experiment.progressive_targets(order_a, n=1_000)
    second = experiment.progressive_targets(order_b, n=2_000)

    assert order_a == order_b
    assert [len(first[key]) for key in ("stage1", "stage2", "stage3")] == [1_000, 100, 10]
    assert [len(second[key]) for key in ("stage1", "stage2", "stage3")] == [2_000, 200, 20]
    assert first["stage3"] < first["stage2"] < first["stage1"]
    assert first["stage1"] < second["stage1"]
    assert first["stage2"] < second["stage2"]
    assert first["stage3"] < second["stage3"]
    assert len(second["stage1"] - first["stage1"]) == 1_000
    assert len(second["stage2"] - first["stage2"]) == 100
    assert len(second["stage3"] - first["stage3"]) == 10


def test_record_id_adds_submission_prefix_only_when_needed():
    assert experiment.record_id({"id": "abc123"}) == "t3_abc123"
    assert experiment.record_id({"name": "t3_abc123"}) == "t3_abc123"


def test_cohen_kappa_handles_perfect_and_chance_corrected_agreement():
    assert experiment.cohen_kappa([True, False], [True, False]) == 1.0
    assert experiment.cohen_kappa([], []) is None
    assert experiment.cohen_kappa([True, True], [True, True]) == 1.0


def test_agreement_uses_pt_sets_not_verbatim_raw_spans():
    left = {
        "a_self_use": True,
        "b_products_mentioned": ["botox"],
        "d_side_effect_pts": [{"raw_span": "my eyelid drooped", "meddra_pt": "Ptosis"}],
    }
    right = {
        "a_self_use": True,
        "b_products_mentioned": ["botox"],
        "d_side_effect_pts": [{"raw_span": "droopy lid", "meddra_pt": "ptosis"}],
    }

    agreement = experiment.compare_extraction_pairs([(left, right)])

    assert agreement["fields"]["d_side_effect_pts"]["exact_agreement"] == 1.0
    assert agreement["fields"]["d_side_effect_pts"]["mean_jaccard"] == 1.0
    assert agreement["side_effect_presence"]["exact_agreement"] == 1.0
    assert agreement["full_record_exact_agreement"] == 1.0


def test_response_usage_reports_cost_cache_and_reasoning():
    measured = experiment.response_usage(
        {
            "usage": {
                "prompt_tokens": 1_500,
                "completion_tokens": 600,
                "cost": 0.0123,
                "prompt_tokens_details": {"cached_tokens": 1_000},
                "completion_tokens_details": {"reasoning_tokens": 400},
            }
        }
    )

    assert measured == {
        "prompt_tokens": 1_500,
        "completion_tokens": 600,
        "cached_tokens": 1_000,
        "reasoning_tokens": 400,
        "cost_usd": 0.0123,
    }

    aggregated = experiment.response_usage(
        {
            "usage": {"cost": 99},
            "accounting": {
                "prompt_tokens": 3_000,
                "completion_tokens": 700,
                "cached_tokens": 2_000,
                "reasoning_tokens": 500,
                "cost_usd": 0.045,
            },
        }
    )
    assert aggregated["cost_usd"] == 0.045
    assert aggregated["prompt_tokens"] == 3_000


def test_request_enables_response_healing_and_healing_metadata_is_normalized():
    stage = experiment.stage_configs("high", "low")["stage2"]
    payload = experiment.build_request_payload(stage, {"title": "x", "selftext": "y"})
    assert payload["plugins"] == [{"id": "response-healing"}]

    status = experiment.response_healing_status(
        {
            "openrouter_metadata": {
                "pipeline": [
                    {
                        "type": "response_healing",
                        "name": "response-healing",
                        "data": {"mode": "json_schema", "improved": True},
                    }
                ]
            }
        }
    )
    assert status["requested"] is True
    assert status["telemetry_available"] is True
    assert status["used"] is True
    assert status["improved"] is True
    assert len(status["pipeline_stages"]) == 1

    cached_status = experiment.response_healing_status({})
    assert cached_status["telemetry_available"] is False
    assert cached_status["used"] is None


def test_malformed_attempt_usage_is_retained_before_healed_success():
    bodies = [
        {
            "id": "bad",
            "model": experiment.DEEPSEEK_MODEL,
            "provider": "test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"a_self_use" false}'},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
            "openrouter_metadata": {"pipeline": []},
        },
        {
            "id": "healed",
            "model": experiment.DEEPSEEK_MODEL,
            "provider": "test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(non_user_extraction())},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "cost": 0.002},
            "openrouter_metadata": {
                "pipeline": [
                    {
                        "type": "response_healing",
                        "name": "response-healing",
                        "data": {"improved": True},
                    }
                ]
            },
        },
    ]

    async def run_call():
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=bodies.pop(0))

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://example.test"
        ) as client:
            return await experiment.call_openrouter(
                client,
                experiment.stage_configs("high", "low")["stage2"],
                "t3_test",
                {"title": "test", "selftext": "test"},
                max_retries=5,
                max_output_attempts=2,
            )

    result = asyncio.run(run_call())
    assert result.succeeded
    assert result.metadata["attempt_count"] == 2
    assert result.metadata["accounting"]["cost_usd"] == 0.003
    assert result.metadata["accounting"]["prompt_tokens"] == 20
    assert "response_preview" in result.metadata["attempts"][0]
    assert result.metadata["response_healing"]["used"] is True
    assert result.metadata["response_healing"]["improved"] is True


def test_output_failures_stop_at_configured_attempt_limit():
    calls = 0

    async def run_call():
        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={
                    "id": f"bad-{calls}",
                    "model": experiment.DEEPSEEK_MODEL,
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "{broken"}}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "cost": 0.001,
                    },
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://example.test"
        ) as client:
            return await experiment.call_openrouter(
                client,
                experiment.stage_configs("high", "low")["stage2"],
                "t3_test",
                {"title": "test", "selftext": "test"},
                max_retries=5,
                max_output_attempts=2,
            )

    result = asyncio.run(run_call())
    assert not result.succeeded
    assert calls == 2
    assert result.metadata["attempt_count"] == 2
    assert result.metadata["accounting"]["cost_usd"] == 0.002
    assert result.response_preview == "{broken"


def test_content_filter_is_preserved_as_terminal_without_retry():
    calls = 0

    async def run_call():
        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={
                    "id": "filtered",
                    "model": experiment.LUNA_MODEL,
                    "provider": "Azure",
                    "choices": [
                        {
                            "finish_reason": "content_filter",
                            "message": {
                                "content": "I'm sorry, but I cannot assist with that request."
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": 0,
                    },
                    "openrouter_metadata": {"pipeline": []},
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://example.test"
        ) as client:
            return await experiment.call_openrouter(
                client,
                experiment.stage_configs("high", "low")["stage3"],
                "t3_filtered",
                {"title": "test", "selftext": "test"},
                max_retries=5,
                max_output_attempts=2,
            )

    result = asyncio.run(run_call())
    assert not result.succeeded
    assert calls == 1
    assert result.failure_kind == "content_filter"
    assert result.terminal is True
    assert result.error == "ContentFilterError: provider blocked the response"
    assert result.metadata["attempt_count"] == 1
    assert result.metadata["finish_reason"] == "content_filter"


def test_async_main_continues_after_retryable_stage_failure(
    tmp_path: Path, monkeypatch
):
    visited: list[str] = []

    async def fake_run_stage(*args, **_kwargs):
        visited.append(args[2].key)
        return 0, 1

    monkeypatch.setattr(experiment, "run_stage", fake_run_stage)
    monkeypatch.setattr(experiment, "export_stage", lambda *_args: 0)
    monkeypatch.setattr(experiment, "export_stage_failures", lambda *_args: 1)
    monkeypatch.setattr(experiment, "write_run_summary", lambda *_args: {})
    monkeypatch.setattr(experiment, "write_agreement_summary", lambda *_args: {})

    connection = experiment.open_database(tmp_path)
    args = SimpleNamespace(concurrency=1, stage="all", results_dir=tmp_path)
    settings = experiment.Settings(openrouter_api_key="test")
    targets = {stage: {f"t3_{stage}"} for stage in ("stage1", "stage2", "stage3")}
    try:
        asyncio.run(
            experiment.async_main(
                args,
                settings,
                experiment.stage_configs("high", "low"),
                targets,
                connection,
            )
        )
    finally:
        connection.close()

    assert visited == ["stage1", "stage2", "stage3"]


def test_open_database_migrates_and_preserves_legacy_failure(tmp_path: Path):
    database = tmp_path / "private_extraction.sqlite3"
    legacy = sqlite3.connect(database)
    legacy.execute(
        """
        CREATE TABLE stage_failures (
            stage TEXT NOT NULL,
            record_id TEXT NOT NULL,
            error TEXT NOT NULL,
            status_code INTEGER,
            attempted_at TEXT NOT NULL,
            PRIMARY KEY (stage, record_id)
        )
        """
    )
    legacy.execute(
        "INSERT INTO stage_failures VALUES (?, ?, ?, ?, ?)",
        ("stage2", "t3_bad", "bad JSON", 200, "2026-01-01T00:00:00+00:00"),
    )
    legacy.commit()
    legacy.close()

    migrated = experiment.open_database(tmp_path)
    columns = {row[1] for row in migrated.execute("PRAGMA table_info(stage_failures)")}
    history = migrated.execute(
        "SELECT record_id, response_preview, attempt_count FROM stage_failure_history"
    ).fetchone()
    migrated.close()

    assert {
        "metadata_json",
        "response_preview",
        "attempt_count",
        "failure_kind",
        "terminal",
    } <= columns
    assert history == ("t3_bad", None, 1)


def test_raw_span_metrics_measure_paraphrases_without_rejecting_them():
    extraction = experiment.ExtractedFields.model_validate(
        {
            "a_self_use": True,
            "b_products_mentioned": ["botox"],
            "c_indication_category": "cosmetic",
            "c_indication_subtype": "crows_feet",
            "d_side_effect_pts": [
                {"raw_span": "horrible bags", "meddra_pt": "Periorbital swelling"},
                {"raw_span": "bags under my eyes", "meddra_pt": "Periorbital bags"},
            ],
            "e1_post_overall_sentiment": "negative",
            "e2_sentiment_about_botox": "negative",
            "f_sentiment_about_side_effect": "distressed",
            "g_treatment_area_freetext": "crows feet",
            "h_dose_units": None,
            "i_provider_type": None,
            "j_first_time_or_repeat": None,
            "j_prior_treatment_count": None,
            "k_time_to_onset": "1_2w",
            "l_duration_status": "ongoing",
            "m_sought_medical_attention": None,
            "n_discontinuation_intent": None,
            "o_age_range": "30-39",
            "o_sex": None,
        }
    )
    source = {"title": "Baggy eyes", "selftext": "I get horrible bags when I smile."}

    assert experiment.raw_span_metrics(extraction, source) == {
        "total_raw_spans": 2,
        "verbatim_raw_spans": 1,
        "all_raw_spans_verbatim": False,
        "raw_span_verbatim_rate": 0.5,
    }
