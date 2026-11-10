from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from .fields import canonical_decision_key, field_label
from .importer import DEFAULT_STAGES
from .models import Annotation, Annotator, ModelOutput, db


def _percent(numerator: int | float, denominator: int | float) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def fleiss_kappa(ratings: list[list[str]]) -> float | None:
    """Fleiss' kappa for complete, fixed-size binary rating rows."""
    if not ratings or len(ratings[0]) < 2:
        return None
    n = len(ratings[0])
    if any(len(row) != n for row in ratings):
        raise ValueError("Fleiss' kappa requires the same number of ratings per unit")
    counts = [
        (sum(value == "correct" for value in row), sum(value == "incorrect" for value in row))
        for row in ratings
    ]
    observed = sum((yes * yes + no * no - n) / (n * (n - 1)) for yes, no in counts) / len(counts)
    all_ratings = len(counts) * n
    p_yes = sum(yes for yes, _ in counts) / all_ratings
    p_no = 1.0 - p_yes
    expected = p_yes * p_yes + p_no * p_no
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else None
    return (observed - expected) / (1.0 - expected)


def dashboard_statistics() -> dict[str, Any]:
    users = db.session.scalars(select(Annotator).order_by(Annotator.created_at)).all()
    outputs = db.session.scalars(
        select(ModelOutput).where(ModelOutput.model_key.in_(DEFAULT_STAGES))
    ).all()
    output_ids_by_submission: dict[int, set[int]] = defaultdict(set)
    for output in outputs:
        output_ids_by_submission[output.submission_id].add(output.id)
    eligible_outputs = {
        submission_id: output_ids
        for submission_id, output_ids in output_ids_by_submission.items()
        if len(output_ids) == len(DEFAULT_STAGES)
    }
    total_items = sum(len(output_ids) for output_ids in eligible_outputs.values())
    total_comparisons = len(eligible_outputs)

    annotations = db.session.scalars(
        select(Annotation)
        .join(Annotation.model_output)
        .options(joinedload(Annotation.model_output))
        .where(ModelOutput.model_key.in_(DEFAULT_STAGES))
    ).all()
    reviewed_by_user_submission: dict[tuple[int, int], set[int]] = defaultdict(set)
    forms_by_user: dict[int, int] = defaultdict(int)
    for annotation in annotations:
        output = annotation.model_output
        if output.submission_id in eligible_outputs:
            reviewed_by_user_submission[(annotation.annotator_id, output.submission_id)].add(
                output.id
            )
            forms_by_user[annotation.annotator_id] += 1
    progress = []
    for user in users:
        completed = sum(
            reviewed_by_user_submission.get((user.id, submission_id), set())
            == expected_output_ids
            for submission_id, expected_output_ids in eligible_outputs.items()
        )
        progress.append(
            {
                "user": user,
                "completed": completed,
                "total": total_comparisons,
                "forms_completed": forms_by_user[user.id],
                "percent": _percent(completed, total_comparisons),
            }
        )
    model_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    field_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for annotation in annotations:
        output = annotation.model_output
        model_bucket = model_totals[(output.model_key, output.model_label)]
        for exact_key, decision in annotation.decisions.items():
            if decision not in {"correct", "incorrect"}:
                continue
            correct = int(decision == "correct")
            model_bucket[0] += correct
            model_bucket[1] += 1
            canonical = canonical_decision_key(exact_key)
            field_totals[canonical][0] += correct
            field_totals[canonical][1] += 1

    model_accuracy = [
        {
            "key": key,
            "label": label,
            "correct": counts[0],
            "ratings": counts[1],
            "percent": _percent(counts[0], counts[1]),
        }
        for (key, label), counts in sorted(model_totals.items())
    ]

    active_ids = {user.id for user in users if user.is_active}
    units: dict[tuple[int, str], dict[int, str]] = defaultdict(dict)
    for annotation in annotations:
        if annotation.annotator_id not in active_ids:
            continue
        for exact_key, decision in annotation.decisions.items():
            if decision in {"correct", "incorrect"}:
                units[(annotation.model_output_id, exact_key)][annotation.annotator_id] = decision

    complete_by_field: dict[str, list[list[str]]] = defaultdict(list)
    if len(active_ids) >= 2:
        ordered_ids = sorted(active_ids)
        for (_output_id, exact_key), decisions in units.items():
            if set(decisions) == active_ids:
                complete_by_field[canonical_decision_key(exact_key)].append(
                    [decisions[user_id] for user_id in ordered_ids]
                )

    all_field_keys = set(field_totals) | set(complete_by_field)
    field_agreement = []
    for key in sorted(all_field_keys):
        ratings = complete_by_field.get(key, [])
        pair_total = 0
        pair_agree = 0
        unanimous = 0
        for row in ratings:
            unanimous += int(len(set(row)) == 1)
            for left, right in combinations(row, 2):
                pair_total += 1
                pair_agree += int(left == right)
        correct, total = field_totals.get(key, [0, 0])
        field_agreement.append(
            {
                "key": key,
                "label": field_label(key),
                "complete_units": len(ratings),
                "accuracy": _percent(correct, total),
                "unanimous": _percent(unanimous, len(ratings)),
                "pairwise": _percent(pair_agree, pair_total),
                "kappa": fleiss_kappa(ratings),
            }
        )

    return {
        "users": users,
        "progress": progress,
        "total_items": total_items,
        "total_comparisons": total_comparisons,
        "total_annotations": len(annotations),
        "active_annotators": len(active_ids),
        "model_accuracy": model_accuracy,
        "field_agreement": field_agreement,
    }
