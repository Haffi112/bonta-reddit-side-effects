from __future__ import annotations

import pytest

from eval_app.stats import fleiss_kappa


def test_fleiss_kappa_perfect_agreement():
    ratings = [
        ["correct", "correct", "correct"],
        ["incorrect", "incorrect", "incorrect"],
    ]
    assert fleiss_kappa(ratings) == pytest.approx(1.0)


def test_fleiss_kappa_rejects_ragged_rows():
    with pytest.raises(ValueError):
        fleiss_kappa([["correct", "correct"], ["incorrect"]])
