from __future__ import annotations

import pytest

from mosaic_rec.p2.alignment_summary import alignment_score


def test_alignment_score_uses_comparator_units_and_warm_margin() -> None:
    result = alignment_score(
        cold=0.02,
        warm=0.09,
        cold_comparator=0.01,
        warm_comparator=0.10,
    )
    assert result["cold_unit"] == 2.0
    assert result["warm_unit"] == pytest.approx(0.0)
    assert result["selection_score"] == pytest.approx(1.0)
    assert result["warm_margin"] == pytest.approx(0.01)


def test_alignment_score_zero_cold_denominator_uses_absolute_delta() -> None:
    result = alignment_score(
        cold=0.02,
        warm=0.10,
        cold_comparator=0.0,
        warm_comparator=0.10,
    )
    assert result["cold_unit"] == 0.02
    assert result["warm_unit"] == 1.0
