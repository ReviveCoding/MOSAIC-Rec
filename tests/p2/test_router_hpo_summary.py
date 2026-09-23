from __future__ import annotations

import pytest

from mosaic_rec.p2.router_hpo_summary import router_objective


def test_router_objective_is_worst_of_frozen_cold_and_warm_terms() -> None:
    result = router_objective(
        cold=0.012,
        warm=0.095,
        cold_comparator=0.010,
        warm_comparator=0.100,
    )
    assert result["cold_gate_term"] == pytest.approx(0.2)
    assert result["warm_gate_term"] == pytest.approx(0.05)
    assert result["router_selection_score"] == pytest.approx(0.05)


def test_router_objective_zero_cold_denominator_uses_absolute_delta() -> None:
    result = router_objective(
        cold=0.002,
        warm=0.1,
        cold_comparator=0.0,
        warm_comparator=0.1,
    )
    assert result["cold_gate_term"] == 0.002
    assert result["router_selection_score"] == 0.002
