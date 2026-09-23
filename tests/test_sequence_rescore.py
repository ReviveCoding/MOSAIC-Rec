from __future__ import annotations

import pytest

from mosaic_rec.score_sequence_validation import original_all_metric


def test_original_all_metric_accepts_legacy_and_sliced_receipts() -> None:
    assert original_all_metric({"ndcg_at_20": 0.125}) == 0.125
    assert (
        original_all_metric(
            {"validation_slices": {"all": {"ndcg_at_20": 0.25, "n": 4}}}
        )
        == 0.25
    )


def test_original_all_metric_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="no overall NDCG"):
        original_all_metric({"validation_slices": {}})
