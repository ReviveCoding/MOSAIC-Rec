from __future__ import annotations

import pytest
import torch

from mosaic_rec.p2.runtime import exact_metric_summary


def test_exact_metric_summary_masks_history_and_preserves_repeat_target() -> None:
    item_ids = ["i1", "i2", "i3"]
    queries = {
        "user_id": ["u1"],
        "query_timestamp": [10],
        "target_item_id": ["i2"],
        "is_controlled_zero": [True],
        "is_natural_zero": [False],
        "is_warm": [False],
    }
    histories = {"u1": [1, 2]}

    def scores(_: list[str]) -> torch.Tensor:
        return torch.tensor([[100.0, 5.0, 4.0]])

    result = exact_metric_summary(
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=torch.tensor([1, 1, 1]),
        score_batch=scores,
        batch_size=1,
    )
    assert result["all"]["ndcg_at_20"] == pytest.approx(1.0)
    assert result["controlled_zero"]["n"] == 1
