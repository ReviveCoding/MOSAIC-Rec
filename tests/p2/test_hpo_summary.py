from __future__ import annotations

from mosaic_rec.p2.hpo_summary import select_family


def _receipt(trial: str, overall: float, warm: float, cold: float) -> dict:
    return {
        "trial_id": trial,
        "status": "PASS",
        "checkpoint": f"{trial}.pt",
        "checkpoint_sha256": trial,
        "validation": {
            "slices": {
                "all": {"ndcg_at_20": overall},
                "warm": {"ndcg_at_20": warm},
                "controlled_zero": {"ndcg_at_20": cold},
            }
        },
    }


def test_selection_uses_frozen_metric_and_tie_order() -> None:
    receipts = [
        _receipt("T3", 0.2, 0.1, 0.3),
        _receipt("T2", 0.2, 0.2, 0.1),
        _receipt("T1", 0.2, 0.2, 0.1),
    ]
    assert select_family(receipts)["selected_trial_id"] == "T1"


def test_selection_preserves_unqualified_family() -> None:
    assert select_family([{"trial_id": "T1", "status": "FAIL"}]) == {
        "status": "UNQUALIFIED_NO_PASS",
        "selected_trial_id": None,
    }
