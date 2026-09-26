from __future__ import annotations

from mosaic_rec.p2.ablations import ablation_overrides


def test_e04_ablation_overrides_cover_frozen_family() -> None:
    assert [ablation_overrides(f"A{index}") for index in range(8)] == [
        {},
        {"image_enabled": False},
        {"text_enabled": False},
        {"gate_feature_indices_removed": [1]},
        {"gate_feature_indices_removed": [2, 3, 4, 5]},
        {"distillation_weight": 0.0},
        {"hard_zero_support_override": False},
        {"router_type": "hard_zero_support", "retrain": False},
    ]
