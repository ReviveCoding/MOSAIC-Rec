"""Frozen executable definitions for the p2 A0-A7 ablation family."""

from __future__ import annotations

from typing import Any

import torch


def ablation_overrides(ablation_id: str) -> dict[str, Any]:
    definitions = {
        "A0": {},
        "A1": {"image_enabled": False},
        "A2": {"text_enabled": False},
        "A3": {"gate_feature_indices_removed": [1]},
        "A4": {"gate_feature_indices_removed": [2, 3, 4, 5]},
        "A5": {"distillation_weight": 0.0},
        "A6": {"hard_zero_support_override": False},
        "A7": {"router_type": "hard_zero_support", "retrain": False},
    }
    try:
        return definitions[ablation_id].copy()
    except KeyError as error:
        raise ValueError(f"Unknown ablation: {ablation_id}") from error


def remove_gate_features(features: torch.Tensor, ablation_id: str) -> torch.Tensor:
    removed = ablation_overrides(ablation_id).get("gate_feature_indices_removed", [])
    if not removed:
        return features
    retained = [index for index in range(features.shape[-1]) if index not in removed]
    return features[..., retained]
