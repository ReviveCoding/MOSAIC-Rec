from __future__ import annotations

import numpy as np
import pytest
import torch

from mosaic_rec.p2.ablations import ablation_overrides, remove_gate_features
from mosaic_rec.p2.robustness import (
    apply_missingness,
    deterministic_item_mask,
    perturb_content_features,
)


def test_missingness_is_deterministic_and_joint_uses_one_mask() -> None:
    items = [f"i{index}" for index in range(200)]
    mask = deterministic_item_mask(
        items, model_id="R4", role="LOCKED_FINAL", modality="image", intensity=0.25
    )
    assert np.array_equal(
        mask,
        deterministic_item_mask(
            items,
            model_id="R4",
            role="LOCKED_FINAL",
            modality="image",
            intensity=0.25,
        ),
    )
    image = np.ones((200, 3), dtype=np.float32)
    text = np.ones((200, 2), dtype=np.float32)
    present = np.ones(200, dtype=np.bool_)
    image_out, text_out, image_flag, text_flag = apply_missingness(
        image,
        text,
        present,
        present,
        item_ids=items,
        model_id="R4",
        role="LOCKED_FINAL",
        intervention="IMAGE_TEXT_JOINT_MISSING",
        intensity=0.25,
    )
    assert np.array_equal(image_flag, text_flag)
    assert np.array_equal(image_out.sum(axis=1) == 0, text_out.sum(axis=1) == 0)


def test_feature_perturbation_is_deterministic_and_normalized() -> None:
    values = np.ones((3, 4), dtype=np.float32)
    kwargs = {
        "item_ids": ["a", "b", "c"],
        "modality": "image",
        "intensity": 0.1,
        "fit_standard_deviation": np.ones(4, dtype=np.float32),
    }
    first = perturb_content_features(values, **kwargs)
    second = perturb_content_features(values, **kwargs)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_ablation_feature_removal_and_unknown_id() -> None:
    features = torch.arange(16).reshape(2, 8)
    assert remove_gate_features(features, "A3").shape[-1] == 7
    assert remove_gate_features(features, "A4").shape[-1] == 4
    assert ablation_overrides("A7")["retrain"] is False
    with pytest.raises(ValueError, match="Unknown ablation"):
        ablation_overrides("A8")
