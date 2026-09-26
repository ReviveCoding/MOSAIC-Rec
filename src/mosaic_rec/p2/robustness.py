"""Frozen deterministic p2 robustness transformations."""

from __future__ import annotations

import hashlib

import numpy as np

from mosaic_rec.p2 import PROTOCOL_ID


def deterministic_item_mask(
    item_ids: list[str],
    *,
    model_id: str,
    role: str,
    modality: str,
    intensity: float,
    base_seed: int = 2026092303,
) -> np.ndarray:
    threshold = int(float(intensity) * (1 << 256))
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{PROTOCOL_ID}:{model_id}:{role}:{modality}:{intensity}:{item_id}:{base_seed}".encode()
                ).digest(),
                "big",
            )
            < threshold
            for item_id in item_ids
        ],
        dtype=np.bool_,
    )


def apply_missingness(
    image: np.ndarray,
    text: np.ndarray,
    image_present: np.ndarray,
    text_present: np.ndarray,
    *,
    item_ids: list[str],
    model_id: str,
    role: str,
    intervention: str,
    intensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_out = np.array(image, copy=True)
    text_out = np.array(text, copy=True)
    image_flag = np.array(image_present, copy=True, dtype=np.bool_)
    text_flag = np.array(text_present, copy=True, dtype=np.bool_)
    if intervention == "IMAGE_MISSING":
        image_mask = deterministic_item_mask(
            item_ids,
            model_id=model_id,
            role=role,
            modality="image",
            intensity=intensity,
        )
        text_mask = np.zeros(len(item_ids), dtype=np.bool_)
    elif intervention == "TEXT_MISSING":
        image_mask = np.zeros(len(item_ids), dtype=np.bool_)
        text_mask = deterministic_item_mask(
            item_ids, model_id=model_id, role=role, modality="text", intensity=intensity
        )
    elif intervention == "IMAGE_TEXT_JOINT_MISSING":
        joint = deterministic_item_mask(
            item_ids,
            model_id=model_id,
            role=role,
            modality="joint",
            intensity=intensity,
        )
        image_mask = joint
        text_mask = joint
    else:
        raise ValueError(f"Unknown missingness intervention: {intervention}")
    image_out[image_mask] = 0
    text_out[text_mask] = 0
    image_flag[image_mask] = False
    text_flag[text_mask] = False
    return image_out, text_out, image_flag, text_flag


def perturb_content_features(
    values: np.ndarray,
    *,
    item_ids: list[str],
    modality: str,
    intensity: float,
    fit_standard_deviation: np.ndarray,
    base_seed: int = 2026092303,
) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    scale = np.asarray(fit_standard_deviation, dtype=np.float32) * float(intensity)
    for row, item_id in enumerate(item_ids):
        digest = hashlib.sha256(
            f"{base_seed}:{item_id}:{modality}:{intensity}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        output[row] += rng.normal(0.0, scale, size=output.shape[1]).astype(np.float32)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    return output / np.maximum(norms, np.finfo(np.float32).eps)
