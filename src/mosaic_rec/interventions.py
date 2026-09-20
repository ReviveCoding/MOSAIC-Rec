"""Deterministic embedding-level interventions for scoped robustness tests."""

from __future__ import annotations

import numpy as np


def mismatched_image_index(n_items: int, seed: int) -> np.ndarray:
    """Map each item to a different item without changing vector values.

    A seeded random order is rotated by one position, making a single cycle.
    This tests mismatched cached image embeddings; it is not pixel corruption.
    """
    if n_items < 2:
        raise ValueError("At least two items are required for a mismatch intervention")
    order = np.random.default_rng(seed).permutation(n_items)
    mapping = np.empty(n_items, dtype=np.int64)
    mapping[order] = np.roll(order, 1)
    if np.any(mapping == np.arange(n_items)):
        raise AssertionError("Mismatch mapping contains a fixed point")
    return mapping


def apply_mismatched_images(
    features: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a separate cached-feature array and the item-to-source map."""
    features = np.asarray(features)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Image features must be a finite matrix")
    mapping = mismatched_image_index(features.shape[0], seed)
    return features[mapping].copy(), mapping
