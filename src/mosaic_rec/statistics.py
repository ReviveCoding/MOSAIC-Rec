"""Predeclared paired cluster uncertainty for full-catalog ranking outcomes.

This module operates on per-example metrics created by a frozen evaluator. It
does not select models, change cohorts, or access protected outcomes itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ClusterInterval:
    point: float
    lower: float
    upper: float
    draws: int
    clusters: int
    cluster_kind: Literal["user", "item"]
    seed: int


def _validate(model: np.ndarray, comparator: np.ndarray,
              cluster_ids: np.ndarray, draws: int) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(model, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    if model.ndim != 1 or comparator.ndim != 1 or cluster_ids.ndim != 1:
        raise ValueError("Scores and cluster IDs must be one-dimensional")
    if not (len(model) == len(comparator) == len(cluster_ids)) or not len(model):
        raise ValueError("Paired scores and cluster IDs must have the same nonzero length")
    if not np.isfinite(model).all() or not np.isfinite(comparator).all():
        raise ValueError("Scores must be finite")
    if np.any((model < 0) | (model > 1) | (comparator < 0) | (comparator > 1)):
        raise ValueError("Ranking metrics must lie in [0, 1]")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    return model - comparator, inverse


def paired_cluster_interval(model: np.ndarray, comparator: np.ndarray,
                            cluster_ids: np.ndarray, *,
                            cluster_kind: Literal["user", "item"],
                            draws: int = 10_000, seed: int = 20260920,
                            chunk_size: int = 128) -> ClusterInterval:
    """Percentile CI for a paired, example-weighted difference by cluster.

    Each resampled cluster contributes all of its examples. For the primary
    one-event-per-user evaluator this is the paired user-cluster bootstrap.
    Item clustering is the cold-item sensitivity analysis. The point estimate
    and every replicate use the same example-weighted denominator.
    """
    if cluster_kind not in ("user", "item"):
        raise ValueError("cluster_kind must be user or item")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    diff, inverse = _validate(model, comparator, cluster_ids, draws)
    n_clusters = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=diff, minlength=n_clusters)
    counts = np.bincount(inverse, minlength=n_clusters)
    point = float(diff.mean())
    rng = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        sampled = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        replicates[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return ClusterInterval(point, float(lower), float(upper), draws,
                           n_clusters, cluster_kind, seed)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm familywise adjusted p values in original hypothesis order."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or not len(p_values):
        raise ValueError("p_values must be a nonempty vector")
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be finite probabilities")
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    adjusted = np.maximum.accumulate(ordered * np.arange(len(ordered), 0, -1))
    out = np.empty_like(p_values)
    out[order] = np.clip(adjusted, 0, 1)
    return out
