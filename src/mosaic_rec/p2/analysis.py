"""Frozen p2 ranking, exposure, pairwise, and bootstrap calculations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np


def ranking_from_rank(rank: np.ndarray, cutoff: int = 20) -> dict[str, np.ndarray]:
    values = np.asarray(rank, dtype=np.int64)
    hit = values <= cutoff
    return {
        "ndcg_at_20": np.where(hit, 1.0 / np.log2(values + 1), 0.0),
        "recall_at_20": hit.astype(np.float64),
        "mrr_at_20": np.where(hit, 1.0 / values, 0.0),
        "hit_at_20": hit.astype(np.float64),
        "precision_at_20": hit.astype(np.float64) / cutoff,
    }


def paired_percentile_bootstrap(
    candidate: np.ndarray,
    comparator: np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    if candidate.shape != comparator.shape or candidate.ndim != 1:
        raise ValueError("Paired arrays must be one-dimensional with equal shape")
    if len(candidate) == 0 or not np.isfinite(candidate - comparator).all():
        raise ValueError("Paired bootstrap requires nonempty finite values")
    delta = candidate - comparator
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    # Bound transient memory while retaining exact deterministic resampling.
    block = 256
    for offset in range(0, draws, block):
        count = min(block, draws - offset)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        samples[offset : offset + count] = delta[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "n": len(delta),
        "draws": draws,
        "mean_delta": float(delta.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def cluster_percentile_bootstrap(
    candidate: np.ndarray,
    comparator: np.ndarray,
    clusters: Sequence[str],
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    cluster_values = np.asarray(clusters)
    if len(candidate) != len(comparator) or len(candidate) != len(cluster_values):
        raise ValueError("Cluster bootstrap inputs must align")
    unique, inverse = np.unique(cluster_values, return_inverse=True)
    sums = np.bincount(inverse, weights=candidate - comparator)
    counts = np.bincount(inverse)
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = generator.integers(0, len(unique), size=len(unique))
        samples[draw] = sums[selected].sum() / counts[selected].sum()
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "n": len(candidate),
        "clusters": len(unique),
        "draws": draws,
        "mean_delta": float((candidate - comparator).mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def pairwise_behavior(
    candidate_rank: np.ndarray, comparator_rank: np.ndarray, cutoff: int = 20
) -> dict[str, float | int]:
    candidate = np.asarray(candidate_rank, dtype=np.int64)
    comparator = np.asarray(comparator_rank, dtype=np.int64)
    if candidate.shape != comparator.shape:
        raise ValueError("Paired ranks must align")
    candidate_metric = ranking_from_rank(candidate, cutoff)["ndcg_at_20"]
    comparator_metric = ranking_from_rank(comparator, cutoff)["ndcg_at_20"]
    delta = candidate_metric - comparator_metric
    candidate_hit = candidate <= cutoff
    comparator_hit = comparator <= cutoff
    count = len(candidate)
    return {
        "n": count,
        "win_rate": float(np.mean(delta > 0)),
        "tie_rate": float(np.mean(delta == 0)),
        "loss_rate": float(np.mean(delta < 0)),
        "rescue_rate": float(np.mean(candidate_hit & ~comparator_hit)),
        "regression_rate": float(np.mean(~candidate_hit & comparator_hit)),
        "both_hit_rate": float(np.mean(candidate_hit & comparator_hit)),
        "both_miss_rate": float(np.mean(~candidate_hit & ~comparator_hit)),
    }


def deterministic_user_pairs(
    user_ids: Sequence[str], count: int
) -> list[tuple[int, int]]:
    ordered = sorted(
        range(len(user_ids)),
        key=lambda index: hashlib.sha256(str(user_ids[index]).encode()).digest(),
    )
    if len(ordered) < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for index in range(min(count, len(ordered) * (len(ordered) - 1) // 2)):
        left = ordered[index % len(ordered)]
        step = 1 + index // len(ordered)
        right = ordered[(index + step) % len(ordered)]
        if left != right:
            pairs.append((left, right))
    return pairs


def _gini(values: np.ndarray) -> float:
    if values.sum() == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float64))
    n = len(ordered)
    return float((2 * np.dot(np.arange(1, n + 1), ordered) / ordered.sum() - n - 1) / n)


def non_accuracy_metrics(
    top20: Sequence[Sequence[str]],
    *,
    user_ids: Sequence[str],
    eligible_item_ids: Sequence[str],
    fit_counts: Mapping[str, int],
    tail_items: set[str],
    representation: Mapping[str, np.ndarray] | None = None,
    pair_count: int = 10_000,
) -> dict[str, float | int | None]:
    eligible = list(eligible_item_ids)
    exposure = {item: 0 for item in eligible}
    popularity: list[float] = []
    novelty: list[float] = []
    tail = 0
    total_fit = sum(fit_counts.get(item, 0) for item in eligible)
    denominator = total_fit + len(eligible)
    for recommendations in top20:
        for item in recommendations:
            if item in exposure:
                exposure[item] += 1
            count = fit_counts.get(item, 0)
            popularity.append(math.log1p(count))
            novelty.append(-math.log2((count + 1) / denominator))
            tail += item in tail_items
    counts = np.asarray(list(exposure.values()), dtype=np.float64)
    shares = counts / counts.sum() if counts.sum() else counts
    nonzero = shares[shares > 0]
    pairs = deterministic_user_pairs(user_ids, pair_count)
    overlap = [
        len(set(top20[left]) & set(top20[right]))
        / len(set(top20[left]) | set(top20[right]))
        for left, right in pairs
    ]
    ild_values: list[float] = []
    if representation is not None:
        for recommendations in top20:
            vectors = [
                representation[item]
                for item in recommendations
                if item in representation
            ]
            if len(vectors) >= 2:
                matrix = np.stack(vectors)
                similarity = matrix @ matrix.T
                upper = similarity[np.triu_indices(len(matrix), k=1)]
                ild_values.append(float(np.mean(1.0 - upper)))
    return {
        "item_coverage_at_20": float(np.count_nonzero(counts) / len(eligible)),
        "average_popularity_at_20": float(np.mean(popularity)),
        "tail_percentage_at_20": float(tail / len(popularity)),
        "exposure_gini": _gini(counts),
        "exposure_hhi": float(np.square(shares).sum()),
        "exposure_entropy": float(-(nonzero * np.log(nonzero)).sum()),
        "novelty": float(np.mean(novelty)),
        "intra_list_diversity": float(np.mean(ild_values)) if ild_values else None,
        "personalization": float(1.0 - np.mean(overlap)) if overlap else None,
        "personalization_pairs": len(overlap),
    }
