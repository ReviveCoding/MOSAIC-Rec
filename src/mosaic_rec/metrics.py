"""Exact single-positive rank metrics used by protocol evaluations."""

from __future__ import annotations

from collections.abc import Iterable
from math import log2


def single_positive_metrics(
    ranks: Iterable[int | None], k: int = 20
) -> dict[str, float | int]:
    """Aggregate one held-out positive per query; ranks are 1-based, None is missed."""
    if k < 1:
        raise ValueError("k must be positive")
    values = list(ranks)
    if not values:
        raise ValueError("no queries")
    if any(rank is not None and rank < 1 for rank in values):
        raise ValueError("ranks must be 1-based")
    hits = [rank for rank in values if rank is not None and rank <= k]
    n = len(values)
    return {
        "n": n,
        "ndcg_at_k": sum(1 / log2(rank + 1) for rank in hits) / n,
        "recall_at_k": len(hits) / n,
        "mrr_at_k": sum(1 / rank for rank in hits) / n,
        "hit_rate_at_k": len(hits) / n,
    }
