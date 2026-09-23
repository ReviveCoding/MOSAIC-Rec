from __future__ import annotations

import numpy as np
import pytest

from mosaic_rec.p2.analysis import (
    cluster_percentile_bootstrap,
    non_accuracy_metrics,
    paired_percentile_bootstrap,
    pairwise_behavior,
    ranking_from_rank,
)


def test_ranking_and_pairwise_known_answer() -> None:
    ranks = np.array([1, 20, 21])
    metrics = ranking_from_rank(ranks)
    assert metrics["recall_at_20"].tolist() == [1.0, 1.0, 0.0]
    assert metrics["precision_at_20"].tolist() == [0.05, 0.05, 0.0]
    result = pairwise_behavior(np.array([1, 30, 9]), np.array([2, 10, 9]))
    assert result["win_rate"] == 1 / 3
    assert result["regression_rate"] == 1 / 3
    assert result["tie_rate"] == 1 / 3


def test_bootstraps_are_deterministic_and_paired() -> None:
    candidate = np.array([1.0, 0.0, 0.5, 0.5])
    comparator = np.array([0.0, 0.0, 0.0, 0.5])
    first = paired_percentile_bootstrap(candidate, comparator, draws=100, seed=7)
    second = paired_percentile_bootstrap(candidate, comparator, draws=100, seed=7)
    assert first == second
    clustered = cluster_percentile_bootstrap(
        candidate, comparator, ["a", "a", "b", "c"], draws=100, seed=8
    )
    assert clustered["clusters"] == 3
    assert clustered["mean_delta"] == 0.375


def test_non_accuracy_known_answer() -> None:
    top = [["a", "b"], ["a", "c"]]
    result = non_accuracy_metrics(
        top,
        user_ids=["u1", "u2"],
        eligible_item_ids=["a", "b", "c", "d"],
        fit_counts={"a": 3, "b": 1, "c": 0, "d": 0},
        tail_items={"c", "d"},
        pair_count=1,
    )
    assert result["item_coverage_at_20"] == 0.75
    assert result["tail_percentage_at_20"] == 0.25
    assert result["personalization"] == pytest.approx(2 / 3)
