"""Frozen p2 primary-gate analysis over immutable LOCKED_FINAL rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def percentile_bootstrap(
    differences: np.ndarray, *, draws: int, seed: int
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires a finite nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    chunk = 256
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        estimates[start : start + count] = values[indices].mean(axis=1)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def item_cluster_bootstrap(
    differences: np.ndarray,
    item_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    items = np.asarray(item_ids)
    unique, inverse = np.unique(items, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    chunk = 512
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        sampled = rng.integers(0, unique.size, size=(count, unique.size))
        estimates[start : start + count] = sums[sampled].sum(axis=1) / counts[
            sampled
        ].sum(axis=1)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "clusters": int(unique.size),
        "mean": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _load(path: Path) -> dict[str, np.ndarray]:
    columns = [
        "user_id",
        "query_timestamp",
        "target_item_id",
        "ndcg_at_20",
        "recall_at_20",
        "mrr_at_20",
        "hit_at_20",
        "precision_at_20",
        "is_controlled_zero",
        "is_natural_zero",
        "is_warm",
    ]
    table = pq.read_table(path, columns=columns)
    return {name: table[name].to_numpy(zero_copy_only=False) for name in columns}


def _assert_paired(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    for column in ("user_id", "query_timestamp", "target_item_id"):
        if not np.array_equal(left[column], right[column]):
            raise RuntimeError(f"Locked outputs are not paired on {column}")


def analyze_locked(
    *,
    cseq_path: Path,
    ccold_path: Path,
    router_path: Path,
    margin: float,
    draws: int = 10_000,
    user_seed: int = 2026092301,
    item_seed: int = 2026092302,
) -> dict[str, Any]:
    models = {
        "C_seq": _load(cseq_path),
        "C_cold": _load(ccold_path),
        "R4": _load(router_path),
    }
    _assert_paired(models["R4"], models["C_seq"])
    _assert_paired(models["R4"], models["C_cold"])
    masks = {
        "overall": np.ones(models["R4"]["user_id"].size, dtype=np.bool_),
        "controlled_zero": models["R4"]["is_controlled_zero"].astype(bool),
        "natural_zero": models["R4"]["is_natural_zero"].astype(bool),
        "warm": models["R4"]["is_warm"].astype(bool),
    }
    metrics: dict[str, Any] = {}
    for model_id, data in models.items():
        metrics[model_id] = {}
        for population, mask in masks.items():
            metrics[model_id][population] = {
                "n": int(mask.sum()),
                **{
                    name: float(data[name][mask].mean())
                    for name in (
                        "ndcg_at_20",
                        "recall_at_20",
                        "mrr_at_20",
                        "hit_at_20",
                        "precision_at_20",
                    )
                },
            }

    cold = masks["controlled_zero"]
    cold_delta = models["R4"]["ndcg_at_20"][cold] - models["C_cold"]["ndcg_at_20"][cold]
    h1_user = percentile_bootstrap(cold_delta, draws=draws, seed=user_seed)
    h1_item = item_cluster_bootstrap(
        cold_delta,
        models["R4"]["target_item_id"][cold],
        draws=draws,
        seed=item_seed,
    )
    warm = masks["warm"]
    warm_delta = models["R4"]["ndcg_at_20"][warm] - models["C_seq"]["ndcg_at_20"][warm]
    h2_user = percentile_bootstrap(warm_delta, draws=draws, seed=user_seed)
    h1_pass = h1_user["lower_95"] > 0.0
    h2_pass = h2_user["lower_95"] > -float(margin)
    return {
        "metrics": metrics,
        "H1": {
            "comparison": "R4_minus_C_cold",
            "population": "CONTROLLED_ZERO_SUPPORT",
            "metric": "NDCG@20",
            "paired_user_bootstrap": h1_user,
            "item_cluster_sensitivity": h1_item,
            "rule": "paired_user_lower_95 > 0",
            "pass": h1_pass,
        },
        "H2": {
            "comparison": "R4_minus_C_seq",
            "population": "FIT_support_gt_20",
            "metric": "NDCG@20",
            "margin": float(margin),
            "paired_user_bootstrap": h2_user,
            "rule": "paired_user_lower_95 > -margin",
            "pass": h2_pass,
        },
        "promotion": {
            "rule": "H1_PASS_AND_H2_PASS",
            "pass": bool(h1_pass and h2_pass),
        },
    }
