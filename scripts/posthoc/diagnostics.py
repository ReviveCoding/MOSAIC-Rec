"""Pure helpers for MOSAIC-Rec read-only industry diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from common import BOOTSTRAP_SEED_BASE

METRICS = ("ndcg_at_20", "recall_at_20", "mrr_at_20", "hit_at_20", "precision_at_20")
IDENTITY = ("user_id", "query_timestamp", "target_item_id")


def support_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 50:
        return "21-50"
    return ">50"


def age_bucket(value: float) -> str:
    if value < 1:
        return "<1 day"
    if value <= 7:
        return "1-7"
    if value <= 30:
        return "8-30"
    if value <= 90:
        return "31-90"
    if value <= 365:
        return "91-365"
    return ">365"


def history_bucket(value: int) -> str:
    if value <= 5:
        return "1-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 50:
        return "21-50"
    return ">50"


def recency_bucket(value: float) -> str:
    if value <= 1:
        return "<=1 day"
    if value <= 7:
        return "1-7"
    if value <= 30:
        return "8-30"
    if value <= 90:
        return "31-90"
    return ">90"


def modality_bucket(image: bool, text: bool) -> str:
    if image and text:
        return "image+text"
    if image:
        return "image-only"
    if text:
        return "text-only"
    return "neither"


def quantile_edges(values: np.ndarray, groups: int = 4) -> list[float]:
    return [
        float(value)
        for value in np.quantile(values.astype(float), np.linspace(0, 1, groups + 1))
    ]


def quantile_label(value: float, edges: list[float]) -> str:
    index = int(np.searchsorted(np.asarray(edges[1:-1]), value, side="right")) + 1
    return f"Q{index}"


def temporal_quartiles(values: np.ndarray) -> tuple[list[int], np.ndarray]:
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.size, dtype=np.int8)
    for quartile, indices in enumerate(np.array_split(order, 4), start=1):
        labels[indices] = quartile
    boundaries = [int(values[labels == quartile].max()) for quartile in (1, 2, 3)]
    return boundaries, labels


def deterministic_seed(namespace: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED_BASE}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def paired_bootstrap(
    values: np.ndarray, namespace: str, draws: int = 10_000
) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("paired bootstrap requires finite nonempty vector")
    unique, counts = np.unique(vector, return_counts=True)
    rng = np.random.default_rng(deterministic_seed(namespace))
    estimates = np.empty(draws, dtype=np.float64)
    probabilities = counts / counts.sum()
    chunk = 512
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        sampled = rng.multinomial(vector.size, probabilities, size=count)
        estimates[start : start + count] = sampled @ unique / vector.size
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "point_estimate": float(vector.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "n": int(vector.size),
    }


def assert_paired(frames: Iterable[pd.DataFrame]) -> None:
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    reference = frames[0].loc[:, IDENTITY].reset_index(drop=True)
    if reference.duplicated().any():
        raise ValueError("duplicate query identity")
    for frame in frames[1:]:
        if not reference.equals(frame.loc[:, IDENTITY].reset_index(drop=True)):
            raise ValueError("persisted model rows are not paired")


def win_tie_loss(delta: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(delta)
    positive = int((values > 0).sum())
    zero = int((values == 0).sum())
    negative = int((values < 0).sum())
    total = values.size
    if total == 0:
        return {
            "n": 0,
            "win_n": 0,
            "tie_n": 0,
            "loss_n": 0,
            "win_rate": 0.0,
            "tie_rate": 0.0,
            "loss_rate": 0.0,
        }
    return {
        "n": total,
        "win_n": positive,
        "tie_n": zero,
        "loss_n": negative,
        "win_rate": positive / total,
        "tie_rate": zero / total,
        "loss_rate": negative / total,
    }


def jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def oracle_values(
    sequence: np.ndarray, content: np.ndarray, router: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    oracle = np.maximum(np.asarray(sequence), np.asarray(content))
    return oracle, oracle - np.asarray(router)


def concentration(values: np.ndarray) -> dict[str, float]:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0:
        return {
            "top_1": 0.0,
            "top_5": 0.0,
            "top_10": 0.0,
            "top_1pct": 0.0,
            "top_5pct": 0.0,
            "top_10pct": 0.0,
        }
    ordered = np.sort(positive)[::-1]
    total = ordered.sum()

    def share(count: int) -> float:
        return float(ordered[: min(count, ordered.size)].sum() / total)

    return {
        "top_1": share(1),
        "top_5": share(5),
        "top_10": share(10),
        "top_1pct": share(max(1, int(np.ceil(0.01 * ordered.size)))),
        "top_5pct": share(max(1, int(np.ceil(0.05 * ordered.size)))),
        "top_10pct": share(max(1, int(np.ceil(0.10 * ordered.size)))),
    }


def gini(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=float)
    vector = vector[vector >= 0]
    if vector.size == 0 or vector.sum() == 0:
        return 0.0
    vector = np.sort(vector)
    n = vector.size
    return float((2 * np.dot(np.arange(1, n + 1), vector) / vector.sum() - n - 1) / n)


def validate_output_path(path: Path, repo: Path) -> None:
    resolved = path.resolve()
    allowed = [
        (repo / "scripts/posthoc").resolve(),
        (repo / "analysis/posthoc").resolve(),
        (repo / "reports/posthoc").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise PermissionError(f"Posthoc output outside allowed roots: {path}")


def descriptive_classification(
    effect: float, *, strong: float = 0.01, weak: float = 0.001
) -> str:
    magnitude = abs(effect)
    if magnitude >= strong:
        return "SUPPORTED_ASSOCIATION"
    if magnitude >= weak:
        return "WEAK_ASSOCIATION"
    return "NOT_SUPPORTED"
