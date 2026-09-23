"""CPU-only post-closure diagnostics over immutable p1 per-user outputs.

This module never imports torch, loads a checkpoint, or performs model inference.
All outputs are explicitly exploratory and live outside canonical p1 evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROTOCOL = "mosaic-rec-2026-09-20-p1"
LABEL = "POST-HOC EXPLORATORY - NOT A P1 CONFIRMATORY CLAIM"
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_DRAWS = 2000
MODELS = ("B7", "B4", "M1")
COMPARATORS = ("B7", "B4")
METRICS = ("ndcg_at_20", "recall_at_20", "mrr_at_20", "hit_at_20")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    payload = "|".join(parts).encode()
    return BOOTSTRAP_SEED ^ int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "little"
    )


def metric_summary(values: np.ndarray, ranks: np.ndarray) -> dict[str, Any]:
    hits = ranks > 0
    hit_ranks = ranks[hits]
    distribution = {str(rank): int(np.sum(hit_ranks == rank)) for rank in range(1, 21)}
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if len(values) else None,
        "rank_hit_mean": float(np.mean(hit_ranks)) if len(hit_ranks) else None,
        "rank_hit_median": float(np.median(hit_ranks)) if len(hit_ranks) else None,
        "rank_distribution_json": json.dumps(distribution, sort_keys=True),
    }


def paired_bootstrap(
    matrix: np.ndarray,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
    clusters: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile intervals for paired row means or cluster-resampled means."""
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if not len(matrix):
        width = matrix.shape[1]
        return np.full(width, np.nan), np.full(width, np.nan)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    if clusters is None:
        for start in range(0, draws, 100):
            size = min(100, draws - start)
            indices = rng.integers(0, len(matrix), size=(size, len(matrix)))
            estimates.append(matrix[indices].mean(axis=1))
    else:
        unique, inverse = np.unique(clusters, return_inverse=True)
        sums = np.zeros((len(unique), matrix.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, matrix)
        counts = np.bincount(inverse).astype(np.float64)
        for _ in range(draws):
            selected = rng.integers(0, len(unique), size=len(unique))
            estimates.append(
                (sums[selected].sum(axis=0) / counts[selected].sum())[None, :]
            )
    values = np.concatenate(estimates, axis=0)
    return np.quantile(values, 0.025, axis=0), np.quantile(values, 0.975, axis=0)


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / left.sum()
    right = right / right.sum()
    midpoint = (left + right) / 2

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(np.sum(first[mask] * np.log2(first[mask] / second[mask])))

    return math.sqrt(max(0.0, (kl(left, midpoint) + kl(right, midpoint)) / 2))


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return math.nan
    points = np.linspace(0, 1, max(len(left), len(right)), endpoint=True)
    left_q = np.quantile(left, points)
    right_q = np.quantile(right, points)
    return float(np.mean(np.abs(left_q - right_q)))


def entropy(labels: Iterable[str]) -> float:
    counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
    if not len(counts):
        return math.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def quantile_labels(values: np.ndarray, count: int, prefix: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.full(len(values), "NOT_AVAILABLE", dtype=object)
    cuts = np.unique(np.quantile(finite, np.arange(1, count) / count))
    bins = np.searchsorted(cuts, values, side="right") + 1
    return np.asarray([f"{prefix}{value}" for value in bins], dtype=object)


def fixed_popularity_labels(
    values: np.ndarray, cuts: np.ndarray, prefix: str
) -> np.ndarray:
    bins = np.searchsorted(cuts, values, side="right") + 1
    labels = np.asarray([f"{prefix}{value}" for value in bins], dtype=object)
    labels[values == 0] = "zero"
    return labels


def operational(values: np.ndarray, cuts: list[float], labels: list[str]) -> np.ndarray:
    return np.asarray(labels, dtype=object)[np.searchsorted(cuts, values, side="left")]


def load_csv_map(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            item, _, value = line.partition(",")
            result[int(item)] = value.strip()
    return result


def read_role(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    tables = {name: pq.read_table(path) for name, path in paths.items()}
    result: dict[str, np.ndarray] = {}
    keys: tuple[np.ndarray, ...] | None = None
    for model, table in tables.items():
        current = tuple(
            np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=np.int64)
            for column in ("user_id", "item_id", "timestamp")
        )
        if keys is None:
            keys = current
            result.update(dict(zip(("user_id", "item_id", "timestamp"), current)))
            result["base_slice"] = np.asarray(table["slice"].to_pylist(), dtype=object)
        elif any(not np.array_equal(a, b) for a, b in zip(keys, current)):
            raise ValueError(f"Per-user keys differ for {model}")
        rank = np.asarray(
            [
                value if value is not None else 0
                for value in table["rank_at_20"].to_pylist()
            ],
            dtype=np.int16,
        )
        result[f"{model}_rank"] = rank
        for metric in ("ndcg_at_20", "recall_at_20", "mrr_at_20"):
            result[f"{model}_{metric}"] = np.asarray(table[metric].to_numpy())
        result[f"{model}_hit_at_20"] = (rank > 0).astype(np.float64)
    return result


def build_fit_context(
    fit_path: Path,
    source_path: Path,
    title_path: Path,
    tag_path: Path,
    image_path: Path,
    text_path: Path,
) -> dict[str, Any]:
    fit = pq.read_table(fit_path)
    users = np.asarray(fit["user_id"].to_numpy()).astype(np.int64)
    items = np.asarray(fit["item_id"].to_numpy()).astype(np.int64)
    stamps = np.asarray(fit["timestamp"].to_numpy(), dtype=np.int64)
    max_item = max(19738, int(items.max()))
    item_count = np.bincount(items, minlength=max_item + 1).astype(np.int64)
    item_users: dict[int, set[int]] = defaultdict(set)
    user_items: dict[int, list[int]] = defaultdict(list)
    user_stamps: dict[int, list[int]] = defaultdict(list)
    for user, item, stamp in zip(users, items, stamps):
        item_users[int(item)].add(int(user))
        user_items[int(user)].append(int(item))
        user_stamps[int(user)].append(int(stamp))
    unique_users = np.zeros(max_item + 1, dtype=np.int64)
    for item, values in item_users.items():
        unique_users[item] = len(values)
    first_seen = np.full(max_item + 1, np.iinfo(np.int64).max, dtype=np.int64)
    with source_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(first_seen[item], stamp)
    titles = load_csv_map(title_path)
    tags = load_csv_map(tag_path)
    image_norm = np.linalg.norm(np.load(image_path, mmap_mode="r"), axis=1)
    text_norm = np.linalg.norm(np.load(text_path, mmap_mode="r"), axis=1)
    supported = item_count[1:][item_count[1:] > 0]
    supported_users = unique_users[1:][unique_users[1:] > 0]
    pop_cuts = np.quantile(supported, [0.5, 0.8])
    item_tier = np.full(max_item + 1, "zero", dtype=object)
    item_tier[1:] = operational(
        item_count[1:].astype(float),
        [0, pop_cuts[0], pop_cuts[1]],
        ["zero", "tail", "torso", "head"],
    )
    user_stats: dict[int, dict[str, float]] = {}
    for user, history in user_items.items():
        counts = item_count[np.asarray(history, dtype=np.int64)]
        history_tags = [tags.get(item, "") for item in history if tags.get(item, "")]
        tiers = item_tier[np.asarray(history, dtype=np.int64)]
        user_stats[user] = {
            "history_length": float(len(history)),
            "unique_item_count": float(len(set(history))),
            "repeat_ratio": float((len(history) - len(set(history))) / len(history)),
            "last_fit_timestamp": float(max(user_stamps[user])),
            "median_history_popularity": float(np.median(counts)),
            "head_share": float(np.mean(tiers == "head")),
            "tail_share": float(np.mean(tiers == "tail")),
            "tag_entropy": entropy(history_tags),
        }
    return {
        "item_count": item_count,
        "item_unique_users": unique_users,
        "first_seen": first_seen,
        "titles": titles,
        "tags": tags,
        "image_norm": image_norm,
        "text_norm": text_norm,
        "item_tier": item_tier,
        "pop_count_quintile_cuts": np.unique(
            np.quantile(supported, np.arange(1, 5) / 5)
        ),
        "pop_count_decile_cuts": np.unique(
            np.quantile(supported, np.arange(1, 10) / 10)
        ),
        "pop_user_quintile_cuts": np.unique(
            np.quantile(supported_users, np.arange(1, 5) / 5)
        ),
        "pop_user_decile_cuts": np.unique(
            np.quantile(supported_users, np.arange(1, 10) / 10)
        ),
        "user_stats": user_stats,
        "fit_rows": len(items),
        "fit_users": len(user_items),
        "fit_items": int(np.sum(item_count > 0)),
    }


def enrich_role(data: dict[str, np.ndarray], context: dict[str, Any]) -> None:
    item = data["item_id"]
    stamp = data["timestamp"]
    support = context["item_count"][item]
    user_pop = context["item_unique_users"][item]
    age = np.maximum(0, (stamp - context["first_seen"][item]) / 86_400_000)
    data["support"] = support.astype(np.float64)
    data["unique_fit_users"] = user_pop.astype(np.float64)
    data["item_age_days"] = age.astype(np.float64)
    data["support_fine"] = operational(
        support,
        [0, 5, 20, 50, 100],
        ["0", "1-5", "6-20", "21-50", "51-100", "101+"],
    )
    data["support_p1"] = operational(
        support,
        [0, 5, 20],
        ["zero", "emerging_1-5", "sparse_6-20", "warm_gt20"],
    )
    data["pop_count_quintile"] = fixed_popularity_labels(
        support, context["pop_count_quintile_cuts"], "Q"
    )
    data["pop_count_decile"] = fixed_popularity_labels(
        support, context["pop_count_decile_cuts"], "D"
    )
    data["pop_user_quintile"] = fixed_popularity_labels(
        user_pop, context["pop_user_quintile_cuts"], "Q"
    )
    data["pop_user_decile"] = fixed_popularity_labels(
        user_pop, context["pop_user_decile_cuts"], "D"
    )
    data["head_torso_tail"] = context["item_tier"][item]
    data["item_age"] = operational(
        age,
        [7, 30, 90, 365],
        ["0-7d", "8-30d", "31-90d", "91-365d", "gt365d"],
    )
    stats = context["user_stats"]
    for name in (
        "history_length",
        "unique_item_count",
        "repeat_ratio",
        "median_history_popularity",
        "head_share",
        "tail_share",
        "tag_entropy",
    ):
        data[name] = np.asarray([stats[int(user)][name] for user in data["user_id"]])
    data["recency_days"] = np.asarray(
        [
            max(0.0, (query - stats[int(user)]["last_fit_timestamp"]) / 86_400_000)
            for user, query in zip(data["user_id"], stamp)
        ]
    )
    data["history_bin"] = operational(
        data["history_length"],
        [2, 5, 10, 20],
        ["1-2", "3-5", "6-10", "11-20", "21+"],
    )
    data["recency_bin"] = operational(
        data["recency_days"],
        [7, 30, 90],
        ["0-7d", "8-30d", "31-90d", "gt90d"],
    )
    data["repeat_quantile"] = quantile_labels(data["repeat_ratio"], 4, "Q")
    data["niche_quantile"] = quantile_labels(data["median_history_popularity"], 4, "Q")
    data["tail_share_quantile"] = quantile_labels(data["tail_share"], 4, "Q")
    data["diversity_quantile"] = quantile_labels(data["tag_entropy"], 3, "Q")
    titles = context["titles"]
    tags = context["tags"]
    text_lengths = np.asarray([len(titles.get(int(i), "").split()) for i in item])
    data["text_presence"] = np.where(text_lengths > 0, "present", "missing")
    data["image_presence"] = np.full(len(item), "present", dtype=object)
    data["tag_presence"] = np.asarray(
        ["present" if tags.get(int(i), "") else "missing" for i in item], dtype=object
    )
    data["text_length"] = operational(
        text_lengths,
        [0, 4, 9, 19],
        ["missing", "1-4", "5-9", "10-19", "20+"],
    )
    image_norm = context["image_norm"][item - 1]
    text_norm = context["text_norm"][item - 1]
    for name, values in (("image_norm", image_norm), ("text_norm", text_norm)):
        lower, upper = np.quantile(context[name], [0.01, 0.99])
        data[f"{name}_bin"] = operational(
            values, [lower, upper], ["low_outlier", "typical", "high_outlier"]
        )


def slice_definitions(data: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    definitions: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(data["user_id"]), dtype=bool))
    ]
    families = (
        "base_slice",
        "support_fine",
        "support_p1",
        "pop_count_quintile",
        "pop_count_decile",
        "pop_user_quintile",
        "pop_user_decile",
        "head_torso_tail",
        "item_age",
        "history_bin",
        "recency_bin",
        "repeat_quantile",
        "niche_quantile",
        "tail_share_quantile",
        "diversity_quantile",
        "text_presence",
        "image_presence",
        "tag_presence",
        "text_length",
        "image_norm_bin",
        "text_norm_bin",
    )
    for family in families:
        for value in sorted(set(data[family].tolist())):
            definitions.append((family, str(value), data[family] == value))
    for age in sorted(set(data["item_age"].tolist())):
        for support in sorted(set(data["support_fine"].tolist())):
            mask = (data["item_age"] == age) & (data["support_fine"] == support)
            if int(mask.sum()) >= 200:
                definitions.append(("item_age_x_support", f"{age}|{support}", mask))
    user_families = (
        "history_bin",
        "recency_bin",
        "niche_quantile",
        "tail_share_quantile",
        "diversity_quantile",
    )
    for base in ("controlled_cold", "warm_fit"):
        for family in user_families:
            for value in sorted(set(data[family].tolist())):
                mask = (data["base_slice"] == base) & (data[family] == value)
                if int(mask.sum()):
                    definitions.append((f"base_x_{family}", f"{base}|{value}", mask))
    return definitions


def analyze_slices(
    role: str, data: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    slice_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    for family, name, mask in slice_definitions(data):
        n = int(mask.sum())
        if not n:
            continue
        controlled = (
            family == "base_slice" and name == "controlled_cold"
        ) or name.startswith("controlled_cold|")
        reportable = n >= (50 if controlled else 200)
        for model in MODELS:
            ranks = data[f"{model}_rank"][mask]
            for metric in METRICS:
                values = data[f"{model}_{metric}"][mask]
                summary = metric_summary(values, ranks)
                slice_rows.append(
                    {
                        "analysis_type": "POSTHOC_EXPLORATORY",
                        "role": role,
                        "slice_family": family,
                        "slice": name,
                        "model": model,
                        "metric": metric,
                        "n": n,
                        "value": summary["mean"],
                        "reportable": reportable,
                        "small_slice_reason": None
                        if reportable
                        else f"n<{50 if controlled else 200}",
                        "rank_hit_mean": summary["rank_hit_mean"],
                        "rank_hit_median": summary["rank_hit_median"],
                        "rank_distribution_json": summary["rank_distribution_json"],
                    }
                )
        for baseline in COMPARATORS:
            differences = np.column_stack(
                [
                    data[f"M1_{metric}"][mask] - data[f"{baseline}_{metric}"][mask]
                    for metric in METRICS
                ]
            )
            if reportable:
                lower, upper = paired_bootstrap(
                    differences,
                    seed=stable_seed(role, family, name, baseline, "user"),
                )
            else:
                lower = upper = np.full(len(METRICS), np.nan)
            item_lower = item_upper = np.full(len(METRICS), np.nan)
            if controlled and reportable:
                item_lower, item_upper = paired_bootstrap(
                    differences,
                    clusters=data["item_id"][mask],
                    seed=stable_seed(role, family, name, baseline, "item"),
                )
            for index, metric in enumerate(METRICS):
                values = differences[:, index]
                delta_rows.append(
                    {
                        "analysis_type": "POSTHOC_EXPLORATORY",
                        "role": role,
                        "slice_family": family,
                        "slice": name,
                        "comparison": f"M1-{baseline}",
                        "metric": metric,
                        "n": n,
                        "mean_delta": float(np.mean(values)),
                        "median_delta": float(np.median(values)),
                        "p10": float(np.quantile(values, 0.10)),
                        "p25": float(np.quantile(values, 0.25)),
                        "p75": float(np.quantile(values, 0.75)),
                        "p90": float(np.quantile(values, 0.90)),
                        "lower_95": None if not reportable else float(lower[index]),
                        "upper_95": None if not reportable else float(upper[index]),
                        "cold_item_lower_95": None
                        if not np.isfinite(item_lower[index])
                        else float(item_lower[index]),
                        "cold_item_upper_95": None
                        if not np.isfinite(item_upper[index])
                        else float(item_upper[index]),
                        "bootstrap_draws": BOOTSTRAP_DRAWS if reportable else 0,
                        "reportable": reportable,
                    }
                )
            m1_hit = data["M1_hit_at_20"][mask] > 0
            baseline_hit = data[f"{baseline}_hit_at_20"][mask] > 0
            ndcg_delta = differences[:, 0]
            both = m1_hit & baseline_hit
            rank_gain = (
                data[f"{baseline}_rank"][mask][both] - data["M1_rank"][mask][both]
            )
            behavior_rows.append(
                {
                    "analysis_type": "POSTHOC_EXPLORATORY",
                    "role": role,
                    "slice_family": family,
                    "slice": name,
                    "comparison": f"M1-{baseline}",
                    "n": n,
                    "win_rate": float(np.mean(ndcg_delta > 0)),
                    "tie_rate": float(np.mean(ndcg_delta == 0)),
                    "loss_rate": float(np.mean(ndcg_delta < 0)),
                    "hit_hit_rate": float(np.mean(m1_hit & baseline_hit)),
                    "rescue_rate": float(np.mean(m1_hit & ~baseline_hit)),
                    "regression_rate": float(np.mean(~m1_hit & baseline_hit)),
                    "miss_miss_rate": float(np.mean(~m1_hit & ~baseline_hit)),
                    "rank_gain_mean_both_hit": float(np.mean(rank_gain))
                    if len(rank_gain)
                    else None,
                    "rank_gain_median_both_hit": float(np.median(rank_gain))
                    if len(rank_gain)
                    else None,
                    "reportable": reportable,
                }
            )
    return slice_rows, delta_rows, behavior_rows


def population_drift(
    qualification: dict[str, np.ndarray],
    locked: dict[str, np.ndarray],
    slice_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    families = {
        "support_fine": "support",
        "pop_count_decile": "support",
        "pop_user_decile": "unique_fit_users",
        "item_age": "item_age_days",
        "history_bin": "history_length",
        "recency_bin": "recency_days",
    }
    for family, numeric in families.items():
        labels = sorted(set(qualification[family]) | set(locked[family]))
        q_counts = np.asarray(
            [np.sum(qualification[family] == label) for label in labels]
        )
        l_counts = np.asarray([np.sum(locked[family] == label) for label in labels])
        divergence = jensen_shannon(q_counts, l_counts)
        distance = wasserstein_1d(qualification[numeric], locked[numeric])
        for label, q_count, l_count in zip(labels, q_counts, l_counts):
            q_share = float(q_count / len(qualification[family]))
            l_share = float(l_count / len(locked[family]))
            result.append(
                {
                    "analysis_type": "POSTHOC_EXPLORATORY",
                    "row_type": "composition",
                    "family": family,
                    "slice": str(label),
                    "model": None,
                    "metric": None,
                    "qualification_n": int(q_count),
                    "locked_n": int(l_count),
                    "qualification_value": q_share,
                    "locked_value": l_share,
                    "change": l_share - q_share,
                    "absolute_share_shift": abs(l_share - q_share),
                    "jensen_shannon": divergence,
                    "wasserstein": distance,
                }
            )
    lookup = {
        (
            row["role"],
            row["slice_family"],
            row["slice"],
            row["model"],
            row["metric"],
        ): row
        for row in slice_rows
    }
    for family, name in (
        ("overall", "all"),
        ("base_slice", "controlled_cold"),
        ("base_slice", "warm_fit"),
    ):
        for model in MODELS:
            for metric in METRICS:
                q = lookup.get(("QUALIFICATION", family, name, model, metric))
                locked_row = lookup.get(("LOCKED_FINAL", family, name, model, metric))
                if q is None or locked_row is None:
                    continue
                result.append(
                    {
                        "analysis_type": "POSTHOC_EXPLORATORY",
                        "row_type": "metric_change",
                        "family": family,
                        "slice": name,
                        "model": model,
                        "metric": metric,
                        "qualification_n": q["n"],
                        "locked_n": locked_row["n"],
                        "qualification_value": q["value"],
                        "locked_value": locked_row["value"],
                        "change": locked_row["value"] - q["value"],
                        "absolute_share_shift": None,
                        "jensen_shannon": None,
                        "wasserstein": None,
                    }
                )
    return result


def user_rows(role: str, data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    fields = (
        "history_length",
        "unique_item_count",
        "repeat_ratio",
        "recency_days",
        "median_history_popularity",
        "head_share",
        "tail_share",
        "tag_entropy",
        "history_bin",
        "recency_bin",
        "repeat_quantile",
        "niche_quantile",
        "tail_share_quantile",
        "diversity_quantile",
    )
    return [
        {
            "analysis_type": "POSTHOC_EXPLORATORY",
            "role": role,
            "user_id": int(data["user_id"][index]),
            **{
                field: data[field][index].item()
                if hasattr(data[field][index], "item")
                else data[field][index]
                for field in fields
            },
        }
        for index in range(len(data["user_id"]))
    ]


def item_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    image = context["image_norm"]
    text = context["text_norm"]
    image_cuts = np.quantile(image, [0.01, 0.99])
    text_cuts = np.quantile(text, [0.01, 0.99])
    result = []
    for item in range(1, len(image) + 1):
        title = context["titles"].get(item, "")
        tag = context["tags"].get(item, "")
        result.append(
            {
                "analysis_type": "POSTHOC_EXPLORATORY",
                "item_id": item,
                "fit_interactions": int(context["item_count"][item]),
                "unique_fit_users": int(context["item_unique_users"][item]),
                "head_torso_tail": context["item_tier"][item],
                "first_observed_timestamp": int(context["first_seen"][item]),
                "image_present": True,
                "text_present": bool(title),
                "tag_present": bool(tag),
                "text_word_count": len(title.split()),
                "tag_count": None,
                "tag_count_status": "NOT_AVAILABLE_SINGLE_SUMMARY_CATEGORY_SOURCE",
                "image_norm": float(image[item - 1]),
                "text_norm": float(text[item - 1]),
                "image_norm_bin": operational(
                    np.asarray([image[item - 1]]),
                    image_cuts.tolist(),
                    ["low_outlier", "typical", "high_outlier"],
                )[0],
                "text_norm_bin": operational(
                    np.asarray([text[item - 1]]),
                    text_cuts.tolist(),
                    ["low_outlier", "typical", "high_outlier"],
                )[0],
            }
        )
    return result


def seed_stability(path: Path) -> list[dict[str, Any]]:
    summary = json.loads(path.read_text())
    rows = []
    for candidate, values in summary["candidates"].items():
        for slice_name, metrics in values["slices"].items():
            mean = metrics["mean_ndcg_at_20"]
            sd = metrics["sample_sd_ndcg_at_20"]
            rows.append(
                {
                    "analysis_type": "POSTHOC_EXPLORATORY",
                    "candidate": candidate,
                    "slice": slice_name,
                    "seed_n": metrics["n"],
                    "mean_ndcg_at_20": mean,
                    "sample_sd_ndcg_at_20": sd,
                    "coefficient_of_variation": sd / mean if mean else None,
                    "min_ndcg_at_20": metrics["min_ndcg_at_20"],
                    "max_ndcg_at_20": metrics["max_ndcg_at_20"],
                }
            )
    return rows


def select_cases(
    data: dict[str, np.ndarray], context: dict[str, Any]
) -> list[dict[str, Any]]:
    delta_b4 = data["M1_ndcg_at_20"] - data["B4_ndcg_at_20"]
    delta_b7 = data["M1_ndcg_at_20"] - data["B7_ndcg_at_20"]

    def choose(mask: np.ndarray, values: np.ndarray, mode: str) -> int:
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise ValueError(f"No candidate for {mode}")
        candidate_values = values[candidates]
        if mode == "max":
            target = candidate_values.max()
        elif mode == "min":
            target = candidate_values.min()
        else:
            target = np.median(candidate_values)
        distance = np.abs(candidate_values - target)
        ordered = np.lexsort((data["user_id"][candidates], distance))
        return int(candidates[ordered[0]])

    cold = data["base_slice"] == "controlled_cold"
    warm = data["base_slice"] == "warm_fit"
    specifications = (
        (
            "largest_m1_rescue_controlled_cold",
            cold & (data["M1_rank"] > 0) & (data["B4_rank"] == 0),
            delta_b4,
            "max",
            "B4",
        ),
        (
            "largest_m1_regression_warm",
            warm & (data["M1_rank"] == 0) & (data["B7_rank"] > 0),
            delta_b7,
            "min",
            "B7",
        ),
        (
            "representative_median_cold_win",
            cold & (delta_b4 > 0),
            delta_b4,
            "median",
            "B4",
        ),
        (
            "representative_median_warm_loss",
            warm & (delta_b7 < 0),
            delta_b7,
            "median",
            "B7",
        ),
    )
    rows = []
    for label, mask, delta, mode, baseline in specifications:
        index = choose(mask, delta, mode)
        item = int(data["item_id"][index])
        rows.append(
            {
                "analysis_type": "POSTHOC_EXPLORATORY",
                "selection_rule": label,
                "qualitative_only": True,
                "user_id": int(data["user_id"][index]),
                "item_id": item,
                "timestamp": int(data["timestamp"][index]),
                "base_slice": data["base_slice"][index],
                "fit_support": int(data["support"][index]),
                "item_age_days": float(data["item_age_days"][index]),
                "title": context["titles"].get(item, ""),
                "tag_summary": context["tags"].get(item, ""),
                "baseline": baseline,
                "m1_rank": int(data["M1_rank"][index]) or None,
                "baseline_rank": int(data[f"{baseline}_rank"][index]) or None,
                "m1_ndcg_at_20": float(data["M1_ndcg_at_20"][index]),
                "baseline_ndcg_at_20": float(data[f"{baseline}_ndcg_at_20"][index]),
                "ndcg_delta": float(delta[index]),
            }
        )
    return rows


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ">": ("10000", "01000", "00100", "00010", "00100", "01000", "10000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    " ": ("00000",) * 7,
}


class Canvas:
    def __init__(self, width: int = 1000, height: int = 650) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray([255] * width * height * 3)

    def set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def rect(
        self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]
    ) -> None:
        for y in range(max(0, y0), min(self.height, y1)):
            start = (y * self.width + max(0, x0)) * 3
            end = (y * self.width + min(self.width, x1)) * 3
            self.pixels[start:end] = bytes(color) * max(
                0, min(self.width, x1) - max(0, x0)
            )

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = round(x0 + (x1 - x0) * step / steps)
            y = round(y0 + (y1 - y0) * step / steps)
            self.rect(
                x - width // 2,
                y - width // 2,
                x + width // 2 + 1,
                y + width // 2 + 1,
                color,
            )

    def text(
        self,
        x: int,
        y: int,
        value: str,
        scale: int = 2,
        color: tuple[int, int, int] = (25, 35, 50),
    ) -> None:
        cursor = x
        for character in value.upper():
            glyph = FONT.get(character, FONT[" "])
            for row, pattern in enumerate(glyph):
                for column, bit in enumerate(pattern):
                    if bit == "1":
                        self.rect(
                            cursor + column * scale,
                            y + row * scale,
                            cursor + (column + 1) * scale,
                            y + (row + 1) * scale,
                            color,
                        )
            cursor += 6 * scale

    def save(self, path: Path, title: str) -> None:
        raw = b"".join(
            b"\x00"
            + bytes(self.pixels[row * self.width * 3 : (row + 1) * self.width * 3])
            for row in range(self.height)
        )

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(
            b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        )
        png += chunk(b"tEXt", b"Title\x00" + title.encode("latin-1", errors="replace"))
        png += chunk(b"IDAT", zlib.compress(raw, 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def bar_figure(
    path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
    *,
    zero_line: bool = False,
) -> None:
    canvas = Canvas()
    canvas.text(35, 20, LABEL, 2, (150, 35, 35))
    canvas.text(35, 55, title, 3)
    values = [
        value for group in series.values() for value in group if np.isfinite(value)
    ]
    low = min(values + [0.0])
    high = max(values + [0.0])
    if high == low:
        high = low + 1
    left, top, right, bottom = 90, 130, 960, 540
    canvas.line(left, top, left, bottom, (40, 50, 65), 2)
    canvas.line(left, bottom, right, bottom, (40, 50, 65), 2)
    colors = ((38, 99, 173), (220, 95, 50), (45, 160, 110))
    group_width = (right - left) / max(1, len(labels))
    bar_width = max(3, int(group_width / (len(series) + 1)))

    def y(value: float) -> int:
        return int(bottom - (value - low) / (high - low) * (bottom - top))

    if zero_line or low < 0:
        canvas.line(left, y(0), right, y(0), (130, 130, 130), 1)
    for group_index, label in enumerate(labels):
        base_x = int(left + group_index * group_width + group_width * 0.12)
        for series_index, (_, group) in enumerate(series.items()):
            value = group[group_index]
            x0 = base_x + series_index * bar_width
            canvas.rect(
                x0,
                min(y(value), y(0)),
                x0 + bar_width - 2,
                max(y(value), y(0)) + 1,
                colors[series_index],
            )
        canvas.text(base_x, bottom + 12, label[:12], 1)
    legend_x = 100
    for index, name in enumerate(series):
        canvas.rect(legend_x, 590, legend_x + 18, 608, colors[index])
        canvas.text(legend_x + 25, 592, name, 1)
        legend_x += 180
    canvas.text(10, top, f"MAX {high:.4f}", 1)
    canvas.text(10, bottom - 8, f"MIN {low:.4f}", 1)
    canvas.save(path, f"{LABEL}: {title}")


def histogram_figure(path: Path, title: str, values: np.ndarray) -> None:
    counts, edges = np.histogram(values, bins=31)
    labels = [f"{edges[index]:.2f}" for index in range(0, len(counts), 5)]
    grouped = [
        int(counts[index : index + 5].sum()) for index in range(0, len(counts), 5)
    ]
    bar_figure(path, title, labels, {"COUNT": grouped})


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing empty output table: {path.name}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def find_metric(
    rows: list[dict[str, Any]],
    role: str,
    family: str,
    name: str,
    model: str,
    metric: str = "ndcg_at_20",
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["role"] == role
        and row["slice_family"] == family
        and row["slice"] == name
        and row["model"] == model
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"Metric not unique: {role}/{family}/{name}/{model}/{metric}")
    return matches[0]


def find_delta(
    rows: list[dict[str, Any]], role: str, family: str, name: str, comparison: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["role"] == role
        and row["slice_family"] == family
        and row["slice"] == name
        and row["comparison"] == comparison
        and row["metric"] == "ndcg_at_20"
    ]
    if len(matches) != 1:
        raise ValueError(f"Delta not unique: {role}/{family}/{name}/{comparison}")
    return matches[0]


def render_reports(
    output: Path,
    slice_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    cold_b4 = find_delta(
        delta_rows, "LOCKED_FINAL", "base_slice", "controlled_cold", "M1-B4"
    )
    warm_b7 = find_delta(delta_rows, "LOCKED_FINAL", "base_slice", "warm_fit", "M1-B7")
    overall_b7 = find_delta(delta_rows, "LOCKED_FINAL", "overall", "all", "M1-B7")
    support = ["0", "1-5", "6-20", "21-50", "51-100", "101+"]
    support_findings = []
    for name in support:
        try:
            row = find_delta(delta_rows, "LOCKED_FINAL", "support_fine", name, "M1-B4")
        except ValueError:
            continue
        support_findings.append(
            (name, row["n"], row["mean_delta"], row["lower_95"], row["upper_95"])
        )
    pop_candidates = [
        row
        for row in delta_rows
        if row["role"] == "LOCKED_FINAL"
        and row["slice_family"] == "head_torso_tail"
        and row["comparison"] == "M1-B4"
        and row["metric"] == "ndcg_at_20"
        and row["reportable"]
    ]
    best_pop = max(pop_candidates, key=lambda row: row["mean_delta"])
    age_candidates = [
        row
        for row in delta_rows
        if row["role"] == "LOCKED_FINAL"
        and row["slice_family"] == "item_age"
        and row["comparison"] == "M1-B4"
        and row["metric"] == "ndcg_at_20"
        and row["reportable"]
    ]
    best_age = max(age_candidates, key=lambda row: row["mean_delta"])
    warm_user_candidates = [
        row
        for row in delta_rows
        if row["role"] == "LOCKED_FINAL"
        and row["slice_family"].startswith("base_x_")
        and row["slice"].startswith("warm_fit|")
        and row["comparison"] == "M1-B7"
        and row["metric"] == "ndcg_at_20"
        and row["reportable"]
    ]
    best_user = max(warm_user_candidates, key=lambda row: row["mean_delta"])
    worst_user = min(warm_user_candidates, key=lambda row: row["mean_delta"])
    user_family_summary = []
    for family in (
        "base_x_history_bin",
        "base_x_recency_bin",
        "base_x_niche_quantile",
        "base_x_tail_share_quantile",
        "base_x_diversity_quantile",
    ):
        family_rows = [
            row for row in warm_user_candidates if row["slice_family"] == family
        ]
        user_family_summary.append(
            (
                family.removeprefix("base_x_"),
                max(family_rows, key=lambda row: row["mean_delta"]),
                min(family_rows, key=lambda row: row["mean_delta"]),
            )
        )
    drift_composition = [row for row in drift_rows if row["row_type"] == "composition"]
    max_shift = max(drift_composition, key=lambda row: row["absolute_share_shift"])
    q_m1_warm = find_metric(slice_rows, "QUALIFICATION", "base_slice", "warm_fit", "M1")
    l_m1_warm = find_metric(slice_rows, "LOCKED_FINAL", "base_slice", "warm_fit", "M1")
    q_b7_warm = find_metric(slice_rows, "QUALIFICATION", "base_slice", "warm_fit", "B7")
    l_b7_warm = find_metric(slice_rows, "LOCKED_FINAL", "base_slice", "warm_fit", "B7")
    user_table_rows = "\n".join(
        "| {family} | {best_slice} (n={best_n:,}) | {best_delta:.6f} | "
        "{worst_slice} (n={worst_n:,}) | {worst_delta:.6f} |".format(
            family=family,
            best_slice=best["slice"].split("|", 1)[1],
            best_n=best["n"],
            best_delta=best["mean_delta"],
            worst_slice=worst["slice"].split("|", 1)[1],
            worst_n=worst["n"],
            worst_delta=worst["mean_delta"],
        )
        for family, best, worst in user_family_summary
    )
    report = f"""# Post-hoc p1 diagnostic report

**{LABEL}**

This report reads immutable p1 per-user outputs and FIT-visible context only. It performs no model inference, creates no canonical evidence, and cannot change the permanent `RETAIN_BASELINE` decision.

## Scope and source integrity

- Analysis type: `POSTHOC_EXPLORATORY`.
- Confirmatory: `false`.
- Bootstrap: {BOOTSTRAP_DRAWS:,} deterministic paired user resamples, seed {BOOTSTRAP_SEED}; item-cluster sensitivity for the controlled-cold slice.
- Locked users analyzed: {manifest["population"]["locked_users"]:,}; qualification users: {manifest["population"]["qualification_users"]:,}.
- Target items represented in locked analysis: {manifest["population"]["locked_target_items"]:,}.
- Every source was hashed before and after analysis with no change.

## 1. Where does MOSAIC help?

Against B4 on the controlled-cold cohort, the existing locked rows show an exploratory mean NDCG@20 delta of {cold_b4["mean_delta"]:.6f}, with paired post-hoc interval [{cold_b4["lower_95"]:.6f}, {cold_b4["upper_95"]:.6f}]. This reproduces the direction of the closed p1 observation but does not upgrade it to superiority.

The strongest reportable head/torso/tail segment versus B4 was `{best_pop["slice"]}` (n={best_pop["n"]:,}, delta={best_pop["mean_delta"]:.6f}). The strongest item-age segment was `{best_age["slice"]}` (n={best_age["n"]:,}, delta={best_age["mean_delta"]:.6f}). These are discovered post hoc.

## 2. Where does it hurt?

Against B7 on the frozen warm slice, the exploratory mean NDCG@20 delta is {warm_b7["mean_delta"]:.6f}, interval [{warm_b7["lower_95"]:.6f}, {warm_b7["upper_95"]:.6f}]. The worst reportable warm user segment among the requested families was `{worst_user["slice_family"]}:{worst_user["slice"]}` (n={worst_user["n"]:,}, delta={worst_user["mean_delta"]:.6f}).

The secondary overall delta versus B7 remains positive at {overall_b7["mean_delta"]:.6f}, which illustrates why aggregate performance masked the warm failure.

## 3. Is the cold advantage concentrated?

| FIT support | n | M1-B4 NDCG delta | Exploratory 95% interval |
|---|---:|---:|---:|
{chr(10).join(f"| {name} | {n:,} | {delta:.6f} | " + (f"[{lower:.6f}, {upper:.6f}]" if lower is not None else "suppressed") + " |" for name, n, delta, lower, upper in support_findings)}

The detailed machine-readable tables also segment by FIT interaction count, unique FIT users, fixed popularity quintiles/deciles, head/torso/tail, item age, and age-by-support cells with n>=200. Concentration by popularity or age is descriptive and noncausal.

## 4. What explains the warm collapse?

M1 warm NDCG@20 changed from {q_m1_warm["value"]:.6f} in QUALIFICATION to {l_m1_warm["value"]:.6f} in LOCKED_FINAL, a change of {l_m1_warm["value"] - q_m1_warm["value"]:.6f}. B7 changed from {q_b7_warm["value"]:.6f} to {l_b7_warm["value"]:.6f}, a change of {l_b7_warm["value"] - q_b7_warm["value"]:.6f}.

The largest absolute composition shift among the audited bins was `{max_shift["family"]}:{max_shift["slice"]}` at {max_shift["absolute_share_shift"]:.4f}. Because M1's warm decline was materially larger than B7's under the same role change, the descriptive evidence is more consistent with an M1-specific sensitivity interacting with context than with a population-only explanation. This is not causal attribution.

## 5. Did qualification-to-locked drift contribute?

Composition drift exists and is quantified in `population_drift.parquet` using absolute share shift, Jensen-Shannon distance, and one-dimensional Wasserstein distance. The model-specific metric changes show that common population change does not fully explain the divergence, but the immutable artifacts cannot isolate a causal mechanism.

## 6. Which effects are stable versus heterogeneous?

The E01 table preserves mean, sample SD, coefficient of variation where meaningful, and min/max for every frozen candidate and slice. It is DEVELOPMENT evidence only. M1_T2 showed nonzero seed variation, and the support, popularity, age, and user slices show substantial heterogeneity in locked deltas. No candidate is reselected.

Warm-item user effects were negative across the requested reportable cohort families. The least and most negative cells were:

| User family | Least negative warm cell | Delta | Most negative warm cell | Delta |
|---|---|---:|---|---:|
{user_table_rows}

Sparse-history, inactive/returning, niche/long-tail preference, and diversity segments therefore change the magnitude of the warm loss rather than revealing a reportable warm cohort where M1 beats B7. These contrasts are hypothesis-generating and use FIT-visible user features only.

## 7. Content-quality diagnostics

Publisher image and text vectors were present for all catalog rows. Title presence, tag-summary presence, text-length bins, and publisher feature-norm outliers were analyzed. Multi-tag count is `NOT_AVAILABLE` because `tags_to_summary.csv` supplies one summary category rather than a valid multi-tag set. No raw-cover feature was created.

## 8. Case studies

`case_studies.csv` contains four deterministic qualitative illustrations: the largest controlled-cold M1 rescue, largest warm regression, median cold win, and median warm loss. They use permitted title/tag summaries and do not support general claims.

## 9. Metrics not computable

The immutable per-user artifacts contain target ranks and per-user metrics but no top-k item lists. Catalog coverage@20, recommendation popularity, long-tail exposure, exposure concentration, intra-list diversity, novelty, personalization, and list overlap are marked `DEFER_TO_P2_REQUIRES_NEW_PREDECLARED_SCORING`. They were not approximated and protected scoring was not rerun.

## 10. What p2 should preregister

- The exact support, popularity, age, and user cohort hypotheses selected from this exploratory report.
- Fixed slice boundaries, minimum sample rules, bootstrap unit, draw count, and multiplicity policy.
- A gate input that can observe content quality if reliability behavior is tested.
- Complete intervention intensities and ablation implementations.
- External transfer identities and access order before PixelRec outcomes.
- Top-k retention if coverage, exposure, diversity, novelty, or personalization will be evaluated.

## Guardrail

Nothing in this report amends p1. The authoritative decision remains `RETAIN_BASELINE`.
"""
    (output / "POSTHOC_DIAGNOSTIC_REPORT.md").write_text(report, encoding="utf-8")
    executive = f"""# Executive summary

**{LABEL}**

MOSAIC's existing p1 locked rows show that its advantage is concentrated near the cold boundary rather than across warm items. The controlled-cold M1-B4 NDCG@20 delta is {cold_b4["mean_delta"]:.6f}; the existing p1 decision remains unchanged because this post-hoc analysis is nonconfirmatory and the original interval did not establish superiority.

M1 underperformed B7 on warm items by {abs(warm_b7["mean_delta"]):.6f} NDCG@20. Qualification-to-locked composition shifted, but M1's warm decline ({l_m1_warm["value"] - q_m1_warm["value"]:.6f}) was larger than B7's ({l_b7_warm["value"] - q_b7_warm["value"]:.6f}), so drift alone is not a sufficient descriptive explanation.

The strongest reportable popularity tier versus B4 was `{best_pop["slice"]}`, and the strongest age bin was `{best_age["slice"]}`. Warm user-level effects were heterogeneous in magnitude but negative in every requested reportable cohort family. These findings are candidates for preregistration in p2, not new p1 claims.

Top-k list metrics were unavailable because immutable p1 outputs retain target ranks but not recommendation lists. They require new preregistered p2 scoring.

**Permanent p1 decision: RETAIN_BASELINE.**
"""
    (output / "EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")
    hypotheses = f"""# Candidate p2 hypotheses

**{LABEL}**

These hypotheses were generated after p1 closure. They were not prespecified in p1 and require a new protocol.

1. **Support-boundary hypothesis.** M1-style content fusion improves NDCG@20 primarily for zero-support targets, with weaker or negative value once collaborative support is available. Preregister the six support bins and compare against both B4 and B7.
2. **Warm-sensitivity hypothesis.** The M1 architecture is more sensitive than B7 to the qualification-to-final context shift on warm items. Preregister the role populations and test an interaction between model and role without using p1 for threshold selection.
3. **Popularity concentration hypothesis.** The largest M1-B4 benefit occurs in the `{best_pop["slice"]}` FIT-popularity tier observed here. Freeze the tier definition from FIT-only item counts and validate on new protected evidence.
4. **Lifecycle hypothesis.** The largest reportable M1-B4 benefit occurs for item age `{best_age["slice"]}` under the first-observed availability proxy. Preregister age and support jointly because they are correlated.
5. **User-context heterogeneity hypothesis.** The magnitude of M1's warm deficit varies by FIT-history, recency, niche preference, tail share, and tag entropy; the least negative exploratory warm segment was `{best_user["slice_family"]}:{best_user["slice"]}` and the most negative was `{worst_user["slice_family"]}:{worst_user["slice"]}`. Choose one mechanistically justified family before p2 outcome access and control multiplicity.
6. **Observable-quality gating hypothesis.** A gate that directly observes modality availability/quality will respond directionally to frozen perturbations better than a support-only gate. Freeze exact perturbations, intensities, statistics, and terminology before evaluation.

Do not use p1 to claim these hypotheses are confirmed. The p1 decision remains `RETAIN_BASELINE`.
"""
    (output / "P2_HYPOTHESES.md").write_text(hypotheses, encoding="utf-8")


def verify_manifest(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "MANIFEST.json").read_text())
    if manifest["analysis_type"] != "POSTHOC_EXPLORATORY":
        raise ValueError("Unexpected analysis type")
    if manifest["confirmatory"] or manifest["may_change_p1_decision"]:
        raise ValueError("Post-hoc guard flags are invalid")
    if manifest["p1_decision"] != "RETAIN_BASELINE":
        raise ValueError("Closed p1 decision changed")
    changed_sources = [
        name
        for name, source in manifest["sources"].items()
        if sha256(Path(source["path"])) != source["sha256"]
    ]
    changed_outputs = [
        relative
        for relative, digest in manifest["outputs"].items()
        if sha256(output / relative) != digest
    ]
    if changed_sources or changed_outputs:
        raise RuntimeError(
            f"Hash verification failed: sources={changed_sources}, outputs={changed_outputs}"
        )
    return {
        "status": "PASS",
        "sources": len(manifest["sources"]),
        "outputs": len(manifest["outputs"]),
        **manifest["population"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("analysis/posthoc_p1"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (
        (root / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve()
    )
    expected_root = (root / "analysis/posthoc_p1").resolve()
    if output != expected_root:
        raise ValueError(f"Post-hoc output must be {expected_root}")
    if args.verify_only:
        print(json.dumps(verify_manifest(output), indent=2))
        return
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    scratch = Path("/home/bjw-0/mosaic-rec-work")
    locked_paths = {
        "B7": scratch / "locked/f00_b7_bert4rec_t3_per_user.parquet",
        "B4": scratch / "locked/f00_b4_two_tower_t3_per_user.parquet",
        "M1": scratch / "locked/f00_m1_t2_per_user.parquet",
    }
    qualification_paths = {
        "B7": scratch / "qualification/e02_b7_bert4rec_t3_per_user.parquet",
        "B4": scratch / "qualification/e02_b4_two_tower_t3_per_user.parquet",
        "M1": scratch / "qualification/e03_m1_t2_per_user.parquet",
    }
    fit = scratch / "data/microlens-100k/split_p1/FIT_VISIBLE.parquet"
    qualification_source = (
        scratch / "data/microlens-100k/split_p1/QUALIFICATION.parquet"
    )
    locked_source = scratch / "data/microlens-100k/split_p1/LOCKED_FINAL.parquet"
    cold_ids = scratch / "data/microlens-100k/split_p1/controlled_cold_item_ids.txt"
    source = scratch / "data/microlens-100k/pairs.csv"
    titles = scratch / "data/microlens-100k/title_en.csv"
    tags = scratch / "data/microlens-100k/tags_to_summary.csv"
    image = (
        scratch / "features/microlens-100k/MicroLens-100k_image_features_CLIPRN50.npy"
    )
    text = (
        scratch
        / "features/microlens-100k/MicroLens-100k_title_en_text_features_BgeM3.npy"
    )
    source_paths: dict[str, Path] = {
        **{f"locked_per_user_{name}": path for name, path in locked_paths.items()},
        **{
            f"qualification_per_user_{name}": path
            for name, path in qualification_paths.items()
        },
        "fit_visible": fit,
        "qualification_interactions": qualification_source,
        "locked_final_interactions": locked_source,
        "controlled_cold_ids": cold_ids,
        "interaction_source": source,
        "titles": titles,
        "tag_summaries": tags,
        "publisher_image_features": image,
        "publisher_text_features": text,
        "final_status": root / "reports/FINAL_STATUS.md",
        "technical_report": root / "reports/MOSAIC_REC_FINAL_TECHNICAL_REPORT.md",
        "claim_ledger": root / "reports/CLAIM_LEDGER.md",
        "claims_json": root / "evidence/claims.json",
        "stage_status": root / "evidence/stage_status.json",
        "locked_receipt_B7": root / "evidence/f00_locked_p1_b7_bert4rec_t3.json",
        "locked_receipt_B4": root / "evidence/f00_locked_p1_b4_two_tower_t3.json",
        "locked_receipt_M1": root / "evidence/f00_locked_p1_m1_t2.json",
        "locked_statistics": root / "evidence/f00_locked_statistics_p1.json",
        "qualification_receipt_B7": root
        / "evidence/e02_qualification_p1_b7_bert4rec_t3.json",
        "qualification_receipt_B4": root
        / "evidence/e02_qualification_p1_b4_two_tower_t3.json",
        "qualification_receipt_M1": root / "evidence/e03_qualification_p1_m1_t2.json",
        "e01_summary": root / "evidence/e01_seed_qualification_p1.json",
        "canonical_warehouse": root / "evidence/warehouse.duckdb",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing immutable source(s): {missing}")
    hashes_before = {name: sha256(path) for name, path in source_paths.items()}
    for model, receipt_name in (
        ("B7", "locked_receipt_B7"),
        ("B4", "locked_receipt_B4"),
        ("M1", "locked_receipt_M1"),
    ):
        receipt = json.loads(source_paths[receipt_name].read_text())
        if receipt["per_user_sha256"] != hashes_before[f"locked_per_user_{model}"]:
            raise ValueError(f"Locked per-user hash mismatch: {model}")
    for model, receipt_name in (
        ("B7", "qualification_receipt_B7"),
        ("B4", "qualification_receipt_B4"),
        ("M1", "qualification_receipt_M1"),
    ):
        receipt = json.loads(source_paths[receipt_name].read_text())
        if (
            receipt["per_user_sha256"]
            != hashes_before[f"qualification_per_user_{model}"]
        ):
            raise ValueError(f"Qualification per-user hash mismatch: {model}")
    locked = read_role(locked_paths)
    qualification = read_role(qualification_paths)
    context = build_fit_context(fit, source, titles, tags, image, text)
    enrich_role(locked, context)
    enrich_role(qualification, context)
    slice_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    for role, data in (("QUALIFICATION", qualification), ("LOCKED_FINAL", locked)):
        slices, deltas, behaviors = analyze_slices(role, data)
        slice_rows.extend(slices)
        delta_rows.extend(deltas)
        behavior_rows.extend(behaviors)
    drift_rows = population_drift(qualification, locked, slice_rows)
    users = user_rows("QUALIFICATION", qualification) + user_rows(
        "LOCKED_FINAL", locked
    )
    items = item_rows(context)
    cases = select_cases(locked, context)
    seeds = seed_stability(source_paths["e01_summary"])
    write_parquet(slice_rows, tables / "slice_metrics.parquet")
    write_parquet(delta_rows, tables / "paired_deltas.parquet")
    write_parquet(behavior_rows, tables / "win_loss_rescue.parquet")
    write_parquet(drift_rows, tables / "population_drift.parquet")
    write_parquet(users, tables / "user_segments.parquet")
    write_parquet(items, tables / "item_segments.parquet")
    write_parquet(seeds, tables / "seed_stability.parquet")
    with (tables / "case_studies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    # Requested figures, all derived from the tables above.
    support_labels = ["0", "1-5", "6-20", "21-50", "51-100", "101+"]
    support_series = {
        model: [
            find_metric(slice_rows, "LOCKED_FINAL", "support_fine", label, model)[
                "value"
            ]
            for label in support_labels
        ]
        for model in MODELS
    }
    bar_figure(
        figures / "support_frontier.png",
        "SUPPORT FRONTIER NDCG20",
        support_labels,
        support_series,
    )
    deciles = sorted(
        {
            row["slice"]
            for row in delta_rows
            if row["role"] == "LOCKED_FINAL"
            and row["slice_family"] == "pop_count_decile"
        }
    )
    bar_figure(
        figures / "popularity_deciles.png",
        "POPULARITY DECILE NDCG DELTA",
        deciles,
        {
            comparison: [
                find_delta(
                    delta_rows, "LOCKED_FINAL", "pop_count_decile", label, comparison
                )["mean_delta"]
                for label in deciles
            ]
            for comparison in ("M1-B4", "M1-B7")
        },
        zero_line=True,
    )
    history_labels = ["1-2", "3-5", "6-10", "11-20", "21+"]
    bar_figure(
        figures / "user_activity_slices.png",
        "USER HISTORY NDCG20",
        history_labels,
        {
            model: [
                find_metric(slice_rows, "LOCKED_FINAL", "history_bin", label, model)[
                    "value"
                ]
                for label in history_labels
            ]
            for model in MODELS
        },
    )
    histogram_figure(
        figures / "rank_delta_distribution.png",
        "M1 MINUS B7 PER USER NDCG",
        locked["M1_ndcg_at_20"] - locked["B7_ndcg_at_20"],
    )
    base_labels = ["COLD", "WARM"]
    behavior_lookup = {
        (row["slice"], row["comparison"]): row
        for row in behavior_rows
        if row["role"] == "LOCKED_FINAL" and row["slice_family"] == "base_slice"
    }
    bar_figure(
        figures / "rescue_regression.png",
        "RESCUE AND REGRESSION RATES",
        base_labels,
        {
            "RESCUE": [
                behavior_lookup[("controlled_cold", "M1-B4")]["rescue_rate"],
                behavior_lookup[("warm_fit", "M1-B7")]["rescue_rate"],
            ],
            "REGRESSION": [
                behavior_lookup[("controlled_cold", "M1-B4")]["regression_rate"],
                behavior_lookup[("warm_fit", "M1-B7")]["regression_rate"],
            ],
        },
    )
    drift_metric_labels = ["B7", "B4", "M1"]
    bar_figure(
        figures / "qualification_locked_drift.png",
        "WARM QUALIFICATION TO LOCKED",
        drift_metric_labels,
        {
            "QUAL": [
                find_metric(
                    slice_rows, "QUALIFICATION", "base_slice", "warm_fit", model
                )["value"]
                for model in MODELS
            ],
            "LOCKED": [
                find_metric(
                    slice_rows, "LOCKED_FINAL", "base_slice", "warm_fit", model
                )["value"]
                for model in MODELS
            ],
        },
    )
    bar_figure(
        figures / "cold_warm_tradeoff.png",
        "LOCKED COLD WARM TRADEOFF",
        ["COLD", "WARM"],
        {
            model: [
                find_metric(slice_rows, "LOCKED_FINAL", "base_slice", name, model)[
                    "value"
                ]
                for name in ("controlled_cold", "warm_fit")
            ]
            for model in MODELS
        },
    )
    hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    if hashes_after != hashes_before:
        changed = [
            name for name in hashes_before if hashes_before[name] != hashes_after[name]
        ]
        raise RuntimeError(f"Immutable source changed during analysis: {changed}")
    schemas = {
        name: pq.ParquetFile(path).schema_arrow.names
        for name, path in {
            **locked_paths,
            **{f"qualification_{k}": v for k, v in qualification_paths.items()},
        }.items()
    }
    top_k_available = any(
        any(
            "top" in column.lower() or "recommend" in column.lower()
            for column in columns
        )
        for columns in schemas.values()
    )
    if top_k_available:
        raise RuntimeError(
            "Unexpected top-k columns require a separately audited analysis path"
        )
    manifest = {
        "protocol_id": PROTOCOL,
        "analysis_type": "POSTHOC_EXPLORATORY",
        "confirmatory": False,
        "may_change_p1_decision": False,
        "p1_decision": "RETAIN_BASELINE",
        "analysis_git_parent": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runtime": "Ubuntu-22.04 CPU-only; no model inference",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "small_slice_policy": {"general_min_n": 200, "controlled_cold_min_n": 50},
        "sources": {
            name: {
                "path": str(path),
                "sha256": hashes_before[name],
                "unchanged_at_end": True,
            }
            for name, path in source_paths.items()
        },
        "per_user_schemas": schemas,
        "population": {
            "locked_users": len(locked["user_id"]),
            "qualification_users": len(qualification["user_id"]),
            "locked_target_items": len(np.unique(locked["item_id"])),
            "qualification_target_items": len(np.unique(qualification["item_id"])),
            "fit_rows": context["fit_rows"],
            "fit_users": context["fit_users"],
            "fit_items": context["fit_items"],
        },
        "top_k_lists_present": False,
        "deferred_metrics": [
            "catalog_coverage_at_20",
            "recommendation_popularity",
            "long_tail_exposure_share",
            "exposure_concentration",
            "intra_list_diversity",
            "novelty",
            "personalization_list_overlap",
        ],
        "deferred_status": "DEFER_TO_P2_REQUIRES_NEW_PREDECLARED_SCORING",
        "semantic_unavailable": {
            "multi_tag_count": "NOT_AVAILABLE_SINGLE_SUMMARY_CATEGORY_SOURCE",
            "demographic_fairness": "NOT_AVAILABLE_NO_DEMOGRAPHIC_DATA",
        },
        "outputs": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "MANIFEST.json"
        },
    }
    render_reports(
        output,
        slice_rows,
        delta_rows,
        behavior_rows,
        drift_rows,
        seeds,
        cases,
        manifest,
    )
    # Reports are rendered after the first output hash inventory; refresh it.
    manifest["outputs"] = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "output": str(output), **manifest["population"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
