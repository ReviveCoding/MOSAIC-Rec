"""Frozen p2 slice labels and descriptive metric aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pyarrow as pa


def support_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 20:
        return "6-20"
    if value <= 50:
        return "21-50"
    if value <= 100:
        return "51-100"
    return "101+"


def age_bin(days: float) -> str:
    if days <= 7:
        return "0-7d"
    if days <= 30:
        return "8-30d"
    if days <= 90:
        return "31-90d"
    if days <= 365:
        return "91-365d"
    return ">365d"


def recency_bin(days: float) -> str:
    return age_bin(days)


def slice_memberships(row: dict[str, Any]) -> list[tuple[str, str]]:
    support = support_bin(int(row["fit_interactions"]))
    age = age_bin(float(row["item_age_days"]))
    memberships = [
        ("overall", "all"),
        ("support", support),
        ("interaction_quintile", f"Q{int(row['interaction_quintile'])}"),
        ("interaction_decile", f"D{int(row['interaction_decile'])}"),
        ("unique_user_quintile", f"Q{int(row['unique_user_quintile'])}"),
        ("popularity_tier", str(row["popularity_tier"])),
        ("item_age", age),
        ("history_length", f"Q{int(row['history_length_quartile'])}"),
        ("recency", recency_bin(float(row["user_recency_days"]))),
        ("tail_preference", f"Q{int(row['tail_share_quartile'])}"),
        ("niche_preference", f"Q{int(row['niche_quartile'])}"),
        ("history_diversity", f"Q{int(row['diversity_quartile'])}"),
        ("support_x_item_age", f"{support}|{age}"),
    ]
    if bool(row["is_controlled_zero"]):
        memberships.append(("primary_population", "controlled_zero"))
    if bool(row["is_natural_zero"]):
        memberships.append(("secondary_population", "natural_zero"))
    if bool(row["is_warm"]):
        memberships.append(("primary_population", "warm"))
    return memberships


def aggregate_slices(rows: Iterable[dict[str, Any]]) -> pa.Table:
    metrics = (
        "ndcg_at_20",
        "recall_at_20",
        "mrr_at_20",
        "hit_at_20",
        "precision_at_20",
    )
    values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        for family, label in slice_memberships(row):
            bucket = values.setdefault(
                (family, label), {metric: [] for metric in metrics}
            )
            for metric in metrics:
                bucket[metric].append(float(row[metric]))
    output: list[dict[str, Any]] = []
    for (family, label), bucket in sorted(values.items()):
        count = len(bucket[metrics[0]])
        controlled = family == "primary_population" and label == "controlled_zero"
        minimum = 50 if controlled else 200
        record: dict[str, Any] = {
            "slice_family": family,
            "slice": label,
            "n": count,
            "inferentially_eligible": count >= minimum,
            "minimum_n": minimum,
        }
        for metric in metrics:
            record[metric] = float(np.mean(bucket[metric]))
        output.append(record)
    return pa.Table.from_pylist(output)


def validate_joined_output(rows: Iterable[dict[str, Any]]) -> None:
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        identity = (
            str(row["user_id"]),
            int(row["query_timestamp"]),
            str(row["target_item_id"]),
        )
        if identity in seen:
            raise ValueError(f"Duplicate scored query identity: {identity}")
        seen.add(identity)
        rank = int(row["target_rank"])
        expected_hit = float(rank <= 20)
        if float(row["hit_at_20"]) != expected_hit:
            raise ValueError("Stored hit metric disagrees with target rank")
