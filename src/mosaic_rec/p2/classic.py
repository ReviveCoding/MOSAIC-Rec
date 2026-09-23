"""Exact CPU popularity baselines under the common p2 candidate contract."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.values = np.zeros(size + 1, dtype=np.int64)

    def add(self, position: int) -> None:
        index = position + 1
        while index < len(self.values):
            self.values[index] += 1
            index += index & -index

    def prefix_before(self, position: int) -> int:
        total = 0
        index = position
        while index:
            total += int(self.values[index])
            index -= index & -index
        return total


def _exact_target_ranks(
    *,
    scores: np.ndarray,
    order: np.ndarray,
    order_position: np.ndarray,
    first_seen: np.ndarray,
    query_timestamps: list[int],
    targets: list[int],
    histories: list[set[int]],
) -> list[int]:
    ranks = [0] * len(targets)
    positive_count = int((scores > 0).sum())
    zero_order = order[scores[order] == 0]
    zero_position = np.full(len(scores), -1, dtype=np.int64)
    zero_position[zero_order] = np.arange(len(zero_order))
    zero_by_time = sorted(zero_order, key=lambda item: (first_seen[item], int(item)))
    zero_queries = sorted(
        (query_timestamps[row], row)
        for row, target in enumerate(targets)
        if scores[target] == 0
    )
    fenwick = _Fenwick(len(zero_order))
    cursor = 0
    for stamp, row in zero_queries:
        while cursor < len(zero_by_time) and first_seen[zero_by_time[cursor]] <= stamp:
            fenwick.add(int(zero_position[zero_by_time[cursor]]))
            cursor += 1
        target = targets[row]
        history = histories[row]
        positive_masked = sum(scores[item] > 0 for item in history)
        target_zero_position = int(zero_position[target])
        zero_masked_ahead = sum(
            scores[item] == 0 and zero_position[item] < target_zero_position
            for item in history
        )
        ranks[row] = (
            positive_count
            - positive_masked
            + fenwick.prefix_before(target_zero_position)
            - zero_masked_ahead
            + 1
        )
    for row, target in enumerate(targets):
        if scores[target] == 0:
            continue
        position = int(order_position[target])
        masked_ahead = sum(order_position[item] < position for item in histories[row])
        ranks[row] = position + 1 - masked_ahead
    return ranks


def _model_scores(
    model_id: str,
    item_ids: list[str],
    fit_path: Path,
    item_to_index: dict[str, int],
) -> np.ndarray:
    if model_id == "GLOBAL_POPULARITY":
        counts = np.zeros(len(item_ids), dtype=np.int64)
        for item in pq.read_table(fit_path, columns=["item_id"]).column(0).to_pylist():
            counts[item_to_index[item]] += 1
        return counts
    if model_id == "RECENT_POPULARITY_30D":
        table = pq.read_table(fit_path, columns=["item_id", "timestamp"])
        stamps = table.column("timestamp").to_numpy()
        cutoff = int(stamps.max()) - 30 * 86400
        counts = np.zeros(len(item_ids), dtype=np.int64)
        for item, stamp in zip(table.column("item_id").to_pylist(), stamps):
            if stamp >= cutoff:
                counts[item_to_index[item]] += 1
        return counts
    raise ValueError(model_id)


def score_popularity(
    *,
    model_id: str,
    fit_path: Path,
    query_path: Path,
    item_context_path: Path,
    item_ids_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    item_ids = item_ids_path.read_text().splitlines()
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    scores = _model_scores(model_id, item_ids, fit_path, item_to_index)
    order = np.asarray(
        sorted(range(len(item_ids)), key=lambda i: (-int(scores[i]), item_ids[i])),
        dtype=np.int64,
    )
    order_position = np.empty(len(item_ids), dtype=np.int64)
    order_position[order] = np.arange(len(order))
    first_seen = (
        pq.read_table(item_context_path, columns=["first_observed_timestamp"])
        .column(0)
        .to_numpy()
    )
    fit = pq.read_table(fit_path, columns=["user_id", "item_id"])
    histories: dict[str, set[int]] = defaultdict(set)
    for user, item in zip(
        fit.column("user_id").to_pylist(), fit.column("item_id").to_pylist()
    ):
        histories[user].add(item_to_index[item])
    queries = pq.read_table(query_path)
    values = {name: queries.column(name).to_pylist() for name in queries.column_names}
    targets = [item_to_index[item] for item in values["target_item_id"]]
    history_rows = [
        histories[user] - {target} for user, target in zip(values["user_id"], targets)
    ]
    target_ranks = _exact_target_ranks(
        scores=scores,
        order=order,
        order_position=order_position,
        first_seen=first_seen,
        query_timestamps=values["query_timestamp"],
        targets=targets,
        histories=history_rows,
    )
    rows: dict[str, list[Any]] = defaultdict(list)
    ndcg = defaultdict(float)
    counts = defaultdict(int)
    started = time.perf_counter()
    for row, user in enumerate(values["user_id"]):
        stamp = values["query_timestamp"][row]
        history = history_rows[row]
        top: list[int] = []
        for candidate in order:
            if first_seen[candidate] > stamp or int(candidate) in history:
                continue
            top.append(int(candidate))
            if len(top) == 100:
                break
        rank = target_ranks[row]
        hit = rank <= 20
        row_ndcg = 1 / math.log2(rank + 1) if hit else 0.0
        labels = ["all"]
        if values["is_controlled_zero"][row]:
            labels.append("controlled_zero")
        if values["is_natural_zero"][row]:
            labels.append("natural_zero")
        if values["is_warm"][row]:
            labels.append("warm")
        for label in labels:
            ndcg[label] += row_ndcg
            counts[label] += 1
        rows["user_id"].append(user)
        rows["query_timestamp"].append(stamp)
        rows["target_item_id"].append(values["target_item_id"][row])
        rows["target_rank"].append(rank)
        rows["ndcg_at_20"].append(row_ndcg)
        rows["recall_at_20"].append(float(hit))
        rows["mrr_at_20"].append(1 / rank if hit else 0.0)
        rows["hit_at_20"].append(float(hit))
        rows["precision_at_20"].append(0.05 if hit else 0.0)
        rows["top20_item_ids"].append([item_ids[item] for item in top[:20]])
        rows["top20_scores"].append([float(scores[item]) for item in top[:20]])
        rows["top100_item_ids"].append([item_ids[item] for item in top])
        rows["top100_scores"].append([float(scores[item]) for item in top])
        rows["model_id"].append(model_id)
        rows["protocol_id"].append(PROTOCOL_ID)
        rows["fit_support"].append(values["fit_interactions"][row])
        rows["is_controlled_zero"].append(values["is_controlled_zero"][row])
        rows["is_natural_zero"].append(values["is_natural_zero"][row])
        rows["is_warm"].append(values["is_warm"][row])
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(dict(rows)), output, compression="zstd")
    duration = time.perf_counter() - started
    return {
        "model_id": model_id,
        "rows": len(values["user_id"]),
        "seconds": duration,
        "users_per_second": len(values["user_id"]) / duration,
        "slices": {
            label: {
                "n": counts[label],
                "ndcg_at_20": ndcg[label] / counts[label] if counts[label] else None,
            }
            for label in ("all", "controlled_zero", "natural_zero", "warm")
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def run_classic(
    *,
    repo: Path,
    config_path: Path,
    fit_path: Path,
    query_path: Path,
    item_context_path: Path,
    item_ids_path: Path,
    output_dir: Path,
    receipt: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Classic baseline config protocol mismatch")
    results = []
    for model in config["models"]:
        output = output_dir / f"{model['model_id'].lower()}_validation.parquet"
        results.append(
            score_popularity(
                model_id=model["model_id"],
                fit_path=fit_path,
                query_path=query_path,
                item_context_path=item_context_path,
                item_ids_path=item_ids_path,
                output=output,
            )
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-C00",
        "status": "PASS_DEVELOPMENT",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "config_sha256": sha256_file(config_path),
        "models": results,
        "evidence_class": "p2_measured_development_classic_baselines",
    }
    write_json_new(receipt, result)
    return result
