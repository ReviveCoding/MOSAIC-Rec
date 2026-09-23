"""CUDA training and exact validation for the p1-style M1 p2 control."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new
from mosaic_rec.p2.models import LegacyM1
from mosaic_rec.p2.runtime import (
    exact_metric_summary,
    exact_score,
    load_catalog,
    sequence_tensor,
    validation_queries,
)


def train_and_score_legacy(
    *,
    repo: Path,
    config_path: Path,
    fit_path: Path,
    query_path: Path,
    context_dir: Path,
    feature_dir: Path,
    checkpoint: Path,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
    hpo_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Legacy config protocol mismatch")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    item_ids, image_np, text_np, tag_np = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    fit = pq.read_table(fit_path, columns=["user_id", "item_id", "timestamp"])
    events = sorted(
        zip(
            fit.column("timestamp").to_pylist(),
            fit.column("user_id").to_pylist(),
            fit.column("item_id").to_pylist(),
        )
    )
    item_context = pq.read_table(context_dir / "item_context.parquet")
    first_seen_np = item_context.column("first_observed_timestamp").to_numpy()
    final_support_np = item_context.column("fit_interactions").to_numpy()
    available = np.flatnonzero(final_support_np > 0)
    available = available[np.argsort(first_seen_np[available], kind="stable")]
    available_times = first_seen_np[available]
    counts = np.zeros(len(item_ids), dtype=np.int64)
    histories: dict[str, list[int]] = defaultdict(list)
    history_supports: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)
    training: list[tuple[list[int], list[int], int, int, int, int]] = []
    rng = np.random.default_rng(seed)
    max_length = int(config["max_length"])
    for stamp, user, item in events:
        item_index = item_to_index[item]
        model_id = item_index + 1
        if histories[user]:
            eligible = int(np.searchsorted(available_times, stamp, side="right"))
            forbidden = seen[user] | {model_id}
            if eligible > len(forbidden):
                while True:
                    negative_index = int(available[rng.integers(eligible)])
                    negative = negative_index + 1
                    if negative not in forbidden:
                        break
                training.append(
                    (
                        histories[user][-max_length:],
                        history_supports[user][-max_length:],
                        model_id,
                        int(counts[item_index]),
                        negative,
                        int(counts[negative_index]),
                    )
                )
        histories[user].append(model_id)
        history_supports[user].append(int(counts[item_index]))
        seen[user].add(model_id)
        counts[item_index] += 1
    sequences, lengths = sequence_tensor([row[0] for row in training], max_length)
    supports = np.zeros_like(sequences.numpy())
    for index, row in enumerate(training):
        values = row[1][-max_length:]
        supports[index, : len(values)] = values
    dataset = TensorDataset(
        sequences,
        torch.from_numpy(supports),
        lengths,
        torch.tensor([row[2] for row in training]),
        torch.tensor([row[3] for row in training]),
        torch.tensor([row[4] for row in training]),
        torch.tensor([row[5] for row in training]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    device = torch.device("cuda:0")
    image = torch.tensor(np.asarray(image_np), device=device)
    text = torch.tensor(np.asarray(text_np), device=device)
    tags = torch.tensor(tag_np.astype(np.int64) + 1, device=device)
    image_present = torch.tensor(
        np.load(feature_dir / "image_present.npy"), device=device
    )
    text_present = torch.tensor(
        np.load(feature_dir / "text_present.npy"), device=device
    )
    model = LegacyM1(
        image,
        text,
        tags,
        image_present,
        text_present,
        width=int(config["width"]),
        length=max_length,
        tag_count=int(tags.max()),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    queries = validation_queries(query_path)
    final_support = torch.tensor(final_support_np, device=device)
    first_seen = torch.tensor(first_seen_np, device=device)
    catalog_ids = torch.arange(1, len(item_ids) + 1, device=device)

    def build_score_batch() -> Any:
        model.eval()
        with torch.inference_mode():
            catalog = model.fuse(catalog_ids, final_support)

        def score_batch(users: list[str]) -> torch.Tensor:
            sequence, length = sequence_tensor(
                [histories[user][-max_length:] for user in users], max_length
            )
            support_values = np.zeros_like(sequence.numpy())
            for row, user in enumerate(users):
                values = history_supports[user][-max_length:]
                support_values[row, : len(values)] = values
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = model.user_state(
                    sequence.to(device),
                    torch.from_numpy(support_values).to(device),
                    length.to(device),
                )
                return state @ catalog.T

        return score_batch

    log = []
    best_metric = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    stopping_reason = "scheduled_epochs_complete"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        model.train()
        total = 0.0
        seen_examples = 0
        for (
            sequence,
            support,
            length,
            positive,
            pos_support,
            negative,
            neg_support,
        ) in loader:
            sequence = sequence.to(device, non_blocking=True)
            support = support.to(device, non_blocking=True)
            length = length.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            pos_support = pos_support.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            neg_support = neg_support.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = model.user_state(sequence, support, length)
                positive_score = (state * model.fuse(positive, pos_support)).sum(-1)
                negative_score = (state * model.fuse(negative, neg_support)).sum(-1)
                loss = torch.nn.functional.softplus(
                    negative_score - positive_score
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(sequence)
            seen_examples += len(sequence)
        log.append({"epoch": epoch + 1, "loss": total / seen_examples})
        if hpo_policy is not None:
            summary = exact_metric_summary(
                queries=queries,
                histories=dict(histories),
                item_ids=item_ids,
                first_seen=first_seen,
                score_batch=build_score_batch(),
                batch_size=int(config["score_batch_size"]),
            )
            metric = float(summary["all"]["ndcg_at_20"] or 0.0)
            log[-1]["validation_ndcg_at_20"] = metric
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
            if stale_epochs >= int(hpo_policy["early_stopping_patience"]):
                stopping_reason = "early_stopping_patience"
                break
            elapsed_hours = (time.perf_counter() - started) / 3600.0
            if elapsed_hours >= float(hpo_policy["maximum_gpu_hours_per_trial"]):
                stopping_reason = "gpu_hour_ceiling"
                break
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    score_batch = build_score_batch()

    scoring = exact_score(
        model_id=config["model_id"],
        role="VALIDATION",
        queries=queries,
        histories=dict(histories),
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        output=output,
        batch_size=int(config["score_batch_size"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": config["stage_id"],
        "status": "PASS",
        "model_id": config["model_id"],
        "trial_id": str(config.get("trial_id", "")),
        "family": str(config.get("family", "LEGACY_M1")),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "config_sha256": sha256_file(config_path),
        "seed": seed,
        "training_examples": len(training),
        "training_log": log,
        "epochs_completed": len(log),
        "best_epoch": best_epoch if hpo_policy is not None else len(log),
        "stopping_reason": stopping_reason,
        "train_seconds": train_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_qualification": gpu_qualification,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation": scoring,
        "evidence_class": str(
            config.get("evidence_class", "p2_measured_development_legacy_pilot")
        ),
    }
    write_json_new(receipt, result)
    return result
