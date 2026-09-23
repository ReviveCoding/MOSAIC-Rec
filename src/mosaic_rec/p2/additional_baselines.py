"""CUDA pilots for p2 BPR-MF, SASRec, and VBPR baselines."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new
from mosaic_rec.p2.models import BPRMF, VBPR, LightGCN, SASRecExpert
from mosaic_rec.p2.runtime import (
    exact_score,
    load_catalog,
    sequence_tensor,
    validation_queries,
)


def _training_rows(
    fit_path: Path,
    item_to_index: dict[str, int],
    max_length: int,
    seed: int,
    negatives_per_user: int,
    first_seen: np.ndarray,
    fit_support: np.ndarray,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    table = pq.read_table(fit_path, columns=["user_id", "item_id", "timestamp"])
    events: dict[str, list[tuple[int, int]]] = {}
    for user, item, stamp in zip(
        table.column("user_id").to_pylist(),
        table.column("item_id").to_pylist(),
        table.column("timestamp").to_pylist(),
    ):
        events.setdefault(user, []).append((stamp, item_to_index[item] + 1))
    users: list[str] = []
    histories: list[list[int]] = []
    targets: list[int] = []
    target_stamps: list[int] = []
    all_histories: dict[str, list[int]] = {}
    for user, rows in events.items():
        rows.sort()
        all_histories[user] = [item for _, item in rows]
        if len(rows) < 2:
            continue
        users.append(user)
        histories.append([item for _, item in rows[:-1]][-max_length:])
        target_stamps.append(rows[-1][0])
        targets.append(rows[-1][1])
    candidates = np.flatnonzero(fit_support > 0)
    candidates = candidates[np.argsort(first_seen[candidates], kind="stable")]
    candidate_times = first_seen[candidates]
    rng = np.random.default_rng(seed)
    negatives = np.zeros((len(users), negatives_per_user), dtype=np.int64)
    for row, (history, target, stamp) in enumerate(
        zip(histories, targets, target_stamps)
    ):
        eligible = int(np.searchsorted(candidate_times, stamp, side="right"))
        forbidden = set(history)
        forbidden.add(target)
        for column in range(negatives_per_user):
            while True:
                candidate = int(candidates[rng.integers(eligible)]) + 1
                if candidate not in forbidden:
                    negatives[row, column] = candidate
                    break
    history_tensor, _ = sequence_tensor(histories, max_length)
    return (
        users,
        history_tensor,
        torch.tensor(targets, dtype=torch.long),
        torch.from_numpy(negatives),
        all_histories,
    )


def run_baseline(
    *,
    repo: Path,
    baseline: str,
    config_path: Path,
    fit_path: Path,
    query_path: Path,
    context_dir: Path,
    feature_dir: Path,
    checkpoint: Path,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Additional baseline protocol mismatch")
    common = config["common"]
    model_config = config["models"][baseline]
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    item_ids, image_np, _, _ = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    item_context = pq.read_table(context_dir / "item_context.parquet")
    first_seen_np = item_context.column("first_observed_timestamp").to_numpy()
    fit_support = item_context.column("fit_interactions").to_numpy()
    users, histories_tensor, targets, negatives, all_histories = _training_rows(
        fit_path,
        item_to_index,
        int(common["max_length"]),
        seed,
        int(common["negatives_per_user"]),
        first_seen_np,
        fit_support,
    )
    user_to_index = {user: index for index, user in enumerate(sorted(all_histories))}
    user_indices = torch.tensor([user_to_index[user] for user in users])
    device = torch.device("cuda:0")
    width = int(common["width"])
    adjacency: torch.Tensor | None = None
    if baseline == "BPR_MF":
        model: torch.nn.Module = BPRMF(len(user_to_index), len(item_ids), width).to(
            device
        )
    elif baseline == "LIGHTGCN":
        model = LightGCN(
            len(user_to_index),
            len(item_ids),
            width,
            layers=int(model_config["layers"]),
        ).to(device)
        fit_table = pq.read_table(fit_path, columns=["user_id", "item_id"])
        edge_users = torch.tensor(
            [user_to_index[user] for user in fit_table.column("user_id").to_pylist()],
            device=device,
        )
        edge_items = torch.tensor(
            [item_to_index[item] for item in fit_table.column("item_id").to_pylist()],
            device=device,
        )
        item_nodes = edge_items + len(user_to_index)
        source = torch.cat([edge_users, item_nodes])
        target_nodes = torch.cat([item_nodes, edge_users])
        node_count = len(user_to_index) + len(item_ids)
        degree = torch.bincount(source, minlength=node_count).float().clamp_min(1)
        weights = degree[source].rsqrt() * degree[target_nodes].rsqrt()
        adjacency = torch.sparse_coo_tensor(
            torch.stack([source, target_nodes]),
            weights,
            (node_count, node_count),
            device=device,
        ).coalesce()
    elif baseline == "VBPR":
        image = torch.tensor(np.asarray(image_np), device=device)
        model = VBPR(len(user_to_index), len(item_ids), image, width).to(device)
    elif baseline == "SASREC":
        model = SASRecExpert(
            len(item_ids),
            width=width,
            length=int(common["max_length"]),
            layers=int(model_config["layers"]),
            heads=int(model_config["heads"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
    else:
        raise ValueError(baseline)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(user_indices, histories_tensor, targets, negatives),
        batch_size=int(common["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    fit_classes = torch.tensor(np.flatnonzero(fit_support > 0) + 1, device=device)
    class_lookup = torch.full((len(item_ids) + 1,), -1, dtype=torch.long, device=device)
    class_lookup[fit_classes] = torch.arange(len(fit_classes), device=device)
    log = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(int(common["epochs"])):
        model.train()
        total = 0.0
        seen = 0
        if baseline == "LIGHTGCN":
            assert adjacency is not None
            user_all, item_all = model(adjacency)
            all_users = user_indices.to(device)
            all_targets = targets.to(device) - 1
            all_negatives = negatives.to(device) - 1
            positive = (user_all[all_users] * item_all[all_targets]).sum(dim=-1)
            negative_score = (
                user_all[all_users, None, :] * item_all[all_negatives]
            ).sum(dim=-1)
            loss = torch.nn.functional.softplus(
                -(positive[:, None] - negative_score)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total = float(loss.detach()) * len(all_users)
            seen = len(all_users)
        for user_index, history, target, negative in (
            [] if baseline == "LIGHTGCN" else loader
        ):
            user_index = user_index.to(device, non_blocking=True)
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if baseline == "SASREC":
                    lengths = history.ne(0).sum(dim=1)
                    state = model.query(history, lengths)
                    logits = state @ model.item(fit_classes).T
                    loss = torch.nn.functional.cross_entropy(
                        logits.float(), class_lookup[target]
                    )
                else:
                    # MF and VBPR use zero-based catalog indices. Training rows
                    # use one-based model IDs so that zero remains sequence padding.
                    item_target = target - 1
                    item_negative = negative - 1
                    positive = model.scores(user_index, item_target[:, None]).squeeze(1)
                    negative_score = model.scores(user_index, item_negative)
                    loss = torch.nn.functional.softplus(
                        -(positive[:, None] - negative_score)
                    ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(history)
            seen += len(history)
        log.append({"epoch": epoch + 1, "loss": total / seen})
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "baseline": baseline, "config": model_config},
        checkpoint,
    )
    model.eval()
    queries = validation_queries(query_path)
    first_seen = torch.tensor(first_seen_np, device=device)
    if baseline == "SASREC":
        catalog = (
            model.item(torch.arange(1, len(item_ids) + 1, device=device))
            .detach()
            .clone()
        )
        unseen = torch.tensor(fit_support == 0, device=device)
        catalog[unseen] = 0

        def score_batch(batch_users: list[str]) -> torch.Tensor:
            tensor, lengths = sequence_tensor(
                [all_histories[user] for user in batch_users], int(common["max_length"])
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = model.query(tensor.to(device), lengths.to(device))
                return state @ catalog.T

    elif baseline == "BPR_MF":
        catalog = model.item.weight.detach().clone()
        catalog[torch.tensor(fit_support == 0, device=device)] = 0

        def score_batch(batch_users: list[str]) -> torch.Tensor:
            indices = torch.tensor(
                [user_to_index[user] for user in batch_users], device=device
            )
            return model.user(indices) @ catalog.T

    elif baseline == "LIGHTGCN":
        assert adjacency is not None
        with torch.inference_mode():
            user_catalog, item_catalog = model(adjacency)
            item_catalog = item_catalog.clone()
            item_catalog[torch.tensor(fit_support == 0, device=device)] = 0

        def score_batch(batch_users: list[str]) -> torch.Tensor:
            indices = torch.tensor(
                [user_to_index[user] for user in batch_users], device=device
            )
            return user_catalog[indices] @ item_catalog.T

    else:
        catalog_ids = torch.arange(len(item_ids), device=device)

        def score_batch(batch_users: list[str]) -> torch.Tensor:
            indices = torch.tensor(
                [user_to_index[user] for user in batch_users], device=device
            )
            return model.scores(indices, catalog_ids[None, :].expand(len(indices), -1))

    scoring = exact_score(
        model_id=model_config["model_id"],
        role="VALIDATION",
        queries=queries,
        histories=all_histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        output=output,
        batch_size=int(common["score_batch_size"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": model_config["stage_id"],
        "status": "PASS",
        "baseline": baseline,
        "model_id": model_config["model_id"],
        "trial_id": str(config.get("trial_id", "")),
        "family": str(config.get("family", baseline)),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "config_sha256": sha256_file(config_path),
        "seed": seed,
        "training_examples": len(users),
        "training_log": log,
        "train_seconds": train_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_qualification": gpu_qualification,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation": scoring,
        "evidence_class": str(
            config.get("evidence_class", "p2_measured_development_baseline_pilot")
        ),
    }
    write_json_new(receipt, result)
    return result
