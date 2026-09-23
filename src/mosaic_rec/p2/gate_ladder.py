"""Train and exact-score the p2 R2-R4 router ladder on open data only."""

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
from mosaic_rec.p2.models import (
    BERT4RecExpert,
    ContentTwoTower,
    ContextRouterGate,
    MonotonicSupportGate,
)
from mosaic_rec.p2.router import _standardize
from mosaic_rec.p2.runtime import (
    exact_score,
    load_catalog,
    load_histories,
    sequence_tensor,
    validation_queries,
)


def _gate_features(
    *,
    candidate_ids: torch.Tensor,
    query_timestamps: torch.Tensor,
    user_features: torch.Tensor,
    support: torch.Tensor,
    first_seen: torch.Tensor,
    image_present: torch.Tensor,
    text_present: torch.Tensor,
) -> torch.Tensor:
    indices = candidate_ids - 1
    candidate_support = support[indices].float()
    age_days = (query_timestamps[..., None] - first_seen[indices]).clamp_min(
        0
    ).float() / 86400.0
    shape = candidate_support.shape
    history = user_features[:, 0, None].expand(shape)
    recency = user_features[:, 1, None].expand(shape)
    median_popularity = user_features[:, 2, None].expand(shape)
    tail_share = user_features[:, 3, None].expand(shape)
    return torch.stack(
        [
            torch.log1p(candidate_support) / 10.0,
            torch.log1p(age_days) / 10.0,
            torch.log1p(history) / 10.0,
            torch.log1p(recency) / 10.0,
            torch.log1p(median_popularity) / 10.0,
            tail_share,
            image_present[indices].float(),
            text_present[indices].float(),
        ],
        dim=-1,
    )


def _load_experts(
    *,
    ladder: dict[str, Any],
    pilot: dict[str, Any],
    feature_dir: Path,
    catalog_size: int,
    device: torch.device,
) -> tuple[BERT4RecExpert, ContentTwoTower, np.ndarray, np.ndarray, np.ndarray]:
    _, image_np, text_np, tag_np = load_catalog(feature_dir)
    sequence_config = pilot["sequence"]
    content_config = pilot["content"]
    sequence = BERT4RecExpert(
        catalog_size,
        width=int(sequence_config["width"]),
        length=int(sequence_config["max_length"]),
        layers=int(sequence_config["layers"]),
        heads=int(sequence_config["heads"]),
        dropout=float(sequence_config["dropout"]),
    ).to(device)
    sequence.load_state_dict(
        torch.load(ladder["experts"]["sequence_checkpoint"], map_location=device)[
            "model"
        ]
    )
    image = torch.tensor(np.asarray(image_np), device=device)
    text = torch.tensor(np.asarray(text_np), device=device)
    tags = torch.tensor(tag_np.astype(np.int64) + 1, device=device)
    content = ContentTwoTower(
        image,
        text,
        tags,
        width=int(content_config["width"]),
        tag_count=int(tags.max().item()),
    ).to(device)
    content.load_state_dict(
        torch.load(ladder["experts"]["content_checkpoint"], map_location=device)[
            "model"
        ]
    )
    sequence.eval()
    content.eval()
    for parameter in sequence.parameters():
        parameter.requires_grad_(False)
    for parameter in content.parameters():
        parameter.requires_grad_(False)
    return sequence, content, image_np, text_np, tag_np


def train_and_score_gate(
    *,
    repo: Path,
    router_name: str,
    ladder_path: Path,
    pilot_path: Path,
    fit_path: Path,
    query_path: Path,
    context_dir: Path,
    feature_dir: Path,
    checkpoint: Path,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    ladder = json.loads(ladder_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    if ladder["protocol_id"] != PROTOCOL_ID or pilot["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Router ladder protocol mismatch")
    model_config = ladder["models"][router_name]
    common = ladder["common"]
    seed = int(ladder["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0")
    item_ids, _, _, _ = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    max_length = int(pilot["sequence"]["max_length"])
    histories = load_histories(fit_path, item_to_index, max_length + 1)
    sequence, content, _, _, _ = _load_experts(
        ladder=ladder,
        pilot=pilot,
        feature_dir=feature_dir,
        catalog_size=len(item_ids),
        device=device,
    )
    item_context = pq.read_table(context_dir / "item_context.parquet")
    support_np = item_context.column("fit_interactions").to_numpy()
    first_seen_np = item_context.column("first_observed_timestamp").to_numpy()
    image_present_np = item_context.column("image_present").to_numpy()
    text_present_np = item_context.column("text_present").to_numpy()
    support = torch.tensor(support_np, device=device)
    first_seen = torch.tensor(first_seen_np, device=device)
    image_present = torch.tensor(image_present_np, device=device)
    text_present = torch.tensor(text_present_np, device=device)
    tail_np = item_context.column("positive_support_tail").to_numpy(
        zero_copy_only=False
    )

    fit_table = pq.read_table(fit_path, columns=["user_id", "item_id", "timestamp"])
    user_events: dict[str, list[tuple[int, int]]] = {}
    for user, item, stamp in zip(
        fit_table.column("user_id").to_pylist(),
        fit_table.column("item_id").to_pylist(),
        fit_table.column("timestamp").to_pylist(),
    ):
        user_events.setdefault(user, []).append((stamp, item_to_index[item] + 1))
    training = []
    for user, events in user_events.items():
        events.sort()
        if len(events) < 2:
            continue
        context = [item for _, item in events[:-1]][-(max_length - 1) :]
        target_stamp, target = events[-1]
        context_indices = np.asarray(context, dtype=np.int64) - 1
        recency = max(0.0, (target_stamp - events[-2][0]) / 86400.0)
        user_feature = [
            float(len(context)),
            recency,
            float(np.median(support_np[context_indices])),
            float(np.mean(tail_np[context_indices])),
        ]
        training.append((user, context, target, target_stamp, user_feature))

    fit_candidates = np.flatnonzero(support_np > 0)
    fit_candidates = fit_candidates[
        np.argsort(first_seen_np[fit_candidates], kind="stable")
    ]
    candidate_times = first_seen_np[fit_candidates]
    rng = np.random.default_rng(seed)
    negative_count = int(common["negatives_per_user"])
    negatives = np.zeros((len(training), negative_count), dtype=np.int64)
    for row, (_, context, target, stamp, _) in enumerate(training):
        eligible = int(np.searchsorted(candidate_times, stamp, side="right"))
        forbidden = set(context)
        forbidden.add(target)
        for column in range(negative_count):
            while True:
                candidate = int(fit_candidates[rng.integers(eligible)]) + 1
                if candidate not in forbidden:
                    negatives[row, column] = candidate
                    break
    history_tensor, _ = sequence_tensor([row[1] for row in training], max_length)
    target_tensor = torch.tensor([row[2] for row in training])
    stamp_tensor = torch.tensor([row[3] for row in training])
    user_feature_tensor = torch.tensor(
        [row[4] for row in training], dtype=torch.float32
    )
    loader = DataLoader(
        TensorDataset(
            history_tensor,
            target_tensor,
            torch.from_numpy(negatives),
            stamp_tensor,
            user_feature_tensor,
        ),
        batch_size=int(common["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    gate: torch.nn.Module
    if model_config["gate"] == "monotonic":
        gate = MonotonicSupportGate().to(device)
    else:
        gate = ContextRouterGate(hidden=int(model_config["hidden"])).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
    )
    log = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(common["epochs"])):
        gate.train()
        total = 0.0
        seen = 0
        for history, target, negative, stamp, user_feature in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            stamp = stamp.to(device, non_blocking=True)
            user_feature = user_feature.to(device, non_blocking=True)
            candidates = torch.cat([target[:, None], negative], dim=1)
            lengths = history.ne(0).sum(dim=1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                with_mask = history.clone()
                positions = lengths
                with_mask[torch.arange(len(history), device=device), positions] = (
                    sequence.mask_id
                )
                sequence_state = sequence.query(with_mask, lengths + 1)
                sequence_scores = (
                    sequence_state[:, None, :] * sequence.item(candidates)
                ).sum(dim=-1)
                content_state = content.user_vectors(history)
                content_scores = (
                    content_state[:, None, :] * content.item_vectors(candidates)
                ).sum(dim=-1)
                aligned = _standardize(
                    torch.stack([sequence_scores, content_scores], dim=1).float()
                )
            features = _gate_features(
                candidate_ids=candidates,
                query_timestamps=stamp,
                user_features=user_feature,
                support=support,
                first_seen=first_seen,
                image_present=image_present,
                text_present=text_present,
            )
            weight = gate(features)
            combined = weight * aligned[:, 0, :] + (1 - weight) * aligned[:, 1, :]
            ranking_loss = torch.nn.functional.softplus(
                -(combined[:, :1] - combined[:, 1:])
            ).mean()
            distillation = (combined[:, 0] - aligned[:, 0, 0]).square().mean()
            loss = (
                ranking_loss + float(model_config["distillation_weight"]) * distillation
            )
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
        {"model": gate.state_dict(), "router": router_name, "config": model_config},
        checkpoint,
    )

    sequence_catalog_ids = torch.arange(1, len(item_ids) + 1, device=device)
    sequence_catalog = sequence.item(sequence_catalog_ids).detach().clone()
    fit_classes = torch.tensor(
        sorted({item for history in histories.values() for item in history}),
        device=device,
    )
    unseen = torch.ones(len(item_ids) + 1, dtype=torch.bool, device=device)
    unseen[fit_classes] = False
    sequence_catalog[unseen[1:]] = 0
    content_parts = []
    with torch.inference_mode():
        for offset in range(0, len(item_ids), 4096):
            ids = torch.arange(
                offset + 1, min(len(item_ids), offset + 4096) + 1, device=device
            )
            content_parts.append(content.item_vectors(ids).float())
    content_catalog = torch.cat(content_parts)
    queries = validation_queries(query_path)
    query_by_user = {user: row for row, user in enumerate(queries["user_id"])}
    pending_weight: torch.Tensor | None = None

    def score_batch(users: list[str]) -> torch.Tensor:
        nonlocal pending_weight
        sequence_histories = [histories[user][-max_length + 1 :] for user in users]
        sequence_with_mask = [
            history + [sequence.mask_id] for history in sequence_histories
        ]
        seq_tensor, seq_lengths = sequence_tensor(sequence_with_mask, max_length)
        content_tensor, _ = sequence_tensor(
            [
                histories[user][-int(pilot["content"]["max_history"]) :]
                for user in users
            ],
            int(pilot["content"]["max_history"]),
        )
        rows = [query_by_user[user] for user in users]
        stamps = torch.tensor(
            [queries["query_timestamp"][row] for row in rows], device=device
        )
        user_features = torch.tensor(
            [
                [
                    queries["fit_history_length"][row],
                    queries["user_recency_days"][row],
                    queries["median_history_popularity"][row],
                    queries["history_tail_share"][row],
                ]
                for row in rows
            ],
            device=device,
        )
        all_candidates = sequence_catalog_ids[None, :].expand(len(users), -1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            seq_state = sequence.query(seq_tensor.to(device), seq_lengths.to(device))
            sequence_scores = seq_state @ sequence_catalog.T
            content_state = content.user_vectors(content_tensor.to(device))
            content_scores = content_state @ content_catalog.T
            feature = _gate_features(
                candidate_ids=all_candidates,
                query_timestamps=stamps,
                user_features=user_features,
                support=support,
                first_seen=first_seen,
                image_present=image_present,
                text_present=text_present,
            )
            pending_weight = gate(feature).float()
        return torch.stack([sequence_scores, content_scores], dim=1)

    def combine(scores: torch.Tensor) -> torch.Tensor:
        if pending_weight is None:
            raise RuntimeError("Gate weights unavailable")
        aligned = _standardize(scores)
        return (
            pending_weight * aligned[:, 0, :] + (1 - pending_weight) * aligned[:, 1, :]
        )

    gate.eval()
    scoring = exact_score(
        model_id=model_config["model_id"],
        role="VALIDATION",
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        combine_scores=combine,
        output=output,
        batch_size=int(common["score_batch_size"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": f"P2-D0{int(router_name[1])}-PILOT",
        "status": "PASS",
        "router": router_name,
        "model_id": model_config["model_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "ladder_config_sha256": sha256_file(ladder_path),
        "pilot_config_sha256": sha256_file(pilot_path),
        "sequence_checkpoint_sha256": sha256_file(
            Path(ladder["experts"]["sequence_checkpoint"])
        ),
        "content_checkpoint_sha256": sha256_file(
            Path(ladder["experts"]["content_checkpoint"])
        ),
        "seed": seed,
        "training_examples": len(training),
        "training_log": log,
        "train_seconds": train_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_qualification": gpu_qualification,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation": scoring,
        "evidence_class": "p2_measured_development_router_pilot",
    }
    write_json_new(receipt, result)
    return result
