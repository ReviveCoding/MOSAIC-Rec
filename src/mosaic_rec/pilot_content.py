"""CUDA cold-capable content baselines on frozen MicroLens FIT/validation roles."""

from __future__ import annotations

import argparse
import bisect
import csv
import fcntl
import hashlib
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn


class ContentBPR(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_tags: int,
        text: torch.Tensor,
        image: torch.Tensor,
        tags: torch.Tensor,
        mode: str,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.user = nn.Embedding(n_users + 1, 64)
        self.item_id = nn.Embedding(n_items + 1, 64)
        self.text_projection = nn.Linear(text.shape[1], 64, bias=False)
        self.image_projection = nn.Linear(image.shape[1], 64, bias=False)
        self.tag_embedding = nn.Embedding(n_tags, 64)
        self.register_buffer("text", text)
        self.register_buffer("image", image)
        self.register_buffer("tags", tags)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item_id.weight, std=0.02)

    def item_vector(
        self, ids: torch.Tensor, cold: torch.Tensor | None = None
    ) -> torch.Tensor:
        visual = self.image_projection(self.image[ids - 1])
        if self.mode == "vbpr":
            collaborative = self.item_id(ids)
            if cold is not None:
                collaborative = collaborative.masked_fill(cold[..., None], 0)
            return collaborative + visual
        textual = self.text_projection(self.text[ids - 1])
        metadata = self.tag_embedding(self.tags[ids - 1])
        return (textual + visual + metadata) / 3

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item_vector(items)).sum(-1)


def parquet_events(path: Path) -> list[tuple[int, int, int]]:
    table = pq.read_table(path, columns=["user_id", "item_id", "timestamp"])
    return sorted(
        (int(t), int(u), int(i))
        for u, i, t in zip(
            table["user_id"].to_pylist(),
            table["item_id"].to_pylist(),
            table["timestamp"].to_pylist(),
        )
    )


def load_features(feature_dir: Path, tag_file: Path, device: torch.device):
    text = np.load(feature_dir / "MicroLens-100k_title_en_text_features_BgeM3.npy")
    image = np.load(feature_dir / "MicroLens-100k_image_features_CLIPRN50.npy")
    if text.shape != image.shape or text.shape[0] != 19738:
        raise ValueError("Feature shape or item mapping mismatch")
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-8)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
    tag_values: dict[int, str] = {}
    with tag_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key, _, tag = line.partition(",")
            tag_values[int(key)] = tag.strip()
    if set(tag_values) != set(range(1, 19739)):
        raise ValueError("Tag item mapping mismatch")
    tag_vocab = {
        tag: index for index, tag in enumerate(sorted(set(tag_values.values())))
    }
    tags = torch.tensor(
        [tag_vocab[tag_values[i]] for i in range(1, 19739)], device=device
    )
    return (
        torch.as_tensor(text, device=device),
        torch.as_tensor(image, device=device),
        tags,
        len(tag_vocab),
    )


def run(
    mode: str,
    split_dir: Path,
    interaction_csv: Path,
    feature_dir: Path,
    tag_file: Path,
    lock: Path,
    checkpoint: Path,
    epochs: int,
    seed: int,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    if mode not in {"vbpr", "two_tower_content"}:
        raise ValueError(mode)
    device = torch.device("cuda:0")
    fit = parquet_events(split_dir / "FIT_VISIBLE.parquet")
    validation = parquet_events(split_dir / "VALIDATION.parquet")
    first_seen: dict[int, int] = {}
    with interaction_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    warm_first: dict[int, int] = {}
    for stamp, _, item in fit:
        warm_first[item] = min(stamp, warm_first.get(item, stamp))
    available = sorted(warm_first, key=lambda i: (warm_first[i], i))
    available_times = [warm_first[i] for i in available]
    fit_seen: dict[int, set[int]] = defaultdict(set)
    rng = random.Random(seed)
    user_ids, positives, negative_by_epoch = [], [], [[] for _ in range(epochs)]
    for stamp, user, item in fit:
        limit = bisect.bisect_right(available_times, stamp)
        excluded = fit_seen[user] | {item}
        if len(excluded) < limit:
            user_ids.append(user)
            positives.append(item)
            for negatives in negative_by_epoch:
                neg = available[rng.randrange(limit)]
                while neg in excluded:
                    neg = available[rng.randrange(limit)]
                negatives.append(neg)
        fit_seen[user].add(item)
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_val.items()
        if user in fit_seen
    ]
    text, image, tags, n_tags = load_features(feature_dir, tag_file, device)
    torch.manual_seed(seed)
    model = ContentBPR(
        max(user for _, user, _ in fit), 19738, n_tags, text, image, tags, mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    user_tensor = torch.tensor(user_ids, device=device)
    pos_tensor = torch.tensor(positives, device=device)
    log = []
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(f"pid={__import__('os').getpid()} model={mode}\n")
        lock_handle.flush()
        train_start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            neg_tensor = torch.tensor(negative_by_epoch[epoch], device=device)
            order = torch.randperm(len(user_tensor), device=device)
            loss_sum = 0.0
            for offset in range(0, len(order), 8192):
                idx = order[offset : offset + 8192]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pos = model.score(user_tensor[idx], pos_tensor[idx])
                    neg = model.score(user_tensor[idx], neg_tensor[idx])
                    loss = torch.nn.functional.softplus(neg - pos).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(idx)
            log.append(
                {"epoch": epoch + 1, "fit_bpr_loss": loss_sum / len(user_tensor)}
            )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        model.eval()
        item_ids = torch.arange(1, 19739, device=device)
        cold_mask = torch.tensor(
            [i not in warm_first for i in range(1, 19739)], device=device
        )
        first_tensor = torch.tensor(
            [first_seen[i] for i in range(1, 19739)], device=device
        )
        cold_cohort = {
            int(line)
            for line in (split_dir / "controlled_cold_item_ids.txt")
            .read_text()
            .splitlines()
        }
        metrics = {
            "all": [0.0, 0.0, 0.0, 0],
            "controlled_cold": [0.0, 0.0, 0.0, 0],
            "warm_fit": [0.0, 0.0, 0.0, 0],
        }
        score_start = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                catalog = model.item_vector(item_ids, cold_mask).to(torch.float32)
            for offset in range(0, len(eval_rows), 256):
                batch = eval_rows[offset : offset + 256]
                users = torch.tensor([r[0] for r in batch], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores = model.user(users) @ catalog.T
                scores = scores.to(torch.float64)
                scores -= (
                    torch.arange(19738, device=device, dtype=torch.float64)[None, :]
                    * 1e-12
                )
                stamps = torch.tensor([r[1] for r in batch], device=device)
                scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
                for row, (user, _, _) in enumerate(batch):
                    scores[
                        row, torch.tensor(list(fit_seen[user]), device=device) - 1
                    ] = -torch.inf
                top = scores.topk(20, dim=1).indices.cpu().tolist()
                for ranked, (_, _, target) in zip(top, batch):
                    slices = ["all"]
                    if target in cold_cohort:
                        slices.append("controlled_cold")
                    elif target in warm_first:
                        slices.append("warm_fit")
                    for slice_name in slices:
                        metric = metrics[slice_name]
                        metric[3] += 1
                        if target - 1 in ranked:
                            rank = ranked.index(target - 1) + 1
                            metric[0] += 1 / log2(rank + 1)
                            metric[1] += 1
                            metric[2] += 1 / rank
        torch.cuda.synchronize()
        score_seconds = time.perf_counter() - score_start
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "mode": mode, "seed": seed}, checkpoint
        )
        peak_vram = torch.cuda.max_memory_allocated()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    slices = {
        key: {
            "n": int(v[3]),
            "ndcg_at_20": v[0] / v[3] if v[3] else None,
            "recall_at_20": v[1] / v[3] if v[3] else None,
            "mrr_at_20": v[2] / v[3] if v[3] else None,
        }
        for key, v in metrics.items()
    }
    return {
        "stage_id": "C04" if mode == "vbpr" else "C02",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "PILOT_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model": mode.upper(),
        "dataset": "microlens-100k",
        "device": torch.cuda.get_device_name(0),
        "seed": seed,
        "fit_samples": len(user_ids),
        "epochs": epochs,
        "training_log": log,
        "train_seconds": train_seconds,
        "score_seconds": score_seconds,
        "peak_vram_bytes": peak_vram,
        "validation_slices": slices,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": [
            "Development pilot only; no equal-budget HPO, reference parity, or locked result.",
            "Untimestamped historical content revisions are not ruled out.",
        ],
        "evidence_class": "measured_development_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["vbpr", "two_tower_content"])
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("interaction_csv", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("tag_file", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    result = run(
        args.mode,
        args.split_dir,
        args.interaction_csv,
        args.feature_dir,
        args.tag_file,
        args.lock,
        args.checkpoint,
        args.epochs,
        args.seed,
    )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
