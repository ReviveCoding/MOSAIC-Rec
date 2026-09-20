"""CUDA LightGCN pilot with FIT-only graph and exact validation ranking."""

from __future__ import annotations

import argparse
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

import pyarrow.parquet as pq
import torch
from torch import nn


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, graph: torch.Tensor) -> None:
        super().__init__()
        self.users = nn.Embedding(n_users + 1, 64)
        self.items = nn.Embedding(n_items + 1, 64)
        self.register_buffer("graph", graph)
        nn.init.normal_(self.users.weight, std=0.02)
        nn.init.normal_(self.items.weight, std=0.02)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        initial = torch.cat([self.users.weight, self.items.weight], dim=0)
        first = torch.sparse.mm(self.graph, initial)
        second = torch.sparse.mm(self.graph, first)
        all_layers = (initial + first + second) / 3
        return all_layers[:self.users.num_embeddings], all_layers[self.users.num_embeddings:]


def read_events(path: Path) -> list[tuple[int, int, int]]:
    table = pq.read_table(path)
    return sorted((int(t), int(u), int(i)) for u, i, t in zip(
        table["user_id"].to_pylist(), table["item_id"].to_pylist(),
        table["timestamp"].to_pylist()))


def build_graph(fit: list[tuple[int, int, int]], n_users: int,
                n_items: int, device: torch.device) -> torch.Tensor:
    edges = sorted({(user, item) for _, user, item in fit})
    n_nodes = n_users + n_items + 2
    users = torch.tensor([u for u, _ in edges], dtype=torch.long, device=device)
    items = torch.tensor([n_users + 1 + i for _, i in edges], dtype=torch.long, device=device)
    rows = torch.cat((users, items))
    cols = torch.cat((items, users))
    degrees = torch.bincount(rows, minlength=n_nodes).float()
    values = torch.rsqrt(degrees[rows] * degrees[cols])
    graph = torch.sparse_coo_tensor(torch.stack((rows, cols)), values,
                                    (n_nodes, n_nodes), device=device).coalesce()
    return graph


def run(split_dir: Path, source: Path, lock: Path, checkpoint: Path,
        epochs: int, seed: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    fit = read_events(split_dir / "FIT_VISIBLE.parquet")
    validation = read_events(split_dir / "VALIDATION.parquet")
    n_users = max(user for _, user, _ in fit)
    n_items = 19738
    device = torch.device("cuda:0")
    graph = build_graph(fit, n_users, n_items, device)
    torch.manual_seed(seed)
    model = LightGCN(n_users, n_items, graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    seen: dict[int, set[int]] = defaultdict(set)
    warm = set()
    for _, user, item in fit:
        seen[user].add(item)
        warm.add(item)
    warm_items = sorted(warm)
    rng = random.Random(seed)
    negative = []
    for _, user, _ in fit:
        item = rng.choice(warm_items)
        while item in seen[user]:
            item = rng.choice(warm_items)
        negative.append(item)
    users = torch.tensor([u for _, u, _ in fit], device=device)
    positives = torch.tensor([i for _, _, i in fit], device=device)
    negatives = torch.tensor(negative, device=device)
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [(user, stamp, item) for user, (stamp, item) in first_val.items() if user in seen]
    cohort = {int(line) for line in (split_dir / "controlled_cold_item_ids.txt").read_text().splitlines()}
    log = []
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(f"pid={__import__('os').getpid()} model=LIGHTGCN\n")
        lock_handle.flush()
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            user_vec, item_vec = model.propagate()
            pos = (user_vec[users] * item_vec[positives]).sum(-1)
            neg = (user_vec[users] * item_vec[negatives]).sum(-1)
            bpr = torch.nn.functional.softplus(neg - pos).mean()
            regularizer = (model.users(users).square().sum() +
                           model.items(positives).square().sum() +
                           model.items(negatives).square().sum()) / (2 * len(fit))
            loss = bpr + 1e-4 * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.append({"epoch": epoch + 1, "fit_bpr_loss": float(bpr.detach())})
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        model.eval()
        with torch.inference_mode():
            user_vec, item_vec = model.propagate()
            catalog = item_vec[1:].clone()
            cold_mask = torch.tensor([i not in warm for i in range(1, n_items + 1)], device=device)
            catalog[cold_mask] = 0
            first_tensor = torch.tensor([first_seen[i] for i in range(1, n_items + 1)], device=device)
            buckets = {"all": [0.0, 0.0, 0], "controlled_cold": [0.0, 0.0, 0],
                       "warm_fit": [0.0, 0.0, 0]}
            score_start = time.perf_counter()
            for offset in range(0, len(eval_rows), 256):
                batch = eval_rows[offset:offset + 256]
                batch_users = torch.tensor([r[0] for r in batch], device=device)
                scores = (user_vec[batch_users] @ catalog.T).to(torch.float64)
                scores -= torch.arange(n_items, device=device, dtype=torch.float64)[None, :] * 1e-12
                stamps = torch.tensor([r[1] for r in batch], device=device)
                scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
                for row, (user, _, _) in enumerate(batch):
                    scores[row, torch.tensor(list(seen[user]), device=device) - 1] = -torch.inf
                top = scores.topk(20, dim=1).indices.cpu().tolist()
                for ranked, (_, _, target) in zip(top, batch):
                    labels = ["all"]
                    if target in cohort:
                        labels.append("controlled_cold")
                    elif target in warm:
                        labels.append("warm_fit")
                    for label in labels:
                        buckets[label][2] += 1
                        if target - 1 in ranked:
                            rank = ranked.index(target - 1) + 1
                            buckets[label][0] += 1 / log2(rank + 1)
                            buckets[label][1] += 1
            torch.cuda.synchronize()
            score_seconds = time.perf_counter() - score_start
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs}, checkpoint)
        peak_vram = torch.cuda.max_memory_allocated()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    slices = {
        label: {"n": int(v[2]), "ndcg_at_20": v[0] / v[2] if v[2] else None,
                "recall_at_20": v[1] / v[2] if v[2] else None}
        for label, v in buckets.items()
    }
    return {
        "stage_id": "C01", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "PILOT_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model": "LIGHTGCN_2L_PILOT", "dataset": "microlens-100k",
        "device": torch.cuda.get_device_name(0), "seed": seed,
        "fit_edges": len(fit), "graph_nnz": graph._nnz(),
        "epochs": epochs, "training_log": log, "train_seconds": train_seconds,
        "score_seconds": score_seconds, "peak_vram_bytes": peak_vram,
        "validation_slices": slices,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": ["Development pilot only; no fair HPO, reference parity, or locked result.",
                     "ID-only graph model is not a sole cold comparator."],
        "evidence_class": "measured_development_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    result = run(args.split_dir, args.source, args.lock,
                 args.checkpoint, args.epochs, args.seed)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
