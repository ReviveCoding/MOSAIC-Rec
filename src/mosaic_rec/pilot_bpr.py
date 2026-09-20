"""CUDA BPR-MF pilot on MicroLens FIT, validation only."""

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

import numpy as np
import torch


def run(source: Path, lock: Path, checkpoint: Path, fit_cutoff: int,
        validation_cutoff: int, epochs: int, seed: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is prohibited")
    device = torch.device("cuda:0")
    rng = random.Random(seed)
    events: list[tuple[int, int, int]] = []
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            user, item, stamp = int(row["user"]), int(row["item"]), int(row["timestamp"])
            events.append((stamp, user, item))
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    events.sort()
    users = sorted({user for _, user, _ in events})
    items = sorted(first_seen)
    user_to_idx = {user: i for i, user in enumerate(users)}
    item_to_idx = {item: i for i, item in enumerate(items)}
    fit_pairs: list[tuple[int, int]] = []
    seen: dict[int, set[int]] = defaultdict(set)
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in events:
        if stamp <= fit_cutoff:
            fit_pairs.append((user_to_idx[user], item_to_idx[item]))
            seen[user_to_idx[user]].add(item_to_idx[item])
        elif stamp <= validation_cutoff and user not in first_val:
            first_val[user] = (stamp, item)
    warm_items = sorted({i for _, i in fit_pairs})
    samples = np.asarray(fit_pairs, dtype=np.int64)
    torch.manual_seed(seed)
    user_emb = torch.nn.Embedding(len(users), 64, device=device)
    item_emb = torch.nn.Embedding(len(items), 64, device=device)
    torch.nn.init.normal_(user_emb.weight, std=0.01)
    torch.nn.init.normal_(item_emb.weight, std=0.01)
    optimizer = torch.optim.Adam([*user_emb.parameters(), *item_emb.parameters()], lr=0.003)
    training_log = []
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(f"pid={__import__('os').getpid()} model=BPR_MF_PILOT\n")
        lock_handle.flush()
        start = time.perf_counter()
        for epoch in range(epochs):
            order = np.random.default_rng(seed + epoch).permutation(len(samples))
            total_loss = 0.0
            for offset in range(0, len(order), 8192):
                batch = samples[order[offset:offset + 8192]]
                negatives = []
                for u, _ in batch:
                    negative = rng.choice(warm_items)
                    while negative in seen[int(u)]:
                        negative = rng.choice(warm_items)
                    negatives.append(negative)
                u = torch.as_tensor(batch[:, 0], device=device)
                p = torch.as_tensor(batch[:, 1], device=device)
                n = torch.as_tensor(negatives, device=device)
                positive = (user_emb(u) * item_emb(p)).sum(-1)
                negative = (user_emb(u) * item_emb(n)).sum(-1)
                loss = torch.nn.functional.softplus(negative - positive).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            training_log.append({"epoch": epoch + 1, "fit_bpr_loss": total_loss / len(samples)})
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        with torch.no_grad():
            # ID-only items without FIT evidence receive a zero vector, not a
            # randomly initialized candidate representation.
            item_vectors = item_emb.weight.detach().clone()
            cold_mask = torch.ones(len(items), dtype=torch.bool, device=device)
            cold_mask[torch.as_tensor(warm_items, device=device)] = False
            item_vectors[cold_mask] = 0
            first_tensor = torch.as_tensor([first_seen[i] for i in items], device=device)
            eligible = [(user_to_idx[u], stamp, item_to_idx[item])
                        for u, (stamp, item) in first_val.items() if user_to_idx[u] in seen]
            hit = reciprocal = ndcg = 0.0
            for offset in range(0, len(eligible), 256):
                batch = eligible[offset:offset + 256]
                u = torch.as_tensor([r[0] for r in batch], device=device)
                stamp = torch.as_tensor([r[1] for r in batch], device=device)
                scores = (user_emb(u) @ item_vectors.T).to(torch.float64)
                scores -= torch.arange(len(items), device=device, dtype=torch.float64)[None, :] * 1e-12
                scores.masked_fill_(first_tensor[None, :] > stamp[:, None], -torch.inf)
                for j, (user_idx, _, _) in enumerate(batch):
                    scores[j, list(seen[user_idx])] = -torch.inf
                top = scores.topk(20, dim=1).indices.cpu().tolist()
                for row, (_, _, target) in zip(top, batch):
                    if target in row:
                        rank = row.index(target) + 1
                        hit += 1
                        reciprocal += 1 / rank
                        ndcg += 1 / log2(rank + 1)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"user": user_emb.state_dict(), "item": item_emb.state_dict(),
                    "seed": seed, "fit_cutoff": fit_cutoff}, checkpoint)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    count = len(eligible)
    return {
        "stage_id": "C00", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "PILOT", "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model": "BPR_MF_PILOT", "dataset": "microlens-100k",
        "device": torch.cuda.get_device_name(0), "seed": seed,
        "epochs": epochs, "fit_events": len(fit_pairs),
        "train_seconds": training_seconds, "training_log": training_log,
        "validation_users": count, "ndcg_at_20": ndcg / count,
        "recall_at_20": hit / count, "mrr_at_20": reciprocal / count,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": ["Pilot only; no HPO, seed qualification, cold-cohort protocol, or locked evaluation.",
                     "ID-only cold item vector is zero; cold comparison needs a content-capable model."],
        "evidence_class": "measured_pilot_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-cutoff", type=int, required=True)
    parser.add_argument("--validation-cutoff", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    result = run(args.source, args.lock, args.checkpoint, args.fit_cutoff,
                 args.validation_cutoff, args.epochs, args.seed)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
