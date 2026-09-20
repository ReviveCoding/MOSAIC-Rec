"""Common FIT-visible BPR-MF pilot with exact CUDA validation scoring."""

from __future__ import annotations

import argparse
import bisect
import csv
import fcntl
import hashlib
import json
import random
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.pilot_lightgcn import read_events


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64) -> None:
        super().__init__()
        self.users = nn.Embedding(n_users + 1, dim)
        self.items = nn.Embedding(n_items + 1, dim)
        nn.init.normal_(self.users.weight, std=0.02)
        nn.init.normal_(self.items.weight, std=0.02)


def time_safe_negatives(
    fit: list[tuple[int, int, int]], seed: int
) -> tuple[list[int], list[int], list[int], dict[int, set[int]], set[int]]:
    first_fit: dict[int, int] = {}
    for stamp, _, item in fit:
        first_fit[item] = min(stamp, first_fit.get(item, stamp))
    warm = set(first_fit)
    available = sorted(warm, key=lambda item: (first_fit[item], item))
    available_times = [first_fit[item] for item in available]
    seen: dict[int, set[int]] = defaultdict(set)
    rng = random.Random(seed)
    training_users, training_items, negatives = [], [], []
    for stamp, user, item in fit:
        limit = bisect.bisect_right(available_times, stamp)
        excluded = seen[user] | {item}
        if len(excluded) < limit:
            negative = available[rng.randrange(limit)]
            while negative in excluded:
                negative = available[rng.randrange(limit)]
            training_users.append(user)
            training_items.append(item)
            negatives.append(negative)
        seen[user].add(item)
    if not training_items:
        raise ValueError("No time-eligible FIT negative samples")
    return training_users, training_items, negatives, seen, warm


def run(
    split_dir: Path, source: Path, checkpoint: Path, epochs: int, seed: int
) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU busy before BPR split pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    fit = read_events(split_dir / "FIT_VISIBLE.parquet")
    validation = read_events(split_dir / "VALIDATION.parquet")
    n_users, n_items = max(user for _, user, _ in fit), 19738
    training_users, training_items, negatives, seen, warm = time_safe_negatives(
        fit, seed
    )
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item) for user, (stamp, item) in first_val.items() if user in seen
    ]
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    cohort = {
        int(x)
        for x in (split_dir / "controlled_cold_item_ids.txt").read_text().splitlines()
    }
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    model = BPRMF(n_users, n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(training_users, dtype=torch.long),
            torch.tensor(training_items, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        ),
        batch_size=8192,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    training_log = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for users, positives, negatives_batch in loader:
            users = users.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            negatives_batch = negatives_batch.to(device, non_blocking=True)
            user_vec = model.users(users)
            pos_vec = model.items(positives)
            neg_vec = model.items(negatives_batch)
            bpr = torch.nn.functional.softplus(
                (user_vec * (neg_vec - pos_vec)).sum(-1)
            ).mean()
            regularizer = (
                user_vec.square().sum(-1)
                + pos_vec.square().sum(-1)
                + neg_vec.square().sum(-1)
            ).mean() / 2
            loss = bpr + 1e-4 * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(bpr.detach()) * len(users)
        training_log.append(
            {"epoch": epoch + 1, "fit_bpr_loss": loss_sum / len(training_items)}
        )
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    model.eval()
    score_start = time.perf_counter()
    buckets = {name: [0.0, 0.0, 0] for name in ("all", "controlled_cold", "warm_fit")}
    with torch.inference_mode():
        catalog = model.items.weight[1:].clone()
        catalog[
            torch.tensor([i not in warm for i in range(1, n_items + 1)], device=device)
        ] = 0
        first = torch.tensor(
            [first_seen[i] for i in range(1, n_items + 1)], device=device
        )
        tie = torch.arange(n_items, device=device, dtype=torch.float64) * 1e-12
        for offset in range(0, len(eval_rows), 256):
            batch = eval_rows[offset : offset + 256]
            users = torch.tensor([user for user, _, _ in batch], device=device)
            scores = (model.users(users) @ catalog.T).to(torch.float64) - tie[None, :]
            stamps = torch.tensor([stamp for _, stamp, _ in batch], device=device)
            scores.masked_fill_(first[None, :] > stamps[:, None], -torch.inf)
            for row, (user, _, _) in enumerate(batch):
                scores[
                    row, torch.tensor(list(seen[user]), device=device) - 1
                ] = -torch.inf
            top = scores.topk(20, dim=1).indices.cpu().tolist()
            for ranked, (_, _, target) in zip(top, batch):
                labels = ["all"]
                if target in cohort:
                    labels.append("controlled_cold")
                elif target in warm:
                    labels.append("warm_fit")
                for name in labels:
                    buckets[name][2] += 1
                    if target - 1 in ranked:
                        rank = ranked.index(target - 1) + 1
                        buckets[name][0] += 1 / log2(rank + 1)
                        buckets[name][1] += 1
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - score_start
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "seed": seed, "epochs": epochs}, checkpoint
    )
    slices = {
        name: {
            "n": int(v[2]),
            "ndcg_at_20": v[0] / v[2] if v[2] else None,
            "recall_at_20": v[1] / v[2] if v[2] else None,
        }
        for name, v in buckets.items()
    }
    return {
        "stage_id": "C00",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "PILOT_PASS",
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "model": "B2_BPR_MF_FIT_VISIBLE",
        "seed": seed,
        "epochs": epochs,
        "fit_samples": len(training_items),
        "device": torch.cuda.get_device_name(0),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": hashlib.sha256(
            Path("configs/bpr_split_pilot_p1.json").read_bytes()
        ).hexdigest(),
        "input_sha256": {
            "fit_visible": hashlib.sha256(
                (split_dir / "FIT_VISIBLE.parquet").read_bytes()
            ).hexdigest(),
            "validation": hashlib.sha256(
                (split_dir / "VALIDATION.parquet").read_bytes()
            ).hexdigest(),
        },
        "fit_interactions": len(fit),
        "validation_slices": slices,
        "training_log": training_log,
        "train_seconds": train_seconds,
        "score_seconds": score_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": [
            "Development pilot only; no HPO, reference parity, or seed qualification.",
            "ID-only model is not a sole cold comparator.",
        ],
        "evidence_class": "measured_development_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.receipt.exists() or args.checkpoint.exists():
        raise FileExistsError("Pilot output already exists; preserve it")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with args.lock.open("a+") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.write(
                f"pid={__import__('os').getpid()} model=BPR_MF_FIT_VISIBLE\n"
            )
            lock_handle.flush()
            result = run(
                args.split_dir, args.source, args.checkpoint, args.epochs, args.seed
            )
    except Exception as exc:
        result = {
            "stage_id": "C00",
            "protocol_id": "mosaic-rec-2026-09-20-p1",
            "state": "DEVELOPMENT",
            "status": "FAIL_RUNTIME",
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "microlens-100k",
            "model": "B2_BPR_MF_FIT_VISIBLE",
            "seed": args.seed,
            "epochs": args.epochs,
            "failure_class": "gpu_contention"
            if "GPU busy" in str(exc)
            else "runtime_unclassified",
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "evidence_class": "measured_development_failure",
        }
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
