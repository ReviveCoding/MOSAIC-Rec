"""CUDA SASRec-style sequential baseline pilot on the frozen MicroLens split."""

from __future__ import annotations

import argparse
import bisect
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
from torch.utils.data import DataLoader, TensorDataset


class SASRec(nn.Module):
    def __init__(self, catalog_size: int, length: int = 20, width: int = 64) -> None:
        super().__init__()
        self.length = length
        self.item = nn.Embedding(catalog_size + 1, width, padding_idx=0)
        self.position = nn.Embedding(length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(seq.shape[1], device=seq.device)
        x = self.item(seq) + self.position(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq.shape[1], seq.shape[1], device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=mask, src_key_padding_mask=seq.eq(0))
        last = x[torch.arange(len(seq), device=seq.device), lengths - 1]
        return self.norm(last)


class GRU4Rec(nn.Module):
    def __init__(self, catalog_size: int, length: int = 20, width: int = 64) -> None:
        super().__init__()
        self.item = nn.Embedding(catalog_size + 1, width, padding_idx=0)
        self.gru = nn.GRU(width, width, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.item(seq))
        last = states[torch.arange(len(seq), device=seq.device), lengths - 1]
        return self.norm(last)


def load_rows(path: Path) -> list[tuple[int, int, int]]:
    table = pq.read_table(path, columns=["user_id", "item_id", "timestamp"])
    users = table["user_id"].to_pylist()
    items = table["item_id"].to_pylist()
    stamps = table["timestamp"].to_pylist()
    return sorted((int(t), int(u), int(i)) for u, i, t in zip(users, items, stamps))


def training_availability(
    fit: list[tuple[int, int, int]], cold_items: set[int]
) -> tuple[list[int], list[int]]:
    first_fit: dict[int, int] = {}
    for stamp, _, item in fit:
        first_fit[item] = min(stamp, first_fit.get(item, stamp))
    if cold_items.intersection(first_fit):
        raise ValueError("Controlled-cold item has model-visible FIT support")
    order = sorted(first_fit, key=lambda item: (first_fit[item], item))
    return order, [first_fit[item] for item in order]


def run(
    split_dir: Path,
    source_first_seen: Path,
    lock: Path,
    checkpoint: Path,
    epochs: int,
    seed: int,
    architecture: str,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    fit = load_rows(split_dir / "FIT_VISIBLE.parquet")
    validation = load_rows(split_dir / "VALIDATION.parquet")
    # First-observed item times are used only for availability <= query time.
    first_seen: dict[int, int] = {}
    import csv

    with source_first_seen.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    catalog_size = max(first_seen)
    cold_items = {
        int(item)
        for item in (split_dir / "controlled_cold_item_ids.txt")
        .read_text()
        .splitlines()
    }
    available_order, available_times = training_availability(fit, cold_items)
    histories: dict[int, list[int]] = defaultdict(list)
    sequences: list[list[int]] = []
    lengths: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    rng = random.Random(seed)
    for stamp, user, item in fit:
        history = histories[user]
        if history:
            seq = history[-20:]
            seen = set(history)
            seen.add(item)
            limit = bisect.bisect_right(available_times, stamp)
            if len(seen) < limit:
                neg = available_order[rng.randrange(limit)]
                while neg in seen:
                    neg = available_order[rng.randrange(limit)]
                sequences.append(seq + [0] * (20 - len(seq)))
                lengths.append(len(seq))
                positives.append(item)
                negatives.append(neg)
        history.append(item)
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_val.items()
        if histories.get(user)
    ]
    dataset = TensorDataset(
        torch.tensor(np.asarray(sequences), dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    model = (
        SASRec(catalog_size) if architecture == "sasrec" else GRU4Rec(catalog_size)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    log: list[dict] = []
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(
            f"pid={__import__('os').getpid()} model={architecture.upper()}_PILOT\n"
        )
        lock_handle.flush()
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            for seq, length, positive, negative in loader:
                seq = seq.to(device, non_blocking=True)
                length = length.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model(seq, length)
                    pos_score = (state * model.item(positive)).sum(-1)
                    neg_score = (state * model.item(negative)).sum(-1)
                    loss = torch.nn.functional.softplus(neg_score - pos_score).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(seq)
            log.append({"epoch": epoch + 1, "fit_bpr_loss": loss_sum / len(dataset)})
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        model.eval()
        warm = {item for _, _, item in fit}
        catalog = model.item.weight[1:].detach().clone()
        cold_mask = torch.tensor(
            [item not in warm for item in range(1, catalog_size + 1)], device=device
        )
        catalog[cold_mask] = 0
        first_tensor = torch.tensor(
            [first_seen[i] for i in range(1, catalog_size + 1)], device=device
        )
        hit = reciprocal = ndcg = 0.0
        score_start = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(eval_rows), 256):
                batch = eval_rows[offset : offset + 256]
                seqs = [histories[user][-20:] for user, _, _ in batch]
                length = torch.tensor([len(seq) for seq in seqs], device=device)
                seq = torch.tensor(
                    [s + [0] * (20 - len(s)) for s in seqs], device=device
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model(seq, length)
                    scores = (state @ catalog.T).to(torch.float32)
                scores = scores.to(torch.float64)
                scores -= (
                    torch.arange(catalog_size, device=device, dtype=torch.float64)[
                        None, :
                    ]
                    * 1e-12
                )
                stamps = torch.tensor([r[1] for r in batch], device=device)
                scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
                for row, (user, _, _) in enumerate(batch):
                    scores[
                        row, torch.tensor(histories[user], device=device) - 1
                    ] = -torch.inf
                top = scores.topk(20, dim=1).indices.cpu().tolist()
                for ranked, (_, _, target) in zip(top, batch):
                    if target - 1 in ranked:
                        rank = ranked.index(target - 1) + 1
                        hit += 1
                        reciprocal += 1 / rank
                        ndcg += 1 / log2(rank + 1)
        torch.cuda.synchronize()
        score_seconds = time.perf_counter() - score_start
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "seed": seed, "epochs": epochs}, checkpoint
        )
        peak_vram = torch.cuda.max_memory_allocated()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return {
        "stage_id": "C03",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "PILOT_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"{architecture.upper()}_ID_PILOT",
        "dataset": "microlens-100k",
        "device": torch.cuda.get_device_name(0),
        "seed": seed,
        "fit_samples": len(dataset),
        "epochs": epochs,
        "training_log": log,
        "train_seconds": train_seconds,
        "score_seconds": score_seconds,
        "peak_vram_bytes": peak_vram,
        "validation_users": len(eval_rows),
        "ndcg_at_20": ndcg / len(eval_rows),
        "recall_at_20": hit / len(eval_rows),
        "mrr_at_20": reciprocal / len(eval_rows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": [
            "Pilot only: no fair HPO, reference parity, seed qualification, or locked result.",
            "ID-only model is not a valid sole cold comparator.",
            "Training negatives use only model-visible FIT items available by the event time.",
        ],
        "evidence_class": "measured_development_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source_first_seen", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--architecture", choices=["sasrec", "gru4rec"], default="sasrec"
    )
    args = parser.parse_args()
    if args.receipt.exists() or args.checkpoint.exists():
        raise FileExistsError(
            "Pilot output exists; preserve prior evidence and use new paths"
        )
    result = run(
        args.split_dir,
        args.source_first_seen,
        args.lock,
        args.checkpoint,
        args.epochs,
        args.seed,
        args.architecture,
    )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
