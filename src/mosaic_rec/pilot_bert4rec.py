"""Local BERT4Rec Cloze pilot on the common MicroLens temporal protocol."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.pilot_mosaic import events


class BERT4Rec(nn.Module):
    def __init__(self, catalog_size: int, width: int = 64, length: int = 20) -> None:
        super().__init__()
        self.catalog_size = catalog_size
        self.mask_id = catalog_size + 1
        self.item = nn.Embedding(catalog_size + 2, width, padding_idx=0)
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

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        values = self.item(sequence) + self.position(positions)[None, :, :]
        encoded = self.encoder(values, src_key_padding_mask=sequence.eq(0))
        return self.norm(encoded)


def cloze_batch(
    sequence: torch.Tensor, lengths: torch.Tensor, mask_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask observed positions, always including the next-item position."""
    observed = sequence.ne(0)
    mask = (torch.rand(sequence.shape, device=sequence.device) < 0.2) & observed
    mask[torch.arange(len(sequence), device=sequence.device), lengths - 1] = True
    labels = sequence[mask]
    masked = sequence.clone()
    masked[mask] = mask_id
    return masked, mask, labels


def run(
    split_dir: Path,
    source: Path,
    checkpoint: Path,
    config: Path,
    epochs: int,
    seed: int,
) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    if epochs < 1:
        raise ValueError("epochs must be positive")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU busy before BERT4Rec pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    fit = events(split_dir / "FIT_VISIBLE.parquet")
    validation = events(split_dir / "VALIDATION.parquet")
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    catalog_size = max(first_seen)
    cold = {
        int(item)
        for item in (split_dir / "controlled_cold_item_ids.txt")
        .read_text()
        .splitlines()
    }
    fit_items = {item for _, _, item in fit}
    if cold.intersection(fit_items):
        raise ValueError("Controlled-cold item has model-visible FIT support")
    train_ids = sorted(fit_items)
    histories: dict[int, list[int]] = defaultdict(list)
    sequences: list[list[int]] = []
    lengths: list[int] = []
    for _, user, item in fit:
        history = histories[user]
        if history:
            window = history[-19:] + [item]
            sequences.append(window + [0] * (20 - len(window)))
            lengths.append(len(window))
        history.append(item)
    if not sequences:
        raise ValueError("No FIT sequence examples")
    first_validation: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_validation.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_validation.items()
        if histories.get(user)
    ]
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    model = BERT4Rec(catalog_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    class_lookup = torch.full((catalog_size + 2,), -1, dtype=torch.long, device=device)
    class_lookup[torch.tensor(train_ids, device=device)] = torch.arange(
        len(train_ids), device=device
    )
    train_ids_tensor = torch.tensor(train_ids, device=device)
    dataset = TensorDataset(
        torch.tensor(np.asarray(sequences), dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    training_log: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        label_count = 0
        for sequence, length in loader:
            sequence = sequence.to(device, non_blocking=True)
            length = length.to(device, non_blocking=True)
            masked, selected, labels = cloze_batch(sequence, length, model.mask_id)
            target_classes = class_lookup[labels]
            if (target_classes < 0).any():
                raise ValueError("Masked target absent from FIT training classes")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = model(masked)[selected]
                logits = states @ model.item(train_ids_tensor).T
                loss = torch.nn.functional.cross_entropy(logits.float(), target_classes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            label_count += len(labels)
        training_log.append(
            {"epoch": epoch + 1, "fit_cloze_loss": loss_sum / label_count}
        )
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    model.eval()
    item_ids = torch.arange(1, catalog_size + 1, device=device)
    catalog = model.item(item_ids).detach().clone()
    unseen = torch.ones(catalog_size + 1, dtype=torch.bool, device=device)
    unseen[train_ids_tensor] = False
    catalog[unseen[1:]] = 0
    first_tensor = torch.tensor(
        [first_seen[item] for item in range(1, catalog_size + 1)], device=device
    )
    buckets = {name: [0.0, 0.0, 0] for name in ("all", "controlled_cold", "warm_fit")}
    score_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(eval_rows), 256):
            batch = eval_rows[offset : offset + 256]
            contexts = [histories[user][-19:] for user, _, _ in batch]
            lengths_batch = torch.tensor(
                [len(context) + 1 for context in contexts], device=device
            )
            sequence = torch.tensor(
                [
                    context + [model.mask_id] + [0] * (20 - len(context) - 1)
                    for context in contexts
                ],
                device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model(sequence)
                states = encoded[
                    torch.arange(len(batch), device=device), lengths_batch - 1
                ]
                scores = states @ catalog.T
            scores = scores.to(torch.float64)
            scores -= (
                torch.arange(catalog_size, device=device, dtype=torch.float64)[None, :]
                * 1e-12
            )
            stamps = torch.tensor([stamp for _, stamp, _ in batch], device=device)
            scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
            for row, (user, _, _) in enumerate(batch):
                scores[
                    row, torch.tensor(histories[user], device=device) - 1
                ] = -torch.inf
            top = scores.topk(20, dim=1).indices.cpu().tolist()
            for ranked, (_, _, target) in zip(top, batch):
                labels = ["all"]
                if target in cold:
                    labels.append("controlled_cold")
                elif target in fit_items:
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
            "n": int(value[2]),
            "ndcg_at_20": value[0] / value[2] if value[2] else None,
            "recall_at_20": value[1] / value[2] if value[2] else None,
        }
        for name, value in buckets.items()
    }
    return {
        "stage_id": "C03",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "LOCAL_COMMON_PROTOCOL_PILOT",
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "model": "B7_BERT4REC_LOCAL",
        "device": torch.cuda.get_device_name(0),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "seed": seed,
        "epochs": epochs,
        "fit_samples": len(dataset),
        "training_log": training_log,
        "train_seconds": train_seconds,
        "score_seconds": score_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "validation_slices": slices,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_sha256": config_sha256,
        "input_sha256": {
            "fit_visible": hashlib.sha256(
                (split_dir / "FIT_VISIBLE.parquet").read_bytes()
            ).hexdigest(),
            "validation": hashlib.sha256(
                (split_dir / "VALIDATION.parquet").read_bytes()
            ).hexdigest(),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "warnings": [
            "Local Cloze adaptation; reference/local parity is not established.",
            "Controlled-cold item IDs are absent from training targets and softmax classes.",
            "Development pilot only; no equal-budget HPO, independent seeds, or locked result.",
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.receipt.exists() or args.checkpoint.exists():
        raise FileExistsError(
            "Pilot output exists; preserve it and choose distinct paths"
        )
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with args.lock.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={__import__('os').getpid()} model=B7_BERT4REC_LOCAL\n")
            handle.flush()
            result = run(
                args.split_dir,
                args.source,
                args.checkpoint,
                args.config,
                args.epochs,
                args.seed,
            )
    except Exception as exc:
        result = {
            "stage_id": "C03",
            "protocol_id": "mosaic-rec-2026-09-20-p1",
            "state": "DEVELOPMENT",
            "status": "FAIL_LOCAL_PILOT",
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "microlens-100k",
            "model": "B7_BERT4REC_LOCAL",
            "failure_class": "runtime_or_implementation_unclassified",
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
