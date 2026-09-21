"""Time-safe MOSAIC sequence-fusion development ladder on MicroLens."""

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
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.hpo_runtime import learning_rate


class MosaicSequence(nn.Module):
    def __init__(
        self,
        variant: int,
        text: torch.Tensor,
        image: torch.Tensor,
        tags: torch.Tensor,
        text_present: torch.Tensor,
        n_tags: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        if text.ndim != 2 or image.ndim != 2 or text.shape[0] != image.shape[0]:
            raise ValueError("Text and image buffers must align by item row")
        self.item = nn.Embedding(int(text.shape[0]) + 1, 64, padding_idx=0)
        self.text_projection = nn.Linear(int(text.shape[1]), 64, bias=False)
        self.image_projection = nn.Linear(int(image.shape[1]), 64, bias=False)
        self.tag_embedding = nn.Embedding(n_tags, 64)
        self.position = nn.Embedding(20, 64)
        self.global_logits = nn.Parameter(torch.zeros(4))
        self.support_gate = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 4))
        self.mono_base = nn.Parameter(torch.zeros(4))
        self.mono_slope = nn.Parameter(torch.tensor(0.0))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.register_buffer("text", text)
        self.register_buffer("image", image)
        self.register_buffer("tags", tags)
        self.register_buffer("text_present", text_present)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def fuse(
        self,
        ids: torch.Tensor,
        support: torch.Tensor,
        augment: bool = False,
        *,
        text_available: torch.Tensor | None = None,
        image_available: torch.Tensor | None = None,
        metadata_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = ids.clamp_min(1) - 1
        collaborative = self.item(ids)
        textual = self.text_projection(self.text[safe])
        visual = self.image_projection(self.image[safe])
        metadata = self.tag_embedding(self.tags[safe])
        branches = torch.stack((collaborative, textual, visual, metadata), dim=-2)
        active = ids.ne(0)
        available = torch.stack(
            (active & support.gt(0), active & self.text_present[safe], active, active),
            dim=-1,
        )
        if text_available is not None:
            available[..., 1] &= text_available.bool()
        if image_available is not None:
            available[..., 2] &= image_available.bool()
        if metadata_available is not None:
            available[..., 3] &= metadata_available.bool()
        base_available = available.clone()
        if augment and self.training and self.variant >= 4:
            behavioral = torch.rand_like(support.float()) < 0.2
            available[..., 0] &= ~behavioral
        if augment and self.training and self.variant >= 5:
            dropout = torch.rand((*ids.shape, 3), device=ids.device) < 0.1
            available[..., 1:] &= ~dropout
            none_left = active & ~available.any(-1)
            available[..., 2] |= none_left & base_available[..., 2]
            remaining = none_left & ~base_available[..., 2]
            available[..., 3] |= remaining & base_available[..., 3]
            remaining &= ~base_available[..., 3]
            available[..., 1] |= remaining & base_available[..., 1]
            remaining &= ~base_available[..., 1]
            available[..., 0] |= remaining & base_available[..., 0]
        if self.variant <= 1:
            logits = torch.zeros((*ids.shape, 4), device=ids.device)
        elif self.variant == 2:
            logits = self.global_logits.expand(*ids.shape, 4)
        elif self.variant <= 6:
            logits = self.support_gate(torch.log1p(support.float())[..., None])
        else:
            logits = self.mono_base.expand(*ids.shape, 4).clone()
            logits[..., 0] += torch.nn.functional.softplus(
                self.mono_slope
            ) * torch.log1p(support.float())
        logits = logits.masked_fill(~available, -torch.inf)
        logits = torch.where(
            available.any(-1)[..., None], logits, torch.zeros_like(logits)
        )
        weights = torch.softmax(logits, dim=-1) * available.float()
        if self.variant == 0:
            # Naive sum retains branch count/scale differences by design.
            fused = (branches * available[..., None]).sum(-2)
        else:
            fused = (branches * weights[..., None]).sum(-2)
        fused = fused.masked_fill(~active[..., None], 0)
        weights = weights.masked_fill(~active[..., None], 0)
        return fused, weights

    def user_state(
        self,
        seq: torch.Tensor,
        seq_support: torch.Tensor,
        length: torch.Tensor,
        *,
        text_available: torch.Tensor | None = None,
        image_available: torch.Tensor | None = None,
        metadata_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vectors, _ = self.fuse(
            seq,
            seq_support,
            augment=True,
            text_available=text_available,
            image_available=image_available,
            metadata_available=metadata_available,
        )
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = vectors + self.position(positions)[None, :, :]
        mask = torch.triu(
            torch.ones(seq.shape[1], seq.shape[1], device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(x, mask=mask, src_key_padding_mask=seq.eq(0))
        last = encoded[torch.arange(len(seq), device=seq.device), length - 1]
        return self.norm(last)


def events(path: Path) -> list[tuple[int, int, int]]:
    table = pq.read_table(path)
    return sorted(
        (int(t), int(u), int(i))
        for u, i, t in zip(
            table["user_id"].to_pylist(),
            table["item_id"].to_pylist(),
            table["timestamp"].to_pylist(),
        )
    )


def load_content(
    feature_dir: Path, tag_file: Path, title_file: Path, device: torch.device
):
    text = np.load(feature_dir / "MicroLens-100k_title_en_text_features_BgeM3.npy")
    image = np.load(feature_dir / "MicroLens-100k_image_features_CLIPRN50.npy")
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-8)
    tags: dict[int, str] = {}
    with tag_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            item, _, tag = line.partition(",")
            tags[int(item)] = tag.strip()
    vocab = {tag: index for index, tag in enumerate(sorted(set(tags.values())))}
    titles: dict[int, str] = {}
    with title_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            item, _, title = line.partition(",")
            titles[int(item)] = title.strip()
    if set(tags) != set(range(1, 19739)) or set(titles) != set(tags):
        raise ValueError("Content item mapping mismatch")
    return (
        torch.as_tensor(text, device=device),
        torch.as_tensor(image, device=device),
        torch.tensor([vocab[tags[i]] for i in range(1, 19739)], device=device),
        torch.tensor([bool(titles[i]) for i in range(1, 19739)], device=device),
        len(vocab),
    )


def run(
    variant: int,
    split_dir: Path,
    source: Path,
    feature_dir: Path,
    tag_file: Path,
    title_file: Path,
    lock: Path,
    checkpoint: Path,
    epochs: int,
    seed: int,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    if variant not in range(8):
        raise ValueError("Only M0-M7 are development candidates")
    fit = events(split_dir / "FIT_VISIBLE.parquet")
    validation = events(split_dir / "VALIDATION.parquet")
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    visible_first: dict[int, int] = {}
    for stamp, _, item in fit:
        visible_first[item] = min(stamp, visible_first.get(item, stamp))
    available = sorted(visible_first, key=lambda item: (visible_first[item], item))
    available_times = [visible_first[item] for item in available]
    histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    user_seen: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    seqs, supports, lengths, users, pos_ids, pos_s, neg_ids, neg_s = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rng = random.Random(seed)
    for stamp, user, item in fit:
        history = histories[user]
        if history:
            limit = bisect.bisect_right(available_times, stamp)
            excluded = user_seen[user] | {item}
            if len(excluded) < limit:
                negative = available[rng.randrange(limit)]
                while negative in excluded:
                    negative = available[rng.randrange(limit)]
                recent = history[-20:]
                seqs.append([x[0] for x in recent] + [0] * (20 - len(recent)))
                supports.append([x[1] for x in recent] + [0] * (20 - len(recent)))
                lengths.append(len(recent))
                users.append(user)
                pos_ids.append(item)
                pos_s.append(counts[item])
                neg_ids.append(negative)
                neg_s.append(counts[negative])
        history.append((item, counts[item]))
        user_seen[user].add(item)
        counts[item] += 1
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_val.items()
        if histories.get(user)
    ]
    dataset = TensorDataset(
        torch.tensor(np.asarray(seqs), dtype=torch.long),
        torch.tensor(np.asarray(supports), dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(pos_ids, dtype=torch.long),
        torch.tensor(pos_s, dtype=torch.long),
        torch.tensor(neg_ids, dtype=torch.long),
        torch.tensor(neg_s, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    device = torch.device("cuda:0")
    content = load_content(feature_dir, tag_file, title_file, device)
    torch.manual_seed(seed)
    model = MosaicSequence(variant, *content).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate(0.001), weight_decay=1e-4
    )
    log = []
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(f"pid={__import__('os').getpid()} model=M{variant}\n")
        lock_handle.flush()
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            for (
                seq,
                seq_support,
                length,
                positive,
                positive_support,
                negative,
                negative_support,
            ) in loader:
                seq = seq.to(device, non_blocking=True)
                seq_support = seq_support.to(device, non_blocking=True)
                length = length.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                positive_support = positive_support.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)
                negative_support = negative_support.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model.user_state(seq, seq_support, length)
                    pos_vec, _ = model.fuse(positive, positive_support, augment=True)
                    neg_vec, _ = model.fuse(negative, negative_support, augment=True)
                    loss = torch.nn.functional.softplus(
                        (state * neg_vec).sum(-1) - (state * pos_vec).sum(-1)
                    ).mean()
                    if variant == 6:
                        text_vec = model.text_projection(model.text[positive - 1])
                        image_vec = model.image_projection(model.image[positive - 1])
                        agreement = torch.nn.functional.cosine_similarity(
                            text_vec, image_vec
                        )
                        present = model.text_present[positive - 1].float()
                        loss = loss + 0.01 * (
                            (1 - agreement) * present
                        ).sum() / present.sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(seq)
            log.append({"epoch": epoch + 1, "fit_loss": loss_sum / len(dataset)})
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        model.eval()
        item_ids = torch.arange(1, 19739, device=device)
        final_support = torch.tensor(
            [counts[i] for i in range(1, 19739)], device=device
        )
        first_tensor = torch.tensor(
            [first_seen[i] for i in range(1, 19739)], device=device
        )
        cohort = {
            int(line)
            for line in (split_dir / "controlled_cold_item_ids.txt")
            .read_text()
            .splitlines()
        }
        buckets = {
            "all": [0.0, 0.0, 0],
            "controlled_cold": [0.0, 0.0, 0],
            "warm_fit": [0.0, 0.0, 0],
        }
        score_start = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                catalog, gate_weights = model.fuse(item_ids, final_support)
            catalog = catalog.to(torch.float32)
            for offset in range(0, len(eval_rows), 256):
                batch = eval_rows[offset : offset + 256]
                histories_batch = [histories[user][-20:] for user, _, _ in batch]
                seq = torch.tensor(
                    [[x[0] for x in h] + [0] * (20 - len(h)) for h in histories_batch],
                    device=device,
                )
                seq_support = torch.tensor(
                    [[x[1] for x in h] + [0] * (20 - len(h)) for h in histories_batch],
                    device=device,
                )
                length = torch.tensor([len(h) for h in histories_batch], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model.user_state(seq, seq_support, length)
                    scores = state @ catalog.T
                scores = scores.to(torch.float64)
                scores -= (
                    torch.arange(19738, device=device, dtype=torch.float64)[None, :]
                    * 1e-12
                )
                stamps = torch.tensor([row[1] for row in batch], device=device)
                scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
                for row, (user, _, _) in enumerate(batch):
                    scores[
                        row, torch.tensor(list(user_seen[user]), device=device) - 1
                    ] = -torch.inf
                top = scores.topk(20, dim=1).indices.cpu().tolist()
                for ranked, (_, _, target) in zip(top, batch):
                    labels = ["all"]
                    if target in cohort:
                        labels.append("controlled_cold")
                    elif counts[target] > 0:
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
        torch.save(
            {
                "model": model.state_dict(),
                "variant": variant,
                "seed": seed,
                "epochs": epochs,
            },
            checkpoint,
        )
        peak_vram = torch.cuda.max_memory_allocated()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    slices = {
        label: {
            "n": int(v[2]),
            "ndcg_at_20": v[0] / v[2] if v[2] else None,
            "recall_at_20": v[1] / v[2] if v[2] else None,
        }
        for label, v in buckets.items()
    }
    weights = gate_weights.float().cpu().numpy()
    cold_indices = np.asarray([i - 1 for i in cohort])
    warm_indices = np.asarray([i - 1 for i in counts if counts[i] > 20])
    return {
        "stage_id": f"D0{variant}",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "PILOT_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"M{variant}",
        "dataset": "microlens-100k",
        "device": torch.cuda.get_device_name(0),
        "seed": seed,
        "fit_samples": len(dataset),
        "epochs": epochs,
        "training_log": log,
        "train_seconds": train_seconds,
        "score_seconds": score_seconds,
        "peak_vram_bytes": peak_vram,
        "validation_slices": slices,
        "gate_weights_cold_mean": None
        if variant == 0
        else weights[cold_indices].mean(axis=0).tolist(),
        "gate_weights_warm_gt20_mean": None
        if variant == 0
        else weights[warm_indices].mean(axis=0).tolist(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": [
            "Development pilot only; no HPO, qualification, or locked result.",
            "Gate weights are relative fusion coefficients, not calibrated reliability probabilities.",
            "Untimestamped historical content revisions remain a leakage limitation.",
        ],
        "evidence_class": "measured_development_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", type=int)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("tag_file", type=Path)
    parser.add_argument("title_file", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.receipt.exists() or args.checkpoint.exists():
        raise FileExistsError(
            "Pilot output exists; preserve prior evidence and use new paths"
        )
    result = run(
        args.variant,
        args.split_dir,
        args.source,
        args.feature_dir,
        args.tag_file,
        args.title_file,
        args.lock,
        args.checkpoint,
        args.epochs,
        args.seed,
    )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
