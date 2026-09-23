"""Read-only full-catalog validation replay for frozen sequence checkpoints."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import torch

from mosaic_rec.pilot_bert4rec import BERT4Rec
from mosaic_rec.pilot_sasrec import GRU4Rec, SASRec, load_rows


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def original_all_metric(receipt: dict[str, object]) -> float:
    slices = receipt.get("validation_slices") or {}
    if isinstance(slices, dict) and isinstance(slices.get("all"), dict):
        value = slices["all"].get("ndcg_at_20")
    else:
        value = receipt.get("ndcg_at_20")
    if value is None:
        raise ValueError("Source receipt has no overall NDCG@20")
    return float(value)


def require_idle_gpu() -> None:
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU compute process already active: {processes}")


def score(
    architecture: str,
    split_dir: Path,
    source: Path,
    source_receipt_path: Path,
    checkpoint: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for neural full-catalog replay")
    source_receipt = json.loads(source_receipt_path.read_text())
    expected_hash = source_receipt.get("checkpoint_sha256")
    before_hash = digest(checkpoint)
    if before_hash != expected_hash:
        raise ValueError("Checkpoint hash differs from source HPO receipt")

    fit = load_rows(split_dir / "FIT_VISIBLE.parquet")
    validation = load_rows(split_dir / "VALIDATION.parquet")
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

    histories: dict[int, list[int]] = defaultdict(list)
    for _, user, item in fit:
        histories[user].append(item)
    first_validation: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_validation.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_validation.items()
        if histories.get(user)
    ]

    device = torch.device("cuda:0")
    if architecture == "bert4rec":
        model: torch.nn.Module = BERT4Rec(catalog_size)
    elif architecture == "gru4rec":
        model = GRU4Rec(catalog_size)
    elif architecture == "sasrec":
        model = SASRec(catalog_size)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(payload["seed"]) != int(source_receipt["seed"]):
        raise ValueError("Checkpoint seed differs from source HPO receipt")
    if int(payload["epochs"]) != int(source_receipt["epochs"]):
        raise ValueError("Checkpoint epochs differ from source HPO receipt")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    item_embedding = model.item  # type: ignore[attr-defined]
    catalog = (
        item_embedding(torch.arange(1, catalog_size + 1, device=device))
        .detach()
        .clone()
    )
    warm_mask = torch.tensor(
        [item in fit_items for item in range(1, catalog_size + 1)], device=device
    )
    catalog[~warm_mask] = 0
    first_tensor = torch.tensor(
        [first_seen[item] for item in range(1, catalog_size + 1)], device=device
    )
    buckets = {name: [0.0, 0.0, 0] for name in ("all", "controlled_cold", "warm_fit")}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(eval_rows), 256):
            batch = eval_rows[offset : offset + 256]
            if architecture == "bert4rec":
                bert = model
                contexts = [histories[user][-19:] for user, _, _ in batch]
                lengths = torch.tensor(
                    [len(context) + 1 for context in contexts], device=device
                )
                mask_id = bert.mask_id  # type: ignore[attr-defined]
                sequence = torch.tensor(
                    [
                        context + [mask_id] + [0] * (20 - len(context) - 1)
                        for context in contexts
                    ],
                    device=device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = bert(sequence)
                    states = encoded[
                        torch.arange(len(batch), device=device), lengths - 1
                    ]
                    scores = states @ catalog.T
            else:
                contexts = [histories[user][-20:] for user, _, _ in batch]
                lengths = torch.tensor(
                    [len(context) for context in contexts], device=device
                )
                sequence = torch.tensor(
                    [context + [0] * (20 - len(context)) for context in contexts],
                    device=device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    states = model(sequence, lengths)
                    scores = states @ catalog.T
            scores = scores.to(torch.float64)
            scores -= (
                torch.arange(catalog_size, device=device, dtype=torch.float64)[None, :]
                * 1e-12
            )
            stamps = torch.tensor([stamp for _, stamp, _ in batch], device=device)
            scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
            for row_index, (user, _, _) in enumerate(batch):
                scores[
                    row_index, torch.tensor(histories[user], device=device) - 1
                ] = -torch.inf
            top = scores.topk(20, dim=1).indices.cpu().tolist()
            for ranked, (_, _, target) in zip(top, batch):
                labels = ["all"]
                if target in cold:
                    labels.append("controlled_cold")
                elif target in fit_items:
                    labels.append("warm_fit")
                for label in labels:
                    bucket = buckets[label]
                    bucket[2] += 1
                    if target - 1 in ranked:
                        rank = ranked.index(target - 1) + 1
                        bucket[0] += 1 / log2(rank + 1)
                        bucket[1] += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    slices = {
        name: {
            "n": int(values[2]),
            "ndcg_at_20": values[0] / values[2] if values[2] else None,
            "recall_at_20": values[1] / values[2] if values[2] else None,
        }
        for name, values in buckets.items()
    }
    agreement = abs(
        float(slices["all"]["ndcg_at_20"]) - original_all_metric(source_receipt)
    )
    if agreement > 1e-6:
        raise ValueError(f"Overall replay disagreement {agreement} exceeds 1e-6")
    after_hash = digest(checkpoint)
    if after_hash != before_hash:
        raise ValueError("Read-only replay changed the checkpoint")
    return {
        "stage_id": "E01_SEQUENCE_RESCORE",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "READ_ONLY_RESCORE_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "model": (f"{source_receipt['model']}_T{source_receipt['hpo_trial_index']}"),
        "source_model": source_receipt["model"],
        "source_hpo_trial_index": source_receipt["hpo_trial_index"],
        "architecture": architecture,
        "seed": source_receipt["seed"],
        "epochs": source_receipt["epochs"],
        "device": torch.cuda.get_device_name(0),
        "validation_slices": slices,
        "score_seconds": elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "source_receipt": str(source_receipt_path),
        "source_receipt_sha256": digest(source_receipt_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": after_hash,
        "checkpoint_unchanged": True,
        "overall_agreement_absolute": agreement,
        "agreement_tolerance": 1e-6,
        "input_sha256": {
            "fit_visible": digest(split_dir / "FIT_VISIBLE.parquet"),
            "validation": digest(split_dir / "VALIDATION.parquet"),
            "controlled_cold_ids": digest(split_dir / "controlled_cold_item_ids.txt"),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "warnings": [
            "Read-only DEVELOPMENT re-scoring; this is not comparator selection.",
            "QUALIFICATION and LOCKED_FINAL were not accessed.",
        ],
        "evidence_class": "measured_development_validation_rescore",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture", choices=("gru4rec", "sasrec", "bert4rec"))
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("source_receipt", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(
            "Replay receipt exists; preserve it and use a distinct path"
        )
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    require_idle_gpu()
    with args.lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={__import__('os').getpid()} model=SEQUENCE_RESCORE\n")
        handle.flush()
        result = score(
            args.architecture,
            args.split_dir,
            args.source,
            args.source_receipt,
            args.checkpoint,
        )
    temporary = args.receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.receipt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
