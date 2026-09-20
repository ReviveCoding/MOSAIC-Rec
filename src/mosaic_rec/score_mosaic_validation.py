"""Read-only CUDA replay of existing MOSAIC VALIDATION checkpoints."""

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

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mosaic_rec.pilot_mosaic import MosaicSequence, events, load_content


def run(
    variant: int,
    original_receipt: Path,
    split_dir: Path,
    source: Path,
    feature_dir: Path,
    tag_file: Path,
    title_file: Path,
    output: Path,
) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU busy before validation replay")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for neural full-catalog replay")
    original = json.loads(original_receipt.read_text())
    if (
        original["protocol_id"] != "mosaic-rec-2026-09-20-p1"
        or original["model"] != f"M{variant}"
    ):
        raise ValueError("Original receipt identity mismatch")
    checkpoint = Path(original["checkpoint"])
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != original["checkpoint_sha256"]
    ):
        raise ValueError("Original checkpoint hash mismatch")
    fit = events(split_dir / "FIT_VISIBLE.parquet")
    validation = events(split_dir / "VALIDATION.parquet")
    histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    for _, user, item in fit:
        histories[user].append((item, counts[item]))
        seen[user].add(item)
        counts[item] += 1
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_val.items()
        if user in histories
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
    content = load_content(feature_dir, tag_file, title_file, device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
    if checkpoint_data["variant"] != variant:
        raise ValueError("Checkpoint variant mismatch")
    model = MosaicSequence(variant, *content).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    item_ids = torch.arange(1, 19739, device=device)
    final_support = torch.tensor([counts[i] for i in range(1, 19739)], device=device)
    first_tensor = torch.tensor([first_seen[i] for i in range(1, 19739)], device=device)
    buckets = {name: [0.0, 0] for name in ("all", "controlled_cold", "warm_fit")}
    rows_out: list[dict] = []
    start = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            catalog, _ = model.fuse(item_ids, final_support)
        catalog = catalog.to(torch.float32)
        for offset in range(0, len(eval_rows), 256):
            batch = eval_rows[offset : offset + 256]
            batch_history = [histories[user][-20:] for user, _, _ in batch]
            seq = torch.tensor(
                [
                    [x[0] for x in history] + [0] * (20 - len(history))
                    for history in batch_history
                ],
                device=device,
            )
            support = torch.tensor(
                [
                    [x[1] for x in history] + [0] * (20 - len(history))
                    for history in batch_history
                ],
                device=device,
            )
            length = torch.tensor(
                [len(history) for history in batch_history], device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                user_state = model.user_state(seq, support, length)
                scores = user_state @ catalog.T
            scores = scores.to(torch.float64)
            scores -= (
                torch.arange(19738, device=device, dtype=torch.float64)[None, :] * 1e-12
            )
            stamps = torch.tensor([row[1] for row in batch], device=device)
            scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
            for row, (user, _, _) in enumerate(batch):
                scores[
                    row, torch.tensor(list(seen[user]), device=device) - 1
                ] = -torch.inf
            top = scores.topk(20, dim=1).indices.cpu().tolist()
            for ranked, (user, stamp, target) in zip(top, batch):
                slice_name = (
                    "controlled_cold"
                    if target in cohort
                    else "warm_fit"
                    if counts[target] > 0
                    else "other_cold"
                )
                rank = ranked.index(target - 1) + 1 if target - 1 in ranked else None
                ndcg = 1 / log2(rank + 1) if rank is not None else 0.0
                buckets["all"][0] += ndcg
                buckets["all"][1] += 1
                if slice_name in buckets:
                    buckets[slice_name][0] += ndcg
                    buckets[slice_name][1] += 1
                rows_out.append(
                    {
                        "user_id": user,
                        "item_id": target,
                        "timestamp": stamp,
                        "slice": slice_name,
                        "rank_at_20": rank,
                        "ndcg_at_20": ndcg,
                    }
                )
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - start
    slices = {
        name: {"n": value[1], "ndcg_at_20": value[0] / value[1] if value[1] else None}
        for name, value in buckets.items()
    }
    for name, result in slices.items():
        expected = original["validation_slices"][name]
        if (
            result["n"] != expected["n"]
            or abs(result["ndcg_at_20"] - expected["ndcg_at_20"]) > 1e-6
        ):
            raise ValueError(f"Replay differs from original {name} validation slice")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_out), output, compression="zstd")
    return {
        "stage_id": "D08",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "VALIDATION_REPLAY_PASS",
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "model": f"M{variant}",
        "seed": original["seed"],
        "device": torch.cuda.get_device_name(0),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": hashlib.sha256(
            Path("configs/mosaic_validation_replay_p1.json").read_bytes()
        ).hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": original["checkpoint_sha256"],
        "source_receipt": original_receipt.stem,
        "per_user_path": str(output),
        "per_user_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "per_user_rows": len(rows_out),
        "validation_slices": slices,
        "score_seconds": score_seconds,
        "warnings": [
            "Development validation replay only; not a finalist or locked result.",
            "Timing requires a separate uncontended systems qualification.",
        ],
        "evidence_class": "measured_development_replay",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", type=int, choices=(1, 4, 6, 7))
    parser.add_argument("original_receipt", type=Path)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("tag_file", type=Path)
    parser.add_argument("title_file", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.output.exists():
        raise FileExistsError("Replay output already exists; preserve it")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with args.lock.open("a+") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.write(
                f"pid={__import__('os').getpid()} model=M{args.variant}_VALIDATION_REPLAY\n"
            )
            lock_handle.flush()
            result = run(
                args.variant,
                args.original_receipt,
                args.split_dir,
                args.source,
                args.feature_dir,
                args.tag_file,
                args.title_file,
                args.output,
            )
    except Exception as exc:
        result = {
            "stage_id": "D08",
            "protocol_id": "mosaic-rec-2026-09-20-p1",
            "state": "DEVELOPMENT",
            "status": "FAIL_VALIDATION_REPLAY",
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "microlens-100k",
            "model": f"M{args.variant}",
            "failure_class": "evaluator_or_runtime_unclassified",
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "evidence_class": "measured_development_replay_failure",
        }
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
