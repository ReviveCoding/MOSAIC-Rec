"""MMRec BM3 reference architecture on the MicroLens temporal development protocol."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset


class GPUContentionError(RuntimeError):
    """Another compute process was present before this pilot claimed CUDA."""


class DatasetInfo:
    def __init__(self, users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> None:
        self.users = users
        self.items = items
        self.n_users = n_users
        self.n_items = n_items

    def get_user_num(self) -> int:
        return self.n_users

    def get_item_num(self) -> int:
        return self.n_items

    def inter_matrix(self, form: str = "coo"):
        matrix = sp.coo_matrix((np.ones(len(self.users), dtype=np.float32),
                                (self.users, self.items)),
                               shape=(self.n_users, self.n_items))
        return matrix if form == "coo" else matrix.asformat(form)


class LoaderProxy:
    def __init__(self, info: DatasetInfo) -> None:
        self.dataset = info

    def inter_matrix(self, form: str = "coo"):
        return self.dataset.inter_matrix(form)


def rows(path: Path) -> list[tuple[int, int, int]]:
    table = pq.read_table(path, columns=["timestamp", "user_id", "item_id"])
    return sorted((int(t), int(u), int(i)) for t, u, i in zip(
        table["timestamp"].to_pylist(), table["user_id"].to_pylist(),
        table["item_id"].to_pylist()))


def patch_scipy_dok() -> None:
    # SciPy 1.15 removed MMRec's private DOK _update. Assigning entries
    # individually preserves the reference adjacency's exact dictionary values.
    if not hasattr(sp.dok_matrix, "_update"):
        def restore_update(matrix, entries):
            for key, value in entries.items():
                matrix[key] = value
        sp.dok_matrix._update = restore_update  # type: ignore[attr-defined,method-assign]


def run(reference_root: Path, feature_dir: Path, split_dir: Path, source: Path,
        checkpoint: Path, lock: Path, epochs: int, seed: int) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    gpu_processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if gpu_processes:
        raise GPUContentionError(f"GPU busy before BM3 pilot: {gpu_processes}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for BM3 reference protocol pilot")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    fit = rows(split_dir / "FIT_VISIBLE.parquet")
    validation = rows(split_dir / "VALIDATION.parquet")
    users_np = np.asarray([user for _, user, _ in fit], dtype=np.int64)
    items_np = np.asarray([item for _, _, item in fit], dtype=np.int64)
    n_users, n_items = max(users_np) + 1, 19739
    histories: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    for _, user, item in fit:
        histories[user].add(item)
        counts[item] += 1
    first_val: dict[int, tuple[int, int]] = {}
    for stamp, user, item in validation:
        first_val.setdefault(user, (stamp, item))
    eval_rows = [(user, stamp, item) for user, (stamp, item) in first_val.items()
                 if user in histories]
    first_seen = np.full(n_items, np.iinfo(np.int64).max, dtype=np.int64)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(first_seen[item], stamp)
    cohort = {int(x) for x in (split_dir / "controlled_cold_item_ids.txt").read_text().splitlines()}
    patch_scipy_dok()
    sys.path.insert(0, str(reference_root / "src"))
    from models.bm3 import BM3  # type: ignore[import-not-found]

    device = torch.device("cuda:0")
    config = {
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "NEG_PREFIX": "neg_",
        "train_batch_size": 2048, "device": device, "end2end": False,
        "is_multimodal_model": True,
        "data_path": str(feature_dir.parent) + "/", "dataset": feature_dir.name,
        "vision_feature_file": "vision.npy", "text_feature_file": "text.npy",
        "embedding_size": 64, "n_layers": 1, "reg_weight": 0.1,
        "cl_weight": 2.0, "dropout": 0.3,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader_proxy = LoaderProxy(DatasetInfo(users_np, items_np, n_users, n_items))
    model = BM3(config, loader_proxy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(torch.from_numpy(users_np), torch.from_numpy(items_np)),
                        batch_size=2048, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    lock.parent.mkdir(parents=True, exist_ok=True)
    training_log = []
    with lock.open("w") as handle:
        handle.write(f"pid={__import__('os').getpid()} model=BM3_MMREC_REFERENCE\n")
        handle.flush()
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for users, items in loader:
                users = users.to(device, non_blocking=True)
                items = items.to(device, non_blocking=True)
                loss = model.calculate_loss((users, items))
                if not torch.isfinite(loss):
                    raise FloatingPointError("BM3 produced nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(users)
            training_log.append({"epoch": epoch + 1, "fit_loss": total / len(fit)})
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        model.eval()
        score_start = time.perf_counter()
        buckets = {name: [0.0, 0.0, 0] for name in ("all", "controlled_cold", "warm_fit")}
        with torch.inference_mode():
            user_vectors, item_vectors = model.forward()
            user_vectors = model.predictor(user_vectors)
            item_vectors = model.predictor(item_vectors)
            first = torch.as_tensor(first_seen, device=device)
            for offset in range(0, len(eval_rows), 256):
                batch = eval_rows[offset:offset + 256]
                batch_users = torch.tensor([user for user, _, _ in batch], device=device)
                scores = (user_vectors[batch_users] @ item_vectors.T).to(torch.float64)
                scores -= torch.arange(n_items, device=device, dtype=torch.float64)[None, :] * 1e-12
                stamps = torch.tensor([stamp for _, stamp, _ in batch], device=device)
                scores.masked_fill_(first[None, :] > stamps[:, None], -torch.inf)
                scores[:, 0] = -torch.inf
                for row, (user, _, _) in enumerate(batch):
                    scores[row, torch.tensor(list(histories[user]), device=device)] = -torch.inf
                top = scores.topk(20, dim=1).indices.cpu().tolist()
                for ranked, (_, _, target) in zip(top, batch):
                    labels = ["all"]
                    if target in cohort:
                        labels.append("controlled_cold")
                    elif counts[target] > 0:
                        labels.append("warm_fit")
                    for name in labels:
                        buckets[name][2] += 1
                        if target in ranked:
                            rank = ranked.index(target) + 1
                            buckets[name][0] += 1 / log2(rank + 1)
                            buckets[name][1] += 1
        torch.cuda.synchronize()
        score_seconds = time.perf_counter() - score_start
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs}, checkpoint)
        peak_vram = torch.cuda.max_memory_allocated()
    slices = {name: {"n": int(v[2]),
                     "ndcg_at_20": v[0] / v[2] if v[2] else None,
                     "recall_at_20": v[1] / v[2] if v[2] else None}
              for name, v in buckets.items()}
    return {
        "stage_id": "C04", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "REFERENCE_PROTOCOL_PILOT",
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k", "model": "B9_BM3_MMREC_REFERENCE",
        "reference_commit": "c68ee7c9b0de1ca7c5150258322edb4994be29d9",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_sha256": {
            "fit_visible": hashlib.sha256((split_dir / "FIT_VISIBLE.parquet").read_bytes()).hexdigest(),
            "validation": hashlib.sha256((split_dir / "VALIDATION.parquet").read_bytes()).hexdigest(),
            "vision": hashlib.sha256((feature_dir / "vision.npy").read_bytes()).hexdigest(),
            "text": hashlib.sha256((feature_dir / "text.npy").read_bytes()).hexdigest(),
        },
        "environment": {"torch": torch.__version__, "scipy": scipy.__version__},
        "device": torch.cuda.get_device_name(0), "seed": seed, "epochs": epochs,
        "training_log": training_log, "train_seconds": train_seconds,
        "score_seconds": score_seconds, "peak_vram_bytes": peak_vram,
        "validation_slices": slices, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warnings": [
            "Reference BM3 architecture evaluated on common temporal validation; no equal-budget HPO or seed qualification.",
            "BM3 inference uses graph/ID embeddings, not a direct content-only cold item representation.",
            "SciPy DOK private _update was restored by element assignment; adjacency values are unchanged.",
            "Sparse reference path uses FP32; no CPU fallback occurred.",
        ],
        "evidence_class": "measured_development_reference_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.receipt.exists() or args.checkpoint.exists():
        raise FileExistsError("Pilot output exists; preserve the prior attempt and choose new paths")
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        with args.lock.open("a+") as heavy_lock:
            fcntl.flock(heavy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(args.reference_root, args.feature_dir, args.split_dir, args.source,
                         args.checkpoint, args.lock, args.epochs, args.seed)
    except Exception as exc:
        result = {
            "stage_id": "C04", "protocol_id": "mosaic-rec-2026-09-20-p1",
            "state": "DEVELOPMENT", "status": "FAIL_REFERENCE_RUNTIME",
            "started_utc": started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "microlens-100k", "model": "B9_BM3_MMREC_REFERENCE",
            "seed": args.seed, "epochs": args.epochs,
            "failure_class": ("external_gpu_contention" if isinstance(exc, GPUContentionError)
                              else "implementation_or_reference_compatibility"),
            "exception": repr(exc), "traceback": traceback.format_exc(),
            "evidence_class": "measured_development_reference_failure",
        }
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    result["config_sha256"] = hashlib.sha256(
        Path("configs/bm3_reference_pilot_p1.json").read_bytes()
    ).hexdigest()
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
