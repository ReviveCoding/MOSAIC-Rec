"""Isolated MMRec FREEDOM runtime pilot on MicroLens temporal VALIDATION."""

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
import scipy
import torch
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.pilot_mmrec_bm3 import DatasetInfo, LoaderProxy, patch_scipy_dok, rows


def run(reference_root: Path, feature_dir: Path, split_dir: Path, source: Path,
        checkpoint: Path, lock: Path, epochs: int, seed: int) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU busy before FREEDOM pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for FREEDOM pilot")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    fit = rows(split_dir / "FIT_VISIBLE.parquet")
    validation = rows(split_dir / "VALIDATION.parquet")
    users_np = np.asarray([user for _, user, _ in fit], dtype=np.int64)
    items_np = np.asarray([item for _, _, item in fit], dtype=np.int64)
    n_users, n_items = int(users_np.max()) + 1, 19739
    histories: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    for _, user, item in fit:
        histories[user].add(item)
        counts[item] += 1
    candidates = sorted(counts)
    rng = random.Random(seed)
    negatives = []
    for user, item in zip(users_np, items_np):
        if len(histories[int(user)]) >= len(candidates):
            raise ValueError("User exhausted the FIT-only negative candidate pool")
        neg = candidates[rng.randrange(len(candidates))]
        while neg in histories[int(user)] or neg == int(item):
            neg = candidates[rng.randrange(len(candidates))]
        negatives.append(neg)
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
    from models.freedom import FREEDOM  # type: ignore[import-not-found]

    class PaddedFREEDOM(FREEDOM):
        def get_knn_adj_mat(self, mm_embeddings):
            # Item 0 is an adapter-only pad. Build the reference's exact cosine
            # kNN among real items, then add an isolated pad self-edge. This
            # avoids the reference 0/0 cosine normalization on the zero row.
            real = mm_embeddings[1:]
            normalized = real / torch.linalg.vector_norm(real, dim=1, keepdim=True).clamp_min(1e-12)
            similarity = normalized @ normalized.T
            neighbors = similarity.topk(self.knn_k, dim=-1).indices + 1
            del similarity
            source_ids = torch.arange(1, mm_embeddings.shape[0], device=self.device)
            source_ids = source_ids[:, None].expand_as(neighbors)
            edges = torch.stack((source_ids.reshape(-1), neighbors.reshape(-1)), dim=0)
            pad_edge = torch.zeros((2, 1), dtype=torch.long, device=self.device)
            edges = torch.cat((pad_edge, edges), dim=1)
            return edges, self.compute_normalized_laplacian(
                edges, torch.Size((mm_embeddings.shape[0], mm_embeddings.shape[0])))

    device = torch.device("cuda:0")
    config = {
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "NEG_PREFIX": "neg_",
        "train_batch_size": 2048, "device": device, "end2end": False,
        "is_multimodal_model": True, "data_path": str(feature_dir.parent) + "/",
        "dataset": feature_dir.name, "vision_feature_file": "vision.npy",
        "text_feature_file": "text.npy", "embedding_size": 64, "feat_embed_dim": 64,
        "knn_k": 10, "lambda_coeff": 0.9, "cf_model": "lightgcn",
        "n_mm_layers": 1, "n_ui_layers": 2, "reg_weight": 1e-4,
        "mm_image_weight": 0.1, "dropout": 0.8, "degree_ratio": 1.0,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph_cache = feature_dir / "mm_adj_freedomdsp_10_1.pt"
    if graph_cache.exists():
        raise FileExistsError(
            f"Reference graph cache already exists; verify its input identity before reuse: {graph_cache}"
        )
    loader_proxy = LoaderProxy(DatasetInfo(users_np, items_np, n_users, n_items))
    # Reference initialization constructs dense n_items^2 similarities on CUDA.
    model = PaddedFREEDOM(config, loader_proxy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(torch.from_numpy(users_np), torch.from_numpy(items_np),
                                      torch.tensor(negatives, dtype=torch.long)),
                        batch_size=2048, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    lock.parent.mkdir(parents=True, exist_ok=True)
    training_log = []
    with lock.open("w") as handle:
        handle.write(f"pid={__import__('os').getpid()} model=FREEDOM_MMREC_REFERENCE\n")
        handle.flush()
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            model.pre_epoch_processing()
            total = 0.0
            for users, positive, negative in loader:
                users = users.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)
                loss = model.calculate_loss((users, positive, negative))
                if not torch.isfinite(loss):
                    raise FloatingPointError("FREEDOM produced nonfinite loss")
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
            user_vectors, item_vectors = model.forward(model.norm_adj)
            first = torch.as_tensor(first_seen, device=device)
            item_index = torch.arange(n_items, device=device, dtype=torch.float64)
            for offset in range(0, len(eval_rows), 256):
                batch = eval_rows[offset:offset + 256]
                batch_users = torch.tensor([user for user, _, _ in batch], device=device)
                scores = (user_vectors[batch_users] @ item_vectors.T).to(torch.float64)
                scores -= item_index[None, :] * 1e-12
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
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k", "model": "B10_FREEDOM_MMREC_REFERENCE",
        "reference_commit": "c68ee7c9b0de1ca7c5150258322edb4994be29d9",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": hashlib.sha256(Path("configs/freedom_reference_pilot_p1.json").read_bytes()).hexdigest(),
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
        "item_graph_cache": str(graph_cache),
        "item_graph_sha256": hashlib.sha256(graph_cache.read_bytes()).hexdigest(),
        "warnings": [
            "One-epoch reference runtime pilot; no equal-budget HPO or seed qualification.",
            "Publisher content enters item kNN graph; cold capability requires measured validation.",
            "FIT-only fixed negative pool and exact first-validation-event available-catalog scoring.",
            "Reference SciPy DOK private _update restored by element assignment.",
            "The item-zero adapter pad is excluded from real-item cosine kNN; an isolated pad self-edge is added.",
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
    started = datetime.now(timezone.utc).isoformat()
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
            "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "microlens-100k", "model": "B10_FREEDOM_MMREC_REFERENCE",
            "seed": args.seed, "epochs": args.epochs,
            "failure_class": ("external_gpu_contention" if "GPU busy" in str(exc)
                              else "implementation_or_reference_compatibility"),
            "exception": repr(exc), "traceback": traceback.format_exc(),
            "evidence_class": "measured_development_reference_failure",
        }
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
