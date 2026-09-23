"""Frozen three-seed ensemble scoring on the LOCKED_FINAL role."""

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

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mosaic_rec.pilot_bert4rec import BERT4Rec
from mosaic_rec.pilot_content import ContentBPR, load_features
from mosaic_rec.pilot_mosaic import MosaicSequence, events, load_content

PROTOCOL_ID = "mosaic-rec-2026-09-20-p1"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def require_idle_gpu() -> None:
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU compute process already active: {processes}")


def locked_state_guard(
    root: Path, frozen: dict[str, object], manifest_path: Path
) -> None:
    status = json.loads((root / "evidence/stage_status.json").read_text())
    if status["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Protocol identity mismatch")
    if status["scientific_state"] != "FROZEN":
        raise RuntimeError("FROZEN state guard is not open")
    if status["stages"]["E05"] != "FROZEN_GUARD_PASS":
        raise RuntimeError("E05 FROZEN guard has not passed")
    guard = json.loads(
        (root / "evidence/e05_frozen_guard_p1_amendment1.json").read_text()
    )
    if guard["status"] != "FROZEN_AMENDMENT1_GUARD_PASS":
        raise RuntimeError("Frozen guard receipt is invalid")
    guarded_manifest = root / guard["e05_manifest"]
    if digest(guarded_manifest) != guard["e05_manifest_sha256"]:
        raise ValueError("E05 manifest changed after its guard")
    if guarded_manifest.resolve() != manifest_path.resolve():
        raise ValueError("Locked scorer did not receive the guarded E05 amendment")
    if frozen["locked_access"] != (
        "PROHIBITED_UNTIL_AMENDMENT1_IS_COMMITTED_AND_PROTOCOL_GUARD_PASSES"
    ):
        raise ValueError("Frozen locked-access contract mismatch")


def candidate_from_manifest(
    manifest: dict[str, object], config_id: str
) -> dict[str, object]:
    matches = [row for row in manifest["candidates"] if row["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"Candidate identity not unique: {config_id}")
    return matches[0]


def load_first_seen(source: Path) -> dict[int, int]:
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item, stamp = int(row["item"]), int(row["timestamp"])
            first_seen[item] = min(stamp, first_seen.get(item, stamp))
    return first_seen


def score(
    root: Path,
    manifest_path: Path,
    config_id: str,
    split_dir: Path,
    source: Path,
    feature_dir: Path,
    tags: Path,
    titles: Path,
    per_user_output: Path,
) -> dict[str, object]:
    started_utc = datetime.now(timezone.utc).isoformat()
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    frozen = json.loads(manifest_path.read_text())
    if frozen["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Frozen manifest protocol mismatch")
    locked_state_guard(root, frozen, manifest_path)
    qualification_manifest_path = root / frozen["qualification_manifest"]
    if digest(qualification_manifest_path) != frozen["qualification_manifest_sha256"]:
        raise ValueError("Qualification manifest changed after freeze")
    manifest = json.loads(qualification_manifest_path.read_text())
    if config_id not in {
        frozen["C_seq"],
        frozen["C_cold"],
        frozen["MOSAIC_finalist"],
    }:
        raise ValueError("Model is not a frozen locked-execution identity")
    candidate = candidate_from_manifest(manifest, config_id)
    locked_path = split_dir / "LOCKED_FINAL.parquet"
    if locked_path.name != "LOCKED_FINAL.parquet":
        raise ValueError("Only LOCKED_FINAL role is authorized")
    if digest(locked_path) != frozen["data"]["locked_final_sha256"]:
        raise ValueError("LOCKED_FINAL data hash mismatch")
    if (
        digest(split_dir / "FIT_VISIBLE.parquet")
        != manifest["data"]["fit_visible"]["sha256"]
    ):
        raise ValueError("FIT_VISIBLE data hash mismatch")
    if (
        digest(split_dir / "controlled_cold_item_ids.txt")
        != manifest["data"]["controlled_cold_ids"]["sha256"]
    ):
        raise ValueError("Controlled-cold cohort hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for exact neural locked scoring")

    fit = events(split_dir / "FIT_VISIBLE.parquet")
    locked_final = events(locked_path)
    first_seen = load_first_seen(source)
    catalog_size = max(first_seen)
    cold = {
        int(item)
        for item in (split_dir / "controlled_cold_item_ids.txt")
        .read_text()
        .splitlines()
    }
    seen: dict[int, set[int]] = defaultdict(set)
    histories: dict[int, list[int]] = defaultdict(list)
    supported_histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for _, user, item in fit:
        histories[user].append(item)
        supported_histories[user].append((item, counts[item]))
        seen[user].add(item)
        counts[item] += 1
    if cold.intersection({item for _, _, item in fit}):
        raise ValueError("Controlled-cold item has FIT_VISIBLE support")
    first_locked: dict[int, tuple[int, int]] = {}
    for stamp, user, item in locked_final:
        first_locked.setdefault(user, (stamp, item))
    eval_rows = [
        (user, stamp, item)
        for user, (stamp, item) in first_locked.items()
        if histories.get(user)
    ]
    if not eval_rows:
        raise ValueError("No eligible LOCKED_FINAL rows")

    device = torch.device("cuda:0")
    architecture = candidate["architecture"]
    checkpoint_rows = candidate["checkpoints"]
    if [int(row["seed"]) for row in checkpoint_rows] != [
        20260924,
        20260925,
        20260926,
    ]:
        raise ValueError("Frozen locked seed set mismatch")
    models: list[torch.nn.Module] = []
    catalogs: list[torch.Tensor] = []
    checkpoint_hashes_before: list[str] = []
    item_ids = torch.arange(1, catalog_size + 1, device=device)
    warm_mask = torch.tensor(
        [counts[item] > 0 for item in range(1, catalog_size + 1)], device=device
    )
    final_support = torch.tensor(
        [counts[item] for item in range(1, catalog_size + 1)], device=device
    )
    content_cache = None
    baseline_content_cache = None
    for checkpoint_row in checkpoint_rows:
        checkpoint = Path(checkpoint_row["checkpoint"])
        checkpoint_hash = digest(checkpoint)
        if checkpoint_hash != checkpoint_row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        receipt_path = root / checkpoint_row["receipt"]
        if digest(receipt_path) != checkpoint_row["receipt_sha256"]:
            raise ValueError(f"Source receipt hash mismatch: {receipt_path}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if int(payload["seed"]) != int(checkpoint_row["seed"]):
            raise ValueError("Checkpoint seed mismatch")
        if architecture == "bert4rec":
            model: torch.nn.Module = BERT4Rec(catalog_size)
            model.load_state_dict(payload["model"], strict=True)
            model.to(device).eval()
            catalog = model.item(item_ids).detach().clone()  # type: ignore[attr-defined]
            catalog[~warm_mask] = 0
        elif architecture == "two_tower_content":
            if baseline_content_cache is None:
                baseline_content_cache = load_features(feature_dir, tags, device)
            text, image, tag_ids, n_tags = baseline_content_cache
            model = ContentBPR(
                max(user for _, user, _ in fit),
                catalog_size,
                n_tags,
                text,
                image,
                tag_ids,
                "two_tower_content",
            )
            model.load_state_dict(payload["model"], strict=True)
            model.to(device).eval()
            catalog = model.item_vector(item_ids, ~warm_mask).detach()  # type: ignore[attr-defined]
        elif architecture == "mosaic":
            if content_cache is None:
                content_cache = load_content(feature_dir, tags, titles, device)
            variant = int(candidate["variant"])
            if int(payload["variant"]) != variant:
                raise ValueError("MOSAIC checkpoint variant mismatch")
            model = MosaicSequence(variant, *content_cache)
            model.load_state_dict(payload["model"], strict=True)
            model.to(device).eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                catalog, _ = model.fuse(item_ids, final_support)  # type: ignore[attr-defined]
            catalog = catalog.detach()
        else:
            raise ValueError(f"Unsupported locked architecture: {architecture}")
        models.append(model)
        catalogs.append(catalog.to(torch.float32))
        checkpoint_hashes_before.append(checkpoint_hash)

    first_tensor = torch.tensor(
        [first_seen[item] for item in range(1, catalog_size + 1)], device=device
    )
    metrics = {
        name: [0.0, 0.0, 0.0, 0] for name in ("all", "controlled_cold", "warm_fit")
    }
    per_user_rows: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats()
    scoring_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(eval_rows), 256):
            batch = eval_rows[offset : offset + 256]
            ensemble_scores: torch.Tensor | None = None
            for model, catalog in zip(models, catalogs):
                if architecture == "bert4rec":
                    contexts = [histories[user][-19:] for user, _, _ in batch]
                    lengths = torch.tensor(
                        [len(context) + 1 for context in contexts], device=device
                    )
                    mask_id = model.mask_id  # type: ignore[attr-defined]
                    sequence = torch.tensor(
                        [
                            context + [mask_id] + [0] * (20 - len(context) - 1)
                            for context in contexts
                        ],
                        device=device,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        encoded = model(sequence)
                        state = encoded[
                            torch.arange(len(batch), device=device), lengths - 1
                        ]
                        model_scores = state @ catalog.T
                elif architecture == "two_tower_content":
                    users = torch.tensor([row[0] for row in batch], device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model_scores = model.user(users) @ catalog.T  # type: ignore[attr-defined]
                else:
                    batch_history = [
                        supported_histories[user][-20:] for user, _, _ in batch
                    ]
                    sequence = torch.tensor(
                        [
                            [row[0] for row in history] + [0] * (20 - len(history))
                            for history in batch_history
                        ],
                        device=device,
                    )
                    support = torch.tensor(
                        [
                            [row[1] for row in history] + [0] * (20 - len(history))
                            for history in batch_history
                        ],
                        device=device,
                    )
                    lengths = torch.tensor(
                        [len(history) for history in batch_history], device=device
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        state = model.user_state(sequence, support, lengths)  # type: ignore[attr-defined]
                        model_scores = state @ catalog.T
                model_scores = model_scores.to(torch.float32)
                ensemble_scores = (
                    model_scores
                    if ensemble_scores is None
                    else ensemble_scores + model_scores
                )
            if ensemble_scores is None:
                raise AssertionError("No ensemble scores")
            scores = (ensemble_scores / len(models)).to(torch.float64)
            scores -= (
                torch.arange(catalog_size, device=device, dtype=torch.float64)[None, :]
                * 1e-12
            )
            stamps = torch.tensor([stamp for _, stamp, _ in batch], device=device)
            scores.masked_fill_(first_tensor[None, :] > stamps[:, None], -torch.inf)
            for row_index, (user, _, _) in enumerate(batch):
                scores[
                    row_index, torch.tensor(sorted(seen[user]), device=device) - 1
                ] = -torch.inf
            top = scores.topk(20, dim=1).indices.cpu().tolist()
            for ranked, (user, stamp, target) in zip(top, batch):
                slice_name = (
                    "controlled_cold"
                    if target in cold
                    else "warm_fit"
                    if counts[target] > 0
                    else "other_cold"
                )
                rank = ranked.index(target - 1) + 1 if target - 1 in ranked else None
                ndcg = 1 / log2(rank + 1) if rank is not None else 0.0
                recall = 1.0 if rank is not None else 0.0
                reciprocal = 1 / rank if rank is not None else 0.0
                labels = ["all"] + ([slice_name] if slice_name in metrics else [])
                for label in labels:
                    metric = metrics[label]
                    metric[0] += ndcg
                    metric[1] += recall
                    metric[2] += reciprocal
                    metric[3] += 1
                per_user_rows.append(
                    {
                        "user_id": user,
                        "item_id": target,
                        "timestamp": stamp,
                        "slice": slice_name,
                        "rank_at_20": rank,
                        "ndcg_at_20": ndcg,
                        "recall_at_20": recall,
                        "mrr_at_20": reciprocal,
                    }
                )
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - scoring_started
    slices = {
        name: {
            "n": int(values[3]),
            "ndcg_at_20": values[0] / values[3] if values[3] else None,
            "recall_at_20": values[1] / values[3] if values[3] else None,
            "mrr_at_20": values[2] / values[3] if values[3] else None,
        }
        for name, values in metrics.items()
    }
    per_user_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(per_user_rows), per_user_output, compression="zstd"
    )
    checkpoint_hashes_after = [
        digest(Path(row["checkpoint"])) for row in checkpoint_rows
    ]
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise ValueError("Read-only locked scoring changed a checkpoint")
    return {
        "stage_id": "F00_F02",
        "protocol_id": PROTOCOL_ID,
        "state": "LOCKED_EXECUTION",
        "status": "LOCKED_SCORE_PASS",
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "role": "LOCKED_FINAL",
        "model": config_id,
        "architecture": architecture,
        "device": torch.cuda.get_device_name(0),
        "ensemble_seeds": [int(row["seed"]) for row in checkpoint_rows],
        "ensemble_size": len(models),
        "validation_slices": slices,
        "score_seconds": score_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_sha256": checkpoint_hashes_after,
        "checkpoint_unchanged": True,
        "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "manifest_sha256": digest(manifest_path),
        "qualification_manifest_sha256": frozen["qualification_manifest_sha256"],
        "input_sha256": {
            "fit_visible": manifest["data"]["fit_visible"]["sha256"],
            "locked_final": frozen["data"]["locked_final_sha256"],
            "controlled_cold_ids": manifest["data"]["controlled_cold_ids"]["sha256"],
        },
        "per_user_path": str(per_user_output),
        "per_user_sha256": digest(per_user_output),
        "per_user_rows": len(per_user_rows),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "warnings": [
            "This is protected locked evidence under immutable p1 identities.",
            "No tuning, selection, or threshold changes are permitted from this outcome.",
            "First-observed item time is a candidate-availability proxy, not launch time.",
        ],
        "evidence_class": "measured_locked_exact_full_catalog",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("config_id")
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("tags", type=Path)
    parser.add_argument("titles", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--per-user-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.per_user_output.exists():
        raise FileExistsError("Locked output exists; preserve it")
    root = Path(__file__).resolve().parents[2]
    require_idle_gpu()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(
            f"pid={__import__('os').getpid()} model={args.config_id}_LOCKED_FINAL\n"
        )
        lock_handle.flush()
        result = score(
            root,
            args.manifest,
            args.config_id,
            args.split_dir,
            args.source,
            args.feature_dir,
            args.tags,
            args.titles,
            args.per_user_output,
        )
    temporary = args.receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.receipt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
