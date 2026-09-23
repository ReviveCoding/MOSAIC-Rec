"""Dataset loading, exact ranking, and expert training for p2."""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new
from mosaic_rec.p2.models import BERT4RecExpert, ContentTwoTower, cloze_batch


def load_histories(
    fit_path: Path, item_to_index: dict[str, int], max_length: int
) -> dict[str, list[int]]:
    table = pq.read_table(fit_path, columns=["user_id", "item_id", "timestamp"])
    users = table.column("user_id").to_pylist()
    items = table.column("item_id").to_pylist()
    stamps = table.column("timestamp").to_pylist()
    order = sorted(range(len(users)), key=lambda i: (stamps[i], users[i], items[i]))
    histories: dict[str, list[int]] = defaultdict(list)
    for index in order:
        histories[users[index]].append(item_to_index[items[index]] + 1)
    return {user: values[-max_length:] for user, values in histories.items()}


def sequence_tensor(
    histories: list[list[int]], length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.zeros((len(histories), length), dtype=np.int64)
    lengths = np.zeros(len(histories), dtype=np.int64)
    for row, history in enumerate(histories):
        values = history[-length:]
        array[row, : len(values)] = values
        lengths[row] = len(values)
    return torch.from_numpy(array), torch.from_numpy(lengths)


def load_catalog(
    feature_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    item_ids = (feature_dir / "publisher_image_item_ids.txt").read_text().splitlines()
    image = np.load(feature_dir / "publisher_image_features.npy", mmap_mode="r")
    text = np.load(feature_dir / "authorized_minilm_text_features.npy", mmap_mode="r")
    tag = np.load(feature_dir / "tag_ids.npy")
    if not (len(item_ids) == len(image) == len(text) == len(tag)):
        raise RuntimeError("Catalog feature alignment failure")
    return item_ids, image, text, tag


def validation_queries(query_path: Path) -> dict[str, list[Any]]:
    table = pq.read_table(query_path)
    return {name: table.column(name).to_pylist() for name in table.column_names}


def exact_score(
    *,
    model_id: str,
    role: str,
    queries: dict[str, list[Any]],
    histories: dict[str, list[int]],
    item_ids: list[str],
    first_seen: torch.Tensor,
    score_batch: Callable[[list[str]], torch.Tensor],
    output: Path,
    batch_size: int,
    top_k: int = 100,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    device = first_seen.device
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    rows: dict[str, list[Any]] = {
        "user_id": [],
        "query_timestamp": [],
        "target_item_id": [],
        "target_rank": [],
        "ndcg_at_20": [],
        "recall_at_20": [],
        "mrr_at_20": [],
        "hit_at_20": [],
        "precision_at_20": [],
        "top20_item_ids": [],
        "top20_scores": [],
        "top100_item_ids": [],
        "top100_scores": [],
        "model_id": [],
        "protocol_id": [],
        "fit_support": [],
        "is_controlled_zero": [],
        "is_natural_zero": [],
        "is_warm": [],
    }
    ndcg_sums = defaultdict(float)
    count_sums = defaultdict(int)
    start = time.perf_counter()
    catalog_tie = (
        torch.arange(len(item_ids), device=device, dtype=torch.float64) * 1e-12
    )
    with torch.inference_mode():
        for offset in range(0, len(queries["user_id"]), batch_size):
            users = queries["user_id"][offset : offset + batch_size]
            scores = score_batch(users).float().to(torch.float64)
            stamps = torch.tensor(
                queries["query_timestamp"][offset : offset + len(users)],
                device=device,
            )
            scores.masked_fill_(first_seen[None, :] > stamps[:, None], -torch.inf)
            targets = [
                item_to_index[item]
                for item in queries["target_item_id"][offset : offset + len(users)]
            ]
            for row, user in enumerate(users):
                history = histories[user]
                target_model_id = targets[row] + 1
                masked = [item - 1 for item in history if item != target_model_id]
                if masked:
                    scores[row, torch.tensor(masked, device=device)] = -torch.inf
            scores -= catalog_tie[None, :]
            target_tensor = torch.tensor(targets, device=device)
            target_scores = scores[
                torch.arange(len(users), device=device), target_tensor
            ]
            ranks = ((scores > target_scores[:, None]).sum(dim=1) + 1).cpu().tolist()
            top_scores, top_indices = scores.topk(top_k, dim=1)
            top_scores_list = top_scores.cpu().tolist()
            top_indices_list = top_indices.cpu().tolist()
            for local, user in enumerate(users):
                global_row = offset + local
                rank = int(ranks[local])
                hit = rank <= 20
                ndcg = 1.0 / math.log2(rank + 1) if hit else 0.0
                mrr = 1.0 / rank if hit else 0.0
                controlled = bool(queries["is_controlled_zero"][global_row])
                natural = bool(queries["is_natural_zero"][global_row])
                warm = bool(queries["is_warm"][global_row])
                labels = ["all"]
                if controlled:
                    labels.append("controlled_zero")
                if natural:
                    labels.append("natural_zero")
                if warm:
                    labels.append("warm")
                for label in labels:
                    ndcg_sums[label] += ndcg
                    count_sums[label] += 1
                indices = top_indices_list[local]
                values = top_scores_list[local]
                rows["user_id"].append(user)
                rows["query_timestamp"].append(queries["query_timestamp"][global_row])
                rows["target_item_id"].append(queries["target_item_id"][global_row])
                rows["target_rank"].append(rank)
                rows["ndcg_at_20"].append(ndcg)
                rows["recall_at_20"].append(float(hit))
                rows["mrr_at_20"].append(mrr)
                rows["hit_at_20"].append(float(hit))
                rows["precision_at_20"].append(0.05 if hit else 0.0)
                rows["top20_item_ids"].append(
                    [item_ids[index] for index in indices[:20]]
                )
                rows["top20_scores"].append(values[:20])
                rows["top100_item_ids"].append([item_ids[index] for index in indices])
                rows["top100_scores"].append(values)
                rows["model_id"].append(model_id)
                rows["protocol_id"].append(PROTOCOL_ID)
                rows["fit_support"].append(queries["fit_interactions"][global_row])
                rows["is_controlled_zero"].append(controlled)
                rows["is_natural_zero"].append(natural)
                rows["is_warm"].append(warm)
    torch.cuda.synchronize()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), output, compression="zstd")
    return {
        "role": role,
        "rows": len(rows["user_id"]),
        "seconds": time.perf_counter() - start,
        "users_per_second": len(rows["user_id"]) / (time.perf_counter() - start),
        "slices": {
            label: {
                "n": count_sums[label],
                "ndcg_at_20": ndcg_sums[label] / count_sums[label]
                if count_sums[label]
                else None,
            }
            for label in ("all", "controlled_zero", "natural_zero", "warm")
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def train_sequence(
    *,
    repo: Path,
    config: dict[str, Any],
    fit_path: Path,
    query_path: Path,
    context_dir: Path,
    feature_dir: Path,
    checkpoint: Path,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    section = config["sequence"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    item_ids, _, _, _ = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    histories = load_histories(fit_path, item_to_index, int(section["max_length"]))
    train_histories = [values for values in histories.values() if len(values) >= 2]
    sequences, lengths = sequence_tensor(train_histories, int(section["max_length"]))
    device = torch.device("cuda:0")
    model = BERT4RecExpert(
        len(item_ids),
        width=int(section["width"]),
        length=int(section["max_length"]),
        layers=int(section["layers"]),
        heads=int(section["heads"]),
        dropout=float(section["dropout"]),
    ).to(device)
    fit_item_ids = sorted({item for history in train_histories for item in history})
    fit_classes = torch.tensor(fit_item_ids, device=device)
    lookup = torch.full((len(item_ids) + 2,), -1, device=device, dtype=torch.long)
    lookup[fit_classes] = torch.arange(len(fit_classes), device=device)
    loader = DataLoader(
        TensorDataset(sequences, lengths),
        batch_size=int(section["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    log: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(int(section["epochs"])):
        model.train()
        loss_sum = 0.0
        labels_seen = 0
        for batch, batch_lengths in loader:
            batch = batch.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            masked, selected, labels = cloze_batch(
                batch,
                batch_lengths,
                model.mask_id,
                float(section["mask_probability"]),
            )
            targets = lookup[labels]
            if (targets < 0).any():
                raise RuntimeError("Sequence target absent from FIT classes")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = model(masked)[selected]
                logits = states @ model.item(fit_classes).T
                loss = torch.nn.functional.cross_entropy(logits.float(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            labels_seen += len(labels)
        log.append({"epoch": epoch + 1, "loss": loss_sum / labels_seen})
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": section, "seed": seed}, checkpoint
    )
    queries = validation_queries(query_path)
    first_seen_table = pq.read_table(
        context_dir / "item_context.parquet", columns=["first_observed_timestamp"]
    )
    first_seen = torch.tensor(first_seen_table.column(0).to_numpy(), device=device)
    catalog_ids = torch.arange(1, len(item_ids) + 1, device=device)
    catalog = model.item(catalog_ids).detach().clone()
    unseen = torch.ones(len(item_ids) + 1, dtype=torch.bool, device=device)
    unseen[fit_classes] = False
    catalog[unseen[1:]] = 0

    def score_batch(users: list[str]) -> torch.Tensor:
        batch_histories = [
            histories[user][-int(section["max_length"]) + 1 :] for user in users
        ]
        with_mask = [history + [model.mask_id] for history in batch_histories]
        tensor, batch_lengths = sequence_tensor(with_mask, int(section["max_length"]))
        tensor = tensor.to(device)
        batch_lengths = batch_lengths.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = model.query(tensor, batch_lengths)
            return states @ catalog.T

    model.eval()
    scoring = exact_score(
        model_id=section["model_id"],
        role="VALIDATION",
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        output=output,
        batch_size=int(config["scoring"]["batch_size"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-C03-PILOT",
        "status": "PASS",
        "model_id": section["model_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "config_sha256": sha256_file(repo / "configs/p2/development_pilots.json"),
        "seed": seed,
        "training_examples": len(train_histories),
        "training_log": log,
        "train_seconds": train_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_qualification": gpu_qualification,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation": scoring,
        "evidence_class": "p2_measured_development_pilot",
    }
    write_json_new(receipt, result)
    return result


def train_content(
    *,
    repo: Path,
    config: dict[str, Any],
    fit_path: Path,
    query_path: Path,
    context_dir: Path,
    feature_dir: Path,
    checkpoint: Path,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    section = config["content"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    item_ids, image_np, text_np, tag_np = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    histories = load_histories(fit_path, item_to_index, int(section["max_history"]) + 1)
    training = [
        (user, history[:-1], history[-1])
        for user, history in histories.items()
        if len(history) >= 2
    ]
    item_context = pq.read_table(context_dir / "item_context.parquet")
    first_seen_np = item_context.column("first_observed_timestamp").to_numpy()
    fit_support_np = item_context.column("fit_interactions").to_numpy()
    fit_candidates = np.flatnonzero(fit_support_np > 0)
    fit_candidates = fit_candidates[
        np.argsort(first_seen_np[fit_candidates], kind="stable")
    ]
    fit_candidate_times = first_seen_np[fit_candidates]
    rng = np.random.default_rng(seed)
    negatives = np.zeros(
        (len(training), int(section["negatives_per_user"])), dtype=np.int64
    )
    fit_table = pq.read_table(fit_path, columns=["user_id", "item_id", "timestamp"])
    last_time: dict[str, int] = {}
    for user, stamp in zip(
        fit_table.column("user_id").to_pylist(),
        fit_table.column("timestamp").to_pylist(),
    ):
        last_time[user] = max(last_time.get(user, stamp), stamp)
    for row, (user, history, target) in enumerate(training):
        eligible_n = int(
            np.searchsorted(fit_candidate_times, last_time[user], side="right")
        )
        forbidden = set(history)
        forbidden.add(target)
        for column in range(negatives.shape[1]):
            while True:
                candidate = int(fit_candidates[rng.integers(eligible_n)]) + 1
                if candidate not in forbidden:
                    negatives[row, column] = candidate
                    break
    history_tensor, _ = sequence_tensor(
        [row[1] for row in training], int(section["max_history"])
    )
    target_tensor = torch.tensor([row[2] for row in training], dtype=torch.long)
    negative_tensor = torch.from_numpy(negatives)
    device = torch.device("cuda:0")
    image = torch.tensor(np.asarray(image_np), device=device)
    text = torch.tensor(np.asarray(text_np), device=device)
    tags = torch.tensor(tag_np.astype(np.int64) + 1, device=device)
    model = ContentTwoTower(
        image,
        text,
        tags,
        width=int(section["width"]),
        tag_count=int(tags.max().item()),
    ).to(device)
    loader = DataLoader(
        TensorDataset(history_tensor, target_tensor, negative_tensor),
        batch_size=int(section["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    log: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(int(section["epochs"])):
        model.train()
        loss_sum = 0.0
        count = 0
        for history, target, negative in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                user = model.user_vectors(history)
                positive = (user * model.item_vectors(target)).sum(dim=-1)
                negative_score = (user[:, None, :] * model.item_vectors(negative)).sum(
                    dim=-1
                )
                loss = torch.nn.functional.softplus(
                    -(positive[:, None] - negative_score)
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(history)
            count += len(history)
        log.append({"epoch": epoch + 1, "loss": loss_sum / count})
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": section, "seed": seed}, checkpoint
    )
    model.eval()
    with torch.inference_mode():
        catalog = []
        for offset in range(0, len(item_ids), 4096):
            ids = torch.arange(
                offset + 1, min(len(item_ids), offset + 4096) + 1, device=device
            )
            catalog.append(model.item_vectors(ids).float())
        catalog_tensor = torch.cat(catalog)
    queries = validation_queries(query_path)
    first_seen = torch.tensor(first_seen_np, device=device)

    def score_batch(users: list[str]) -> torch.Tensor:
        tensor, _ = sequence_tensor(
            [histories[user][-int(section["max_history"]) :] for user in users],
            int(section["max_history"]),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            user = model.user_vectors(tensor.to(device))
            return user @ catalog_tensor.T

    scoring = exact_score(
        model_id=section["model_id"],
        role="VALIDATION",
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        output=output,
        batch_size=int(config["scoring"]["batch_size"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-C04-PILOT",
        "status": "PASS",
        "model_id": section["model_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "config_sha256": sha256_file(repo / "configs/p2/development_pilots.json"),
        "seed": seed,
        "training_examples": len(training),
        "training_log": log,
        "train_seconds": train_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_qualification": gpu_qualification,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation": scoring,
        "evidence_class": "p2_measured_development_pilot",
    }
    write_json_new(receipt, result)
    return result


def load_pilot_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("p2 pilot config protocol mismatch")
    return config
