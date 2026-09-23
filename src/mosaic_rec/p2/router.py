"""p2 support-aware router scoring over frozen expert checkpoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new
from mosaic_rec.p2.models import BERT4RecExpert, ContentTwoTower
from mosaic_rec.p2.runtime import (
    exact_score,
    load_catalog,
    load_histories,
    sequence_tensor,
    validation_queries,
)


def _standardize(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    count = finite.sum(dim=-1, keepdim=True).clamp_min(1)
    safe = torch.where(finite, scores, 0)
    mean = safe.sum(dim=-1, keepdim=True) / count
    centered = torch.where(finite, scores - mean, 0)
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    scale = variance.sqrt().clamp_min(1e-8)
    return torch.where(finite, centered / scale, -torch.inf)


def hard_zero_combine(
    expert_scores: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    if expert_scores.ndim != 3 or expert_scores.shape[1] != 2:
        raise ValueError("Expected [batch, sequence/content, catalog] scores")
    aligned = _standardize(expert_scores)
    return torch.where(support[None, :].eq(0), aligned[:, 1, :], aligned[:, 0, :])


def score_r1(
    *,
    repo: Path,
    config_path: Path,
    pilot_config_path: Path,
    fit_path: Path,
    query_path: Path,
    context_dir: Path,
    feature_dir: Path,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    pilot = json.loads(pilot_config_path.read_text())
    if config["protocol_id"] != PROTOCOL_ID or pilot["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Router config protocol mismatch")
    sequence_config = pilot["sequence"]
    content_config = pilot["content"]
    item_ids, image_np, text_np, tag_np = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    histories = load_histories(
        fit_path, item_to_index, int(sequence_config["max_length"])
    )
    device = torch.device("cuda:0")

    sequence = BERT4RecExpert(
        len(item_ids),
        width=int(sequence_config["width"]),
        length=int(sequence_config["max_length"]),
        layers=int(sequence_config["layers"]),
        heads=int(sequence_config["heads"]),
        dropout=float(sequence_config["dropout"]),
    ).to(device)
    sequence_checkpoint = Path(config["experts"]["sequence_checkpoint"])
    sequence.load_state_dict(
        torch.load(sequence_checkpoint, map_location=device)["model"]
    )
    sequence.eval()
    image = torch.tensor(np.asarray(image_np), device=device)
    text = torch.tensor(np.asarray(text_np), device=device)
    tags = torch.tensor(tag_np.astype(np.int64) + 1, device=device)
    content = ContentTwoTower(
        image,
        text,
        tags,
        width=int(content_config["width"]),
        tag_count=int(tags.max().item()),
    ).to(device)
    content_checkpoint = Path(config["experts"]["content_checkpoint"])
    content.load_state_dict(
        torch.load(content_checkpoint, map_location=device)["model"]
    )
    content.eval()

    item_context = pq.read_table(context_dir / "item_context.parquet")
    first_seen = torch.tensor(
        item_context.column("first_observed_timestamp").to_numpy(), device=device
    )
    support = torch.tensor(
        item_context.column("fit_interactions").to_numpy(), device=device
    )
    fit_classes = torch.tensor(
        sorted({item for history in histories.values() for item in history}),
        device=device,
    )
    sequence_catalog_ids = torch.arange(1, len(item_ids) + 1, device=device)
    sequence_catalog = sequence.item(sequence_catalog_ids).detach().clone()
    unseen = torch.ones(len(item_ids) + 1, dtype=torch.bool, device=device)
    unseen[fit_classes] = False
    sequence_catalog[unseen[1:]] = 0
    content_catalog_parts = []
    with torch.inference_mode():
        for offset in range(0, len(item_ids), 4096):
            ids = torch.arange(
                offset + 1, min(len(item_ids), offset + 4096) + 1, device=device
            )
            content_catalog_parts.append(content.item_vectors(ids).float())
    content_catalog = torch.cat(content_catalog_parts)
    queries = validation_queries(query_path)

    def score_batch(users: list[str]) -> torch.Tensor:
        sequence_histories = [
            histories[user][-int(sequence_config["max_length"]) + 1 :] for user in users
        ]
        sequence_with_mask = [
            history + [sequence.mask_id] for history in sequence_histories
        ]
        sequence_tensor_batch, sequence_lengths = sequence_tensor(
            sequence_with_mask, int(sequence_config["max_length"])
        )
        content_tensor, _ = sequence_tensor(
            [histories[user][-int(content_config["max_history"]) :] for user in users],
            int(content_config["max_history"]),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sequence_state = sequence.query(
                sequence_tensor_batch.to(device), sequence_lengths.to(device)
            )
            sequence_scores = sequence_state @ sequence_catalog.T
            content_state = content.user_vectors(content_tensor.to(device))
            content_scores = content_state @ content_catalog.T
        return torch.stack([sequence_scores, content_scores], dim=1)

    scoring = exact_score(
        model_id=config["router"]["model_id"],
        role="VALIDATION",
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        combine_scores=lambda scores: hard_zero_combine(scores, support),
        output=output,
        batch_size=int(config["scoring"]["batch_size"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-D01-PILOT",
        "status": "PASS",
        "model_id": config["router"]["model_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "config_sha256": sha256_file(config_path),
        "pilot_config_sha256": sha256_file(pilot_config_path),
        "sequence_checkpoint_sha256": sha256_file(sequence_checkpoint),
        "content_checkpoint_sha256": sha256_file(content_checkpoint),
        "alignment": config["router"]["alignment"],
        "gpu_qualification": gpu_qualification,
        "validation": scoring,
        "evidence_class": "p2_measured_development_router_pilot",
    }
    write_json_new(receipt, result)
    return result
