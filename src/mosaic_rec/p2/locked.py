"""Exact p2 LOCKED_FINAL scoring for the E05-frozen ensembles."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import (
    assert_outcome_access,
    load_json,
    sha256_file,
    write_json_new,
)
from mosaic_rec.p2.gate_ladder import _gate_features
from mosaic_rec.p2.models import (
    BERT4RecExpert,
    ContentTwoTower,
    ContextRouterGate,
    MonotonicSupportGate,
)
from mosaic_rec.p2.router import align_expert_scores
from mosaic_rec.p2.runtime import (
    exact_score,
    load_catalog,
    load_histories,
    sequence_tensor,
    validation_queries,
)


def _sequence(
    checkpoint: Path, catalog_size: int, device: torch.device
) -> BERT4RecExpert:
    payload = torch.load(checkpoint, map_location=device)
    config = payload["config"]
    model = BERT4RecExpert(
        catalog_size,
        width=int(config["width"]),
        length=int(config["max_length"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _content(
    checkpoint: Path,
    image_np: np.ndarray,
    text_np: np.ndarray,
    tag_np: np.ndarray,
    device: torch.device,
) -> ContentTwoTower:
    payload = torch.load(checkpoint, map_location=device)
    config = payload["config"]
    image = torch.tensor(np.asarray(image_np), device=device)
    text = torch.tensor(np.asarray(text_np), device=device)
    tags = torch.tensor(tag_np.astype(np.int64) + 1, device=device)
    model = ContentTwoTower(
        image,
        text,
        tags,
        width=int(config["width"]),
        tag_count=int(tags.max().item()),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _content_catalog(
    model: ContentTwoTower, size: int, device: torch.device
) -> torch.Tensor:
    parts = []
    with torch.inference_mode():
        for offset in range(0, size, 4096):
            ids = torch.arange(offset + 1, min(size, offset + 4096) + 1, device=device)
            parts.append(model.item_vectors(ids).float())
    return torch.cat(parts)


def _ensemble(scores: torch.Tensor) -> torch.Tensor:
    return align_expert_scores(scores, "rank_percentile_normalization").mean(dim=1)


def score_locked(
    *,
    repo: Path,
    scratch: Path,
    family: str,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    assert_outcome_access(repo, "LOCKED_FINAL")
    if output.exists() or receipt.exists():
        raise FileExistsError(f"Preserve existing LOCKED_FINAL artifact for {family}")
    freeze_path = repo / "evidence/p2/e05_freeze_manifest.json"
    freeze = load_json(freeze_path)
    candidates = freeze["candidates"]
    if family == "CSEQ_BERT4REC":
        runs = candidates["C_seq"]["runs"]
    elif family == "CCOLD_TWO_TOWER":
        runs = candidates["C_cold"]["runs"]
    else:
        matches = [row for row in candidates["routers"] if row["family"] == family]
        if len(matches) != 1:
            raise ValueError(f"Unknown E05-frozen LOCKED_FINAL family: {family}")
        runs = matches[0]["runs"]
    for run in runs:
        if sha256_file(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed: {run['run_id']}")

    device = torch.device("cuda:0")
    feature_dir = scratch / "features/pixelrec50k"
    context_dir = scratch / "contexts/pixelrec50k"
    fit_path = scratch / "data/pixelrec50k/FIT_VISIBLE.parquet"
    query_path = context_dir / "LOCKED_FINAL_QUERIES.parquet"
    item_ids, image_np, text_np, tag_np = load_catalog(feature_dir)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    histories = load_histories(fit_path, item_to_index, 51)
    queries = validation_queries(query_path)
    item_context = pq.read_table(context_dir / "item_context.parquet")
    first_seen_np = item_context.column("first_observed_timestamp").to_numpy()
    support_np = item_context.column("fit_interactions").to_numpy()
    image_present_np = item_context.column("image_present").to_numpy()
    text_present_np = item_context.column("text_present").to_numpy()
    first_seen = torch.tensor(first_seen_np, device=device)
    support = torch.tensor(support_np, device=device)
    image_present = torch.tensor(image_present_np, device=device)
    text_present = torch.tensor(text_present_np, device=device)
    catalog_ids = torch.arange(1, len(item_ids) + 1, device=device)
    fit_classes = torch.tensor(
        sorted({item for history in histories.values() for item in history}),
        device=device,
    )
    unseen = torch.ones(len(item_ids) + 1, dtype=torch.bool, device=device)
    unseen[fit_classes] = False
    query_by_user = {user: row for row, user in enumerate(queries["user_id"])}
    pending: dict[str, torch.Tensor] = {}
    diagnostic: Callable[[list[str], list[int]], dict[str, list[Any]]] | None = None

    if family == "CSEQ_BERT4REC":
        models = [
            _sequence(Path(run["checkpoint"]), len(item_ids), device) for run in runs
        ]
        catalogs = [model.item(catalog_ids).detach().clone() for model in models]
        for catalog in catalogs:
            catalog[unseen[1:]] = 0

        def score_batch(users: list[str]) -> torch.Tensor:
            values = []
            for model, catalog in zip(models, catalogs):
                histories_batch = [
                    histories[user][-49:] + [model.mask_id] for user in users
                ]
                tensor, lengths = sequence_tensor(histories_batch, 50)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model.query(tensor.to(device), lengths.to(device))
                    values.append(state @ catalog.T)
            return torch.stack(values, dim=1)

    elif family == "CCOLD_TWO_TOWER":
        models = [
            _content(Path(run["checkpoint"]), image_np, text_np, tag_np, device)
            for run in runs
        ]
        catalogs = [_content_catalog(model, len(item_ids), device) for model in models]

        def score_batch(users: list[str]) -> torch.Tensor:
            tensor, _ = sequence_tensor([histories[user][-50:] for user in users], 50)
            values = []
            for model, catalog in zip(models, catalogs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values.append(model.user_vectors(tensor.to(device)) @ catalog.T)
            return torch.stack(values, dim=1)

    else:
        seq_runs = candidates["C_seq"]["runs"]
        content_runs = candidates["C_cold"]["runs"]
        sequences = [
            _sequence(Path(run["checkpoint"]), len(item_ids), device)
            for run in seq_runs
        ]
        contents = [
            _content(Path(run["checkpoint"]), image_np, text_np, tag_np, device)
            for run in content_runs
        ]
        sequence_catalogs = [
            model.item(catalog_ids).detach().clone() for model in sequences
        ]
        for catalog in sequence_catalogs:
            catalog[unseen[1:]] = 0
        content_catalogs = [
            _content_catalog(model, len(item_ids), device) for model in contents
        ]
        gates = []
        for run in runs:
            payload = torch.load(run["checkpoint"], map_location=device)
            gate: torch.nn.Module
            if family == "R2":
                gate = MonotonicSupportGate().to(device)
            else:
                gate = ContextRouterGate(hidden=int(payload["config"]["hidden"])).to(
                    device
                )
            gate.load_state_dict(payload["model"])
            gates.append(gate.eval())

        def score_batch(users: list[str]) -> torch.Tensor:
            seq_histories = [histories[user][-49:] for user in users]
            content_tensor, _ = sequence_tensor(
                [histories[user][-50:] for user in users], 50
            )
            rows = [query_by_user[user] for user in users]
            stamps = torch.tensor(
                [queries["query_timestamp"][row] for row in rows], device=device
            )
            user_features = torch.tensor(
                [
                    [
                        queries["fit_history_length"][row],
                        queries["user_recency_days"][row],
                        queries["median_history_popularity"][row],
                        queries["history_tail_share"][row],
                    ]
                    for row in rows
                ],
                device=device,
            )
            candidates_all = catalog_ids[None, :].expand(len(users), -1)
            feature = _gate_features(
                candidate_ids=candidates_all,
                query_timestamps=stamps,
                user_features=user_features,
                support=support,
                first_seen=first_seen,
                image_present=image_present,
                text_present=text_present,
            )
            final_scores, weights, seq_values, content_values = [], [], [], []
            for sequence, content, seq_catalog, content_catalog, gate in zip(
                sequences, contents, sequence_catalogs, content_catalogs, gates
            ):
                with_mask = [history + [sequence.mask_id] for history in seq_histories]
                seq_tensor, seq_lengths = sequence_tensor(with_mask, 50)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    seq_score = (
                        sequence.query(seq_tensor.to(device), seq_lengths.to(device))
                        @ seq_catalog.T
                    )
                    content_score = (
                        content.user_vectors(content_tensor.to(device))
                        @ content_catalog.T
                    )
                    weight = gate(feature).float()
                aligned = align_expert_scores(
                    torch.stack([seq_score, content_score], dim=1),
                    "rank_percentile_normalization",
                )
                final_scores.append(
                    weight * aligned[:, 0] + (1 - weight) * aligned[:, 1]
                )
                weights.append(weight)
                seq_values.append(aligned[:, 0])
                content_values.append(aligned[:, 1])
            pending["weight"] = torch.stack(weights).mean(0)
            pending["sequence"] = torch.stack(seq_values).mean(0)
            pending["content"] = torch.stack(content_values).mean(0)
            return torch.stack(final_scores, dim=1)

        def diagnostic_batch(
            users: list[str], targets: list[int]
        ) -> dict[str, list[Any]]:
            index = torch.tensor(targets, device=device)
            row = torch.arange(len(users), device=device)
            query_rows = [query_by_user[user] for user in users]
            stamps = np.asarray([queries["query_timestamp"][i] for i in query_rows])
            target_array = np.asarray(targets)
            return {
                "router_weight": pending["weight"][row, index].cpu().tolist(),
                "sequence_score": pending["sequence"][row, index].cpu().tolist(),
                "content_score": pending["content"][row, index].cpu().tolist(),
                "item_age_days": (
                    (stamps - first_seen_np[target_array]) / 86400.0
                ).tolist(),
                "user_fit_history_length": [
                    queries["fit_history_length"][i] for i in query_rows
                ],
                "user_recency_days": [
                    queries["user_recency_days"][i] for i in query_rows
                ],
                "user_tail_share": [
                    queries["history_tail_share"][i] for i in query_rows
                ],
                "image_present": image_present_np[target_array].tolist(),
                "text_present": text_present_np[target_array].tolist(),
            }

        diagnostic = diagnostic_batch

    scoring = exact_score(
        model_id=f"P2-LOCKED-{family}-ENSEMBLE",
        config_id=f"P2-F00-{family}-THREE-SEED-ALIGN-RANK",
        role="LOCKED_FINAL",
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        combine_scores=_ensemble,
        diagnostic_batch=diagnostic,
        output=output,
        batch_size=64,
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-F00",
        "status": "PASS",
        "family": family,
        "role": "LOCKED_FINAL",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": [run["checkpoint_sha256"] for run in runs],
        "gpu_qualification": gpu_qualification,
        "locked_final": scoring,
        "qualification_access_count": 1,
        "locked_final_access_count": 1,
        "evidence_class": "p2_measured_locked_exact_full_catalog_ensemble",
    }
    write_json_new(receipt, result)
    return result
