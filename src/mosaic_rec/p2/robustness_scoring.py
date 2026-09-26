"""Exact locked p2 R4 scoring for the frozen robustness cells."""

from __future__ import annotations

import subprocess
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
from mosaic_rec.p2.locked import _content, _content_catalog, _ensemble, _sequence
from mosaic_rec.p2.models import ContextRouterGate
from mosaic_rec.p2.robustness import apply_missingness, perturb_content_features
from mosaic_rec.p2.router import align_expert_scores
from mosaic_rec.p2.runtime import (
    exact_score,
    load_catalog,
    load_histories,
    sequence_tensor,
    validation_queries,
)


def _cell_is_frozen(repo: Path, intervention: str, intensity: float) -> bool:
    manifest = load_json(repo / "configs/p2/e04_analysis_manifest.json")
    return any(
        row["intervention"] == intervention
        and float(row["intensity"]) == float(intensity)
        for row in manifest["robustness"]["cells"]
    )


def score_robustness_cell(
    *,
    repo: Path,
    scratch: Path,
    intervention: str,
    intensity: float,
    output: Path,
    receipt: Path,
    gpu_qualification: dict[str, object],
) -> dict[str, Any]:
    assert_outcome_access(repo, "LOCKED_FINAL")
    if not _cell_is_frozen(repo, intervention, intensity):
        raise RuntimeError("Robustness cell is not in the E04 freeze")
    if output.exists() or receipt.exists():
        raise FileExistsError("Preserve existing robustness artifact")
    freeze_path = repo / "evidence/p2/e05_freeze_manifest.json"
    freeze = load_json(freeze_path)
    runs = freeze["candidates"]["routers"][0]["runs"]
    seq_runs = freeze["candidates"]["C_seq"]["runs"]
    content_runs = freeze["candidates"]["C_cold"]["runs"]
    for run in [*runs, *seq_runs, *content_runs]:
        if sha256_file(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed: {run['run_id']}")

    device = torch.device("cuda:0")
    feature_dir = scratch / "features/pixelrec50k"
    context_dir = scratch / "contexts/pixelrec50k"
    fit_path = scratch / "data/pixelrec50k/FIT_VISIBLE.parquet"
    query_path = context_dir / "LOCKED_FINAL_QUERIES.parquet"
    item_ids, image_np, text_np, tag_np = load_catalog(feature_dir)
    item_context = pq.read_table(context_dir / "item_context.parquet")
    first_seen_np = item_context.column("first_observed_timestamp").to_numpy()
    support_np = item_context.column("fit_interactions").to_numpy()
    image_present_np = item_context.column("image_present").to_numpy().astype(bool)
    text_present_np = item_context.column("text_present").to_numpy().astype(bool)
    if intervention == "CONTENT_FEATURE_PERTURBATION":
        fit_mask = support_np > 0
        image_np = perturb_content_features(
            image_np,
            item_ids=item_ids,
            modality="image",
            intensity=intensity,
            fit_standard_deviation=np.std(image_np[fit_mask], axis=0),
        )
        text_np = perturb_content_features(
            text_np,
            item_ids=item_ids,
            modality="text",
            intensity=intensity,
            fit_standard_deviation=np.std(text_np[fit_mask], axis=0),
        )
    else:
        image_np, text_np, image_present_np, text_present_np = apply_missingness(
            image_np,
            text_np,
            image_present_np,
            text_present_np,
            item_ids=item_ids,
            model_id="R4",
            role="LOCKED_FINAL",
            intervention=intervention,
            intensity=intensity,
        )

    item_to_index = {item: index for index, item in enumerate(item_ids)}
    histories = load_histories(fit_path, item_to_index, 51)
    queries = validation_queries(query_path)
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

    sequences = [
        _sequence(Path(run["checkpoint"]), len(item_ids), device) for run in seq_runs
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
        gate = ContextRouterGate(hidden=int(payload["config"]["hidden"])).to(device)
        gate.load_state_dict(payload["model"])
        gates.append(gate.eval())
    pending: dict[str, torch.Tensor] = {}

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
                    content.user_vectors(content_tensor.to(device)) @ content_catalog.T
                )
                weight = gate(feature).float()
            aligned = align_expert_scores(
                torch.stack([seq_score, content_score], dim=1),
                "rank_percentile_normalization",
            )
            final_scores.append(weight * aligned[:, 0] + (1 - weight) * aligned[:, 1])
            weights.append(weight)
            seq_values.append(aligned[:, 0])
            content_values.append(aligned[:, 1])
        pending["weight"] = torch.stack(weights).mean(0)
        pending["sequence"] = torch.stack(seq_values).mean(0)
        pending["content"] = torch.stack(content_values).mean(0)
        return torch.stack(final_scores, dim=1)

    def diagnostic_batch(users: list[str], targets: list[int]) -> dict[str, list[Any]]:
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
            "user_recency_days": [queries["user_recency_days"][i] for i in query_rows],
            "user_tail_share": [queries["history_tail_share"][i] for i in query_rows],
            "image_present": image_present_np[target_array].tolist(),
            "text_present": text_present_np[target_array].tolist(),
        }

    cell_id = f"{intervention}-{intensity:g}"
    scoring = exact_score(
        model_id=f"P2-ROBUST-R4-{cell_id}",
        config_id=f"P2-G00-{cell_id}",
        role="LOCKED_FINAL",
        queries=queries,
        histories=histories,
        item_ids=item_ids,
        first_seen=first_seen,
        score_batch=score_batch,
        combine_scores=_ensemble,
        diagnostic_batch=diagnostic_batch,
        output=output,
        batch_size=64,
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-G00",
        "status": "PASS",
        "role": "LOCKED_FINAL",
        "intervention": intervention,
        "intensity": intensity,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "e05_manifest_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": [run["checkpoint_sha256"] for run in runs],
        "gpu_qualification": gpu_qualification,
        "locked_final": scoring,
        "qualification_access_count": 1,
        "locked_final_access_count": 1,
        "evidence_class": "p2_locked_preregistered_robustness",
    }
    write_json_new(receipt, result)
    return result
