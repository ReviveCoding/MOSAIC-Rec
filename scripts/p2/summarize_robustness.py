"""Summarize all preregistered p2 robustness cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import (
    load_json,
    sha256_file,
    verify_p1_immutable,
    write_json_new,
)
from mosaic_rec.p2.locked_analysis import percentile_bootstrap
from mosaic_rec.p2.robustness import deterministic_item_mask


def _stem(intervention: str, intensity: float) -> str:
    return f"{intervention.lower()}_{intensity:g}".replace(".", "p")


def _columns(path: Path) -> dict[str, np.ndarray]:
    names = [
        "user_id",
        "target_item_id",
        "ndcg_at_20",
        "recall_at_20",
        "fit_support",
        "is_controlled_zero",
        "is_warm",
        "router_weight",
    ]
    table = pq.read_table(path, columns=names)
    return {name: table[name].to_numpy(zero_copy_only=False) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    manifest = load_json(args.repo / "configs/p2/e04_analysis_manifest.json")
    base_path = args.scratch / "locked_final/r4_topk.parquet"
    base = _columns(base_path)
    populations = {
        "overall": np.ones(base["user_id"].size, dtype=bool),
        "CONTROLLED_ZERO_SUPPORT": base["is_controlled_zero"].astype(bool),
        "warm_support_gt_20": base["is_warm"].astype(bool),
    }
    rows: list[dict[str, Any]] = []
    for position, cell in enumerate(manifest["robustness"]["cells"], start=1):
        intervention = str(cell["intervention"])
        intensity = float(cell["intensity"])
        stem = _stem(intervention, intensity)
        receipt_path = args.repo / f"evidence/p2/p2-g00-{stem}.json"
        receipt = load_json(receipt_path)
        path = Path(receipt["locked_final"]["output"])
        if sha256_file(path) != receipt["locked_final"]["output_sha256"]:
            raise RuntimeError(f"Robustness hash mismatch: {stem}")
        data = _columns(path)
        for identity in ("user_id", "target_item_id"):
            if not np.array_equal(base[identity], data[identity]):
                raise RuntimeError(f"Robustness pairing mismatch: {stem} {identity}")
        metrics = {}
        for population, mask in populations.items():
            delta = data["ndcg_at_20"][mask] - base["ndcg_at_20"][mask]
            metrics[population] = {
                "n": int(mask.sum()),
                "ndcg_at_20": float(data["ndcg_at_20"][mask].mean()),
                "recall_at_20": float(data["recall_at_20"][mask].mean()),
                "ndcg_delta_vs_A0": percentile_bootstrap(
                    delta, draws=10_000, seed=2026092301
                ),
            }
        positive = base["fit_support"] > 0
        if intervention == "CONTENT_FEATURE_PERTURBATION":
            degraded = np.ones(positive.size, dtype=bool)
        else:
            modality = {
                "IMAGE_MISSING": "image",
                "TEXT_MISSING": "text",
                "IMAGE_TEXT_JOINT_MISSING": "joint",
            }[intervention]
            degraded = deterministic_item_mask(
                base["target_item_id"].tolist(),
                model_id="R4",
                role="LOCKED_FINAL",
                modality=modality,
                intensity=intensity,
            )
        directional = positive & degraded
        base_content = 1.0 - base["router_weight"][directional]
        cell_content = 1.0 - data["router_weight"][directional]
        content_delta = float((cell_content - base_content).mean())
        rows.append(
            {
                "position": position,
                "intervention": intervention,
                "intensity": intensity,
                "receipt_path": receipt_path.as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
                "output_sha256": receipt["locked_final"]["output_sha256"],
                "metrics": metrics,
                "directional_positive_support": {
                    "n_degraded_targets": int(directional.sum()),
                    "base_content_weight": float(base_content.mean()),
                    "cell_content_weight": float(cell_content.mean()),
                    "content_weight_delta": content_delta,
                    "expectation": "content degradation must not increase content reliance",
                    "pass": content_delta <= 0.0,
                },
            }
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-G00",
        "status": "PASS_ANALYSIS",
        "role": "LOCKED_FINAL",
        "qualification_access_count": 1,
        "locked_final_access_count": 1,
        "base_r4_sha256": sha256_file(base_path),
        "cells": rows,
        "all_cells_terminal": len(rows) == 14,
        "directional_pass_count": sum(
            row["directional_positive_support"]["pass"] for row in rows
        ),
        "directional_fail_count": sum(
            not row["directional_positive_support"]["pass"] for row in rows
        ),
        "multiplicity": "Holm was frozen, but no confirmatory p-values were preregistered; intervals are reported without inventing post-lock p-values",
        "terminology": "adaptive routing; reliability-aware prohibited",
        "evidence_class": "p2_locked_preregistered_robustness",
    }
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during robustness summary")
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
