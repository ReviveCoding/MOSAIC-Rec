"""Summarize p2 independent-seed stability without best-seed selection."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("Seed distribution requires at least one value")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "coefficient_of_variation": sample_sd / abs(mean) if mean != 0 else None,
        "min": min(values),
        "max": max(values),
    }


def summarize_seed_qualification(
    *, repo: Path, manifest_path: Path, output: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("E01 manifest protocol mismatch")
    families: dict[str, list[dict[str, Any]]] = {}
    for identity in manifest["runs"]:
        run_id = identity["run_id"]
        receipt_path = repo / f"evidence/p2/p2-e01-{run_id.lower()}.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"Missing E01 receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "PASS":
            raise RuntimeError(f"E01 run did not pass: {receipt_path}")
        if receipt.get("config_sha256") != identity["sha256"]:
            raise RuntimeError(f"E01 config hash mismatch: {receipt_path}")
        checkpoint = Path(receipt["checkpoint"])
        scored = Path(receipt["validation"]["output"])
        if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
            raise RuntimeError(f"E01 checkpoint hash mismatch: {receipt_path}")
        if sha256_file(scored) != receipt["validation"]["output_sha256"]:
            raise RuntimeError(f"E01 output hash mismatch: {receipt_path}")
        families.setdefault(identity["family"], []).append(receipt)
    summaries = {}
    for family, receipts in sorted(families.items()):
        receipts.sort(key=lambda receipt: int(receipt["seed"]))
        summaries[family] = {
            "seeds": [int(receipt["seed"]) for receipt in receipts],
            "checkpoint_sha256": [receipt["checkpoint_sha256"] for receipt in receipts],
            "metrics": {
                label: distribution(
                    [
                        float(
                            receipt["validation"]["slices"][label]["ndcg_at_20"] or 0.0
                        )
                        for receipt in receipts
                    ]
                )
                for label in ("all", "controlled_zero", "natural_zero", "warm")
            },
        }
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E01",
        "status": "PASS_DEVELOPMENT",
        "evidence_class": "p2_measured_development_independent_seed_summary",
        "manifest_sha256": sha256_file(manifest_path),
        "best_seed_selection": False,
        "families": summaries,
        "protected_roles_opened": [],
    }
    write_json_new(output, result)
    return result
