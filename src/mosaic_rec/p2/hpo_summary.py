"""Deterministic summaries for the frozen p2 HPO budget."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new

PRE_ROUTER_FAMILIES = (
    "CSEQ_BERT4REC",
    "CCOLD_TWO_TOWER",
    "LEGACY_M1",
    "BPR_MF",
    "LIGHTGCN",
    "SASREC",
    "VBPR",
)


def metric_key(receipt: dict[str, Any]) -> tuple[float, float, float, str]:
    slices = receipt["validation"]["slices"]
    return (
        float(slices["all"]["ndcg_at_20"] or 0.0),
        float(slices["warm"]["ndcg_at_20"] or 0.0),
        float(slices["controlled_zero"]["ndcg_at_20"] or 0.0),
        str(receipt["trial_id"]),
    )


def select_family(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [receipt for receipt in receipts if receipt.get("status") == "PASS"]
    if not passed:
        return {"status": "UNQUALIFIED_NO_PASS", "selected_trial_id": None}
    # The final tie break is trial_id ascending, so handle it separately from
    # the three descending metric fields.
    best_metrics = max(metric_key(receipt)[:3] for receipt in passed)
    tied = [receipt for receipt in passed if metric_key(receipt)[:3] == best_metrics]
    selected = min(tied, key=lambda receipt: str(receipt["trial_id"]))
    return {
        "status": "SELECTED_DEVELOPMENT",
        "selected_trial_id": selected["trial_id"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "validation": selected["validation"]["slices"],
    }


def summarize_pre_router_hpo(
    *, repo: Path, manifest_path: Path, hpo_path: Path, output: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    hpo = json.loads(hpo_path.read_text())
    if manifest["protocol_id"] != PROTOCOL_ID or hpo["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("p2 HPO protocol mismatch")
    expected = {
        trial["trial_id"]: trial
        for trial in manifest["trials"]
        if trial["family"] in PRE_ROUTER_FAMILIES
    }
    records: dict[str, dict[str, Any]] = {}
    for trial_id, identity in expected.items():
        receipt_path = repo / f"evidence/p2/p2-e00-{trial_id.lower()}.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"Missing HPO receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"Protocol mismatch: {receipt_path}")
        if receipt.get("trial_id") != trial_id:
            raise RuntimeError(f"Trial mismatch: {receipt_path}")
        if receipt.get("config_sha256") != identity["sha256"]:
            raise RuntimeError(f"Config hash mismatch: {receipt_path}")
        if receipt.get("status") == "PASS":
            checkpoint = Path(receipt["checkpoint"])
            scored = Path(receipt["validation"]["output"])
            if sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
                raise RuntimeError(f"Checkpoint hash mismatch: {receipt_path}")
            if sha256_file(scored) != receipt["validation"].get("output_sha256"):
                raise RuntimeError(f"Scored output hash mismatch: {receipt_path}")
        records[trial_id] = receipt
    families: dict[str, Any] = {}
    for family in PRE_ROUTER_FAMILIES:
        family_receipts = [
            records[trial["trial_id"]]
            for trial in manifest["trials"]
            if trial["family"] == family
        ]
        families[family] = {
            "trials": [
                {
                    "trial_id": receipt["trial_id"],
                    "status": receipt["status"],
                    "validation": receipt.get("validation", {}).get("slices"),
                    "best_epoch": receipt.get("best_epoch"),
                    "epochs_completed": receipt.get("epochs_completed"),
                    "stopping_reason": receipt.get("stopping_reason"),
                }
                for receipt in family_receipts
            ],
            "selection": select_family(family_receipts),
        }
    status = (
        "PASS_DEVELOPMENT"
        if all(
            result["selection"]["status"] == "SELECTED_DEVELOPMENT"
            for result in families.values()
        )
        else "HPO_GRID_COMPLETE_UNQUALIFIED"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E00-PRE-ROUTER-SUMMARY",
        "status": status,
        "evidence_class": "p2_measured_development_hpo_summary",
        "protected_roles_opened": [],
        "hpo_sha256": sha256_file(hpo_path),
        "trial_manifest_sha256": sha256_file(manifest_path),
        "trial_count": len(expected),
        "families": families,
    }
    write_json_new(output, result)
    return result
