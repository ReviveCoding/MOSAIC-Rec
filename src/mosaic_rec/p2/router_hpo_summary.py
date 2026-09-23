"""Deterministic p2 router HPO selection under the frozen two-sided criterion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new
from mosaic_rec.p2.router_hpo_configs import ROUTER_FAMILIES


def router_objective(
    *, cold: float, warm: float, cold_comparator: float, warm_comparator: float
) -> dict[str, float]:
    cold_term = (
        cold / cold_comparator - 1.0 if cold_comparator > 0 else cold - cold_comparator
    )
    if warm_comparator <= 0:
        raise RuntimeError("C_seq warm VALIDATION NDCG@20 must be positive")
    warm_term = (warm - warm_comparator) / warm_comparator + 0.10
    return {
        "cold_gate_term": cold_term,
        "warm_gate_term": warm_term,
        "router_selection_score": min(cold_term, warm_term),
    }


def summarize_router_hpo(
    *,
    repo: Path,
    manifest_path: Path,
    pre_router_summary_path: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    pre_router = json.loads(pre_router_summary_path.read_text())
    if (
        manifest["protocol_id"] != PROTOCOL_ID
        or pre_router["protocol_id"] != PROTOCOL_ID
    ):
        raise RuntimeError("Router HPO summary protocol mismatch")
    sequence = pre_router["families"]["CSEQ_BERT4REC"]["selection"]
    content = pre_router["families"]["CCOLD_TWO_TOWER"]["selection"]
    warm_comparator = float(sequence["validation"]["warm"]["ndcg_at_20"])
    cold_comparator = float(content["validation"]["controlled_zero"]["ndcg_at_20"])
    families: dict[str, Any] = {}
    for family in ROUTER_FAMILIES:
        trials = []
        for identity in manifest["trials"]:
            if identity["family"] != family:
                continue
            receipt_path = (
                repo / f"evidence/p2/p2-e00-{identity['trial_id'].lower()}.json"
            )
            if not receipt_path.is_file():
                raise RuntimeError(f"Missing router HPO receipt: {receipt_path}")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "PASS":
                trials.append(
                    {
                        "trial_id": identity["trial_id"],
                        "status": receipt.get("status", "FAIL"),
                    }
                )
                continue
            if receipt.get("config_sha256") != identity["sha256"]:
                raise RuntimeError(f"Router config hash mismatch: {receipt_path}")
            checkpoint = Path(receipt["checkpoint"])
            scored = Path(receipt["validation"]["output"])
            if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"Router checkpoint hash mismatch: {receipt_path}")
            if sha256_file(scored) != receipt["validation"]["output_sha256"]:
                raise RuntimeError(f"Router output hash mismatch: {receipt_path}")
            slices = receipt["validation"]["slices"]
            objective = router_objective(
                cold=float(slices["controlled_zero"]["ndcg_at_20"] or 0.0),
                warm=float(slices["warm"]["ndcg_at_20"] or 0.0),
                cold_comparator=cold_comparator,
                warm_comparator=warm_comparator,
            )
            trials.append(
                {
                    "trial_id": identity["trial_id"],
                    "status": "PASS",
                    "checkpoint": receipt["checkpoint"],
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "validation": slices,
                    **objective,
                }
            )
        passed = [trial for trial in trials if trial["status"] == "PASS"]
        if passed:
            best_score = max(trial["router_selection_score"] for trial in passed)
            tied = [
                trial
                for trial in passed
                if trial["router_selection_score"] == best_score
            ]
            best_overall = max(
                float(trial["validation"]["all"]["ndcg_at_20"] or 0.0) for trial in tied
            )
            tied = [
                trial
                for trial in tied
                if float(trial["validation"]["all"]["ndcg_at_20"] or 0.0)
                == best_overall
            ]
            selected = min(tied, key=lambda trial: trial["trial_id"])
            selection = {
                "status": "SELECTED_DEVELOPMENT",
                "selected_trial_id": selected["trial_id"],
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "validation": selected["validation"],
                "router_selection_score": selected["router_selection_score"],
            }
        else:
            selection = {
                "status": "UNQUALIFIED_NO_PASS",
                "selected_trial_id": None,
            }
        families[family] = {"trials": trials, "selection": selection}
    status = (
        "PASS_DEVELOPMENT"
        if all(
            family["selection"]["status"] == "SELECTED_DEVELOPMENT"
            for family in families.values()
        )
        else "HPO_GRID_COMPLETE_UNQUALIFIED"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E00-ROUTER-SUMMARY",
        "status": status,
        "evidence_class": "p2_measured_development_router_hpo_summary",
        "manifest_sha256": sha256_file(manifest_path),
        "pre_router_summary_sha256": sha256_file(pre_router_summary_path),
        "families": families,
        "protected_roles_opened": [],
    }
    write_json_new(output, result)
    return result
