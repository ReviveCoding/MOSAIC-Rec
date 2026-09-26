"""Materialize the fail-closed p2 pre-QUALIFICATION candidate freeze."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import load_json, sha256_file

FROZEN_FAMILIES = ("CSEQ_BERT4REC", "CCOLD_TWO_TOWER", "R2", "R3", "R4")
ROUTER_FAMILIES = ("R2", "R3", "R4")


def _candidate(repo: Path, identity: dict[str, Any]) -> dict[str, Any]:
    run_id = identity["run_id"]
    config_path = repo / identity["path"]
    if sha256_file(config_path) != identity["sha256"]:
        raise RuntimeError(f"Frozen seed config hash mismatch: {run_id}")
    receipt_path = repo / f"evidence/p2/p2-e01-{run_id.lower()}.json"
    receipt = load_json(receipt_path)
    if receipt.get("status") != "PASS" or receipt.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"Nonterminal seed receipt: {run_id}")
    if (
        receipt.get("seed") != identity["seed"]
        or receipt.get("family") != identity["family"]
    ):
        raise RuntimeError(f"Seed receipt identity mismatch: {run_id}")
    checkpoint = Path(receipt["checkpoint"])
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
        raise RuntimeError(f"Seed checkpoint hash mismatch: {run_id}")
    validation = receipt["validation"]
    output = Path(validation["output"])
    if sha256_file(output) != validation["output_sha256"]:
        raise RuntimeError(f"Seed validation output hash mismatch: {run_id}")
    if validation.get("role") != "VALIDATION":
        raise RuntimeError(f"Protected role in seed receipt: {run_id}")
    return {
        "run_id": run_id,
        "seed": identity["seed"],
        "config_path": identity["path"],
        "config_sha256": identity["sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "validation_output": str(output),
        "validation_output_sha256": validation["output_sha256"],
        "receipt_path": str(receipt_path.relative_to(repo)).replace("\\", "/"),
        "receipt_sha256": sha256_file(receipt_path),
    }


def build_prequalification_freeze(repo: Path) -> dict[str, Any]:
    status = load_json(repo / "evidence/p2/stage_status.json")
    if status.get("scientific_state") != "DEVELOPMENT":
        raise RuntimeError("Pre-QUALIFICATION freeze requires DEVELOPMENT")
    if (
        status.get("qualification_access_count") != 0
        or status.get("locked_final_access_count") != 0
    ):
        raise RuntimeError("Protected outcomes were already accessed")
    if status["stages"].get("P2-E01", {}).get("status") != "PASS_DEVELOPMENT":
        raise RuntimeError("P2-E01 is not complete")

    manifest_path = repo / "configs/p2/seed_trials_manifest.json"
    summary_path = repo / "evidence/p2/p2-e01-seed-summary.json"
    alignment_path = repo / "evidence/p2/p2-b03-alignment-summary.json"
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    alignment = load_json(alignment_path)
    if any(x.get("protocol_id") != PROTOCOL_ID for x in (manifest, summary, alignment)):
        raise RuntimeError("Pre-QUALIFICATION source protocol mismatch")
    if summary.get("best_seed_selection") is not False:
        raise RuntimeError("Best-seed selection is prohibited")
    if alignment.get("selected_alignment_id") != "ALIGN_RANK":
        raise RuntimeError("Frozen alignment changed")

    runs_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in FROZEN_FAMILIES
    }
    for identity in manifest["runs"]:
        family = identity["family"]
        if family in runs_by_family:
            runs_by_family[family].append(_candidate(repo, identity))
    expected_seeds = [20261001, 20261002, 20261003]
    for family, runs in runs_by_family.items():
        runs.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in runs] != expected_seeds:
            raise RuntimeError(f"Independent-seed identity mismatch: {family}")

    cseq_warm = summary["families"]["CSEQ_BERT4REC"]["metrics"]["warm"]["mean"]
    margin = 0.10 * float(cseq_warm)
    frozen_files = [
        "configs/p2/metrics.json",
        "configs/p2/statistics.json",
        "configs/p2/robustness.json",
        "configs/p2/ablations.json",
        "src/mosaic_rec/p2/slices.py",
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E02-PREQUALIFICATION-FREEZE",
        "status": "FROZEN_FOR_QUALIFICATION_ACCESS",
        "scientific_state": "DEVELOPMENT",
        "protected_access_counts": {"QUALIFICATION": 0, "LOCKED_FINAL": 0},
        "alignment": {
            "alignment_id": "ALIGN_RANK",
            "method": "rank_percentile_normalization",
            "summary_path": "evidence/p2/p2-b03-alignment-summary.json",
            "summary_sha256": sha256_file(alignment_path),
        },
        "ensemble_policy": "arithmetic mean of aligned full-catalog scores across all three independent seeds",
        "seed_policy": {
            "independent_seeds": expected_seeds,
            "hpo_seed": 20260930,
            "hpo_seed_excluded": True,
            "best_seed_selection": False,
        },
        "candidates": {
            "C_seq": {
                "family": "CSEQ_BERT4REC",
                "runs": runs_by_family["CSEQ_BERT4REC"],
            },
            "C_cold": {
                "family": "CCOLD_TWO_TOWER",
                "runs": runs_by_family["CCOLD_TWO_TOWER"],
            },
            "routers": [
                {"family": family, "runs": runs_by_family[family]}
                for family in ROUTER_FAMILIES
            ],
        },
        "qualification_selection": {
            "C_seq": "sole frozen CSEQ_BERT4REC three-seed ensemble",
            "C_cold": "sole frozen CCOLD_TWO_TOWER three-seed ensemble",
            "router": "maximize the frozen router objective: min(controlled_zero/C_cold - 1, warm/C_seq - 1 + 0.10); if C_cold controlled-zero is zero use absolute router-minus-C_cold delta; tie overall NDCG@20 then family_id ascending",
        },
        "warm_noninferiority_margin": {
            "rule": "0.10 * C_seq VALIDATION warm NDCG@20",
            "cseq_validation_warm_ndcg_at_20": cseq_warm,
            "numeric_delta": margin,
        },
        "source_hashes": {
            "seed_manifest": sha256_file(manifest_path),
            "seed_summary": sha256_file(summary_path),
            **{path: sha256_file(repo / path) for path in frozen_files},
        },
        "protected_roles_opened": [],
    }
