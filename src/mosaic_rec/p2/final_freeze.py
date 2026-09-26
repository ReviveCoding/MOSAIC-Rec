"""Build deterministic E04 and E05 p2 freeze manifests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.ablations import ablation_overrides
from mosaic_rec.p2.contracts import load_json, sha256_file, verify_p1_immutable
from mosaic_rec.p2.warehouse import validate


def _file(repo: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256_file(repo / relative)}


def build_e04_manifest(repo: Path) -> dict[str, Any]:
    status = load_json(repo / "evidence/p2/stage_status.json")
    summary = load_json(repo / "evidence/p2/p2-e02-e03-qualification-summary.json")
    if (
        status["qualification_access_count"] != 1
        or status["locked_final_access_count"] != 0
    ):
        raise RuntimeError(
            "E04 requires exactly one QUALIFICATION access and sealed LOCKED_FINAL"
        )
    if summary["router_finalist"] != "R4" or summary["selection_is_promotion"]:
        raise RuntimeError("E04 requires the unpromoted frozen R4 finalist")

    robustness = load_json(repo / "configs/p2/robustness.json")
    robustness_cells = [
        {"intervention": row["id"], "intensity": intensity}
        for row in robustness["interventions"]
        for intensity in row["intensities"]
    ]
    ablations = load_json(repo / "configs/p2/ablations.json")
    ablation_cells = []
    for row in ablations["ablations"]:
        ablation_cells.append(
            {
                **row,
                "seeds": ablations["seeds"] if row["retrain"] else [],
                "overrides": ablation_overrides(row["id"]),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E04",
        "status": "FROZEN_ANALYSIS_FAMILIES",
        "scientific_state": "QUALIFICATION",
        "qualification_access_count": 1,
        "locked_final_access_count": 0,
        "finalist": "R4",
        "selection_is_promotion": False,
        "warm_noninferiority_margin": summary["warm_noninferiority_margin"],
        "primary_gates": load_json(repo / "configs/p2/statistics.json")[
            "primary_gates"
        ],
        "robustness": {
            "config": _file(repo, "configs/p2/robustness.json"),
            "implementation": _file(repo, "src/mosaic_rec/p2/robustness.py"),
            "cells": robustness_cells,
            "execution_stage": "P2-G00_AFTER_LOCKED_F00",
            "retraining": False,
        },
        "ablations": {
            "config": _file(repo, "configs/p2/ablations.json"),
            "implementation": _file(repo, "src/mosaic_rec/p2/ablations.py"),
            "cells": ablation_cells,
            "execution_stage": "P2-H00_AFTER_LOCKED_F00",
        },
        "statistics": _file(repo, "configs/p2/statistics.json"),
        "metrics": _file(repo, "configs/p2/metrics.json"),
        "protected_roles_opened": ["QUALIFICATION"],
        "p1_immutable_hashes": verify_p1_immutable(repo),
    }


def build_e05_manifest(repo: Path) -> dict[str, Any]:
    status = load_json(repo / "evidence/p2/stage_status.json")
    if (
        status["qualification_access_count"] != 1
        or status["locked_final_access_count"] != 0
    ):
        raise RuntimeError(
            "E05 requires access counts QUALIFICATION=1 and LOCKED_FINAL=0"
        )
    e04_path = repo / "configs/p2/e04_analysis_manifest.json"
    e04 = load_json(e04_path)
    if e04 != build_e04_manifest(repo):
        raise RuntimeError("E04 manifest no longer rederives exactly")
    prequal = load_json(repo / "configs/p2/prequalification_freeze_v3.json")
    qualification = load_json(
        repo / "evidence/p2/p2-e02-e03-qualification-summary.json"
    )
    selected_router = next(
        row for row in prequal["candidates"]["routers"] if row["family"] == "R4"
    )
    candidates = {
        "C_seq": prequal["candidates"]["C_seq"],
        "C_cold": prequal["candidates"]["C_cold"],
        "routers": [selected_router],
    }
    receipt_names = [
        "p2-e02-qualification-cseq_bert4rec.json",
        "p2-e02-qualification-ccold_two_tower.json",
        "p2-e02-qualification-r4.json",
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E05",
        "status": "COMPLETE_FREEZE_PENDING_GUARD",
        "scientific_state": "FROZEN",
        "qualification_access_count": 1,
        "locked_final_access_count": 0,
        "model_identities": {
            "C_seq": "CSEQ_BERT4REC",
            "C_cold": "CCOLD_TWO_TOWER",
            "router_finalist": "R4",
            "selection_is_promotion": False,
        },
        "candidates": candidates,
        "alignment": prequal["alignment"],
        "ensemble_policy": prequal["ensemble_policy"],
        "qualification_receipts": [
            _file(repo, f"evidence/p2/{name}") for name in receipt_names
        ],
        "qualification_summary": _file(
            repo, "evidence/p2/p2-e02-e03-qualification-summary.json"
        ),
        "warm_noninferiority_margin": qualification["warm_noninferiority_margin"],
        "primary_gates": load_json(repo / "configs/p2/statistics.json")[
            "primary_gates"
        ],
        "scoring_implementation": {
            "entrypoint": _file(repo, "scripts/p2/score_locked.py"),
            "locked_scorer": _file(repo, "src/mosaic_rec/p2/locked.py"),
            "shared_exact_ranker": _file(repo, "src/mosaic_rec/p2/runtime.py"),
            "alignment": _file(repo, "src/mosaic_rec/p2/router.py"),
        },
        "metrics_and_output_schema": _file(repo, "configs/p2/metrics.json"),
        "slices": _file(repo, "src/mosaic_rec/p2/slices.py"),
        "statistics": _file(repo, "configs/p2/statistics.json"),
        "robustness_and_ablations": _file(
            repo, "configs/p2/e04_analysis_manifest.json"
        ),
        "locked_evaluation": {
            "role": "LOCKED_FINAL",
            "query_manifest": "/home/bjw-0/mosaic-rec-work/p2/contexts/pixelrec50k/LOCKED_FINAL_QUERIES.parquet",
            "families": ["CSEQ_BERT4REC", "CCOLD_TWO_TOWER", "R4"],
            "outputs": {
                family: f"/home/bjw-0/mosaic-rec-work/p2/locked_final/{family.lower()}_topk.parquet"
                for family in ["CSEQ_BERT4REC", "CCOLD_TWO_TOWER", "R4"]
            },
            "access_policy": "open once after committed E05 guard PASS",
        },
        "p1_immutable_hashes": verify_p1_immutable(repo),
        "protected_roles_opened": ["QUALIFICATION"],
    }


def e05_guard(repo: Path) -> dict[str, Any]:
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=repo, text=True
    )
    if porcelain:
        raise RuntimeError("E05 guard requires a clean worktree")
    status = load_json(repo / "evidence/p2/stage_status.json")
    if status["scientific_state"] != "FROZEN":
        raise RuntimeError("E05 guard requires FROZEN state")
    if (
        status["qualification_access_count"] != 1
        or status["locked_final_access_count"] != 0
    ):
        raise RuntimeError("E05 protected access counts are invalid")
    manifest_path = repo / "evidence/p2/e05_freeze_manifest.json"
    manifest = load_json(manifest_path)
    if manifest != build_e05_manifest(repo):
        raise RuntimeError("E05 manifest no longer rederives exactly")
    if status.get("frozen_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Stage status does not bind E05 manifest hash")
    warehouse = validate(repo, repo / "evidence/p2/warehouse.duckdb")
    if warehouse["status"] != "PASS" or warehouse["p1_canonical_receipts"] != 0:
        raise RuntimeError("p2-only warehouse validation failed")
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E05-GUARD",
        "status": "PASS",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "worktree_clean": True,
        "e05_manifest_sha256": sha256_file(manifest_path),
        "warehouse_validation": warehouse,
        "p1_immutable_hashes": verify_p1_immutable(repo),
        "protected_access_counts": {"QUALIFICATION": 1, "LOCKED_FINAL": 0},
        "authorized_next_transition": "LOCKED_FINAL_ACCESS_OPEN_ONCE",
        "protected_roles_opened": ["QUALIFICATION"],
    }
