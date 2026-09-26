"""Fail-closed protocol guards for p2 protected outcome access."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    verify_p1_immutable,
)
from mosaic_rec.p2.prequalification import build_prequalification_freeze
from mosaic_rec.p2.warehouse import validate


def prequalification_guard(repo: Path) -> dict[str, Any]:
    status = load_json(repo / "evidence/p2/stage_status.json")
    if status.get("scientific_state") not in {"DEVELOPMENT", "QUALIFICATION"}:
        raise RuntimeError("Guard requires a committed pre-outcome state")
    if (
        status.get("qualification_access_count") != 0
        or status.get("locked_final_access_count") != 0
    ):
        raise RuntimeError("Protected access count is nonzero")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=repo, text=True
    )
    if porcelain:
        raise RuntimeError("Pre-QUALIFICATION guard requires a clean worktree")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    freeze_path = repo / "configs/p2/prequalification_freeze_v3.json"
    frozen = load_json(freeze_path)
    rebuilt = build_prequalification_freeze(repo)
    if canonical_json_sha256(frozen) != canonical_json_sha256(rebuilt):
        raise RuntimeError("Pre-QUALIFICATION freeze no longer rederives exactly")
    warehouse = validate(repo, repo / "evidence/p2/warehouse.duckdb")
    if warehouse.get("status") != "PASS":
        raise RuntimeError("p2 warehouse validation failed")
    if warehouse.get("p1_canonical_receipts") != 0:
        raise RuntimeError("p1 receipt entered p2 warehouse")
    if (
        warehouse.get("qualification_access_count") != 0
        or warehouse.get("locked_final_access_count") != 0
    ):
        raise RuntimeError("Warehouse protected access count is nonzero")
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E02-PREQUALIFICATION-GUARD",
        "status": "PASS",
        "git_commit": head,
        "worktree_clean": True,
        "prequalification_freeze_sha256": sha256_file(freeze_path),
        "prequalification_freeze_canonical_sha256": canonical_json_sha256(frozen),
        "warehouse_validation": warehouse,
        "p1_immutable_hashes": verify_p1_immutable(repo),
        "protected_access_counts": {"QUALIFICATION": 0, "LOCKED_FINAL": 0},
        "authorized_next_transition": "QUALIFICATION_SCORING",
        "protected_roles_opened": [],
    }
