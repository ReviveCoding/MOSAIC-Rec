"""Validate and receipt the p1 FROZEN guard without reading locked outcomes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
E05 = ROOT / "evidence/e05_freeze_manifest_p1.json"
OUTPUT = ROOT / "evidence/e05_frozen_guard_p1.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Frozen guard receipt exists; preserve it")
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if worktree:
        raise RuntimeError("Frozen guard requires a clean committed worktree")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status_path = ROOT / "evidence/stage_status.json"
    status = json.loads(status_path.read_text())
    if status["scientific_state"] != "QUALIFICATION":
        raise RuntimeError("Expected QUALIFICATION before FROZEN transition")
    if status["stages"]["E05"] != "FREEZE_MANIFEST_COMPLETE_PENDING_COMMIT_GUARD":
        raise RuntimeError("E05 is not ready for the commit guard")
    e05 = json.loads(E05.read_text())
    if e05["status"] != "FREEZE_MANIFEST_COMPLETE":
        raise ValueError("E05 manifest is not complete")
    for row in e05["source_freeze_receipts"].values():
        path = ROOT / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Freeze receipt hash mismatch: {path}")
    qualification_manifest = ROOT / e05["qualification_manifest"]
    if digest(qualification_manifest) != e05["qualification_manifest_sha256"]:
        raise ValueError("Qualification manifest hash mismatch")
    for candidate in e05["frozen_candidates"].values():
        for checkpoint in candidate["checkpoints"]:
            path = Path(checkpoint["checkpoint"])
            if digest(path) != checkpoint["checkpoint_sha256"]:
                raise ValueError(f"Frozen checkpoint hash mismatch: {path}")
            receipt = ROOT / checkpoint["receipt"]
            if digest(receipt) != checkpoint["receipt_sha256"]:
                raise ValueError(f"Frozen checkpoint receipt mismatch: {receipt}")
    locked_path = Path(
        "/home/bjw-0/mosaic-rec-work/data/microlens-100k/split_p1/LOCKED_FINAL.parquet"
    )
    if digest(locked_path) != e05["data"]["locked_final_sha256"]:
        raise ValueError("LOCKED_FINAL identity differs from the frozen split receipt")
    warehouse = ROOT / "evidence/warehouse.duckdb"
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        receipt_count = int(
            connection.execute("SELECT count(*) FROM runs").fetchone()[0]
        )
        locked_count = int(
            connection.execute(
                "SELECT count(*) FROM runs WHERE state='LOCKED_EXECUTION'"
            ).fetchone()[0]
        )
    if receipt_count != status["canonical_receipts_current"]:
        raise ValueError("Canonical receipt count mismatch")
    if locked_count != 0:
        raise ValueError("Locked execution evidence already exists")
    result = {
        "stage_id": "E05_FROZEN_GUARD",
        "protocol_id": e05["protocol_id"],
        "state": "FROZEN",
        "status": "FROZEN_GUARD_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "e05_manifest": str(E05.relative_to(ROOT)).replace("\\", "/"),
        "e05_manifest_sha256": digest(E05),
        "qualification_manifest_sha256": e05["qualification_manifest_sha256"],
        "canonical_receipts_before_guard": receipt_count,
        "locked_execution_receipts_before_guard": locked_count,
        "C_seq": e05["C_seq"],
        "C_cold": e05["C_cold"],
        "MOSAIC_finalist": e05["MOSAIC_finalist"],
        "warm_noninferiority_margin": e05["warm_noninferiority_margin"],
        "locked_final_sha256": e05["data"]["locked_final_sha256"],
        "worktree_clean": True,
        "warnings": [
            "This guard hashes LOCKED_FINAL bytes for identity only; it does not deserialize or score protected rows.",
            "Scientific identities are immutable under p1 after locked outcome access.",
        ],
        "evidence_class": "derived_protocol_state_transition",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
