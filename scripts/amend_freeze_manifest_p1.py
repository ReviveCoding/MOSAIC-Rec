"""Correct the pre-locked E05 query-role label while preserving all identities."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "evidence/e05_freeze_manifest_p1.json"
OUTPUT = ROOT / "evidence/e05_freeze_manifest_p1_amendment1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Freeze amendment exists; preserve it")
    status = json.loads((ROOT / "evidence/stage_status.json").read_text())
    if status["scientific_state"] != "FROZEN":
        raise RuntimeError("Expected the guarded pre-locked FROZEN state")
    with __import__("duckdb").connect(
        str(ROOT / "evidence/warehouse.duckdb"), read_only=True
    ) as connection:
        locked = int(
            connection.execute(
                "SELECT count(*) FROM runs WHERE state='LOCKED_EXECUTION'"
            ).fetchone()[0]
        )
    if locked != 0:
        raise RuntimeError("Cannot amend p1 after locked outcome access")
    parent = json.loads(PARENT.read_text())
    amended = dict(parent)
    amended.update(
        {
            "stage_id": "E05_AMENDMENT1",
            "state": "FROZEN",
            "status": "FREEZE_MANIFEST_AMENDMENT_PRELOCK",
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "supersedes_for_locked_execution": str(PARENT.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "superseded_manifest_sha256": digest(PARENT),
            "amendment_reason": (
                "The original E05 copied the qualification query-role label into its "
                "evaluation block. Before any locked receipt existed, correct only that "
                "label to first_LOCKED_FINAL_event_per_user_with_FIT_VISIBLE_history."
            ),
            "scientific_identities_changed": False,
            "locked_access": (
                "PROHIBITED_UNTIL_AMENDMENT1_IS_COMMITTED_AND_PROTOCOL_GUARD_PASSES"
            ),
        }
    )
    amended["evaluation"] = dict(parent["evaluation"])
    amended["evaluation"]["query"] = (
        "first_LOCKED_FINAL_event_per_user_with_FIT_VISIBLE_history"
    )
    OUTPUT.write_text(json.dumps(amended, indent=2) + "\n")
    print(json.dumps({"path": str(OUTPUT), "sha256": digest(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
