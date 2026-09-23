from __future__ import annotations

import json
from pathlib import Path

from mosaic_rec.score_locked import locked_state_guard

ROOT = Path(__file__).resolve().parents[1]


def test_prelocked_amendment_changes_only_locked_query_contract() -> None:
    original = json.loads((ROOT / "evidence/e05_freeze_manifest_p1.json").read_text())
    amended = json.loads(
        (ROOT / "evidence/e05_freeze_manifest_p1_amendment1.json").read_text()
    )
    assert amended["scientific_identities_changed"] is False
    assert amended["C_seq"] == original["C_seq"]
    assert amended["C_cold"] == original["C_cold"]
    assert amended["MOSAIC_finalist"] == original["MOSAIC_finalist"]
    assert (
        amended["warm_noninferiority_margin"] == original["warm_noninferiority_margin"]
    )
    assert amended["evaluation"] == {
        **original["evaluation"],
        "query": "first_LOCKED_FINAL_event_per_user_with_FIT_VISIBLE_history",
    }


def test_locked_scorer_accepts_only_guarded_amendment() -> None:
    path = ROOT / "evidence/e05_freeze_manifest_p1_amendment1.json"
    frozen = json.loads(path.read_text())
    locked_state_guard(ROOT, frozen, path)
