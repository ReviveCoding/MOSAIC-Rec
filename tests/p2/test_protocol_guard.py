from pathlib import Path

from mosaic_rec.p2.protocol_guard import prequalification_guard


def test_clean_repository_prequalification_guard() -> None:
    repo = Path(__file__).resolve().parents[2]
    result = prequalification_guard(repo)
    assert result["status"] == "PASS"
    assert result["worktree_clean"] is True
    assert result["warehouse_validation"]["p1_canonical_receipts"] == 0
    assert result["protected_access_counts"] == {"QUALIFICATION": 0, "LOCKED_FINAL": 0}
