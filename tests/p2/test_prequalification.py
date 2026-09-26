from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic_rec.p2.prequalification import build_prequalification_freeze


def test_repository_prequalification_freeze_is_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[2]
    result = build_prequalification_freeze(repo)
    assert result["status"] == "FROZEN_FOR_QUALIFICATION_ACCESS"
    assert result["alignment"]["alignment_id"] == "ALIGN_RANK"
    assert result["protected_access_counts"] == {"QUALIFICATION": 0, "LOCKED_FINAL": 0}
    assert result["seed_policy"]["best_seed_selection"] is False
    assert [row["family"] for row in result["candidates"]["routers"]] == [
        "R2",
        "R3",
        "R4",
    ]
    assert result["warm_noninferiority_margin"]["numeric_delta"] == pytest.approx(
        0.0013209148732678748
    )


def test_prequalification_rejects_open_protected_counter(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    status_path = repo / "evidence/p2/stage_status.json"
    original = json.loads(status_path.read_text())
    altered = dict(original)
    altered["qualification_access_count"] = 1
    # Exercise the early guard without copying large immutable artifacts.
    target = tmp_path / "evidence/p2"
    target.mkdir(parents=True)
    (target / "stage_status.json").write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="Protected outcomes"):
        build_prequalification_freeze(tmp_path)
