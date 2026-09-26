from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic_rec.p2.prequalification import build_prequalification_freeze


def test_repository_prequalification_freeze_is_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="Protected outcomes"):
        build_prequalification_freeze(repo)


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
