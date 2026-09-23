from __future__ import annotations

import json
from pathlib import Path

from mosaic_rec.p2.warehouse import build, validate


def test_p2_warehouse_excludes_wrong_protocol(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence = repo / "evidence/p2"
    evidence.mkdir(parents=True)
    (evidence / "stage_status.json").write_text(
        json.dumps(
            {
                "protocol_id": "mosaic-rec-2026-09-23-p2",
                "qualification_access_count": 0,
                "locked_final_access_count": 0,
            }
        )
    )
    (evidence / "bad.json").write_text(
        json.dumps({"protocol_id": "mosaic-rec-2026-09-20-p1"})
    )
    try:
        build(repo, tmp_path / "warehouse.duckdb", tmp_path / "tables")
    except RuntimeError as exc:
        assert "Non-p2 receipt" in str(exc)
    else:
        raise AssertionError("p1 receipt entered p2 warehouse")


def test_empty_p2_warehouse_validates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence = repo / "evidence/p2"
    evidence.mkdir(parents=True)
    (evidence / "stage_status.json").write_text(
        json.dumps(
            {
                "protocol_id": "mosaic-rec-2026-09-23-p2",
                "qualification_access_count": 0,
                "locked_final_access_count": 0,
            }
        )
    )
    warehouse = tmp_path / "warehouse.duckdb"
    build(repo, warehouse, tmp_path / "tables")
    assert validate(repo, warehouse)["status"] == "PASS"
