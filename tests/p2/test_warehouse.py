from __future__ import annotations

import json
from pathlib import Path

import duckdb

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


def test_invalidated_receipt_is_retained_but_excluded_from_metrics(
    tmp_path: Path,
) -> None:
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
    (evidence / "model.json").write_text(
        json.dumps(
            {
                "protocol_id": "mosaic-rec-2026-09-23-p2",
                "stage_id": "P2-C03",
                "status": "PASS",
                "model_id": "MODEL",
                "validation": {
                    "role": "VALIDATION",
                    "slices": {"all": {"n": 2, "ndcg_at_20": 0.5}},
                },
            }
        )
    )
    (evidence / "invalidation.json").write_text(
        json.dumps(
            {
                "protocol_id": "mosaic-rec-2026-09-23-p2",
                "stage_id": "P2-C03-AUDIT",
                "status": "INVALID_DEVELOPMENT_SCORING",
                "affected_receipts": ["model.json"],
            }
        )
    )
    warehouse = tmp_path / "warehouse.duckdb"
    build(repo, warehouse, tmp_path / "tables")
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        row = connection.execute(
            "SELECT valid_for_selection, invalidated_by FROM receipts "
            "WHERE receipt = 'model.json'"
        ).fetchone()
        metric_count = connection.execute("SELECT count(*) FROM metrics").fetchone()[0]
    assert row == (False, "invalidation.json")
    assert metric_count == 0
