"""Regression: overall-only sequence HPO receipts remain canonical metrics."""

from __future__ import annotations

import json

import duckdb

from mosaic_rec.warehouse import build


def test_overall_only_sequence_hpo_receipt(tmp_path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    receipt = {
        "stage_id": "E00",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "HPO_TRIAL_PASS",
        "dataset": "microlens-100k",
        "model": "B5_GRU4REC",
        "evidence_class": "measured_development_hpo",
        "training_log": None,
        "validation_slices": None,
        "validation_users": 41694,
        "ndcg_at_20": 0.013264202369016587,
    }
    (evidence / "sequence_trial.json").write_text(json.dumps(receipt))
    warehouse = tmp_path / "warehouse.duckdb"
    result = build(evidence, warehouse, tmp_path / "tables")
    assert result["receipt_count"] == 1
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        metric = connection.execute(
            "SELECT metric, value, n FROM metrics WHERE receipt_id='sequence_trial'"
        ).fetchone()
    assert metric == ("ndcg_at_20", 0.013264202369016587, 41694)
