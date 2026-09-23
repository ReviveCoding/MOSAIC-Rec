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


def test_locked_bootstrap_and_role_are_canonical(tmp_path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    score = {
        "stage_id": "F00_F02",
        "protocol_id": "p1",
        "state": "LOCKED_EXECUTION",
        "status": "LOCKED_SCORE_PASS",
        "dataset": "micro",
        "model": "M1",
        "role": "LOCKED_FINAL",
        "evidence_class": "measured_locked_exact_full_catalog",
        "validation_slices": {"all": {"n": 2, "ndcg_at_20": 0.1, "recall_at_20": 0.2}},
    }
    statistics = {
        "stage_id": "F00_F02_STATISTICS",
        "protocol_id": "p1",
        "state": "ANALYSIS",
        "status": "LOCKED_PRIMARY_INFERENCE_PASS",
        "evidence_class": "derived_locked_confirmatory_inference",
        "cold_superiority": {
            "paired_user_cluster": {"lower": -0.1, "upper": 0.2, "draws": 10000},
            "cold_item_cluster_sensitivity": {
                "lower": -0.2,
                "upper": 0.3,
                "draws": 10000,
            },
        },
        "warm_noninferiority": {
            "paired_user_cluster": {"lower": -0.3, "upper": -0.2, "draws": 10000}
        },
        "overall_vs_C_seq": {"lower": 0.01, "upper": 0.02, "draws": 10000},
    }
    (evidence / "score.json").write_text(json.dumps(score))
    (evidence / "statistics.json").write_text(json.dumps(statistics))
    warehouse = tmp_path / "warehouse.duckdb"
    result = build(evidence, warehouse, tmp_path / "tables")
    assert result["table_rows"]["bootstrap"] == 4
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        role = connection.execute(
            "SELECT DISTINCT role FROM metrics WHERE receipt_id='score'"
        ).fetchone()[0]
    assert role == "LOCKED_FINAL"
