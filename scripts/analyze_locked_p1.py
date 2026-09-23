"""Apply frozen paired inference to immutable p1 locked per-user outcomes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mosaic_rec.statistics import paired_cluster_interval

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "evidence/e05_freeze_manifest_p1_amendment1.json"
OUTPUT = ROOT / "evidence/f00_locked_statistics_p1.json"
RECEIPTS = {
    "C_seq": ROOT / "evidence/f00_locked_p1_b7_bert4rec_t3.json",
    "C_cold": ROOT / "evidence/f00_locked_p1_b4_two_tower_t3.json",
    "MOSAIC": ROOT / "evidence/f00_locked_p1_m1_t2.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(receipt_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "LOCKED_SCORE_PASS":
        raise ValueError(f"Locked score did not pass: {receipt_path}")
    path = Path(receipt["per_user_path"])
    if digest(path) != receipt["per_user_sha256"]:
        raise ValueError(f"Per-user hash mismatch: {path}")
    table = pq.read_table(
        path,
        columns=["user_id", "item_id", "timestamp", "slice", "ndcg_at_20"],
    )
    rows = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
    return receipt, rows


def aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
    for key in ("user_id", "item_id", "timestamp", "slice"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Locked paired-row misalignment: {key}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Locked statistics receipt exists; preserve it")
    frozen = json.loads(FREEZE.read_text())
    draws = int(frozen["statistics"]["bootstrap_draws"])
    if draws != 10_000:
        raise ValueError("Frozen bootstrap draw count changed")
    receipts: dict[str, dict] = {}
    rows: dict[str, dict[str, np.ndarray]] = {}
    for name, path in RECEIPTS.items():
        receipts[name], rows[name] = load_rows(path)
    aligned(rows["MOSAIC"], rows["C_seq"])
    aligned(rows["MOSAIC"], rows["C_cold"])

    cold = rows["MOSAIC"]["slice"] == "controlled_cold"
    warm = rows["MOSAIC"]["slice"] == "warm_fit"
    cold_user = paired_cluster_interval(
        rows["MOSAIC"]["ndcg_at_20"][cold],
        rows["C_cold"]["ndcg_at_20"][cold],
        rows["MOSAIC"]["user_id"][cold],
        cluster_kind="user",
        draws=draws,
        seed=20260920,
    )
    cold_item = paired_cluster_interval(
        rows["MOSAIC"]["ndcg_at_20"][cold],
        rows["C_cold"]["ndcg_at_20"][cold],
        rows["MOSAIC"]["item_id"][cold],
        cluster_kind="item",
        draws=draws,
        seed=20260920,
    )
    warm_user = paired_cluster_interval(
        rows["MOSAIC"]["ndcg_at_20"][warm],
        rows["C_seq"]["ndcg_at_20"][warm],
        rows["MOSAIC"]["user_id"][warm],
        cluster_kind="user",
        draws=draws,
        seed=20260920,
    )
    overall_seq = paired_cluster_interval(
        rows["MOSAIC"]["ndcg_at_20"],
        rows["C_seq"]["ndcg_at_20"],
        rows["MOSAIC"]["user_id"],
        cluster_kind="user",
        draws=draws,
        seed=20260920,
    )
    margin = float(frozen["warm_noninferiority_margin"])
    cold_pass = cold_user.lower > 0
    warm_pass = warm_user.lower > -margin
    if cold_pass and warm_pass:
        decision = "CORE_PROMOTE"
    elif cold_pass:
        decision = "COLD_START_TRADEOFF"
    else:
        decision = "RETAIN_BASELINE"
    result = {
        "stage_id": "F00_F02_STATISTICS",
        "protocol_id": frozen["protocol_id"],
        "state": "ANALYSIS",
        "status": "LOCKED_PRIMARY_INFERENCE_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "frozen_manifest": str(FREEZE.relative_to(ROOT)).replace("\\", "/"),
        "frozen_manifest_sha256": digest(FREEZE),
        "models": {
            "C_seq": frozen["C_seq"],
            "C_cold": frozen["C_cold"],
            "MOSAIC_finalist": frozen["MOSAIC_finalist"],
        },
        "source_receipts": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(path),
                "per_user_sha256": receipts[name]["per_user_sha256"],
            }
            for name, path in RECEIPTS.items()
        },
        "bootstrap_seed": 20260920,
        "bootstrap_draws": draws,
        "cold_superiority": {
            "paired_user_cluster": asdict(cold_user),
            "cold_item_cluster_sensitivity": asdict(cold_item),
            "gate": "lower_95_percent_bound_gt_0",
            "pass": cold_pass,
        },
        "warm_noninferiority": {
            "paired_user_cluster": asdict(warm_user),
            "margin": margin,
            "gate": "lower_95_percent_bound_gt_negative_margin",
            "pass": warm_pass,
        },
        "overall_vs_C_seq": asdict(overall_seq),
        "co_primary_gate_pass": cold_pass and warm_pass,
        "terminal_decision": decision,
        "warnings": [
            "Offline unobserved items are treated as nonrelevant for ranking metrics, not proven negatives.",
            "The cold item-cluster interval is sensitivity evidence; paired user clustering is primary.",
        ],
        "evidence_class": "derived_locked_confirmatory_inference",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
