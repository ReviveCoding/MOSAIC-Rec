"""Check the compact canonical evidence warehouse against source receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


TABLES = (
    "runs", "datasets", "splits", "trials", "checkpoints", "metrics", "slices",
    "robustness", "ablations", "gate_weights", "latency", "gpu", "bootstrap", "claims",
)


def validate(evidence_dir: Path, warehouse: Path) -> dict[str, object]:
    errors: list[str] = []
    con = duckdb.connect(str(warehouse), read_only=True)
    actual = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    errors.extend(f"missing table: {name}" for name in TABLES if name not in actual)
    if errors:
        con.close()
        return {"status": "FAIL", "errors": errors}
    rows = con.execute("SELECT receipt_id, receipt_sha256, status, evidence_class FROM runs").fetchall()
    receipt_ids = set()
    for rid, digest, status, evidence_class in rows:
        if rid in receipt_ids:
            errors.append(f"duplicate receipt ID: {rid}")
        receipt_ids.add(rid)
        path = evidence_dir / f"{rid}.json"
        if not path.is_file():
            errors.append(f"missing receipt: {rid}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"receipt hash mismatch: {rid}")
        if not status or not evidence_class:
            errors.append(f"receipt lacks status/evidence class: {rid}")
    for path in evidence_dir.glob("*.json"):
        if path.name in {"stage_status.json", "gpu_qualification.json", "claims.json"}:
            continue
        data = json.loads(path.read_text())
        if (isinstance(data, dict) and "stage_id" in data
                and "state" in data and "evidence_class" in data
                and path.stem not in receipt_ids):
            errors.append(f"receipt omitted from warehouse: {path.stem}")
    claims = json.loads((evidence_dir / "claims.json").read_text())
    warehouse_claims = {
        row[0]: row[1] for row in con.execute("SELECT claim_id, status FROM claims").fetchall()
    }
    seen_claims: set[str] = set()
    for claim in claims:
        cid = claim["claim_id"]
        if cid in seen_claims:
            errors.append(f"duplicate claim ID: {cid}")
        seen_claims.add(cid)
        if warehouse_claims.get(cid) != claim["status"]:
            errors.append(f"claim mismatch: {cid}")
        if claim["status"] in {"SUPPORTED", "INTERNAL_PILOT"} and claim.get("receipt_id") not in receipt_ids:
            errors.append(f"measured claim lacks receipt: {cid}")
        if claim["status"] == "SUPPORTED" and claim.get("evidence_class") in {
            "unexecuted", "unsupported", "simulated", "projected",
        }:
            errors.append(f"supported claim has invalid evidence class: {cid}")
    bad_metrics = con.execute(
        "SELECT count(*) FROM metrics WHERE value IS NULL OR NOT isfinite(value) "
        "OR n IS NULL OR n <= 0"
    ).fetchone()[0]
    if bad_metrics:
        errors.append(f"invalid metric rows: {bad_metrics}")
    bad_gate = con.execute(
        "SELECT count(*) FROM gate_weights WHERE weight IS NULL OR NOT isfinite(weight) "
        "OR weight < 0 OR weight > 1"
    ).fetchone()[0]
    if bad_gate:
        errors.append(f"invalid gate weights: {bad_gate}")
    bad_trials = con.execute(
        "SELECT count(*) FROM trials t LEFT JOIN runs r ON t.source_receipt_id=r.receipt_id "
        "WHERE (t.source_receipt_id IS NOT NULL AND r.receipt_id IS NULL) "
        "OR (t.validation_ndcg_at_20 IS NOT NULL AND "
        "(NOT isfinite(t.validation_ndcg_at_20) OR t.validation_ndcg_at_20 < 0 "
        "OR t.validation_ndcg_at_20 > 1))"
    ).fetchone()[0]
    if bad_trials:
        errors.append(f"invalid or orphan search trials: {bad_trials}")
    unexecuted_finals = con.execute(
        "SELECT count(*) FROM runs WHERE state='LOCKED_EXECUTION'"
    ).fetchone()[0]
    counts = {name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0] for name in TABLES}
    stage_status = json.loads((evidence_dir / "stage_status.json").read_text())
    if stage_status.get("canonical_receipts_current") != counts["runs"]:
        errors.append("stage_status canonical receipt count differs from warehouse")
    con.close()
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "table_rows": counts,
        "locked_execution_receipts": unexecuted_finals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("warehouse", type=Path)
    args = parser.parse_args()
    result = validate(args.evidence_dir, args.warehouse)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
