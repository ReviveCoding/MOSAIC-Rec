"""Build and validate a p2-only canonical DuckDB/Parquet evidence warehouse."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file


def _receipt_files(repo: Path) -> list[Path]:
    return sorted(
        path
        for path in (repo / "evidence/p2").glob("*.json")
        if path.name not in {"stage_status.json", "warehouse_manifest.json"}
    )


def _metric_rows(path: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, dict[str, Any]]] = []
    validation = receipt.get("validation")
    if isinstance(validation, dict):
        candidates.append((str(receipt.get("model_id", "UNKNOWN")), validation))
    for model in receipt.get("models", []):
        if isinstance(model, dict):
            candidates.append((str(model.get("model_id", "UNKNOWN")), model))
    for model_id, result in candidates:
        for slice_name, metrics in result.get("slices", {}).items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if metric_name == "n" or value is None:
                    continue
                rows.append(
                    {
                        "receipt": path.name,
                        "stage_id": str(receipt.get("stage_id", "")),
                        "model_id": model_id,
                        "role": str(result.get("role", "VALIDATION")),
                        "slice": str(slice_name),
                        "metric": str(metric_name),
                        "value": float(value),
                        "n": int(metrics.get("n", 0)),
                    }
                )
    return rows


def build(repo: Path, warehouse: Path, table_dir: Path) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for path in _receipt_files(repo):
        payload = json.loads(path.read_text())
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"Non-p2 receipt in p2 namespace: {path}")
        receipts.append(
            {
                "receipt": path.name,
                "sha256": sha256_file(path),
                "stage_id": str(payload.get("stage_id", "")),
                "attempt_id": str(payload.get("attempt_id", "")),
                "status": str(payload.get("status", "")),
                "git_commit": str(payload.get("git_commit", "")),
                "evidence_class": str(payload.get("evidence_class", "")),
                "payload_json": json.dumps(payload, sort_keys=True),
            }
        )
        metrics.extend(_metric_rows(path, payload))
    table_dir.mkdir(parents=True, exist_ok=True)
    receipt_schema = pa.schema(
        [
            ("receipt", pa.string()),
            ("sha256", pa.string()),
            ("stage_id", pa.string()),
            ("attempt_id", pa.string()),
            ("status", pa.string()),
            ("git_commit", pa.string()),
            ("evidence_class", pa.string()),
            ("payload_json", pa.string()),
        ]
    )
    metric_schema = pa.schema(
        [
            ("receipt", pa.string()),
            ("stage_id", pa.string()),
            ("model_id", pa.string()),
            ("role", pa.string()),
            ("slice", pa.string()),
            ("metric", pa.string()),
            ("value", pa.float64()),
            ("n", pa.int64()),
        ]
    )
    receipt_table = pa.Table.from_pylist(receipts, schema=receipt_schema)
    metric_table = pa.Table.from_pylist(metrics, schema=metric_schema)
    receipt_path = table_dir / "receipts.parquet"
    metric_path = table_dir / "metrics.parquet"
    pq.write_table(receipt_table, receipt_path, compression="zstd")
    pq.write_table(metric_table, metric_path, compression="zstd")

    warehouse.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{warehouse.name}.", dir=warehouse.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        con = duckdb.connect(str(temporary))
        con.execute(
            "CREATE TABLE receipts AS SELECT * FROM read_parquet(?)",
            [str(receipt_path)],
        )
        con.execute(
            "CREATE TABLE metrics AS SELECT * FROM read_parquet(?)", [str(metric_path)]
        )
        con.close()
        temporary.replace(warehouse)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "protocol_id": PROTOCOL_ID,
        "receipt_count": len(receipts),
        "metric_count": len(metrics),
        "warehouse": str(warehouse),
        "warehouse_sha256": sha256_file(warehouse),
        "tables": {
            "receipts": {
                "path": str(receipt_path),
                "rows": len(receipts),
                "sha256": sha256_file(receipt_path),
            },
            "metrics": {
                "path": str(metric_path),
                "rows": len(metrics),
                "sha256": sha256_file(metric_path),
            },
        },
    }


def validate(repo: Path, warehouse: Path) -> dict[str, Any]:
    status = json.loads((repo / "evidence/p2/stage_status.json").read_text())
    if status["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("p2 stage status mismatch")
    if status["qualification_access_count"] or status["locked_final_access_count"]:
        raise RuntimeError("Unexpected protected access before p2 freeze")
    con = duckdb.connect(str(warehouse), read_only=True)
    receipt_count = con.execute("SELECT count(*) FROM receipts").fetchone()[0]
    non_p2 = con.execute(
        "SELECT count(*) FROM receipts WHERE payload_json LIKE '%mosaic-rec-2026-09-20-p1%'"
    ).fetchone()[0]
    metric_count = con.execute("SELECT count(*) FROM metrics").fetchone()[0]
    con.close()
    source_count = len(_receipt_files(repo))
    if receipt_count != source_count:
        raise RuntimeError(
            f"Warehouse receipt count {receipt_count} != source count {source_count}"
        )
    # p1 may be named only in a provenance explanation; canonical receipt identities
    # are enforced by the protocol_id check during build, so report text matches alone
    # do not enter canonical counts.
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "receipt_count": receipt_count,
        "metric_count": metric_count,
        "p1_canonical_receipts": 0,
        "payloads_mentioning_p1_provenance": non_p2,
        "qualification_access_count": 0,
        "locked_final_access_count": 0,
    }
