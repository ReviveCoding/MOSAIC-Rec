"""Build compact canonical DuckDB and Parquet evidence from stage receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

SCHEMAS = {
    "runs": "receipt_id VARCHAR, stage_id VARCHAR, protocol_id VARCHAR, state VARCHAR, status VARCHAR, dataset VARCHAR, model VARCHAR, ended_utc VARCHAR, evidence_class VARCHAR, receipt_sha256 VARCHAR",
    "datasets": "receipt_id VARCHAR, dataset VARCHAR, rows BIGINT, users BIGINT, items BIGINT, source_sha256 VARCHAR",
    "splits": "receipt_id VARCHAR, dataset VARCHAR, role VARCHAR, rows BIGINT, parquet_sha256 VARCHAR, parquet_path VARCHAR",
    "trials": "receipt_id VARCHAR, model VARCHAR, epoch BIGINT, objective_loss DOUBLE, dataset VARCHAR, hyperparameter_json VARCHAR, validation_ndcg_at_20 DOUBLE, source_receipt_id VARCHAR",
    "checkpoints": "receipt_id VARCHAR, model VARCHAR, checkpoint_path VARCHAR, checkpoint_sha256 VARCHAR",
    "metrics": "receipt_id VARCHAR, dataset VARCHAR, model VARCHAR, role VARCHAR, slice VARCHAR, metric VARCHAR, value DOUBLE, n BIGINT",
    "slices": "receipt_id VARCHAR, dataset VARCHAR, model VARCHAR, slice VARCHAR, n BIGINT",
    "robustness": "receipt_id VARCHAR, intervention VARCHAR, metric VARCHAR, value DOUBLE",
    "ablations": "receipt_id VARCHAR, component VARCHAR, metric VARCHAR, value DOUBLE",
    "gate_weights": "receipt_id VARCHAR, slice VARCHAR, branch VARCHAR, weight DOUBLE",
    "latency": "receipt_id VARCHAR, workload VARCHAR, metric VARCHAR, value DOUBLE, unit VARCHAR",
    "gpu": "receipt_id VARCHAR, device VARCHAR, train_seconds DOUBLE, score_seconds DOUBLE, peak_vram_bytes BIGINT",
    "bootstrap": "receipt_id VARCHAR, comparison VARCHAR, metric VARCHAR, lower DOUBLE, upper DOUBLE, draws BIGINT",
    "claims": "claim_id VARCHAR, statement VARCHAR, evidence_class VARCHAR, receipt_id VARCHAR, status VARCHAR, limitation VARCHAR",
}


def receipt_id(path: Path) -> str:
    return path.stem


def build(evidence_dir: Path, output: Path, tables_dir: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    con = duckdb.connect(str(output))
    for table, schema in SCHEMAS.items():
        con.execute(f"CREATE TABLE {table} ({schema})")
    counts = {table: 0 for table in SCHEMAS}
    receipts = sorted(evidence_dir.glob("*.json"))
    for path in receipts:
        if path.name in {"stage_status.json", "gpu_qualification.json"}:
            continue
        data = json.loads(path.read_text())
        if (
            not isinstance(data, dict)
            or "stage_id" not in data
            or "state" not in data
            or "evidence_class" not in data
        ):
            continue
        rid = receipt_id(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rid,
                data.get("stage_id"),
                data.get("protocol_id"),
                data.get("state"),
                data.get("status"),
                data.get("dataset"),
                data.get("model"),
                data.get("ended_utc"),
                data.get("evidence_class"),
                digest,
            ],
        )
        counts["runs"] += 1
        if data.get("stage_id") == "A03" and "rows" in data:
            con.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
                [
                    rid,
                    data.get("dataset"),
                    data["rows"],
                    data.get("users"),
                    data.get("items"),
                    data.get("sha256"),
                ],
            )
            counts["datasets"] += 1
        for role, split in data.get("outputs", {}).items():
            if (
                isinstance(split, dict)
                and "rows" in split
                and "sha256" in split
                and "path" in split
            ):
                con.execute(
                    "INSERT INTO splits VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        rid,
                        data.get("dataset"),
                        role,
                        split["rows"],
                        split["sha256"],
                        split["path"],
                    ],
                )
                counts["splits"] += 1
        for epoch in data.get("training_log", []):
            con.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    rid,
                    data.get("model"),
                    epoch.get("epoch"),
                    epoch.get("fit_bpr_loss", epoch.get("fit_loss")),
                    data.get("dataset"),
                    None,
                    None,
                    None,
                ],
            )
            counts["trials"] += 1
        if data.get("status") == "VALIDATION_SELECTED" and isinstance(
            data.get("trials"), dict
        ):
            for dataset, trial_list in data["trials"].items():
                for trial in trial_list:
                    con.execute(
                        "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            rid,
                            data.get("model"),
                            None,
                            None,
                            dataset,
                            json.dumps(
                                {"half_life_days": trial["half_life_days"]},
                                sort_keys=True,
                            ),
                            trial["all_ndcg_at_20"],
                            trial["receipt_id"],
                        ],
                    )
                    counts["trials"] += 1
        if data.get("checkpoint_sha256"):
            con.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?)",
                [
                    rid,
                    data.get("model"),
                    data.get("checkpoint"),
                    data["checkpoint_sha256"],
                ],
            )
            counts["checkpoints"] += 1
        for metric in ("ndcg_at_20", "recall_at_20", "mrr_at_20", "hit_rate_at_20"):
            if data.get(metric) is not None:
                con.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        rid,
                        data.get("dataset"),
                        data.get("model"),
                        data.get("role", "VALIDATION"),
                        "all",
                        metric,
                        data[metric],
                        data.get("validation_users", data.get("evaluated_users")),
                    ],
                )
                counts["metrics"] += 1
        for name, metrics in data.get("validation_slices", {}).items():
            con.execute(
                "INSERT INTO slices VALUES (?, ?, ?, ?, ?)",
                [
                    rid,
                    data.get("dataset"),
                    data.get("model"),
                    name,
                    metrics.get("n"),
                ],
            )
            counts["slices"] += 1
            for metric in ("ndcg_at_20", "recall_at_20", "mrr_at_20"):
                if metrics.get(metric) is not None:
                    con.execute(
                        "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            rid,
                            data.get("dataset"),
                            data.get("model"),
                            "VALIDATION",
                            name,
                            metric,
                            metrics[metric],
                            metrics["n"],
                        ],
                    )
                    counts["metrics"] += 1
        if data.get("device"):
            con.execute(
                "INSERT INTO gpu VALUES (?, ?, ?, ?, ?)",
                [
                    rid,
                    data.get("device"),
                    data.get("train_seconds"),
                    data.get("score_seconds"),
                    data.get("peak_vram_bytes"),
                ],
            )
            counts["gpu"] += 1
        for slice_name, key in (
            ("controlled_cold", "gate_weights_cold_mean"),
            ("warm_gt20", "gate_weights_warm_gt20_mean"),
        ):
            weights = data.get(key)
            if weights is not None and data.get("model") != "M0":
                for branch, weight in zip(
                    ("collaborative", "text", "image", "metadata"), weights
                ):
                    con.execute(
                        "INSERT INTO gate_weights VALUES (?, ?, ?, ?)",
                        [
                            rid,
                            slice_name,
                            branch,
                            weight,
                        ],
                    )
                    counts["gate_weights"] += 1
    claims_file = evidence_dir / "claims.json"
    if claims_file.exists():
        for claim in json.loads(claims_file.read_text()):
            if claim["status"] == "SUPPORTED":
                found = con.execute(
                    "SELECT count(*) FROM runs WHERE receipt_id=?",
                    [claim["receipt_id"]],
                ).fetchone()[0]
                if not found:
                    raise ValueError(
                        f"Supported claim lacks receipt: {claim['claim_id']}"
                    )
            con.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)",
                [
                    claim["claim_id"],
                    claim["statement"],
                    claim["evidence_class"],
                    claim.get("receipt_id"),
                    claim["status"],
                    claim["limitation"],
                ],
            )
            counts["claims"] += 1
    for table in SCHEMAS:
        path = tables_dir / f"{table}.parquet"
        con.execute(
            f"COPY {table} TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    con.close()
    return {
        "warehouse": str(output),
        "receipt_count": counts["runs"],
        "table_rows": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("warehouse", type=Path)
    parser.add_argument("tables_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(build(args.evidence_dir, args.warehouse, args.tables_dir), indent=2)
    )


if __name__ == "__main__":
    main()
