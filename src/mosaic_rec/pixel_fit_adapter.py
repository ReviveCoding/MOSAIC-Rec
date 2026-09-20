"""Create a FIT-only dense PixelRec input aligned to qualified content rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    fit_path: Path,
    item_ids_path: Path,
    cold_ids_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    if receipt_path.exists() or output_dir.exists():
        raise FileExistsError("PixelRec FIT adapter output already exists; preserve it")
    item_ids = item_ids_path.read_text().splitlines()
    if len(item_ids) != 82865 or len(set(item_ids)) != len(item_ids):
        raise ValueError("Qualified PixelRec item index is incomplete or duplicated")
    item_index = {item: row + 1 for row, item in enumerate(item_ids)}
    cold_ids = set(cold_ids_path.read_text().splitlines())
    if len(cold_ids) != 3109 or not cold_ids.issubset(item_index):
        raise ValueError("Controlled-cold cohort identity differs from frozen split")
    fit = pq.read_table(fit_path, columns=["user_id", "item_id", "timestamp"])
    raw_users = fit["user_id"].to_pylist()
    raw_items = fit["item_id"].to_pylist()
    stamps = fit["timestamp"].to_pylist()
    if not raw_items or len(raw_users) != len(raw_items):
        raise ValueError("FIT_VISIBLE is empty or malformed")
    if any(item not in item_index for item in raw_items):
        raise ValueError("FIT_VISIBLE contains an item absent from qualified features")
    if cold_ids.intersection(raw_items):
        raise ValueError("Controlled-cold item has model-visible FIT support")
    user_index = {user: row + 1 for row, user in enumerate(sorted(set(raw_users)))}
    dense = pa.table(
        {
            "user_id": pa.array([user_index[user] for user in raw_users], pa.int32()),
            "item_id": pa.array([item_index[item] for item in raw_items], pa.int32()),
            "timestamp": pa.array(stamps, pa.int64()),
        }
    )
    output_dir.mkdir(parents=True)
    dense_path = output_dir / "FIT_DENSE.parquet"
    users_path = output_dir / "user_ids.txt"
    items_path = output_dir / "item_ids.txt"
    pq.write_table(dense, dense_path, compression="zstd")
    users_path.write_text("\n".join(sorted(user_index)) + "\n")
    items_path.write_text("\n".join(item_ids) + "\n")
    return {
        "stage_id": "I01",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "FIT_INPUT_PREP_PASS_NOT_REPLICATION",
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "pixelrec50k",
        "model": "FIT_DENSE_ADAPTER",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "rows": len(raw_items),
        "fit_users": len(user_index),
        "catalog_items": len(item_index),
        "fit_items": len(set(raw_items)),
        "cold_items_with_fit_support": 0,
        "input_sha256": {
            "fit_visible": sha256(fit_path),
            "qualified_item_ids": sha256(item_ids_path),
            "controlled_cold_ids": sha256(cold_ids_path),
        },
        "output_sha256": {
            "fit_dense": sha256(dense_path),
            "user_ids": sha256(users_path),
            "item_ids": sha256(items_path),
        },
        "outputs": {
            "fit_dense": str(dense_path),
            "user_ids": str(users_path),
            "item_ids": str(items_path),
        },
        "warnings": [
            "FIT-only preparation; no validation, qualification, or locked outcome rows accessed.",
            "PixelRec publisher image features remain local only; historical content time is unknown.",
            "No model training or external replication result follows from this adapter.",
        ],
        "evidence_class": "derived_fit_input_preparation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_path", type=Path)
    parser.add_argument("item_ids_path", type=Path)
    parser.add_argument("cold_ids_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("receipt_path", type=Path)
    args = parser.parse_args()
    receipt = prepare(
        args.fit_path,
        args.item_ids_path,
        args.cold_ids_path,
        args.output_dir,
        args.receipt_path,
    )
    args.receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
