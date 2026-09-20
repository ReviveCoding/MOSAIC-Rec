"""Export development-only reference smoke data in RecBole atomic format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def write_atomic(path: Path, rows: list[tuple[str, str, int]]) -> str:
    with path.open("w") as handle:
        handle.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for user, item, stamp in rows:
            handle.write(f"{user}\t{item}\t{stamp}\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_from(path: Path) -> list[tuple[str, str, int]]:
    table = pq.read_table(path)
    return sorted(zip(table["user_id"].to_pylist(),
                      table["item_id"].to_pylist(),
                      table["timestamp"].to_pylist()), key=lambda row: row[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit = rows_from(args.split_dir / "FIT_VISIBLE.parquet")
    validation = rows_from(args.split_dir / "VALIDATION.parquet")
    midpoint = len(validation) // 2
    partitions = {"train": fit, "valid": validation[:midpoint],
                  "test": validation[midpoint:]}
    outputs = {}
    for role, rows in partitions.items():
        path = args.output_dir / f"microlens_p1.{role}.inter"
        outputs[role] = {"rows": len(rows), "sha256": write_atomic(path, rows)}
    result = {
        "stage_id": "C05", "status": "REFERENCE_SMOKE_INPUT",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "data_root": str(args.output_dir.parent), "dataset": "microlens_p1",
        "outputs": outputs,
        "warning": "Validation is halved for reference runtime smoke only; no parity or scientific comparator metric is authorized.",
    }
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
