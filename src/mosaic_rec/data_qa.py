"""Read-only public-dataset qualification; execute in Ubuntu WSL."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualify(name: str, path: Path) -> dict:
    connection = duckdb.connect()
    if name == "microlens-100k":
        relation = f"read_csv('{path.as_posix()}', header=true, auto_detect=true)"
        fields = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        columns = {row[0].lower(): row[0] for row in fields}
        expected = {"user", "item", "timestamp"}
        if set(columns) != expected:
            raise ValueError(f"MicroLens schema mismatch: {fields}")
        user, item, stamp = (columns[x] for x in ("user", "item", "timestamp"))
    elif name == "ml-32m":
        relation = f"read_csv('{path.as_posix()}', header=true, auto_detect=true)"
        fields = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        columns = {row[0].lower(): row[0] for row in fields}
        expected = {"userid", "movieid", "rating", "timestamp"}
        if set(columns) != expected:
            raise ValueError(f"MovieLens schema mismatch: {fields}")
        user, item, stamp = (columns[x] for x in ("userid", "movieid", "timestamp"))
    elif name == "pixelrec50k":
        relation = f"read_csv('{path.as_posix()}', header=true, auto_detect=true)"
        fields = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        columns = {row[0].lower(): row[0] for row in fields}
        expected = {"user_id", "item_id", "timestamp"}
        if set(columns) != expected:
            raise ValueError(f"PixelRec schema mismatch: {fields}")
        user, item, stamp = (columns[x] for x in ("user_id", "item_id", "timestamp"))
    else:
        raise ValueError(name)
    stats = connection.execute(
        f"""SELECT count(*), count(DISTINCT {user}), count(DISTINCT {item}),
        count(*) FILTER (WHERE {user} IS NULL OR {item} IS NULL OR {stamp} IS NULL),
        min({stamp}), max({stamp}),
        approx_quantile({stamp}, [0.7, 0.8, 0.9])
        FROM {relation}"""
    ).fetchone()
    duplicates = connection.execute(
        f"SELECT count(*) - count(DISTINCT ({user}, {item}, {stamp})) FROM {relation}"
    ).fetchone()[0]
    if stats[3] != 0:
        raise ValueError(f"Null keys/timestamps: {stats[3]}")
    return {
        "stage_id": "A03",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "PILOT",
        "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": name,
        "source_file": path.name,
        "sha256": sha256(path),
        "schema": fields,
        "rows": stats[0],
        "users": stats[1],
        "items": stats[2],
        "null_keys_or_timestamps": stats[3],
        "min_timestamp": stats[4],
        "max_timestamp": stats[5],
        "candidate_temporal_quantiles_unfrozen": stats[6],
        "duplicate_user_item_timestamp_rows": duplicates,
        "evidence_class": "measured_data_qa",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["microlens-100k", "ml-32m", "pixelrec50k"])
    parser.add_argument("csv", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = qualify(args.dataset, args.csv)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
