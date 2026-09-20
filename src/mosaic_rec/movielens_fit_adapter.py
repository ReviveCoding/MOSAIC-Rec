"""Build a bounded-memory, FIT-only MovieLens behavioral input."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def prepare(
    fit_path: Path, catalog_path: Path, output_dir: Path, receipt_path: Path
) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    if receipt_path.exists() or output_dir.exists():
        raise FileExistsError(
            "MovieLens FIT adapter output already exists; preserve it"
        )
    output_dir.mkdir(parents=True)
    fit_rows = pq.ParquetFile(fit_path).metadata.num_rows
    con = duckdb.connect(str(output_dir / "adapter.duckdb"))
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute(f"SET temp_directory={sql_path(output_dir / 'tmp')}")
    con.execute(
        "CREATE TABLE users AS SELECT user_id, "
        "row_number() OVER (ORDER BY CAST(user_id AS BIGINT))::INTEGER AS dense_id "
        f"FROM (SELECT DISTINCT user_id FROM read_parquet({sql_path(fit_path)}))"
    )
    con.execute(
        "CREATE TABLE items AS SELECT movieId::BIGINT AS movie_id, "
        "row_number() OVER (ORDER BY movieId::BIGINT)::INTEGER AS dense_id "
        f"FROM read_csv_auto({sql_path(catalog_path)})"
    )
    users = con.execute("SELECT count(*) FROM users").fetchone()[0]
    items = con.execute("SELECT count(*) FROM items").fetchone()[0]
    distinct_items = con.execute(
        "SELECT count(DISTINCT movie_id) FROM items"
    ).fetchone()[0]
    if items != distinct_items:
        raise ValueError("MovieLens catalog contains duplicate movie IDs")
    fit_dense = output_dir / "FIT_DENSE.parquet"
    users_map = output_dir / "users.parquet"
    items_map = output_dir / "items.parquet"
    con.execute(
        "COPY (SELECT u.dense_id AS user_id, i.dense_id AS item_id, f.timestamp "
        f"FROM read_parquet({sql_path(fit_path)}) f "
        "JOIN users u ON f.user_id=u.user_id "
        "JOIN items i ON CAST(f.item_id AS BIGINT)=i.movie_id "
        "ORDER BY f.timestamp, CAST(f.user_id AS BIGINT), CAST(f.item_id AS BIGINT)) "
        f"TO {sql_path(fit_dense)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        "COPY (SELECT user_id, dense_id FROM users ORDER BY dense_id) "
        f"TO {sql_path(users_map)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        "COPY (SELECT movie_id, dense_id FROM items ORDER BY dense_id) "
        f"TO {sql_path(items_map)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    dense_rows = pq.ParquetFile(fit_dense).metadata.num_rows
    if dense_rows != fit_rows:
        raise ValueError("MovieLens FIT rows lost in dense item/user mapping")
    con.close()
    # The working DuckDB file is only a reproducible intermediate; output
    # Parquet and its hashes are the canonical adapter artifacts.
    return {
        "stage_id": "I00",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "FIT_INPUT_PREP_PASS_NOT_REPLICATION",
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "movielens-32m",
        "model": "FIT_DENSE_ADAPTER",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "rows": dense_rows,
        "fit_users": users,
        "catalog_items": items,
        "input_sha256": {
            "fit_visible": sha256(fit_path),
            "official_catalog": sha256(catalog_path),
        },
        "output_sha256": {
            "fit_dense": sha256(fit_dense),
            "users": sha256(users_map),
            "items": sha256(items_map),
        },
        "outputs": {
            "fit_dense": str(fit_dense),
            "users": str(users_map),
            "items": str(items_map),
        },
        "warnings": [
            "FIT-only preparation; no validation, qualification, or locked outcome rows accessed.",
            "The catalog contains all official movie IDs, including items without FIT support.",
            "No model training or behavioral replication result follows from this adapter.",
        ],
        "evidence_class": "derived_fit_input_preparation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_path", type=Path)
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("receipt_path", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.fit_path, args.catalog_path, args.output_dir, args.receipt_path
    )
    args.receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
