"""Materialize time-safe four-role splits and deterministic FIT-hidden items."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa


FIELDS = {
    "microlens-100k": ("user", "item", "timestamp"),
    "pixelrec50k": ("user_id", "item_id", "timestamp"),
    "ml-32m": ("userId", "movieId", "timestamp"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(dataset: str, source: Path, config_path: Path,
                scratch_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    entry = config["datasets"][dataset]
    user, item, stamp = FIELDS[dataset]
    con = duckdb.connect()
    con.execute("SET threads=4")
    relation = f"read_csv('{source.as_posix()}', header=true, auto_detect=true)"
    fit_end, val_end, qual_end = (
        entry[x] for x in ("fit_end", "validation_end", "qualification_end")
    )
    if not fit_end < val_end < qual_end:
        raise ValueError("Cutoffs are not strictly increasing")
    cold_ids: list[str] = []
    if entry["controlled_cold"]:
        fit_items = [str(row[0]) for row in con.execute(
            f"SELECT DISTINCT {item} FROM {relation} WHERE {stamp} <= {fit_end}"
        ).fetchall()]
        threshold = int(config["cold_cohort_fraction"] * (1 << 256))
        for item_id in fit_items:
            digest = hashlib.sha256(f"{config['seed']}:{item_id}".encode()).digest()
            if int.from_bytes(digest, "big") < threshold:
                cold_ids.append(item_id)
    cold_ids.sort()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cold_path = scratch_dir / "controlled_cold_item_ids.txt"
    cold_path.write_text("\n".join(cold_ids) + ("\n" if cold_ids else ""))
    con.register("cold_arrow", pa.table({"item_key": pa.array(cold_ids, type=pa.string())}))
    con.execute("CREATE TEMP TABLE cold_items AS SELECT * FROM cold_arrow")
    projection = f"CAST({user} AS VARCHAR) AS user_id, CAST({item} AS VARCHAR) AS item_id, {stamp} AS timestamp"
    predicates = {
        "FIT_VISIBLE": f"{stamp} <= {fit_end} AND CAST({item} AS VARCHAR) NOT IN (SELECT item_key FROM cold_items)",
        "FIT_WITHHELD_COLD": f"{stamp} <= {fit_end} AND CAST({item} AS VARCHAR) IN (SELECT item_key FROM cold_items)",
        "VALIDATION": f"{stamp} > {fit_end} AND {stamp} <= {val_end}",
        "QUALIFICATION": f"{stamp} > {val_end} AND {stamp} <= {qual_end}",
        "LOCKED_FINAL": f"{stamp} > {qual_end}",
    }
    outputs = {}
    for role, predicate in predicates.items():
        path = scratch_dir / f"{role}.parquet"
        con.execute(
            f"COPY (SELECT {projection} FROM {relation} WHERE {predicate}) "
            f"TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        count = con.execute(f"SELECT count(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        outputs[role] = {"path": str(path), "rows": count, "sha256": file_hash(path)}
    if outputs["FIT_WITHHELD_COLD"]["rows"] and not cold_ids:
        raise ValueError("Withheld rows without a cohort")
    total = sum(row["rows"] for row in outputs.values())
    source_count = con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0]
    if total != source_count:
        raise ValueError(f"Split count {total} != source count {source_count}")
    repo = Path(__file__).resolve().parents[2]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return {
        "stage_id": "A05", "protocol_id": config["protocol_id"],
        "state": "DEVELOPMENT", "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "git_head": git_head,
        "config_sha256": file_hash(config_path), "source_sha256": file_hash(source),
        "cold_cohort_count": len(cold_ids), "cold_cohort_sha256": file_hash(cold_path),
        "cold_cohort_method": "SHA256(seed:item_id) < 5% of 2^256 on FIT-observed item IDs",
        "outputs": outputs, "source_rows": source_count,
        "warnings": [
            "Materializing LOCKED_FINAL role does not authorize opening locked outcome metrics.",
            "First-observed candidate availability is a proxy for item launch.",
        ],
        "evidence_class": "derived_protocol_manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(FIELDS))
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("scratch_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = materialize(args.dataset, args.source, args.config, args.scratch_dir)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
