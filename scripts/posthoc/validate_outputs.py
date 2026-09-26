"""Validate posthoc artifacts and the immutable canonical p2 boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from common import SOURCE_COMMIT, load_json, sha256_file

from mosaic_rec.p2.warehouse import validate as validate_warehouse


def _resolve(repo: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo / path


def validate(repo: Path) -> dict[str, Any]:
    source = load_json(repo / "analysis/posthoc/source_manifest.json")
    manifest = load_json(repo / "analysis/posthoc/posthoc_manifest.json")
    mismatches = []
    for row in source["inputs"]:
        path = _resolve(repo, row["path"])
        observed = sha256_file(path)
        if observed != row["sha256"]:
            mismatches.append(row["path"])
    for row in manifest["artifacts"]:
        path = repo / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            mismatches.append(row["path"])
    tables = 0
    for path in (repo / "analysis/posthoc").glob("*.parquet"):
        table_metadata = pq.read_schema(path).metadata
        if table_metadata is None or b"mosaic_posthoc_metadata" not in table_metadata:
            mismatches.append(path.relative_to(repo).as_posix())
            continue
        metadata = json.loads(table_metadata[b"mosaic_posthoc_metadata"])
        if (
            metadata["source_commit"] != SOURCE_COMMIT
            or metadata["protected_scoring_rerun"]
            or metadata["canonical_p2_mutated"]
            or metadata["row_count"] != pq.read_metadata(path).num_rows
        ):
            mismatches.append(path.relative_to(repo).as_posix())
        tables += 1
    stage = load_json(repo / "evidence/p2/stage_status.json")
    warehouse = validate_warehouse(repo, repo / "evidence/p2/warehouse.duckdb")
    canonical_diff = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            SOURCE_COMMIT,
            "--",
            "evidence/p2",
            "configs/p2",
            "reports/p2",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = "PASS"
    if (
        mismatches
        or canonical_diff
        or stage["scientific_state"] != "CLOSED"
        or stage["final_decision"] != "RETAIN_BASELINE"
        or stage["qualification_access_count"] != 1
        or stage["locked_final_access_count"] != 1
        or stage["p1_canonical_receipts_included"] != 0
        or warehouse["status"] != "PASS"
        or warehouse["p1_canonical_receipts"] != 0
    ):
        status = "FAIL"
    return {
        "status": status,
        "artifact_count": manifest["artifact_count"],
        "parquet_table_count": tables,
        "mismatches": mismatches,
        "canonical_p2_diff": canonical_diff,
        "qualification_access_count": stage["qualification_access_count"],
        "locked_final_access_count": stage["locked_final_access_count"],
        "warehouse_status": warehouse["status"],
        "warehouse_p1_receipts": warehouse["p1_canonical_receipts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    result = validate(parser.parse_args().repo.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
