"""Fail-closed immutable source audit for industry diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from common import (
    SOURCE_COMMIT,
    SOURCE_PROTOCOL,
    load_json,
    metadata,
    sha256_file,
    write_json_new,
)

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.warehouse import validate


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def audit(repo: Path, scratch: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
        cwd=repo,
        check=True,
    )
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    if branch != "posthoc/industry-diagnostics":
        raise RuntimeError(f"Wrong branch: {branch}")
    stage = load_json(repo / "evidence/p2/stage_status.json")
    if (
        stage["protocol_id"] != SOURCE_PROTOCOL
        or stage["scientific_state"] != "CLOSED"
        or stage["final_decision"] != "RETAIN_BASELINE"
        or stage["qualification_access_count"] != 1
        or stage["locked_final_access_count"] != 1
    ):
        raise RuntimeError("Canonical p2 closed-state contract failed")
    warehouse = validate(repo, repo / "evidence/p2/warehouse.duckdb")
    if warehouse["status"] != "PASS" or warehouse["p1_canonical_receipts"] != 0:
        raise RuntimeError("Canonical warehouse audit failed")
    p1 = verify_p1_immutable(repo)

    records: list[dict[str, Any]] = []
    canonical = [
        "evidence/p2/stage_status.json",
        "evidence/p2/warehouse.duckdb",
        "evidence/p2/warehouse_manifest.json",
        "evidence/p2/claims.json",
        "reports/p2/FINAL_STATUS.md",
        "evidence/p2/p2-f00-f02-locked-summary.json",
        "evidence/p2/p2-g00-robustness-summary.json",
        "evidence/p2/p2-j00-j05-systems-summary.json",
    ]
    records.extend(_record(repo / relative) for relative in canonical)
    locked_receipts = {
        "C_seq": repo / "evidence/p2/p2-f00-locked-cseq_bert4rec.json",
        "C_cold": repo / "evidence/p2/p2-f00-locked-ccold_two_tower.json",
        "R4": repo / "evidence/p2/p2-f00-locked-r4.json",
    }
    for receipt_path in locked_receipts.values():
        receipt = load_json(receipt_path)
        output = Path(receipt["locked_final"]["output"])
        if sha256_file(output) != receipt["locked_final"]["output_sha256"]:
            raise RuntimeError(f"Locked artifact hash mismatch: {output}")
        records.extend((_record(receipt_path), _record(output)))

    robustness = load_json(repo / "configs/p2/e04_analysis_manifest.json")[
        "robustness"
    ]["cells"]
    for cell in robustness:
        stem = (
            f"{str(cell['intervention']).lower()}_{float(cell['intensity']):g}".replace(
                ".", "p"
            )
        )
        receipt_path = repo / f"evidence/p2/p2-g00-{stem}.json"
        receipt = load_json(receipt_path)
        output = Path(receipt["locked_final"]["output"])
        if sha256_file(output) != receipt["locked_final"]["output_sha256"]:
            raise RuntimeError(f"Robustness hash mismatch: {stem}")
        records.extend((_record(receipt_path), _record(output)))

    scratch_inputs = [
        scratch / "contexts/pixelrec50k/LOCKED_FINAL_QUERIES.parquet",
        scratch / "contexts/pixelrec50k/item_context.parquet",
        scratch / "features/pixelrec50k/publisher_image_features.npy",
        scratch / "features/pixelrec50k/publisher_image_item_ids.txt",
        scratch / "features/pixelrec50k/authorized_minilm_text_features.npy",
        scratch / "features/pixelrec50k/authorized_minilm_item_ids.txt",
        scratch / "features/pixelrec50k/tag_ids.npy",
        scratch / "features/pixelrec50k/image_present.npy",
        scratch / "features/pixelrec50k/text_present.npy",
        scratch / "features/pixelrec50k/tag_vocabulary.json",
    ]
    records.extend(_record(path) for path in scratch_inputs)
    unique = {row["path"]: row for row in records}
    return list(unique.values()), {
        "head": head,
        "branch": branch,
        "warehouse": warehouse,
        "p1_immutable_hashes": p1,
        "protected_process_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, validation = audit(args.repo, args.scratch)
    source_hashes = {row["path"]: row["sha256"] for row in records}
    result = {
        **metadata(
            script=Path(__file__), source_hashes=source_hashes, row_count=len(records)
        ),
        "status": "PASS",
        "analysis_type": "POSTHOC_EXPLORATORY",
        "confirmatory": False,
        "may_change_p2_decision": False,
        "inputs": records,
        "validation": validation,
    }
    write_json_new(args.output, result)
    print(json.dumps({"status": "PASS", "inputs": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
