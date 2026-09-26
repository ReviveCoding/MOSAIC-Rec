"""Materialize the deterministic p2 E04 analysis-family freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file, write_json_new
from mosaic_rec.p2.final_freeze import build_e04_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_e04_manifest(args.repo)
    write_json_new(args.output, result)
    write_json_new(
        args.receipt,
        {
            "protocol_id": result["protocol_id"],
            "stage_id": "P2-E04",
            "status": "PASS_FREEZE",
            "manifest_path": args.output.as_posix(),
            "manifest_sha256": sha256_file(args.output),
            "qualification_access_count": 1,
            "locked_final_access_count": 0,
            "protected_roles_opened": ["QUALIFICATION"],
            "evidence_class": "p2_frozen_analysis_definition",
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
