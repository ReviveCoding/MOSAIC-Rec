"""Apply the frozen p2 primary gates to immutable LOCKED_FINAL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import (
    load_json,
    sha256_file,
    verify_p1_immutable,
    write_json_new,
)
from mosaic_rec.p2.locked_analysis import analyze_locked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    manifest = load_json(args.repo / "evidence/p2/e05_freeze_manifest.json")
    paths = {
        family: Path(path)
        for family, path in manifest["locked_evaluation"]["outputs"].items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis = analyze_locked(
        cseq_path=paths["CSEQ_BERT4REC"],
        ccold_path=paths["CCOLD_TWO_TOWER"],
        router_path=paths["R4"],
        margin=float(manifest["warm_noninferiority_margin"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-F00-F02",
        "status": "PASS_ANALYSIS",
        "role": "LOCKED_FINAL",
        "qualification_access_count": 1,
        "locked_final_access_count": 1,
        "e05_manifest_sha256": sha256_file(
            args.repo / "evidence/p2/e05_freeze_manifest.json"
        ),
        "input_sha256": {family: sha256_file(path) for family, path in paths.items()},
        "bootstrap_draws": 10_000,
        "user_bootstrap_seed": 2026092301,
        "item_bootstrap_seed": 2026092302,
        **analysis,
        "evidence_class": "p2_locked_confirmatory",
    }
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 locked analysis")
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
