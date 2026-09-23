"""Run one preregistered p2 score-alignment trial on VALIDATION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.router import score_r1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment_id")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    alignment_id = args.alignment_id.upper()
    manifest_path = args.repo / "configs/p2/alignment_trials_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identities = {trial["alignment_id"]: trial for trial in manifest["trials"]}
    if alignment_id not in identities:
        raise ValueError(f"Unknown frozen alignment identity: {alignment_id}")
    identity = identities[alignment_id]
    config_path = args.repo / identity["path"]
    if sha256_file(config_path) != identity["sha256"]:
        raise RuntimeError(f"Alignment config hash mismatch: {alignment_id}")
    output = args.scratch / f"alignment/{alignment_id.lower()}_validation.parquet"
    receipt = args.repo / f"evidence/p2/p2-b03-{alignment_id.lower()}.json"
    for path in (output, receipt):
        if path.exists():
            raise FileExistsError(f"Preserve existing alignment artifact: {path}")
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, alignment_id) as qualification:
        result = score_r1(
            repo=args.repo,
            config_path=config_path,
            pilot_config_path=args.repo / "configs/p2/development_pilots_attempt2.json",
            fit_path=args.scratch / "data/pixelrec50k/FIT_VISIBLE.parquet",
            query_path=args.scratch / "contexts/pixelrec50k/VALIDATION_QUERIES.parquet",
            context_dir=args.scratch / "contexts/pixelrec50k",
            feature_dir=args.scratch / "features/pixelrec50k",
            output=output,
            receipt=receipt,
            gpu_qualification=qualification,
        )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 alignment trial")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
