"""Run one frozen p2 router HPO trial with exact early stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable
from mosaic_rec.p2.gate_ladder import train_and_score_gate
from mosaic_rec.p2.gpu import heavy_gpu_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_id")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    trial_id = args.trial_id.upper()
    manifest = json.loads(args.manifest.read_text())
    identities = {trial["trial_id"]: trial for trial in manifest["trials"]}
    if trial_id not in identities:
        raise ValueError(f"Unknown frozen router trial: {trial_id}")
    identity = identities[trial_id]
    config_path = Path(identity["path"])
    if not config_path.is_absolute():
        config_path = args.repo / config_path
    if sha256_file(config_path) != identity["sha256"]:
        raise RuntimeError(f"Router HPO config hash mismatch: {trial_id}")
    config = json.loads(config_path.read_text())
    family = config["family"]
    hpo_path = args.repo / "configs/p2/hpo.json"
    hpo = json.loads(hpo_path.read_text())
    if sha256_file(hpo_path) != manifest["hpo_sha256"]:
        raise RuntimeError("Frozen HPO hash mismatch")
    checkpoint = args.scratch / f"hpo/checkpoints/{trial_id.lower()}.pt"
    output = args.scratch / f"hpo/results/{trial_id.lower()}_validation.parquet"
    receipt = args.repo / f"evidence/p2/p2-e00-{trial_id.lower()}.json"
    for path in (checkpoint, output, receipt):
        if path.exists():
            raise FileExistsError(f"Preserve existing router HPO artifact: {path}")
    policy = {
        "early_stopping_patience": int(hpo["early_stopping_patience"]),
        "maximum_gpu_hours_per_trial": float(hpo["maximum_gpu_hours_per_trial"]),
        "hpo_manifest_sha256": sha256_file(hpo_path),
    }
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, trial_id) as qualification:
        result = train_and_score_gate(
            repo=args.repo,
            router_name=family,
            ladder_path=config_path,
            pilot_path=args.repo / "configs/p2/development_pilots_attempt2.json",
            fit_path=args.scratch / "data/pixelrec50k/FIT_VISIBLE.parquet",
            query_path=args.scratch / "contexts/pixelrec50k/VALIDATION_QUERIES.parquet",
            context_dir=args.scratch / "contexts/pixelrec50k",
            feature_dir=args.scratch / "features/pixelrec50k",
            checkpoint=checkpoint,
            output=output,
            receipt=receipt,
            gpu_qualification=qualification,
            hpo_policy=policy,
        )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 router HPO trial")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
