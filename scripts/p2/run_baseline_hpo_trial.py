"""Run one frozen p2 additional-baseline HPO trial with exact early stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.additional_baselines import run_baseline
from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.hpo_configs import ADDITIONAL_FAMILIES, validate_trial_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_id")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    trial_id = args.trial_id.upper()
    config_path = args.repo / f"configs/p2/hpo_trials/{trial_id.lower()}.json"
    manifest_path = args.repo / "configs/p2/hpo_trials_manifest.json"
    hpo_path = args.repo / "configs/p2/hpo.json"
    identity = validate_trial_identity(
        manifest=json.loads(manifest_path.read_text()),
        trial_id=trial_id,
        config_path=config_path,
        hpo_path=hpo_path,
    )
    family = identity["family"]
    if family not in ADDITIONAL_FAMILIES:
        raise ValueError(f"Not an additional-baseline HPO trial: {trial_id}")
    hpo = json.loads(hpo_path.read_text())
    checkpoint = args.scratch / f"hpo/checkpoints/{trial_id.lower()}.pt"
    output = args.scratch / f"hpo/results/{trial_id.lower()}_validation.parquet"
    receipt = args.repo / f"evidence/p2/p2-e00-{trial_id.lower()}.json"
    for path in (checkpoint, output, receipt):
        if path.exists():
            raise FileExistsError(f"Preserve existing HPO artifact: {path}")
    before = verify_p1_immutable(args.repo)
    policy = {
        "early_stopping_patience": int(hpo["early_stopping_patience"]),
        "maximum_gpu_hours_per_trial": float(hpo["maximum_gpu_hours_per_trial"]),
        "hpo_manifest_sha256": sha256_file(hpo_path),
    }
    with heavy_gpu_lock(args.lock, trial_id) as qualification:
        result = run_baseline(
            repo=args.repo,
            baseline=family,
            config_path=config_path,
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
        raise RuntimeError("p1 changed during p2 baseline HPO trial")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
