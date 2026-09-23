"""Run one frozen p2 E01 independent-seed trial on VALIDATION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.additional_baselines import run_baseline
from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable
from mosaic_rec.p2.gate_ladder import train_and_score_gate
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.legacy import train_and_score_legacy
from mosaic_rec.p2.router_hpo_configs import ROUTER_FAMILIES
from mosaic_rec.p2.runtime import train_content, train_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    run_id = args.run_id.upper()
    manifest = json.loads(args.manifest.read_text())
    identities = {run["run_id"]: run for run in manifest["runs"]}
    if run_id not in identities:
        raise ValueError(f"Unknown frozen E01 run: {run_id}")
    identity = identities[run_id]
    config_path = Path(identity["path"])
    if not config_path.is_absolute():
        config_path = args.repo / config_path
    if sha256_file(config_path) != identity["sha256"]:
        raise RuntimeError(f"E01 config hash mismatch: {run_id}")
    config = json.loads(config_path.read_text())
    family = config["family"]
    stem = run_id.lower()
    checkpoint = args.scratch / f"seed_qualification/checkpoints/{stem}.pt"
    output = args.scratch / f"seed_qualification/results/{stem}_validation.parquet"
    receipt = args.repo / f"evidence/p2/p2-e01-{stem}.json"
    for path in (checkpoint, output, receipt):
        if path.exists():
            raise FileExistsError(f"Preserve existing E01 artifact: {path}")
    hpo = json.loads((args.repo / "configs/p2/hpo.json").read_text())
    policy = {
        "early_stopping_patience": int(hpo["early_stopping_patience"]),
        "maximum_gpu_hours_per_trial": float(hpo["maximum_gpu_hours_per_trial"]),
        "hpo_manifest_sha256": sha256_file(args.repo / "configs/p2/hpo.json"),
    }
    common = {
        "repo": args.repo,
        "config_path": config_path,
        "fit_path": args.scratch / "data/pixelrec50k/FIT_VISIBLE.parquet",
        "query_path": args.scratch / "contexts/pixelrec50k/VALIDATION_QUERIES.parquet",
        "context_dir": args.scratch / "contexts/pixelrec50k",
        "feature_dir": args.scratch / "features/pixelrec50k",
        "checkpoint": checkpoint,
        "output": output,
        "receipt": receipt,
    }
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, run_id) as qualification:
        if family == "CSEQ_BERT4REC":
            result = train_sequence(
                config=config,
                gpu_qualification=qualification,
                hpo_policy=policy,
                **common,
            )
        elif family == "CCOLD_TWO_TOWER":
            result = train_content(
                config=config,
                gpu_qualification=qualification,
                hpo_policy=policy,
                **common,
            )
        elif family == "LEGACY_M1":
            result = train_and_score_legacy(
                gpu_qualification=qualification,
                hpo_policy=policy,
                **common,
            )
        elif family in {"BPR_MF", "LIGHTGCN", "SASREC", "VBPR"}:
            result = run_baseline(
                baseline=family,
                gpu_qualification=qualification,
                hpo_policy=policy,
                **common,
            )
        elif family in ROUTER_FAMILIES:
            result = train_and_score_gate(
                repo=args.repo,
                router_name=family,
                ladder_path=config_path,
                pilot_path=args.repo / "configs/p2/development_pilots_attempt2.json",
                fit_path=common["fit_path"],
                query_path=common["query_path"],
                context_dir=common["context_dir"],
                feature_dir=common["feature_dir"],
                checkpoint=checkpoint,
                output=output,
                receipt=receipt,
                gpu_qualification=qualification,
                hpo_policy=policy,
            )
        else:
            raise ValueError(f"Unsupported E01 family: {family}")
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 E01 trial")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
