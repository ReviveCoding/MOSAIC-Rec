"""Run one p2 BPR-MF, SASRec, or VBPR CUDA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.additional_baselines import run_baseline
from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=["BPR_MF", "LIGHTGCN", "SASREC", "VBPR"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.checkpoint, args.output, args.receipt):
        if path.exists():
            raise FileExistsError(path)
    config = json.loads(args.config.read_text())
    model_id = config["models"][args.baseline]["model_id"]
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, model_id) as qualification:
        result = run_baseline(
            repo=args.repo,
            baseline=args.baseline,
            config_path=args.config,
            fit_path=args.fit,
            query_path=args.queries,
            context_dir=args.contexts,
            feature_dir=args.features,
            checkpoint=args.checkpoint,
            output=args.output,
            receipt=args.receipt,
            gpu_qualification=qualification,
        )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 baseline pilot")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
