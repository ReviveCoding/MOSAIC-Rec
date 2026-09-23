"""Run the p2 legacy M1 control with the exclusive CUDA lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.legacy import train_and_score_legacy


def main() -> None:
    parser = argparse.ArgumentParser()
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
    before = verify_p1_immutable(args.repo)
    config = json.loads(args.config.read_text())
    with heavy_gpu_lock(args.lock, config["model_id"]) as qualification:
        result = train_and_score_legacy(
            repo=args.repo,
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
        raise RuntimeError("p1 changed during p2 legacy pilot")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
