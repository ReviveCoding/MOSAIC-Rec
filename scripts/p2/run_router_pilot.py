"""Run the preregistered p2 R1 hard zero-support router pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.router import score_r1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists():
            raise FileExistsError(path)
    repo = args.repo.resolve()
    before = verify_p1_immutable(repo)
    config = json.loads(args.config.read_text())
    with heavy_gpu_lock(args.lock, config["router"]["model_id"]) as qualification:
        result = score_r1(
            repo=repo,
            config_path=args.config,
            pilot_config_path=args.pilot_config,
            fit_path=args.fit,
            query_path=args.queries,
            context_dir=args.contexts,
            feature_dir=args.features,
            output=args.output,
            receipt=args.receipt,
            gpu_qualification=qualification,
        )
    if before != verify_p1_immutable(repo):
        raise RuntimeError("p1 changed during p2 R1 pilot")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
