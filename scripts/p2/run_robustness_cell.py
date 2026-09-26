"""Run one frozen p2 robustness cell under the heavy-GPU lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.robustness_scoring import score_robustness_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("intervention")
    parser.add_argument("intensity", type=float)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    stem = f"{args.intervention.lower()}_{args.intensity:g}".replace(".", "p")
    output = args.scratch / f"robustness/{stem}_topk.parquet"
    receipt = args.repo / f"evidence/p2/p2-g00-{stem}.json"
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, f"P2-G00-{stem}") as qualification:
        result = score_robustness_cell(
            repo=args.repo,
            scratch=args.scratch,
            intervention=args.intervention,
            intensity=args.intensity,
            output=output,
            receipt=receipt,
            gpu_qualification=qualification,
        )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 robustness scoring")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
