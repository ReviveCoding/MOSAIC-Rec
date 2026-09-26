"""Score one E05-frozen p2 ensemble on LOCKED_FINAL exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.locked import score_locked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    family = args.family.upper()
    stem = family.lower()
    output = args.scratch / f"locked_final/{stem}_topk.parquet"
    receipt = args.repo / f"evidence/p2/p2-f00-locked-{stem}.json"
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, f"P2-LOCKED-{family}") as qualification:
        result = score_locked(
            repo=args.repo,
            scratch=args.scratch,
            family=family,
            output=output,
            receipt=receipt,
            gpu_qualification=qualification,
        )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 LOCKED_FINAL scoring")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
