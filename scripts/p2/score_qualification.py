"""Score one frozen p2 ensemble on QUALIFICATION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.gpu import heavy_gpu_lock
from mosaic_rec.p2.qualification import score_qualification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    family = args.family.upper()
    stem = family.lower()
    output = args.scratch / f"qualification/{stem}_topk.parquet"
    receipt = args.repo / f"evidence/p2/p2-e02-qualification-{stem}.json"
    before = verify_p1_immutable(args.repo)
    with heavy_gpu_lock(args.lock, f"P2-QUAL-{family}") as qualification:
        result = score_qualification(
            repo=args.repo,
            scratch=args.scratch,
            family=family,
            output=output,
            receipt=receipt,
            gpu_qualification=qualification,
        )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 QUALIFICATION scoring")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
