"""Summarize the completed p2 E01 independent-seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.seed_summary import summarize_seed_qualification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    result = summarize_seed_qualification(
        repo=args.repo, manifest_path=args.manifest, output=args.output
    )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 E01 summary")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
