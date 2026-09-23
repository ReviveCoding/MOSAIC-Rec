"""Summarize and select the preregistered p2 alignment comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.alignment_summary import summarize_alignment
from mosaic_rec.p2.contracts import verify_p1_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    result = summarize_alignment(
        repo=args.repo,
        manifest_path=args.repo / "configs/p2/alignment_trials_manifest.json",
        implementation_path=args.repo / "configs/p2/alignment_implementation_001.json",
        output=args.output,
    )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 alignment summary")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
