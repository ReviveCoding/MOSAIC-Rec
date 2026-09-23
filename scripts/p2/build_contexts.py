"""Build FIT-only p2 contexts and common role query manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contexts import build_contexts
from mosaic_rec.p2.contracts import verify_p1_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    before = verify_p1_immutable(repo)
    receipt = build_contexts(
        repo=repo,
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
    )
    if before != verify_p1_immutable(repo):
        raise RuntimeError("p1 changed during p2 context construction")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
