"""Prepare the fresh PixelRec p2 data roles and p2-only receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.data import prepare_pixelrec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    p1 = verify_p1_immutable(repo)
    receipts = prepare_pixelrec(
        repo=repo,
        source=args.source,
        item_source=args.items,
        scratch=args.scratch,
    )
    if verify_p1_immutable(repo) != p1:
        raise RuntimeError("p1 hashes changed while preparing p2")
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
