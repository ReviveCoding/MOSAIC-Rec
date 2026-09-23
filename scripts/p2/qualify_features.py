"""Materialize and qualify p2-owned PixelRec content feature identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.features import qualify_pixelrec_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-index", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--text-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    before = verify_p1_immutable(repo)
    receipts = qualify_pixelrec_features(
        repo=repo,
        item_source=args.items,
        image_source=args.image,
        image_index_source=args.image_index,
        text_source=args.text,
        text_index_source=args.text_index,
        output_dir=args.output_dir,
    )
    if before != verify_p1_immutable(repo):
        raise RuntimeError("p1 hashes changed during p2 feature qualification")
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
