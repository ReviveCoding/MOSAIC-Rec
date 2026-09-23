"""Build and validate the p2-only canonical evidence warehouse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import replace_json, verify_p1_immutable
from mosaic_rec.p2.warehouse import build, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    result = build(args.repo, args.warehouse, args.table_dir)
    result["validation"] = validate(args.repo, args.warehouse)
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed while building p2 warehouse")
    if args.manifest is not None:
        replace_json(args.manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
