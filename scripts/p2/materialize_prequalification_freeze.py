"""Materialize the p2 E02 candidate and pre-QUALIFICATION freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable, write_json_new
from mosaic_rec.p2.prequalification import build_prequalification_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    result = build_prequalification_freeze(args.repo)
    write_json_new(args.output, result)
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 pre-QUALIFICATION freeze")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
