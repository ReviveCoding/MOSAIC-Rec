"""Summarize the completed frozen p2 pre-router HPO trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable
from mosaic_rec.p2.hpo_summary import summarize_pre_router_hpo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    result = summarize_pre_router_hpo(
        repo=args.repo,
        manifest_path=args.repo / "configs/p2/hpo_trials_manifest.json",
        hpo_path=args.repo / "configs/p2/hpo.json",
        output=args.output,
    )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 HPO summary")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
