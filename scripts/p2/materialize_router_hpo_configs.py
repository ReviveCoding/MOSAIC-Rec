"""Materialize router trials from frozen p2 HPO and completed open-data selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable, write_json_new
from mosaic_rec.p2.router_hpo_configs import (
    materialize_router_configs,
    write_router_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pre-router-summary", type=Path, required=True)
    parser.add_argument("--alignment-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    hpo_path = args.repo / "configs/p2/hpo.json"
    configs = materialize_router_configs(
        hpo=json.loads(hpo_path.read_text()),
        ladder=json.loads((args.repo / "configs/p2/router_ladder.json").read_text()),
        pre_router_summary=json.loads(args.pre_router_summary.read_text()),
        alignment_summary=json.loads(args.alignment_summary.read_text()),
    )
    result = write_router_configs(
        output_dir=args.output_dir,
        configs=configs,
        hpo_path=hpo_path,
        pre_router_summary_path=args.pre_router_summary,
        alignment_summary_path=args.alignment_summary,
    )
    write_json_new(args.manifest, result)
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 router HPO materialization")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
