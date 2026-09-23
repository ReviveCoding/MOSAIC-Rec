"""Materialize p2 E01 independent-seed configs from completed HPO summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable, write_json_new
from mosaic_rec.p2.seed_configs import materialize_seed_configs, write_seed_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--pre-router-summary", type=Path, required=True)
    parser.add_argument("--router-manifest", type=Path, required=True)
    parser.add_argument("--router-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    paths = {
        "hpo": args.repo / "configs/p2/hpo.json",
        "pre_router_manifest": args.repo / "configs/p2/hpo_trials_manifest.json",
        "pre_router_summary": args.pre_router_summary,
        "router_manifest": args.router_manifest,
        "router_summary": args.router_summary,
    }
    loaded = {name: json.loads(path.read_text()) for name, path in paths.items()}
    configs = materialize_seed_configs(
        repo=args.repo,
        scratch=args.scratch,
        hpo=loaded["hpo"],
        pre_router_manifest=loaded["pre_router_manifest"],
        pre_router_summary=loaded["pre_router_summary"],
        router_manifest=loaded["router_manifest"],
        router_summary=loaded["router_summary"],
    )
    result = write_seed_configs(
        output_dir=args.output_dir,
        configs=configs,
        source_hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    write_json_new(args.manifest, result)
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 independent-seed materialization")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
